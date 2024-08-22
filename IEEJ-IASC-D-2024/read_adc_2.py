"""
BMFに流れる電流とBMFにかかる電圧センサデータを連続収集して抵抗値を求めCSVに保存
"""
# sample数取ってから平均を取る
"""
sudo groupadd gpio
sudo usermod -a -G gpio -G spi pi # assumes the pi user
sudo chown root:spi /dev/spidev*
sudo chmod g+rw /dev/spidev*
"""


import os
import sys
import time
import numpy as np
import datetime
from enum import IntEnum
from itertools import starmap

import CTA_feedback.module.skyfish.mcp3208 as adc

isSave = True # csvに保存するかどうか
Vref = 5.0 # 電源電圧
Offset = 2.5 # オフセット電圧
read_samples = 10 # サンプル数 500
Vthreshold = 0.1 # 電圧の閾値

# 画面クリア
os.system("clear")
# ロゴ表示
os.system("paste ./skyfish/pilogo.txt ./skyfish/bmflogo.txt | lolcat")


# チャンネル定義
class channnel(IntEnum):
    # Configure ADC settings
    BMF_current = 0 # BMFに流れる電流
    BMF_voltage = 2 # BMFにかかる電圧
    v33_voltage = 4 # 3.3V電源電圧
    v50_voltage = 6 # 5.0V電源電圧


# 5Vでは2MHzが限界?
try: 
    adc.open(port=0, cs=0, speed=7800000) # 7.8 MHz が限界
# 例外処理(権限がない場合)
except PermissionError:
    os.system("sudo groupadd gpio")
    os.system("sudo usermod -a -G gpio -G spi pi")
    os.system("sudo chown root:spi /dev/spidev*")
    os.system("sudo chmod g+rw /dev/spidev*")
    adc.open(port=0, cs=0, speed=7800000) # 7.8 MHz が限界

# アナログ入力を読み取る関数
def analog_input_read(samples=1):
    """  
    BMF_CVのアナログ入力を読み取る関数です。

    Parameters:
        samples (int): 各チャンネルごとのサンプル数 (デフォルトは1)

    Returns:
        tuple: 読み取った電流と電圧の値のタプル

    """
    read_voltage = np.fromiter(starmap(adc.read_voltage, zip(
        [channnel.BMF_voltage] * samples,
        [adc.convtype.diff] * samples,
        [Vref] * samples
    )), dtype=np.float64, count=samples)

    read_current = np.fromiter(starmap(adc.read_voltage, zip(
        [channnel.BMF_current] * samples, 
        [adc.convtype.sgl] * samples,
        [Vref] * samples
    )), dtype=np.float64, count=samples)

    return read_voltage, read_current

# 現在時刻取得
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(offset=t_delta, name='JST')
now = datetime.datetime.now(JST)

# CSVファイル名
csv_filename = "RPI_" + now.strftime('%Y-%m-%d_%H-%M-%S') + "_MEAN" + str(read_samples)
print("\033[?25l保存CSVファイルン名: " + csv_filename + ".csv")

# CSVヘッダー
csv_header = ["elapsed_S", "supply_V", "current_A", "BMF_resistance"]

try:
    # 空の numpy 配列を作成
    csv_data = np.empty((0, len(csv_header)), dtype=np.float64)
    # 開始時間
    start = time.perf_counter()
    
    while True:
        # 経過時間
        elapse = time.perf_counter() - start
        # 電竜センサと電源の電圧値を取得
        voltage, current_v = analog_input_read(samples=read_samples)
        
        # 電圧の閾値を下回る値を削除
        v0_index = np.where(voltage <= Vthreshold)[0]
        del_index = np.unique((v0_index, 
                               [num - 1 for num in v0_index], 
                               [num + 1 for num in v0_index]))
        del_index = [i for i in del_index if 0 <= i < len(voltage)]
        
        del_voltage = np.delete(voltage, del_index) if len(del_index) != len(voltage) else np.zeros(1)
        del_current_v = np.delete(current_v, del_index) if len(del_index) != len(current_v) else np.zeros(1)
        
        if np.array_equal(del_voltage, np.zeros(1)):
            continue
        else:
            # print(" ", v0_index, del_index, del_voltage, end="")

            # 電竜値に変換
            del_current = (del_current_v - Offset) / Vref

            # 平均値算出(0以下の値は0にする)
            mean_voltage = max(0, np.mean(del_voltage))
            mean_current = max(0, np.mean(del_current))
            # 抵抗値の算出 (R = V / I)
            mean_resistance = mean_voltage / mean_current if mean_current != 0 else 0
            
            # 保存データ作成
            data = np.array([[elapse, mean_voltage, mean_current, mean_resistance]], dtype=np.float64)

            # CSVキャッシュ書き込み
            csv_data = np.concatenate((csv_data, data), axis=0)  # numpy 配列に行を追加
            
            print("\rS:" + f'{elapse:4.2f}' + "s",
                "CSA:" + f'{len(del_current)}' + f'{mean_current:+1.10f}' + "A",
                "BMF:" + f'{len(del_voltage)}' + f'{mean_voltage:+1.10f}' + "V", end="")
        # time.sleep(1)

except KeyboardInterrupt:
    print("\nADCを閉じています.")
    adc.close()
    # 経過時間の最初を0にする
    csv_data[:, 0] = csv_data[:, 0] - csv_data[0][0]
    # CSVに保存
    if isSave:
        print(csv_filename + ".csvに保存中.")
        np.savetxt(f'./{csv_filename}.csv', np.array(csv_data), delimiter=',', header= ','.join(csv_header), comments='')
        print("「\033[31m" + csv_filename + ".csv\033[0m」に保存しました.")
    print('終了\033[?25h')
