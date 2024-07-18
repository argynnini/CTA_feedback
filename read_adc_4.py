"""
BMFに流れる電流とBMFにかかる電圧センサデータを連続収集して抵抗値を求めCSVに保存
"""
# 取ってsample数たまったら平均
"""
sudo groupadd gpio
sudo usermod -a -G gpio -G spi pi # assumes the pi user
sudo chown root:spi /dev/spidev*
sudo chmod g+rw /dev/spidev*
"""

import os
import time
import numpy as np
import datetime
from enum import IntEnum
from itertools import starmap

import mcp3208 as adc

isSave = True # csvに保存するかどうか
Vref = 5.0 # 電源電圧
Offset = 2.5 # オフセット電圧
read_samples = 500 # サンプル数 500
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
    elapse = 0
    while True:
        # キャッシュデータ (BMF電圧, BMF電流)
        cache_data = np.empty((0, 2), dtype=np.float64)
        while len(cache_data) < read_samples:
            # 電竜センサと電源の電圧値を取得
            voltage = adc.read_voltage(channnel.BMF_voltage, adc.conv.diff, Vref)
            current_v = adc.read_voltage(channnel.BMF_current, adc.conv.sgl, Vref)
            # 電圧値が閾値以上の場合
            if (voltage >= Vthreshold):
                # 電流値に変換
                current = (current_v - Offset) / Vref

                # CSVキャッシュ書き込み
                cache_data = np.concatenate((cache_data, [[voltage, current]]), axis=0)

                print("\rS:" + f'{elapse:4.2f}' + "s",
                    "CSA:" + f'{current:+1.10f}' + "A",
                    "BMF:" + f'{voltage:+1.10f}' + "V", end="")
        # 経過時間
        elapse = time.perf_counter() - start
        # 平均値を取得(1列目に経過時間を追加)
        mean = np.mean(cache_data, axis=0)
        # 抵抗値の算出 (R = V / I)
        resistance = np.where(mean[1] != 0, mean[0] / mean[1], 0)

        # CSVキャッシュ書き込み
        csv_data = np.concatenate((csv_data, [[elapse, *mean, resistance]]), axis=0)

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
