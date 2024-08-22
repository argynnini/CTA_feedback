"""
BMFに流れる電流とBMFにかかる電圧センサデータを連続収集して抵抗値を求めCSVに保存
"""

"""
sudo groupadd gpio
sudo usermod -a -G gpio -G spi pi # assumes the pi user
sudo chown root:spi /dev/spidev*
sudo chmod g+rw /dev/spidev*
"""

import os
from re import T
import time
import numpy as np
import datetime
from enum import IntEnum
from itertools import starmap

import CTA_feedback.module.skyfish.mcp3208 as adc

isSave = True # csvに保存するかどうか
Vref = 5.0 # 電源電圧
Offset = 2.5 # オフセット電圧
samples = 500 # サンプル数 500


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


# def analog_input_read(number_of_samples_per_channel = 500):
#     BMF_current = []
#     BMF_voltage = []
#     # v33_voltage = []
#     # v50_voltage = []
#     BMF_current = [adc.read_voltage(channnel.BMF_current, adc.convtype.sgl, Vref) for i in range(number_of_samples_per_channel)]
#     BMF_voltage = [adc.read_voltage(channnel.BMF_voltage, adc.convtype.diff, Vref) for i in range(number_of_samples_per_channel)]
#     # v33_voltage.append(adc.read_channel_voltage(channnel.v33_voltage, adc.convtype.diff, Vref))
#     # v50_voltage.append(adc.read_channel_voltage(channnel.v50_voltage, adc.convtype.diff, Vref))
#     return([BMF_current, BMF_voltage])

# アナログ入力を読み取る関数
def analog_input_read(number_of_samples_per_channel=1):
    """
    BMF_CVのアナログ入力を読み取る関数です。

    Parameters:
        number_of_samples_per_channel (int): 各チャンネルごとのサンプル数 (デフォルトは1)

    Returns:
        tuple: 読み取った電流と電圧の値のタプル

    """
    read_current = np.fromiter(starmap(adc.read_voltage, zip(
        [channnel.BMF_current] * number_of_samples_per_channel, 
        [adc.conv.sgl] * number_of_samples_per_channel,
        [Vref] * number_of_samples_per_channel
    )), dtype=np.float64, count=number_of_samples_per_channel)
    
    read_voltage = np.fromiter(starmap(adc.read_voltage, zip(
        [channnel.BMF_voltage] * number_of_samples_per_channel,
        [adc.conv.diff] * number_of_samples_per_channel,
        [Vref] * number_of_samples_per_channel
    )), dtype=np.float64, count=number_of_samples_per_channel)
    
    return read_current, read_voltage

# 現在時刻取得
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(offset=t_delta, name='JST')
now = datetime.datetime.now(JST)

# CSVファイル名
csv_filename = "RPI_" + now.strftime('%Y-%m-%d_%H-%M-%S') + "_MEAN" + str(samples)
print("保存CSVファイルン名: " + csv_filename + ".csv")

# CSVヘッダー
csv_header = ["elapsed_S", "supply_V", "current_A", "BMF_resistance"]

try:
    # 空の numpy 配列を作成
    csv_data = np.empty((0, len(csv_header)), dtype=np.float64)
    # 開始時間
    start = time.perf_counter()
    
    while True:
        # 経過時間
        elapsed_time = time.perf_counter() - start
        # 電竜センサと電源の電圧値を取得
        current_voltage_data, voltage_data = analog_input_read(number_of_samples_per_channel=samples)

        # 電圧が0のときはスキップ
        # zero_voltage_indices = np.where(voltage_data == 0)[0]
        # voltage_data = voltage_data[zero_voltage_indices]
        # current_voltage_data = current_voltage_data[zero_voltage_indices]

        # 電竜値に変換
        current_data = (current_voltage_data - Offset) / Vref

        # 抵抗値の算出 (R = V / I, I = 0のときは0にする)
        # resistance_data = np.true_divide(voltage_data, current_data, out=np.zeros_like(current_data), where=current_data!=0)

        # 平均値算出
        voltage_data = np.mean(voltage_data)
        # current_voltage_data = np.mean(current_voltage_data)
        current_data = np.mean(current_data)
        # resistance_data = np.mean(resistance_data)
        # 抵抗値の算出 (R = V / I, I = 0のときは0にする)
        resistance_data = voltage_data / current_data if current_data != 0 else 0

        # 保存データ作成
        data = np.array([[elapsed_time, voltage_data, current_data, resistance_data]], dtype=np.float64)
        # print(csv_data)

        # CSVキャッシュ書き込み
        csv_data = np.concatenate((csv_data, data), axis=0)  # numpy 配列に行を追加
        
        print("\rS:" + f'{elapsed_time:4.2f}' + "s",
              "CSA:" + f'{np.mean(current_data):+1.10f}' + "A",
              "BMF:" + f'{np.mean(voltage_data):+1.10f}' + "V", end="")
        # time.sleep(0.1)

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
    print('終了')
