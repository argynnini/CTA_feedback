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
import pigpio
from simple_pid import PID

import mcp3208 as adc
import rotary_code as rc

# --- パラメータ設定 ---
isSave = True # csvに保存するかどうか
Vref = 5.0 # 電源電圧
Offset = 2.5 # オフセット電圧
read_samples = 500 # サンプル数 500
Vthreshold = 1.4 # 電圧の閾値
Kp = -30.0 # 比例ゲイン
Ki = 0.0 # 積分ゲイン
Kd = 0.0 # 微分ゲイン
pwm_freq = 1000
duty_cycle = 50
resistance_range = [7, 12] # 抵抗値の許容範囲(7.5~10.0Ω)
pins = [22, 23, 24, 25] # ロータリコードのピン設定

# --- 定数計算 ---
work_dir = os.getcwd() # 作業ディレクトリのパス

# 抵抗値の範囲
r_range = resistance_range[1] - resistance_range[0]

# 現在時刻取得
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(offset=t_delta, name='JST')
now = datetime.datetime.now(JST)

# --- ロゴ表示 ---
os.system(f"clear && paste {work_dir}/skyfish/pilogo.txt {work_dir}/skyfish/bmflogo.txt | lolcat")


# --- ADC設定 ---
# チャンネル定義
class ch(IntEnum):
    # Configure ADC settings
    BMF_A = 0 # BMFに流れる電流
    BMF_V = 2 # BMFにかかる電圧
    PWR_3V3 = 4 # 3.3V電源電圧
    PWR_5V0 = 6 # 5.0V電源電圧

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

# --- PWM ---
pi = pigpio.pi()
pi.hardware_PWM(12, pwm_freq, int(duty_cycle * 255 / 100))
rc.setup_pin(pi, pins, pigpio.INPUT, pigpio.PUD_DOWN)

# --- PID制御 ---
pid = PID(Kp, Ki, Kd,
          setpoint=((1 - (rc.read_pin(pi, pins) / 9.0)) * r_range) + resistance_range[0],
          output_limits=(10, 80))

# --- CSV ---
# CSVファイル名
csv_filename = "RPI_" + now.strftime('%Y-%m-%d_%H-%M-%S') + "_MEAN" + str(read_samples)
print("\033[?25l保存CSVファイルン名: " + csv_filename + ".csv")

# CSVヘッダー
csv_header = ["elapsed_S", "supply_V", "current_A", "BMF_R", "target_R"]

# --- メイン処理 ---
try:
    # CSV保存用データ
    csv_data = np.empty((0, len(csv_header)), dtype=np.float64)

    # 開始時間
    zero_count = 0
    output = 0
    start = time.perf_counter()
    elapse = 0
    while True:
        pid.setpoint = target_resistance = ((1 - (rc.read_pin(pi, pins) / 9.0)) * r_range) + resistance_range[0]
        # キャッシュデータ (BMF電圧, BMF電流)
        cache_data = np.empty((0, 2), dtype=np.float64)
        while len(cache_data) < read_samples: # サンプル数分だけ取得
            # 電竜と電圧を取得
            voltage = adc.read_voltage(ch.BMF_V, adc.conv.diff, Vref)
            current = (adc.read_voltage(ch.BMF_A, adc.conv.sgl, Vref) - Offset) / Vref

            # 電圧値が閾値以上かつ電流値が0以上の場合
            if (voltage >= Vthreshold) and (current > 0):
                # CSVキャッシュ書き込み
                cache_data = np.concatenate((cache_data, [[voltage, current]]), axis=0)
                zero_count = 0
            else:
                zero_count += 1
                # 電圧値がread_samples回0が連続する場合
                if zero_count >= read_samples:
                    # キャッシュに0を書き込み(データが残ってない場合)
                    if cache_data.size == 0:
                        cache_data = np.zeros((1, 2), dtype=np.float64)
                    zero_count = 0
                    break

            # print(f'\rS:{time.perf_counter() - start:4.2f}s',
            #       f'OHM:{resistance:2.2f}Ω',
            #       f'CSA:{current:+1.10f}A',
            #       f'BMF:{voltage:+1.10f}V', end="")
        # 経過時間
        elapse = time.perf_counter() - start
        # 平均値を取得
        mean = np.mean(cache_data, axis=0)
        # 抵抗値の算出 (R = V / I)
        resistance = np.divide(mean[0], mean[1], out=np.zeros_like(mean[0]), where=mean[1] != 0.0)
        # PID値計算
        output = pid(resistance) # 0-100%
        # PWM制御
        pi.set_PWM_dutycycle(12, int(output * 2.55)) # 0-255

        # 情報開示
        print("\r", f'S:{elapse:4.2f}s', f'PWM:{int(output):03d}',
              f'OHM:{resistance:2.2f}/{target_resistance:2.2f}Ω', end="")

        # CSVキャッシュ書き込み
        csv_data = np.concatenate((csv_data, [[elapse, *mean, resistance, target_resistance]]), axis=0)

except KeyboardInterrupt:
    print("\nADCを閉じています.")
    adc.close()
    pi.set_PWM_dutycycle(12, int(0 * 255 / 100))
    pi.stop()
    # 経過時間の最初を0にする
    csv_data[:, 0] = csv_data[:, 0] - csv_data[0][0]
    # CSVに保存
    if isSave:
        print(csv_filename + ".csvに保存中.")
        np.savetxt(f'{work_dir}/{csv_filename}.csv', np.array(csv_data), delimiter=',', header= ','.join(csv_header), comments='')
        print("「\033[31m" + csv_filename + ".csv\033[0m」に保存しました.")
    print('終了\033[?25h')
