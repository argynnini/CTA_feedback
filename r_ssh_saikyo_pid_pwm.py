"""
BMFに流れる電流とBMFにかかる電圧センサデータを連続収集して抵抗値を求めCSVに保存(Rから呼び出す)
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
import math
import argparse

import mcp3208 as adc
import rotary_code as rc

parser = argparse.ArgumentParser(
    prog="r_ssh_saikyo_pid_pwm.py",
    usage="python r_ssh_saikyo_pid_pwm.py",
    description="BMFに流れる電流とBMFにかかる電圧センサデータを連続収集して抵抗値を求めCSVに保存",
    epilog="end",
    add_help=True,
)


# --- 引数設定 ---
# parser.add_argument(
#     "-s",
#     "--save",
#     action="store_true",
#     help="CSVに保存するかどうか",
# )
parser.add_argument(
    "-s",
    "--read_samples",
    type=int,
    default=500,
    help="サンプル数",
)
parser.add_argument(
    "-v",
    "--vthreshold",
    type=float,
    default=1.0,
    help="電圧の閾値",
)
parser.add_argument(
    "-f",
    "--pwm_freq",
    type=int,
    default=1000,
    help="PWM周波数",
)
parser.add_argument(
    "-d",
    "--duty_range",
    type=int,
    nargs=2,
    default=[7, 100],
    help="PWMの許容範囲",
)
parser.add_argument(
    "-k",
    "--pid",
    type=float,
    nargs=3,
    default=[40.0, 10.0, 0.50],
    help="PIDゲイン",
)
parser.add_argument(
    "-t",
    "--test_distance",
    type=float,
    nargs="+",
    default=[4, 7, 10, 13, 16, 19, 22, 25],
    help="テスト屈曲距離",
)
parser.add_argument(
    "-i",
    "--test_interval",
    type=float,
    default=10,
    help="テスト間隔",
)
parser.add_argument(
    "-a",
    "--resistance_range",
    type=float,
    nargs=2,
    default=[7.4, 9.4],
    help="抵抗値の許容範囲",
)

args = parser.parse_args()

# --- パラメータ設定 ---
# /// PID ///
Kp = args.pid[
    0
]  # 比例ゲイン40 :y = 24.286x + 18.333 : y=41.182e^(0.2377x) : y=4.4643x^2-6.9643x+60
Ki = args.pid[1]  # 積分ゲイン 10
Kd = args.pid[2]  # 微分ゲイン 0.5
# /// PWM ///
pwm_freq = args.pwm_freq  # PWM周波数 1000
test_distance = args.test_distance  # テスト抵抗値
test_resistance = [
    0.39 * math.sqrt(25.36 - x) + 7.4 for x in test_distance
]  # テスト抵抗値
# test_resistance = # list(np.linspace(9.4, 7.4, 10)) # [9.4, 9.0, 8.5, 8.0, 7.5] # テスト抵抗値
test_interval = args.test_interval  # テスト間隔(sec)
resistance_range = args.resistance_range  # 抵抗値の許容範囲(7.5~10.0Ω)
duty_range = args.duty_range  # PWMの許容範囲(0~100%)7
pins = [22, 23, 24, 25]  # ロータリコードのピン設定
# /// その他 ///
isSave = True  # csvに保存するかどうか
read_samples = args.read_samples  # サンプル数 500
Vthreshold = args.vthreshold  # 電圧の閾値(これよりちいさいとデータを取得しない)
Vref = 5.0  # 電源電圧
Offset = 2.5  # オフセット電圧

# --- 定数計算 ---
work_dir = os.getcwd()  # 作業ディレクトリのパス

# 抵抗値の範囲
r_range = resistance_range[1] - resistance_range[0]

# 現在時刻取得
t_delta = datetime.timedelta(hours=9)
JST = datetime.timezone(offset=t_delta, name="JST")
now = datetime.datetime.now(JST)

# --- CSV ---
# CSVファイル名
csv_filename = f'RPI{now.strftime("%y%m%d_%H-%M-%S")}_M{str(read_samples)}_PID{str(Kp)}-{str(Ki)}-{str(Kd)}'
print(csv_filename + ".csv")

# --- ADC設定 ---
# チャンネル定義
class ch(IntEnum):
    # Configure ADC settings
    BMF_A = 0  # BMFに流れる電流
    BMF_V = 2  # BMFにかかる電圧
    PWR_3V3 = 4  # 3.3V電源電圧
    PWR_5V0 = 6  # 5.0V電源電圧


# 5Vでは2MHzが限界?
try:
    adc.open(port=0, cs=0, speed=7800000)  # 7.8 MHz が限界
# 例外処理(権限がない場合)
except PermissionError:
    os.system("sudo groupadd gpio")
    os.system("sudo usermod -a -G gpio -G spi pi")
    os.system("sudo chown root:spi /dev/spidev*")
    os.system("sudo chmod g+rw /dev/spidev*")
    adc.open(port=0, cs=0, speed=7800000)  # 7.8 MHz が限界

# --- PWM ---
pi = pigpio.pi()
try:
    pi.hardware_PWM(12, pwm_freq, int(duty_range[0] * 255 / 100))
except AttributeError:
    print("pigpio initializing...")
    os.system("sudo pigpiod")
    time.sleep(0.1)
    pi = pigpio.pi()
rc.setup_pin(pi, pins, pigpio.INPUT, pigpio.PUD_DOWN)

# --- PID制御 ---
pid = PID(
    -Kp,
    -Ki,
    -Kd,
    setpoint=((1 - (rc.read_pin(pi, pins) / 9.0)) * r_range) + resistance_range[0],
    output_limits=(duty_range[0], duty_range[1]),
)



# CSVヘッダー
csv_header = ["elapsed_S", "supply_V", "current_A", "BMF_R", "target_R", "PWM"]

# --- メイン処理 ---
try:
    # CSV保存用データ
    csv_data = np.empty((0, len(csv_header)), dtype=np.float64)
    print(
        "test resistance data: ",
        test_resistance,
        "take time: ",
        len(test_resistance) * test_interval,
        "s",
    )
    seikaku_count = 0

    # 開始時間
    zero_count = 0
    output = 0
    start = time.perf_counter()
    elapse = 0
    while time.perf_counter() - start < len(test_resistance) * test_interval:
        pid.setpoint = target_resistance = test_resistance[
            int((time.perf_counter() - start) / test_interval) % len(test_resistance)
        ]
        # pid.Kp = -1*41.182 * np.exp(0.2377 * target_resistance)
        # pid.Kp = -4.4643 * (target_resistance ** 2) + 6.9643 * target_resistance - 60
        # キャッシュデータ (BMF電圧, BMF電流)
        cache_data = np.empty((0, 2), dtype=np.float64)
        while len(cache_data) < read_samples:  # サンプル数分だけ取得
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
        # 経過時間
        elapse = time.perf_counter() - start
        # 平均値を取得
        mean = np.mean(cache_data, axis=0)
        # 抵抗値の算出 (R = V / I)
        resistance = np.divide(
            mean[0], mean[1], out=np.zeros_like(mean[0]), where=mean[1] != 0.0
        )
        if abs(resistance - target_resistance) < 0.1:
            seikaku_count += 1
        # PID値計算
        output = pid(resistance)  # 0-100%
        if output is None:
            output = 0.0
        # PWM制御
        pi.set_PWM_dutycycle(12, int(output * 2.55))  # 0-255

        # CSVキャッシュ書き込み
        csv_data = np.concatenate(
            (csv_data, [[elapse, *mean, resistance, target_resistance, output]]), axis=0
        )

    # 終了処理
    print("\nclosing ADC...")
    adc.close()
    pi.set_PWM_dutycycle(12, int(0 * 255 / 100))
    pi.stop()
    # 経過時間の最初を0にする
    csv_data[:, 0] = csv_data[:, 0] - csv_data[0][0]
    # CSVに保存
    if isSave:
        print("saving to " + csv_filename + ".csv")
        np.savetxt(
            f"{work_dir}/{csv_filename}.csv",
            np.array(csv_data),
            delimiter=",",
            header=",".join(csv_header)
            + f',#Kp:{Kp},#Ki:{Ki},#Kd:{Kd},#avg:{read_samples},#freq:{pwm_freq},#duty:[{duty_range[0]}-{duty_range[1]}],#Vth:{Vthreshold},{now.strftime("%Y/%m/%d-%H:%M:%S")}',
            comments="",
        )

except KeyboardInterrupt:
    adc.close()
    pi.set_PWM_dutycycle(12, int(0 * 255 / 100))
    pi.stop()
    # 経過時間の最初を0にする
    csv_data[:, 0] = csv_data[:, 0] - csv_data[0][0]
    # CSVに保存
    if isSave:
        np.savetxt(
            f"{work_dir}/{csv_filename}.csv",
            np.array(csv_data),
            delimiter=",",
            header=",".join(csv_header)
            + f',#Kp:{Kp},#Ki:{Ki},#Kd:{Kd},#avg:{read_samples},#freq:{pwm_freq},#duty:[{duty_range[0]}-{duty_range[1]}],#Vth:{Vthreshold},{now.strftime("%Y/%m/%d-%H:%M:%S")}',
            comments="",
        )
