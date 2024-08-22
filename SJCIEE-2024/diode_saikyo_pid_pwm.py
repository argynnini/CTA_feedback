# PWM 立下りのとき1回を無視する

import CTA_feedback.module.skyfish.mcp3208 as adc
import time
import os
import pigpio
import csv
from simple_pid import PID
import math

Kp = 40.0
Ki = 10.0
Kd = 0.5

diode_current = 0.00963  # 9.63mA
ampR = 1.986  # 1.986 kΩ
ampGain = 1 + 49.4 / ampR

pid = PID(-Kp, -Ki, -Kd, setpoint=0)
pid.output_limits = (0, 30)  # PWM は0~30%の範囲で出力
pid.sample_time = 0.1  # 制御周期は0.1秒
# pid.proportional_on_measurement = True  # 逓減制御

test_distance = [4, 7]  # , 10, 13, 16, 19, 22, 25]
# test_resistance = [
#     0.39 * math.sqrt(25.36 - x) + 7.4 for x in test_distance
# ]  # テスト抵抗値
test_resistance = [9.0, 8.0, 7.5]
test_interval = 5  # テスト抵抗値を変更する間隔

pwm_freq = 1000 # サンプリング周波数
duty_range = [0, 30]  # duty比の範囲

v_thresh = [1.5, 3.0]

# 5Vでは2MHzが限界
try:
    adc.open(port=0, cs=0, speed=2000000)  # 5V のとき 2 MHz が限界
# 例外処理(権限がない場合)
except PermissionError:
    os.system("sudo groupadd gpio")
    os.system("sudo usermod -a -G gpio -G spi pi")
    os.system("sudo chown root:spi /dev/spidev*")
    os.system("sudo chmod g+rw /dev/spidev*")
    adc.open(port=0, cs=0, speed=2000000)  # 5V のとき 2 MHz が限界

try:
    # PWMの設定
    gpio = pigpio.pi()
    gpio.hardware_PWM(12, pwm_freq, int(0 * 255 / 100))
except AttributeError:
    os.system("sudo pigpiod")
    time.sleep(1)
    gpio = pigpio.pi()
    gpio.hardware_PWM(12, pwm_freq, int(0 * 255 / 100))

usevoltage = 2 # 0:生の電圧, 1:移動平均, 2:一次遅れノイズフィルタ

# 移動平均を計算するための設定
window_size = 50  # 移動平均のウィンドウサイズ
voltage_history = []  # 過去の電圧値を保持するリスト
# 一次遅れノイズフィルタの設定
cut_off_freq = 10  # カットオフ周波数 10
tau = 1 / (2 * math.pi * cut_off_freq)  # 時定数

time.sleep(1) # 1秒待つ

# 上限は30%くらい
csv_data = []
voltage = [0.0, 0.0, 0.0]
control = 0
before_time = 0
start_time = time.perf_counter()
elapsed_time = 0
with open("/home/pi/BMF_CV/idiode.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "Voltage", "VoltageMvAvg" + str(window_size), "VoltageFstLag" + str(cut_off_freq), "ResistanceTgt", "Resistance" + str(usevoltage), "PWM"])
    try:
        # 目標抵抗値の設定
        pid.setpoint = test_resistance[0]
        print("target: {0:.2f}Ω v: {1}".format(pid.setpoint, usevoltage))
        while test_interval * len(test_resistance) > elapsed_time:
            pid.setpoint = test_resistance[math.floor(elapsed_time / test_interval)]
            # チャンネル6~7の差分電圧を読み取る
            voltage[0] = adc.read_voltage(channel=6, convtype=0, vref=5.0)
            # 経過時間
            elapsed_time = time.perf_counter() - start_time  # 経過時間
            if (voltage[0] > v_thresh[0] and voltage[0] < v_thresh[1]):  # 1.5V ~ 3.0V の範囲内のとき
                time_diff = elapsed_time - before_time
                # 過去の電圧値を更新
                voltage_history.append(voltage[0])
                if len(voltage_history) > window_size:
                    voltage_history.pop(0)  # 最も古いデータを削除

                # 移動平均を計算
                voltage[1] = sum(voltage_history) / len(voltage_history)
                # print(avg_voltage)

                # 一次遅れノイズフィルタ
                if before_time == 0:
                    voltage[2] = voltage[0]
                # filtered_voltage = alpha * (1 - time_diff / cut_off_freq) * filtered_voltage + time_diff / cut_off_freq * voltage
                voltage[2] = tau / (tau + time_diff) * voltage[2] + time_diff / (tau + time_diff) * voltage[0]
                before_time = elapsed_time

                if (control := pid(voltage[usevoltage] / ampGain / diode_current)) is None:
                    control = 0
                gpio.set_PWM_dutycycle(12, int(control * 255 / 100))
                csv_data.append([elapsed_time, voltage[0], voltage[1], voltage[2], test_resistance[min(math.floor(elapsed_time / test_interval), len(test_resistance) - 1)], (voltage[usevoltage] / ampGain) / diode_current, control])
            print("T: {0:.1f}s, V: {1:.3f}V, mvV: {2:.3f}V, 1lV: {3:.3f}V, R: {4:.1f}Ω, PWM: {5:.1f}%".format(
                elapsed_time, voltage[0], voltage[1], voltage[2], (voltage[usevoltage] / ampGain) / diode_current, control), end="\r")
        gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
        print("\nGPIO cleanup")
        # 経過時間の最初を0にする
        for i in range(len(csv_data)):
            csv_data[i][0] = csv_data[i][0] - csv_data[0][0]
        writer.writerows(csv_data)
        exit(gpio.stop())

    # キーボード割り込み(Ctrl + C)があった場合の処理
    except KeyboardInterrupt:
        gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
        print("\nGPIO cleanup")
        # 経過時間の最初を0にする
        for i in range(len(csv_data)):
            csv_data[i][0] = csv_data[i][0] - csv_data[0][0]
        writer.writerows(csv_data)
        exit(gpio.stop())
