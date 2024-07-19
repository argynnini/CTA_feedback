import mcp3208 as adc
import time
import os
import pigpio
import csv
from simple_pid import PID
import math

Kp = 20.0
Ki = 0.0
Kd = 0.0

diode_current = 0.00963  # 9.63mA
ampR = 1.986  # 1.986 kΩ
ampGain = 1 + 49.4 / ampR

pid = PID(-Kp, -Ki, -Kd, setpoint=0)
pid.output_limits = (0, 30)  # PWM は0~30%の範囲で出力
pid.sample_time = 0.1  # 制御周期は0.1秒
# pid.proportional_on_measurement = True  # 逓減制御

test_distance = [4, 7]  # , 10, 13, 16, 19, 22, 25]
test_resistance = [
    0.39 * math.sqrt(25.36 - x) + 7.4 for x in test_distance
]  # テスト抵抗値

pwm_freq = 1000
duty_range = [0, 30]  # duty比の範囲


gpio = pigpio.pi()

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

# 上限は30%くらい
csv_data = []
control = 0
start_time = time.perf_counter()
with open("/home/pi/BMF_CV/idiode.csv", "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["time", "Voltage", "Resistance", "PWM"])
    try:
        # PWMの設定
        gpio.hardware_PWM(12, pwm_freq, int(0 * 255 / 100))
        # 初期PWMの設定
        gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
        # 目標抵抗値の設定
        pid.setpoint = test_resistance[0]
        print("target: {0:.3f}Ω".format(pid.setpoint))
        while True:
            # チャンネル6~7の差分電圧を読み取る
            voltage = adc.read_voltage(channel=6, convtype=0, vref=5.0)
            # 経過時間
            elapsed_time = time.perf_counter() - start_time  # 経過時間
            if (
                voltage > v_thresh[0] and voltage < v_thresh[1]
            ):  # 1.5V ~ 3.0V の範囲内のとき
                if (control := pid(voltage / ampGain / diode_current)) is None:
                    control = 0
                gpio.set_PWM_dutycycle(12, int(control * 255 / 100))
                csv_data.append(
                    [elapsed_time, voltage, (voltage / ampGain) / diode_current, control]
                )
            print(
                "T: {0:.1f}s, V: {1:.3f}V, R: {2:.1f}Ω, PWM: {3:.1f}%".format(
                    elapsed_time, voltage, (voltage / ampGain) / diode_current, control
                ),
                end="\r",
            )

    # キーボード割り込み(Ctrl + C)があった場合の処理
    except KeyboardInterrupt:
        gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
        print("\nGPIO cleanup")
        # 経過時間の最初を0にする
        for i in range(len(csv_data)):
            csv_data[i][0] = csv_data[i][0] - csv_data[0][0]
        writer.writerows(csv_data)
        exit(gpio.stop())
