import mcp3208 as adc
import time
import os
import pigpio
import csv
from simple_pid import PID

Kp = 0.1
Ki = 0.01
Kd = 0.05

pid = PID(Kp, Ki, Kd, setpoint=0)
pid.output_limits = (0, 30) # PWM は0~30%の範囲で出力



pwm_freq = 1000
duty_cycle = 0

gpio = pigpio.pi()

v_thresh = [1.5, 3.0]

# 5Vでは2MHzが限界?
try: 
    adc.open(port=0, cs=0, speed=2000000) # 2 MHz が限界
# 例外処理(権限がない場合)
except PermissionError:
    os.system("sudo groupadd gpio")
    os.system("sudo usermod -a -G gpio -G spi pi")
    os.system("sudo chown root:spi /dev/spidev*")
    os.system("sudo chmod g+rw /dev/spidev*")
    adc.open(port=0, cs=0, speed=7800000) # 7.8 MHz が限界

# 上限は30%くらい

value = []
start_time = time.time()
with open('/home/pi/BMF_CV/idiode.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["time", "Voltage"])
    try:
        gpio.hardware_PWM(12, pwm_freq, int(duty_cycle * 255 / 100))
        gpio.set_PWM_dutycycle(12, int(10 * 255 / 100))
        while True:
            voltage = adc.read_voltage(7)
            elapsed_time = time.time() - start_time
            print("time: {0:.3f}, voltage: {1:.3f}".format(elapsed_time, voltage), end="\r")
            if voltage > v_thresh[0] and voltage < v_thresh[1]:
                value.append([elapsed_time, voltage])

    # キーボード割り込み(Ctrl + C)があった場合の処理
    except KeyboardInterrupt:
        gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
        gpio.stop()
        print("GPIO cleanup")
        writer.writerows(value)
        raise


