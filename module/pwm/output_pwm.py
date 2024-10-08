import time
import pigpio

"""
sudo groupadd gpio
sudo usermod -a -G gpio pi
sudo chown root.gpio /dev/gpiomem
sudo chmod a+rw /dev/gpiomem
sudo pigpiod
"""

pwm_freq = 1000
duty_cycle = 0

gpio = pigpio.pi()
try:
    gpio.hardware_PWM(12, pwm_freq, int(duty_cycle * 255 / 100))

    print("Start PWM with a duty cycle of 5%")
    gpio.set_PWM_dutycycle(12, int(5 * 255 / 100))
    time.sleep(5)

    # Change the duty cycle to 50%
    for i in range(1, 101):
        print("PWM:", i, end="\r")
        gpio.set_PWM_dutycycle(12, int(i * 255 / 100))
        time.sleep(0.5)

    # Wait for 5 seconds
    print("Wait for 5 seconds")
    time.sleep(5)

    gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
    # Stop PWM(止まらない)
    gpio.stop()

except KeyboardInterrupt:
    gpio.set_PWM_dutycycle(12, int(0 * 255 / 100))
    gpio.stop()
    print("GPIO cleanup")
    raise
