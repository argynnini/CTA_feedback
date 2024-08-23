import time
import pigpio

"""
sudo groupadd gpio
sudo usermod -a -G gpio pi
sudo chown root.gpio /dev/gpiomem
sudo chmod a+rw /dev/gpiomem
sudo pigpiod
"""

pwm_freq = 10
duty_cycle = 0
pwm_pin = 13

gpio = pigpio.pi()
try:
    gpio.hardware_PWM(pwm_pin, pwm_freq, int(duty_cycle * 255 / 100))

    gpio.set_PWM_dutycycle(pwm_pin, int(20 * 255 / 100))

    # Wait for 5 seconds
    print("Wait for 1 minute")
    time.sleep(60)

    gpio.set_PWM_dutycycle(pwm_pin, int(0 * 255 / 100))
    # Stop PWM(止まらない)
    gpio.stop()

except KeyboardInterrupt:
    gpio.set_PWM_dutycycle(pwm_pin, int(0 * 255 / 100))
    gpio.stop()
    print("GPIO cleanup")
    raise
