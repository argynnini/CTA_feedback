import mcp3208 as adc
import time

adc.open(port=0, cs=0, speed=1000) # 7.8 MHz が限界

while True:
    value = adc.read_voltage(7)
    print(value)
    time.sleep(0.1)
