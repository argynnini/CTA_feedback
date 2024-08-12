import pigpio

# Define the GPIO pins
pins = [22, 23, 24, 25]

def setup_pin(pi, pins, inout, pud):
    for pin in pins:
        pi.set_mode(pin, inout)
        pi.set_pull_up_down(pin, pud)

def read_pin(pi, pins):
    value = 0
    for index, pin in enumerate(pins):
        state = pi.read(pin)
        value = value | (state << index)
    return value