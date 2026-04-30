import time
import Jetson.GPIO as GPIO

BUTTON_PIN = 40

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN)

try:
    while True:
        print(GPIO.input(BUTTON_PIN), flush=True)
        time.sleep(0.2)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()

