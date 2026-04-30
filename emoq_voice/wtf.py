import Jetson.GPIO as GPIO
import time

BUTTON_PIN = 40

GPIO.setmode(GPIO.BOARD)
GPIO.setup(BUTTON_PIN, GPIO.IN)

try:
    while True:
        print(f"{time.time():.3f} state={GPIO.input(BUTTON_PIN)}")
        time.sleep(0.02)
except KeyboardInterrupt:
    pass
finally:
    GPIO.cleanup()
