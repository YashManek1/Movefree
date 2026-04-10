import threading
import time
import logging
import numpy as np

try:
    import Jetson.GPIO as GPIO
    LIB_GPIO_PRESENT = True
except ImportError:
    try:
        import RPi.GPIO as GPIO
        LIB_GPIO_PRESENT = True
    except ImportError:
        LIB_GPIO_PRESENT = False

logger = logging.getLogger('HardwareManager')

class HardwareManager:
    def __init__(self):
        self.ultrasonic_dist = 999.0
        self.battery_level = 100.0
        self.running = True
        self.gpio_active = LIB_GPIO_PRESENT

        self.TRIG = 23
        self.ECHO = 24
        self.VIB_LEFT = 17
        self.VIB_RIGHT = 27

        if self.gpio_active:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.TRIG, GPIO.OUT)
                GPIO.setup(self.ECHO, GPIO.IN)
                GPIO.setup(self.VIB_LEFT, GPIO.OUT)
                GPIO.setup(self.VIB_RIGHT, GPIO.OUT)
                GPIO.output(self.VIB_LEFT, False)
                GPIO.output(self.VIB_RIGHT, False)
            except Exception as e:
                logger.error(f'GPIO Setup Failed: {e}')
                self.gpio_active = False

        threading.Thread(target=self._sensor_loop, daemon=True).start()

    def _sensor_loop(self):
        while self.running:
            if self.gpio_active:
                self._read_ultrasonic()
            else:
                time.sleep(0.5)

    def _read_ultrasonic(self):
        try:
            GPIO.output(self.TRIG, False)
            time.sleep(0.05)
            GPIO.output(self.TRIG, True)
            time.sleep(0.00001)
            GPIO.output(self.TRIG, False)

            timeout = time.time() + 0.1
            pulse_start = time.time()
            while GPIO.input(self.ECHO) == 0:
                pulse_start = time.time()
                if pulse_start > timeout:
                    return

            timeout = time.time() + 0.1
            pulse_end = time.time()
            while GPIO.input(self.ECHO) == 1:
                pulse_end = time.time()
                if pulse_end > timeout:
                    return

            pulse_duration = pulse_end - pulse_start
            self.ultrasonic_dist = round(pulse_duration * 17150 / 100, 2)
        except Exception:
            self.ultrasonic_dist = 999.0

    def trigger_haptic(self, code):
        if not self.gpio_active:
            return

        def pulse(pin):
            GPIO.output(pin, True)
            time.sleep(0.3)
            GPIO.output(pin, False)

        if code == 1:
            threading.Thread(target=pulse, args=(self.VIB_LEFT,), daemon=True).start()
        elif code == 2:
            threading.Thread(target=pulse, args=(self.VIB_RIGHT,), daemon=True).start()
        elif code == 3:
            GPIO.output(self.VIB_LEFT, True)
            GPIO.output(self.VIB_RIGHT, True)
            time.sleep(0.5)
            GPIO.output(self.VIB_LEFT, False)
            GPIO.output(self.VIB_RIGHT, False)

    def get_distance(self):
        return self.ultrasonic_dist

    def get_battery_level(self):
        try:
            with open('/sys/class/power_supply/BAT0/capacity', 'r') as f:
                return int(f.read().strip())
        except Exception:
            self.battery_level = max(0, self.battery_level - 0.001)
            return round(self.battery_level, 1)

    def get_temperature(self):
        try:
            with open('/sys/class/thermal/thermal_zone0/temp', 'r') as f:
                return round(int(f.read().strip()) / 1000.0, 1)
        except Exception:
            return round(45 + np.random.uniform(-2, 2), 1)

    def get_sensor_status(self):
        return {
            'sonar_active': self.ultrasonic_dist < 999.0,
            'gpio': self.gpio_active,
        }

    def check_ambient_light(self, frame):
        if frame is None:
            return 'OK'
        gray = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]
        return 'DARK' if np.mean(gray) < 40 else 'OK'

    def stop(self):
        self.running = False
        if self.gpio_active:
            try:
                GPIO.cleanup()
            except Exception:
                pass
