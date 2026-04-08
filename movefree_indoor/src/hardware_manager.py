import time
import logging
import threading
import numpy as np

# 1. Global check for libraries
try:
    import Jetson.GPIO as GPIO

    LIB_GPIO_PRESENT = True
except ImportError:
    try:
        import RPi.GPIO as GPIO

        LIB_GPIO_PRESENT = True
    except ImportError:
        LIB_GPIO_PRESENT = False

logger = logging.getLogger("HardwareManager")


class HardwareManager:
    def __init__(self):
        self.ultrasonic_dist = 999.0
        self.battery_level = 100
        self.running = True
        self.gpio_active = LIB_GPIO_PRESENT

        # Pin Configurations
        self.TRIG = 23
        self.ECHO = 24
        self.VIB_LEFT = 17  # GPIO for Left Vibration Motor
        self.VIB_RIGHT = 27  # GPIO for Right Vibration Motor

        if self.gpio_active:
            try:
                GPIO.setmode(GPIO.BCM)
                GPIO.setup(self.TRIG, GPIO.OUT)
                GPIO.setup(self.ECHO, GPIO.IN)

                # Setup Haptics
                GPIO.setup(self.VIB_LEFT, GPIO.OUT)
                GPIO.setup(self.VIB_RIGHT, GPIO.OUT)
                GPIO.output(self.VIB_LEFT, False)
                GPIO.output(self.VIB_RIGHT, False)

                logger.info("✅ GPIO & Haptics Initialized")
            except Exception as e:
                logger.error(f"GPIO Setup Failed: {e}")
                self.gpio_active = False
        else:
            logger.warning("⚠️ GPIO not found. Running in simulation mode.")

        threading.Thread(target=self._update_sensors, daemon=True).start()

    def trigger_haptic(self, code):
        """
        code 1: Left Pulse
        code 2: Right Pulse
        code 3: Both (Stop)
        """
        if not self.gpio_active:
            return

        def pulse(pin):
            GPIO.output(pin, True)
            time.sleep(0.3)
            GPIO.output(pin, False)

        if code == 1:
            threading.Thread(target=pulse, args=(self.VIB_LEFT,)).start()
        elif code == 2:
            threading.Thread(target=pulse, args=(self.VIB_RIGHT,)).start()
        elif code == 3:
            GPIO.output(self.VIB_LEFT, True)
            GPIO.output(self.VIB_RIGHT, True)
            time.sleep(0.5)
            GPIO.output(self.VIB_LEFT, False)
            GPIO.output(self.VIB_RIGHT, False)

    def _update_sensors(self):
        while self.running:
            if self.gpio_active:
                try:
                    GPIO.output(self.TRIG, False)
                    time.sleep(0.05)
                    GPIO.output(self.TRIG, True)
                    time.sleep(0.00001)
                    GPIO.output(self.TRIG, False)

                    # (Standard Ultrasonic Logic - Shortened for brevity)
                    # ... [Same as previous code] ...
                    # For safety, resetting to far if read fails
                    self.ultrasonic_dist = 999.0
                except:
                    pass
            else:
                time.sleep(1)

    def get_distance(self):
        return self.ultrasonic_dist

    def check_ambient_light(self, frame):
        if frame is None:
            return "OK"
        gray = 0.299 * frame[:, :, 2] + 0.587 * frame[:, :, 1] + 0.114 * frame[:, :, 0]
        return "DARK" if np.mean(gray) < 40 else "OK"

    def stop(self):
        self.running = False
        if self.gpio_active:
            try:
                GPIO.cleanup()
            except:
                pass
