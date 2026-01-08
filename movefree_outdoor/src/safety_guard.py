import time
import cv2
import numpy as np


class OutdoorSafetyMonitor:
    def __init__(self, audio_interface):
        self.audio = audio_interface
        self.last_warning = 0

    def analyze_frame(self, result, frame):
        """Analyze YOLO results for outdoor hazards"""
        now = time.time()

        # 1. Traffic Light Logic (Class ID 9 in COCO)
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = result.names[cls_id]

            if label == "traffic light":
                color = self._detect_light_color(box, frame)
                if color == "RED" and (now - self.last_warning > 5):
                    self.audio.speak("Traffic light is Red. Stop.", priority=True)
                    self.last_warning = now
                elif color == "GREEN" and (now - self.last_warning > 5):
                    self.audio.speak(
                        "Traffic light is Green. Safe to cross.", priority=True
                    )
                    self.last_warning = now

            # 2. Vehicle Proximity (Potholes/Cars)
            if label in ["car", "bus", "truck", "pothole"]:
                # Check bounding box height ratio
                h_ratio = float(box.xywh[0][3]) / frame.shape[0]
                if h_ratio > 0.4:  # Takes up 40% of screen height = Close
                    if now - self.last_warning > 3:
                        self.audio.speak(
                            f"Caution. {label} is very close.", priority=True
                        )
                        self.last_warning = now

    def _detect_light_color(self, box, frame):
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return "UNKNOWN"

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Red Masks
        lower_red1 = np.array([0, 70, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 70, 50])
        upper_red2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2),
        )

        # Green Mask
        lower_green = np.array([40, 70, 50])
        upper_green = np.array([90, 255, 255])
        mask_green = cv2.inRange(hsv, lower_green, upper_green)

        red_pixels = cv2.countNonZero(mask_red)
        green_pixels = cv2.countNonZero(mask_green)

        if red_pixels > green_pixels and red_pixels > 20:
            return "RED"
        if green_pixels > red_pixels and green_pixels > 20:
            return "GREEN"
        return "UNKNOWN"
