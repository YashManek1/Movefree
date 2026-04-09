import cv2
import time
import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance: float
    zone: str
    priority: int
    clock_dir: str = ""


class ZoneBasedNavigator:
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self.left_boundary = int(frame_width * 0.35)
        self.right_boundary = int(frame_width * 0.65)

        # Thresholds
        self.STOP_DIST = 1.0
        self.WARN_DIST = 2.5

        # State Management
        self.last_state = "CLEAR"
        self.state_counter = 0
        self.last_spoken_time = 0
        self.clear_speech_interval = 15.0  # Only say "Path Clear" every 15 seconds

    def classify_zone(self, bbox):
        cx = (bbox[0] + bbox[2]) // 2
        if cx < self.left_boundary:
            return "left"
        elif cx > self.right_boundary:
            return "right"
        return "center"

    def get_critical_warning(self, detections: List[Detection]) -> Optional[str]:
        """Immediate safety override"""
        for d in detections:
            # Fall Hazards
            if d.label in ["stairs", "hole"] and d.distance < 2.5:
                return f"Caution! {d.label} {d.zone}."

            # Immediate Collision
            if d.zone == "center" and d.distance < self.STOP_DIST:
                return f"Stop! {d.label} directly ahead."
        return None

    def analyze_detections(self, detections: List[Detection]) -> dict:
        """Smart navigation with reduced chatter"""
        # Filter only relevant objects (ignore distant ones)
        relevant = [d for d in detections if d.distance < self.WARN_DIST]

        center_objs = [d for d in relevant if d.zone == "center"]
        left_objs = [d for d in relevant if d.zone == "left"]
        right_objs = [d for d in relevant if d.zone == "right"]

        current_state = "CLEAR"
        message = ""
        haptic_code = 0  # 0=None, 1=Left, 2=Right, 3=Stop

        # Logic
        if center_objs:
            # Path blocked
            closest = min(center_objs, key=lambda x: x.distance)

            # Check openings
            if not left_objs and not right_objs:
                current_state = "AVOID"
                message = f"{closest.label} ahead. Go left or right."
                haptic_code = 1  # Pulse to indicate turn
            elif not left_objs:
                current_state = "TURN_LEFT"
                message = f"{closest.label} ahead. Turn left."
                haptic_code = 1
            elif not right_objs:
                current_state = "TURN_RIGHT"
                message = f"{closest.label} ahead. Turn right."
                haptic_code = 2
            else:
                current_state = "BLOCKED"
                message = "Path blocked. Stop."
                haptic_code = 3

        # State Persistence (Anti-Jitter)
        if current_state == self.last_state:
            self.state_counter += 1
        else:
            self.state_counter = 0
            self.last_state = current_state

        result = {"message": None, "haptic": haptic_code}

        # Decision to Speak
        now = time.time()

        # 1. Critical Blockage: Speak immediately
        if current_state == "BLOCKED" and self.state_counter > 2:
            result["message"] = message

        # 2. Turn Instructions: Speak if stable for 5 frames
        elif (
            current_state in ["TURN_LEFT", "TURN_RIGHT", "AVOID"]
            and self.state_counter == 5
        ):
            result["message"] = message

        # 3. Path Clear: Only speak rarely (UX Fix)
        elif current_state == "CLEAR" and (
            now - self.last_spoken_time > self.clear_speech_interval
        ):
            result["message"] = "Path clear."
            self.last_spoken_time = now

        return result

    def draw_zones(self, frame):
        h, w = frame.shape[:2]
        color = (255, 255, 0)  # Cyan

        # Draw transparent overlay logic (Simulated with lines for speed)
        cv2.line(frame, (self.left_boundary, 0), (self.left_boundary, h), color, 1)
        cv2.line(frame, (self.right_boundary, 0), (self.right_boundary, h), color, 1)

        # Labels
        cv2.putText(frame, "L", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(
            frame,
            "C",
            (self.left_boundary + 10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )
        cv2.putText(
            frame,
            "R",
            (self.right_boundary + 10, h - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
        )

        return frame
