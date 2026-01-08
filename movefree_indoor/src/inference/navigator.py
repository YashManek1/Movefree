"""
Advanced Indoor Navigator - Human Autopilot System
Intelligent 3-zone navigation with conversational AI
"""

import numpy as np
import cv2
import logging
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional

logger = logging.getLogger(__name__)


@dataclass
class Detection:
    """Represents a detected object"""

    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    distance: float
    zone: str  # 'left', 'center', 'right'
    priority: int  # 0=low, 1=medium, 2=high (safety critical)


class ZoneBasedNavigator:
    """
    Intelligent 3-zone navigation system
    Divides camera view into left, center, right zones
    Provides human-like navigation guidance
    """

    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height

        # Zone boundaries (configurable)
        self.left_boundary = int(frame_width * 0.33)
        self.right_boundary = int(frame_width * 0.67)

        # Distance thresholds (meters)
        self.CRITICAL_DISTANCE = 0.8  # Immediate stop
        self.WARNING_DISTANCE = 1.5  # Slow down, prepare to turn
        self.CAUTION_DISTANCE = 2.5  # Be aware

        # Priority classes (safety-critical objects)
        self.CRITICAL_CLASSES = {"person", "stairs", "door"}
        self.HAZARD_CLASSES = {"chair", "table", "bed", "sofa"}

        # Temporal smoothing (reduce jitter)
        self.detection_history = deque(maxlen=5)  # Last 5 frames
        self.last_command = ""
        self.command_cooldown = 0

        # State tracking
        self.is_path_blocked = False
        self.last_clear_zone = "center"

    def analyze_detections(self, detections: List[Detection]) -> dict:
        """
        Analyze detections and return navigation guidance

        Returns:
            dict with keys: 'state', 'message', 'urgency', 'action'
        """
        # Add to history for temporal smoothing
        self.detection_history.append(detections)

        # Classify detections by zone
        left_zone = []
        center_zone = []
        right_zone = []

        for det in detections:
            if det.zone == "left":
                left_zone.append(det)
            elif det.zone == "center":
                center_zone.append(det)
            elif det.zone == "right":
                right_zone.append(det)

        # Analyze each zone
        left_status = self._analyze_zone(left_zone, "left")
        center_status = self._analyze_zone(center_zone, "center")
        right_status = self._analyze_zone(right_zone, "right")

        # Decision tree for navigation
        navigation = self._make_navigation_decision(
            left_status, center_status, right_status
        )

        return navigation

    def _analyze_zone(self, detections: List[Detection], zone_name: str) -> dict:
        """Analyze a single zone"""
        if not detections:
            return {
                "clear": True,
                "closest_distance": float("inf"),
                "hazards": [],
                "critical_hazards": [],
            }

        # Find closest object
        closest = min(detections, key=lambda d: d.distance)

        # Categorize hazards
        hazards = [d for d in detections if d.label in self.HAZARD_CLASSES]
        critical = [d for d in detections if d.label in self.CRITICAL_CLASSES]

        # Determine if zone is clear for navigation
        is_clear = closest.distance > self.WARNING_DISTANCE

        return {
            "clear": is_clear,
            "closest_distance": closest.distance,
            "closest_object": closest.label,
            "hazards": hazards,
            "critical_hazards": critical,
        }

    def _make_navigation_decision(self, left: dict, center: dict, right: dict) -> dict:
        """
        Core navigation logic - THE BRAIN
        Makes human-like decisions based on zone analysis
        """

        # === EMERGENCY STOP CONDITIONS ===
        if not center["clear"] and center["closest_distance"] < self.CRITICAL_DISTANCE:
            return {
                "state": "EMERGENCY_STOP",
                "message": f"Stop immediately! {center['closest_object']} directly ahead!",
                "urgency": "critical",
                "action": "stop",
                "priority": 2,
            }

        # Stairs detection (ALWAYS critical)
        stairs_detected = self._check_for_stairs(left, center, right)
        if stairs_detected:
            zone, distance = stairs_detected
            if distance < 2.0:
                return {
                    "state": "STAIRS_WARNING",
                    "message": f"Caution! Stairs {zone}, {distance:.1f} meters away",
                    "urgency": "critical",
                    "action": "stop_and_assess",
                    "priority": 2,
                }

        # Person detected (high priority)
        person_info = self._check_for_people(left, center, right)
        if person_info:
            zone, distance = person_info
            if distance < self.WARNING_DISTANCE:
                return {
                    "state": "PERSON_NEARBY",
                    "message": f"Person {zone}, {distance:.1f} meters",
                    "urgency": "high",
                    "action": "slow_down",
                    "priority": 1,
                }

        # === NAVIGATION GUIDANCE ===

        # Path completely blocked
        if not center["clear"] and not left["clear"] and not right["clear"]:
            return {
                "state": "PATH_BLOCKED",
                "message": "Path blocked on all sides. Please turn around.",
                "urgency": "high",
                "action": "turn_around",
                "priority": 1,
            }

        # Center blocked - navigate around
        if not center["clear"]:
            # Find best alternative path
            if left["clear"] and right["clear"]:
                # Both sides clear - choose based on more space
                if left["closest_distance"] > right["closest_distance"]:
                    direction = "left"
                    distance = left["closest_distance"]
                else:
                    direction = "right"
                    distance = right["closest_distance"]

                return {
                    "state": "NAVIGATE_AROUND",
                    "message": f"{center['closest_object']} ahead. Turn {direction}. Path is clear.",
                    "urgency": "medium",
                    "action": f"turn_{direction}",
                    "priority": 1,
                }

            elif left["clear"]:
                return {
                    "state": "NAVIGATE_LEFT",
                    "message": f"{center['closest_object']} ahead. Turn left. Right is blocked.",
                    "urgency": "medium",
                    "action": "turn_left",
                    "priority": 1,
                }

            elif right["clear"]:
                return {
                    "state": "NAVIGATE_RIGHT",
                    "message": f"{center['closest_object']} ahead. Turn right. Left is blocked.",
                    "urgency": "medium",
                    "action": "turn_right",
                    "priority": 1,
                }

        # Side hazards only
        if not left["clear"] and left["closest_distance"] < self.CAUTION_DISTANCE:
            return {
                "state": "HAZARD_LEFT",
                "message": f"{left['closest_object']} on your left, {left['closest_distance']:.1f} meters",
                "urgency": "low",
                "action": "veer_right",
                "priority": 0,
            }

        if not right["clear"] and right["closest_distance"] < self.CAUTION_DISTANCE:
            return {
                "state": "HAZARD_RIGHT",
                "message": f"{right['closest_object']} on your right, {right['closest_distance']:.1f} meters",
                "urgency": "low",
                "action": "veer_left",
                "priority": 0,
            }

        # All clear - provide encouragement
        return {
            "state": "PATH_CLEAR",
            "message": "Path is clear. Continue forward.",
            "urgency": "info",
            "action": "continue",
            "priority": 0,
        }

    def _check_for_stairs(self, left, center, right) -> Optional[Tuple[str, float]]:
        """Check for stairs in any zone (SAFETY CRITICAL)"""
        for zone_data, zone_name in [
            (left, "on left"),
            (center, "ahead"),
            (right, "on right"),
        ]:
            for hazard in zone_data.get("critical_hazards", []):
                if hazard.label == "stairs":
                    return (zone_name, hazard.distance)
        return None

    def _check_for_people(self, left, center, right) -> Optional[Tuple[str, float]]:
        """Check for people in any zone"""
        for zone_data, zone_name in [
            (left, "on left"),
            (center, "ahead"),
            (right, "on right"),
        ]:
            for hazard in zone_data.get("critical_hazards", []):
                if hazard.label == "person":
                    return (zone_name, hazard.distance)
        return None

    def classify_zone(self, bbox: Tuple[int, int, int, int]) -> str:
        """Classify which zone a bounding box belongs to"""
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2

        if center_x < self.left_boundary:
            return "left"
        elif center_x > self.right_boundary:
            return "right"
        else:
            return "center"

    def draw_zones(self, frame: np.ndarray) -> np.ndarray:
        """Draw zone boundaries on frame for visualization"""
        frame_copy = frame.copy()
        height, width = frame_copy.shape[:2]

        # Draw zone lines
        cv2.line(
            frame_copy,
            (self.left_boundary, 0),
            (self.left_boundary, height),
            (0, 255, 255),
            2,
        )
        cv2.line(
            frame_copy,
            (self.right_boundary, 0),
            (self.right_boundary, height),
            (0, 255, 255),
            2,
        )

        # Label zones
        cv2.putText(
            frame_copy,
            "LEFT",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame_copy,
            "CENTER",
            (width // 2 - 40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            frame_copy,
            "RIGHT",
            (width - 80, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        return frame_copy
