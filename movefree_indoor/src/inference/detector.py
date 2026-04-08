"""
Object detection module for MoveFree
Tuned for stability
"""

import cv2
import numpy as np
from ultralytics import YOLO
import yaml
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IndoorDetector:
    def __init__(
        self,
        model_path="runs/detect/movefree_indoor/weights/best.pt",
        config_path="config/config.yaml",
    ):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = YOLO(model_path)
        self.conf_threshold = 0.25  # Lower threshold catches more objects
        self.class_names = self.config["dataset"]["classes"]

    def detect_objects(self, frame):
        """
        Detect objects with persistence tracking
        """
        results = self.model.track(
            frame,
            conf=self.conf_threshold,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )

        annotated_frame = results[0].plot()
        return results, annotated_frame

    def analyze_scene(self, results):
        scene_analysis = {
            "obstacles": [],
            "danger_level": "safe",
            "navigation_advice": [],
        }

        if not results or not results[0].boxes:
            scene_analysis["navigation_advice"].append("Path clear")
            return scene_analysis

        for box in results[0].boxes:
            if box.id is None:
                continue  # Skip objects without track ID (flicker reduction)

            class_id = int(box.cls[0])
            class_name = self.class_names.get(class_id, "unknown")
            bbox = box.xyxy[0].cpu().numpy()

            obj_info = {
                "class": class_name,
                "bbox": bbox,
                "center_x": (bbox[0] + bbox[2]) / 2,
            }
            scene_analysis["obstacles"].append(obj_info)

        return scene_analysis

    def get_direction_advice(self, scene_analysis, frame_width):
        if not scene_analysis["obstacles"]:
            return "Path clear"

        left_count = 0
        right_count = 0
        center = frame_width / 2

        for obs in scene_analysis["obstacles"]:
            if obs.get("distance", 10) < 2.5:
                if obs["center_x"] < center:
                    left_count += 1
                else:
                    right_count += 1

        if left_count > right_count:
            return "Turn Right"
        elif right_count > left_count:
            return "Turn Left"
        elif left_count > 0:
            return "Stop. Blocked."
        return "Proceed Forward"
