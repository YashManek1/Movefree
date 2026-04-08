"""
Advanced Distance Estimation with Multi-Method Fusion
Camera-only depth estimation optimized for indoor navigation
"""

import numpy as np
import cv2
from typing import Tuple
from collections import deque


class AdvancedDistanceEstimator:
    """
    Estimates distance using multiple methods:
    1. Pinhole camera model (geometric)
    2. Object size heuristics (known object dimensions)
    3. Temporal smoothing (reduce noise)
    4. Perspective-based estimation
    """

    def __init__(self, camera_height: float = 1.5, focal_length: float = 700):
        """
        Args:
            camera_height: Camera height from ground in meters (chest/head level)
            focal_length: Camera focal length in pixels (calibrate per device)
        """
        self.camera_height = camera_height
        self.focal_length = focal_length
        self.horizon_offset = 0  # Auto-calibrated
        self.calibrated = False

        # Temporal smoothing buffers
        self.distance_buffers = {}  # track_id -> deque of distances
        self.buffer_size = 10

        # Known object dimensions (real-world heights in meters)
        self.object_heights = {
            "person": 1.7,
            "door": 2.0,
            "chair": 0.9,
            "table": 0.75,
            "sofa": 0.8,
            "bed": 0.6,
            "cabinet": 1.8,
            "refrigerator": 1.7,
            "window": 1.5,
            "tv": 0.5,
            "wardrobe": 1.9,
            "shelf": 1.2,
        }

    def auto_calibrate(self, detections, frame_height: int):
        """
        Auto-calibrate horizon based on detected objects
        Should be called for first ~60 frames
        """
        if len(detections) < 3:
            return False

        # Use bottom points of detected objects to estimate horizon
        bottom_points = []
        for det in detections:
            if hasattr(det, "bbox"):
                x1, y1, x2, y2 = det.bbox
                bottom_points.append(y2)

        if bottom_points:
            # Median bottom point gives rough horizon estimate
            median_bottom = np.median(bottom_points)
            self.horizon_offset = median_bottom - (frame_height / 2)
            self.calibrated = True
            return True

        return False

    def estimate(
        self,
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        frame_width: int = None,
        class_name: str = None,
        track_id: int = None,
    ) -> float:
        """
        Estimate distance using multiple methods

        Args:
            bbox: Bounding box (x1, y1, x2, y2)
            frame_height: Frame height in pixels
            frame_width: Frame width in pixels
            class_name: Object class for size-based estimation
            track_id: Object tracking ID for temporal smoothing

        Returns:
            Estimated distance in meters
        """
        x1, y1, x2, y2 = bbox

        # Method 1: Geometric estimation (pinhole camera model)
        geometric_dist = self._geometric_estimate(bbox, frame_height)

        # Method 2: Size-based estimation (if known object)
        size_based_dist = None
        if class_name and class_name in self.object_heights:
            size_based_dist = self._size_based_estimate(bbox, class_name)

        # Method 3: Perspective-based estimation
        perspective_dist = None
        if frame_width:
            perspective_dist = self._perspective_estimate(
                bbox, frame_height, frame_width
            )

        # Combine estimates (weighted average)
        distances = []
        weights = []

        if geometric_dist:
            distances.append(geometric_dist)
            weights.append(0.4)

        if size_based_dist:
            distances.append(size_based_dist)
            weights.append(0.5)  # Trust size-based more for known objects

        if perspective_dist:
            distances.append(perspective_dist)
            weights.append(0.1)

        if distances:
            distance = np.average(distances, weights=weights)
        else:
            distance = geometric_dist if geometric_dist else 5.0

        # Clamp to reasonable indoor range
        distance = np.clip(distance, 0.3, 10.0)

        # Temporal smoothing (if tracking available)
        if track_id is not None:
            distance = self._smooth_distance(track_id, distance)

        return float(distance)

    def _geometric_estimate(
        self, bbox: Tuple[int, int, int, int], frame_height: int
    ) -> float:
        """Pinhole camera model estimation"""
        x1, y1, x2, y2 = bbox

        # Use bottom center of bbox (feet/base of object)
        bottom_y = y2

        # Calculate horizon line (adjusted for camera tilt)
        horizon = (frame_height / 2) - self.horizon_offset

        # Pixels from horizon to object base
        pixel_distance = bottom_y - horizon

        # Objects above horizon are far away or floating
        if pixel_distance <= 5:
            return 8.0  # Default far distance

        # Pinhole formula: distance = (camera_height * focal_length) / pixel_offset
        distance = (self.camera_height * self.focal_length) / pixel_distance

        return distance

    def _size_based_estimate(
        self, bbox: Tuple[int, int, int, int], class_name: str
    ) -> float:
        """Estimate distance based on known object dimensions"""
        x1, y1, x2, y2 = bbox

        # Pixel height of object in image
        pixel_height = y2 - y1

        if pixel_height < 10:  # Too small to be reliable
            return None

        # Real-world height of object
        real_height = self.object_heights[class_name]

        # Distance formula: d = (real_height * focal_length) / pixel_height
        distance = (real_height * self.focal_length) / pixel_height

        return distance

    def _perspective_estimate(
        self, bbox: Tuple[int, int, int, int], frame_height: int, frame_width: int
    ) -> float:
        """Estimate based on object position in frame (perspective cues)"""
        x1, y1, x2, y2 = bbox

        # Objects lower in frame are generally closer
        # Normalize y position (0 = top, 1 = bottom)
        center_y = (y1 + y2) / 2
        y_position = center_y / frame_height

        # Simple inverse relationship
        # Objects at bottom (y=1) are closer
        if y_position > 0.7:
            return 1.5  # Close
        elif y_position > 0.4:
            return 3.0  # Medium
        else:
            return 6.0  # Far

    def _smooth_distance(self, track_id: int, current_distance: float) -> float:
        """Apply temporal smoothing using exponential moving average"""
        if track_id not in self.distance_buffers:
            self.distance_buffers[track_id] = deque(maxlen=self.buffer_size)

        buffer = self.distance_buffers[track_id]
        buffer.append(current_distance)

        # Exponential moving average
        if len(buffer) < 3:
            return current_distance

        weights = np.exp(np.linspace(-1, 0, len(buffer)))
        weights /= weights.sum()

        smoothed = np.average(list(buffer), weights=weights)
        return smoothed

    def calibrate_focal_length(
        self,
        known_distance: float,
        bbox: Tuple[int, int, int, int],
        frame_height: int,
        real_height: float = 1.7,
    ):
        """
        Calibrate focal length using a known distance measurement

        Args:
            known_distance: Measured distance to object in meters
            bbox: Bounding box of object at known distance
            frame_height: Frame height
            real_height: Real-world height of object (default: person height)
        """
        x1, y1, x2, y2 = bbox
        pixel_height = y2 - y1

        # Solve for focal length: f = (d * h_pixels) / h_real
        self.focal_length = (known_distance * pixel_height) / real_height

        print(f"✅ Calibrated focal length: {self.focal_length:.1f} pixels")
