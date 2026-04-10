import cv2
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

class ObjectDetector:
    HIGH_PRIORITY = {'person', 'door', 'stairs', 'step', 'hole'}
    MED_PRIORITY = {'chair', 'couch', 'bed', 'toilet', 'refrigerator', 'tv', 'potted plant'}

    OBJECT_HEIGHTS = {
        'person': 1.7, 'door': 2.0, 'chair': 0.9, 'table': 0.75,
        'couch': 0.8, 'bed': 0.6, 'cabinet': 1.8, 'refrigerator': 1.7,
        'window': 1.5, 'tv': 0.5, 'wardrobe': 1.9, 'shelf': 1.2,
    }

    SYNONYMS = {'sofa': 'couch', 'fridge': 'refrigerator', 'plant': 'potted plant'}

    def __init__(self, model_path: str = 'yolov8n.pt', conf: float = 0.45,
                 camera_height: float = 1.5, focal_length: float = 700):
        self.conf = conf
        self.camera_height = camera_height
        self.focal_length = focal_length
        self._dist_buffers = {}
        self.class_names = {}

        if YOLO_AVAILABLE:
            self.model = YOLO(model_path)
            self.class_names = self.model.names
        else:
            self.model = None
            logger.warning('YOLO not available.')

    def update_conf(self, conf: float):
        self.conf = max(0.2, min(0.9, conf))

    def detect(self, frame: np.ndarray, frame_h: int, frame_w: int,
                left_bound: int, right_bound: int) -> list:
        if self.model is None:
            return []
        try:
            results = self.model.track(frame, conf=self.conf, persist=True, verbose=False)
        except Exception:
            try:
                results = self.model.predict(frame, conf=self.conf, verbose=False)
            except Exception:
                return []

        if not results or not results[0].boxes:
            return []

        detections = []
        for box in results[0].boxes:
            try:
                cls = int(box.cls[0])
                label = self.class_names.get(cls, 'object')
                conf = float(box.conf[0])
                track_id = int(box.id[0]) if box.id is not None else None
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                dist = self._estimate_distance(bbox, frame_h, label, track_id)
                zone = self._classify_zone((x1 + x2) // 2, left_bound, right_bound)
                clock = self._clock_direction((x1 + x2) // 2, frame_w)
                prio = 2 if label in self.HIGH_PRIORITY else (1 if label in self.MED_PRIORITY else 0)

                from src.inference.navigator import Detection
                detections.append(Detection(label, conf, bbox, dist, zone, prio, clock))
            except Exception:
                continue

        return detections

    def _estimate_distance(self, bbox, frame_h, label, track_id):
        x1, y1, x2, y2 = bbox
        pixel_h = y2 - y1
        if pixel_h < 10:
            return 5.0

        dists = []
        if label in self.OBJECT_HEIGHTS:
            d = (self.OBJECT_HEIGHTS[label] * self.focal_length) / pixel_h
            dists.append((d, 0.6))

        bottom_y = y2
        horizon = frame_h / 2
        pixel_dist = bottom_y - horizon
        if pixel_dist > 5:
            d = (self.camera_height * self.focal_length) / pixel_dist
            dists.append((d, 0.4))

        if not dists:
            return 5.0

        total_w = sum(w for _, w in dists)
        raw = sum(d * w for d, w in dists) / total_w
        raw = float(np.clip(raw, 0.3, 10.0))

        if track_id is not None:
            buf = self._dist_buffers.setdefault(track_id, deque(maxlen=8))
            buf.append(raw)
            if len(buf) >= 3:
                weights = np.exp(np.linspace(-1, 0, len(buf)))
                weights /= weights.sum()
                raw = float(np.average(list(buf), weights=weights))

        return round(raw, 2)

    def _classify_zone(self, cx, left_bound, right_bound):
        if cx < left_bound:
            return 'left'
        elif cx > right_bound:
            return 'right'
        return 'center'

    def _clock_direction(self, cx, frame_w):
        ratio = cx / frame_w
        if ratio < 0.2: return "10 o'clock"
        if ratio < 0.4: return "11 o'clock"
        if ratio < 0.6: return "12 o'clock"
        if ratio < 0.8: return "1 o'clock"
        return "2 o'clock"

    def normalize_label(self, label: str) -> str:
        return self.SYNONYMS.get(label.lower(), label.lower())
