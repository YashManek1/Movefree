import cv2
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
import time

logger = logging.getLogger(__name__)

@dataclass
class Detection:
    label: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    distance: float
    zone: str
    priority: int
    clock_dir: str = ''

class ZoneBasedNavigator:
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self.left_boundary = int(frame_width * 0.35)
        self.right_boundary = int(frame_width * 0.65)
        self.STOP_DIST = 1.0
        self.WARN_DIST = 2.5
        self.last_state = 'CLEAR'
        self.state_counter = 0
        self.last_clear_time = 0
        self.CLEAR_INTERVAL = 15.0

    def classify_zone(self, bbox: Tuple) -> str:
        cx = (bbox[0] + bbox[2]) // 2
        if cx < self.left_boundary:
            return 'left'
        elif cx > self.right_boundary:
            return 'right'
        return 'center'

    def get_critical_warning(self, detections: List[Detection]) -> Optional[str]:
        for d in detections:
            if d.label in ('stairs', 'hole', 'step') and d.distance < 2.5:
                return f'Caution! {d.label} ahead at {d.distance:.1f} meters.'
            if d.zone == 'center' and d.distance < self.STOP_DIST:
                return f'Stop! {d.label} directly ahead.'
            if d.label == 'person' and d.zone == 'center' and d.distance < 1.8:
                return f'Person very close, {d.clock_dir}.'
        return None

    def detect_crowding(self, detections: List[Detection]) -> Optional[str]:
        people = [d for d in detections if d.label == 'person' and d.distance < 3.0]
        if len(people) >= 3:
            return f'{len(people)} people nearby. Navigate carefully.'
        return None

    def find_exit(self, detections: List[Detection]) -> Optional[str]:
        doors = [d for d in detections if d.label in ('door', 'entrance')]
        if doors:
            closest = min(doors, key=lambda x: x.distance)
            return f'Door at {closest.clock_dir}, {closest.distance:.1f} meters.'
        return None

    def analyze_detections(self, detections: List[Detection]) -> dict:
        relevant = [d for d in detections if d.distance < self.WARN_DIST]
        center = [d for d in relevant if d.zone == 'center']
        left = [d for d in relevant if d.zone == 'left']
        right = [d for d in relevant if d.zone == 'right']

        state = 'CLEAR'
        message = ''
        haptic = 0

        if center:
            closest = min(center, key=lambda x: x.distance)
            if not left and not right:
                state = 'AVOID'
                message = f'{closest.label} ahead. Go left or right.'
                haptic = 1
            elif not left:
                state = 'TURN_LEFT'
                message = f'{closest.label} ahead. Turn left.'
                haptic = 1
            elif not right:
                state = 'TURN_RIGHT'
                message = f'{closest.label} ahead. Turn right.'
                haptic = 2
            else:
                state = 'BLOCKED'
                message = 'Path blocked. Stop.'
                haptic = 3

        crowd_msg = self.detect_crowding(detections)

        if state == self.last_state:
            self.state_counter += 1
        else:
            self.state_counter = 0
            self.last_state = state

        result = {'message': None, 'haptic': haptic}
        now = time.time()

        if state == 'BLOCKED' and self.state_counter > 2:
            result['message'] = message
        elif state in ('TURN_LEFT', 'TURN_RIGHT', 'AVOID') and self.state_counter == 5:
            result['message'] = message
        elif state == 'CLEAR' and now - self.last_clear_time > self.CLEAR_INTERVAL:
            result['message'] = crowd_msg or 'Path clear.'
            self.last_clear_time = now
        elif crowd_msg and self.state_counter == 1:
            result['message'] = crowd_msg

        return result

    def draw_zones(self, frame):
        h, w = frame.shape[:2]
        color = (0, 229, 255)
        cv2.line(frame, (self.left_boundary, 0), (self.left_boundary, h), color, 1)
        cv2.line(frame, (self.right_boundary, 0), (self.right_boundary, h), color, 1)
        cv2.putText(frame, 'L', (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, 'C', (self.left_boundary + 8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        cv2.putText(frame, 'R', (self.right_boundary + 8, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
        return frame
