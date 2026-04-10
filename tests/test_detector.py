"""
Tests: ObjectDetector — distance estimation, zone/clock direction classification,
       label normalisation, confidence gating, priority assignment.

YOLO model loading is patched out completely so no GPU/model file is needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from src.inference.detector import ObjectDetector

@pytest.fixture
def detector():
    """Detector with YOLO patched out."""
    with patch('src.inference.detector.YOLO_AVAILABLE', False):
        d = ObjectDetector(model_path='yolov8n.pt', conf=0.45)
    d.class_names = {
        0: 'person', 1: 'chair', 2: 'door', 3: 'stairs',
        4: 'couch', 5: 'potted plant', 6: 'tv',
    }
    return d

class TestClockDirection:
    """Feature: Object position reported as clock direction (10–2 o'clock)."""

    def test_far_left_is_10_oclock(self, detector):
        assert detector._clock_direction(cx=50, frame_w=640) == "10 o'clock"

    def test_center_left_is_11_oclock(self, detector):
        assert detector._clock_direction(cx=200, frame_w=640) == "11 o'clock"

    def test_center_is_12_oclock(self, detector):
        assert detector._clock_direction(cx=320, frame_w=640) == "12 o'clock"

    def test_center_right_is_1_oclock(self, detector):
        assert detector._clock_direction(cx=450, frame_w=640) == "1 o'clock"

    def test_far_right_is_2_oclock(self, detector):
        assert detector._clock_direction(cx=620, frame_w=640) == "2 o'clock"

class TestDetectorZoneClassification:
    """Feature: Classify detected object into left/center/right zone."""

    def test_left_zone(self, detector):
        assert detector._classify_zone(cx=100, left_bound=224, right_bound=416) == 'left'

    def test_center_zone(self, detector):
        assert detector._classify_zone(cx=320, left_bound=224, right_bound=416) == 'center'

    def test_right_zone(self, detector):
        assert detector._classify_zone(cx=500, left_bound=224, right_bound=416) == 'right'

class TestDistanceEstimation:
    """Feature: Distance in metres reported per detected object using known height model."""

    def test_person_near_distance(self, detector):

        dist = detector._estimate_distance(
            bbox=(0, 0, 100, 700), frame_h=720, label='person', track_id=None
        )
        assert 1.0 < dist < 3.0, f"Expected ~1.7 m, got {dist}"

    def test_chair_distance(self, detector):

        dist = detector._estimate_distance(
            bbox=(100, 0, 300, 350), frame_h=480, label='chair', track_id=None
        )
        assert 0.3 <= dist <= 10.0  

    def test_tiny_bbox_returns_safe_default(self, detector):
        dist = detector._estimate_distance(
            bbox=(0, 0, 5, 5), frame_h=480, label='person', track_id=None
        )
        assert dist == 5.0

    def test_distance_clamped_min(self, detector):
        """Very tall bounding box should be clamped to 0.3m minimum."""
        dist = detector._estimate_distance(
            bbox=(0, 0, 640, 2000), frame_h=480, label='person', track_id=None
        )
        assert dist >= 0.3

    def test_distance_clamped_max(self, detector):
        """Tiny box can't be >10m."""
        dist = detector._estimate_distance(
            bbox=(300, 230, 340, 250), frame_h=480, label='person', track_id=None
        )
        assert dist <= 10.0

    def test_temporal_smoothing_converges(self, detector):
        """Repeated readings with same track_id should converge to a stable value."""
        readings = []
        for i in range(10):
            d = detector._estimate_distance(
                bbox=(100, 50, 250, 400), frame_h=480, label='person', track_id=42
            )
            readings.append(d)
        variance = max(readings) - min(readings)
        assert variance < 0.5, "Temporal smoothing should reduce variance"

    def test_unknown_label_uses_perspective(self, detector):
        """For objects with unknown height, distance comes from perspective."""
        dist = detector._estimate_distance(
            bbox=(200, 0, 300, 300), frame_h=480, label='unknownwidget', track_id=None
        )
        assert 0.3 <= dist <= 10.0

class TestLabelNormalisation:
    """Feature: Synonyms handled — 'sofa' == 'couch' etc."""

    def test_sofa_normalised_to_couch(self, detector):
        assert detector.normalize_label('sofa') == 'couch'

    def test_fridge_normalised(self, detector):
        assert detector.normalize_label('fridge') == 'refrigerator'

    def test_plant_normalised(self, detector):
        assert detector.normalize_label('plant') == 'potted plant'

    def test_unknown_label_passthrough(self, detector):
        assert detector.normalize_label('dragon') == 'dragon'

class TestConfidenceUpdate:
    """Feature: Detection confidence gate can be tuned via remote config."""

    def test_update_conf_clamps_min(self, detector):
        detector.update_conf(-5.0)
        assert detector.conf >= 0.2

    def test_update_conf_clamps_max(self, detector):
        detector.update_conf(100.0)
        assert detector.conf <= 0.9

    def test_update_conf_valid(self, detector):
        detector.update_conf(0.6)
        assert detector.conf == pytest.approx(0.6)

class TestPriorityLabels:
    """Feature: High-priority objects (stairs, person, hole) handled with urgent haptics."""

    def test_person_is_high_priority(self, detector):
        assert 'person' in detector.HIGH_PRIORITY

    def test_stairs_is_high_priority(self, detector):
        assert 'stairs' in detector.HIGH_PRIORITY

    def test_hole_is_high_priority(self, detector):
        assert 'hole' in detector.HIGH_PRIORITY
