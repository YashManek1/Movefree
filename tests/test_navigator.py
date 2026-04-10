"""
Tests: Navigator — Zone classification, critical warnings, guidance messages,
       crowding detection, exit finding, anti-jitter state machine.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import time
import pytest
from src.inference.navigator import ZoneBasedNavigator, Detection

def make_detection(label='chair', distance=2.0, zone='center', priority=0,
                   clock_dir="12 o'clock", confidence=0.80,
                   bbox=(260, 100, 380, 380)):
    return Detection(
        label=label, confidence=confidence, bbox=bbox,
        distance=distance, zone=zone, priority=priority,
        clock_dir=clock_dir,
    )

@pytest.fixture
def nav():
    return ZoneBasedNavigator(frame_width=640, frame_height=480)

class TestZoneClassification:
    """Feature: Object zone (left/center/right) from bounding box x-center."""

    def test_center_object_classified_center(self, nav):

        assert nav.classify_zone((270, 100, 370, 400)) == 'center'

    def test_left_object_classified_left(self, nav):

        assert nav.classify_zone((0, 100, 100, 400)) == 'left'

    def test_right_object_classified_right(self, nav):

        assert nav.classify_zone((550, 100, 640, 400)) == 'right'

    def test_boundary_left_edge(self, nav):
        assert nav.classify_zone((0, 0, 1, 480)) == 'left'

    def test_boundary_right_edge(self, nav):
        assert nav.classify_zone((639, 0, 640, 480)) == 'right'

    def test_exact_left_boundary_is_center(self, nav):

        cx = nav.left_boundary
        assert nav.classify_zone((cx - 1, 0, cx + 1, 480)) == 'center'

class TestCriticalWarnings:
    """Feature: Immediate audible critical warning for high-priority obstacles."""

    def test_stairs_close_triggers_warning(self, nav):
        d = make_detection(label='stairs', distance=2.0, zone='center', priority=2)
        msg = nav.get_critical_warning([d])
        assert msg is not None
        assert 'stairs' in msg.lower()
        assert '2.0' in msg

    def test_stairs_far_no_warning(self, nav):
        d = make_detection(label='stairs', distance=3.0, zone='center', priority=2)
        assert nav.get_critical_warning([d]) is None

    def test_center_object_stop_distance_warning(self, nav):
        d = make_detection(label='wall', distance=0.8, zone='center', priority=0)
        msg = nav.get_critical_warning([d])
        assert msg is not None
        assert 'stop' in msg.lower()

    def test_object_at_stop_threshold_exact(self, nav):

        d = make_detection(label='box', distance=1.0, zone='center', priority=0)
        msg = nav.get_critical_warning([d])
        assert msg is None  

    def test_object_just_inside_stop_threshold(self, nav):

        d = make_detection(label='box', distance=0.99, zone='center', priority=0)
        msg = nav.get_critical_warning([d])
        assert msg is not None

    def test_side_object_does_not_trigger_stop(self, nav):
        d = make_detection(label='wall', distance=0.5, zone='left', priority=0)
        assert nav.get_critical_warning([d]) is None

    def test_person_very_close_triggers_warning(self, nav):
        d = make_detection(label='person', distance=1.5, zone='center', priority=2)
        msg = nav.get_critical_warning([d])
        assert msg is not None
        assert 'person' in msg.lower()

    def test_person_1p8m_boundary_triggers(self, nav):
        d = make_detection(label='person', distance=1.799, zone='center', priority=2)
        msg = nav.get_critical_warning([d])
        assert msg is not None

    def test_person_far_no_critical(self, nav):
        d = make_detection(label='person', distance=2.5, zone='center', priority=2)

        result = nav.get_critical_warning([d])

        assert result is None

    def test_no_detections_no_warning(self, nav):
        assert nav.get_critical_warning([]) is None

    def test_hole_warning(self, nav):
        d = make_detection(label='hole', distance=1.0, zone='center', priority=2)
        msg = nav.get_critical_warning([d])
        assert msg is not None
        assert 'hole' in msg.lower()

class TestNavigationGuidance:
    """Feature: Navigation turn instructions with anti-jitter (state counter).
    Object audio feedback: label, clock direction, distance in meters.
    """

    def _run_n_frames(self, nav, detections, n):
        for _ in range(n):
            result = nav.analyze_detections(detections)
        return result

    def test_blocked_center_only_reports_after_3_frames(self, nav):

        dets = [
            make_detection(label='wall', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='left', priority=0),
            make_detection(label='box', distance=1.5, zone='right', priority=0),
        ]
        for i in range(4):  
            res = nav.analyze_detections(dets)
        assert res['message'] is not None
        assert 'blocked' in res['message'].lower() or 'stop' in res['message'].lower()

    def test_turn_left_suggestion(self, nav):

        dets = [
            make_detection(label='chair', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='right', priority=0),
        ]
        for _ in range(6):
            res = nav.analyze_detections(dets)
        assert res['message'] is not None
        assert 'left' in res['message'].lower()

    def test_turn_right_suggestion(self, nav):

        dets = [
            make_detection(label='chair', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='left', priority=0),
        ]
        for _ in range(6):
            res = nav.analyze_detections(dets)
        assert res['message'] is not None
        assert 'right' in res['message'].lower()

    def test_clear_path_announces_clear(self, nav):
        nav.last_clear_time = 0  
        res = nav.analyze_detections([])
        assert res['message'] is not None
        assert 'clear' in res['message'].lower()

    def test_haptic_code_for_blocked(self, nav):
        dets = [
            make_detection(label='wall', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='left', priority=0),
            make_detection(label='box', distance=1.5, zone='right', priority=0),
        ]
        res = nav.analyze_detections(dets)
        assert res['haptic'] == 3

    def test_haptic_code_for_turn_left(self, nav):
        dets = [
            make_detection(label='chair', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='right', priority=0),
        ]
        res = nav.analyze_detections(dets)
        assert res['haptic'] == 1

    def test_haptic_code_for_turn_right(self, nav):
        dets = [
            make_detection(label='chair', distance=1.5, zone='center', priority=0),
            make_detection(label='box', distance=1.5, zone='left', priority=0),
        ]
        res = nav.analyze_detections(dets)
        assert res['haptic'] == 2

    def test_objects_beyond_warn_dist_ignored(self, nav):
        dets = [make_detection(label='wall', distance=5.0, zone='center', priority=0)]
        res = nav.analyze_detections(dets)
        assert res['haptic'] == 0

class TestCrowdingDetection:
    """Feature: Detect when ≥3 people are nearby and warn."""

    def test_three_people_triggers_crowding(self, nav):
        dets = [make_detection(label='person', distance=2.0, zone='center') for _ in range(3)]
        msg = nav.detect_crowding(dets)
        assert msg is not None
        assert '3' in msg

    def test_two_people_no_crowding(self, nav):
        dets = [make_detection(label='person', distance=2.0, zone='center') for _ in range(2)]
        assert nav.detect_crowding(dets) is None

    def test_people_beyond_3m_not_counted(self, nav):
        dets = [make_detection(label='person', distance=3.5, zone='center') for _ in range(5)]
        assert nav.detect_crowding(dets) is None

    def test_mixed_crowd_count(self, nav):
        close = [make_detection(label='person', distance=2.0) for _ in range(4)]
        far = [make_detection(label='person', distance=4.0) for _ in range(3)]
        msg = nav.detect_crowding(close + far)
        assert msg is not None
        assert '4' in msg

class TestExitFinding:
    """Feature: Voice-announce location of nearest door/exit."""

    def test_finds_door(self, nav):
        d = make_detection(label='door', distance=3.0, zone='left', clock_dir="10 o'clock")
        msg = nav.find_exit([d])
        assert msg is not None
        assert 'door' in msg.lower()
        assert "10 o'clock" in msg

    def test_no_door_returns_none(self, nav):
        d = make_detection(label='chair', distance=1.0, zone='center')
        assert nav.find_exit([d]) is None

    def test_multiple_doors_returns_closest(self, nav):
        d1 = make_detection(label='door', distance=5.0, clock_dir="10 o'clock")
        d2 = make_detection(label='door', distance=2.0, clock_dir="2 o'clock")
        msg = nav.find_exit([d1, d2])
        assert '2.0' in msg
