"""
Tests: StreamLoader — reconnect logic, frame queue, stop behaviour.
       cv2.VideoCapture is mocked so no real camera is needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import time
import queue
import pytest
import numpy as np
from unittest.mock import MagicMock, patch

def _make_cap(frames_ok=5, fail_after=None):
    """Build a mock VideoCapture that yields black frames then fails."""
    cap = MagicMock()
    cap.isOpened.return_value = True
    black = np.zeros((480, 640, 3), dtype=np.uint8)
    call_count = [0]

    def read_side_effect():
        call_count[0] += 1
        if fail_after is not None and call_count[0] > fail_after:
            return False, None
        return True, black.copy()

    cap.read.side_effect = read_side_effect
    return cap

class TestStreamLoader:
    """Feature: Continuous MJPEG stream with auto-reconnect and frame queue."""

    def test_read_returns_frame(self):
        cap = _make_cap()
        with patch('cv2.VideoCapture', return_value=cap):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('0', reconnect_delay=0.1)
            time.sleep(0.3)
            frame = sl.read(timeout=1.0)
            sl.stop()
        assert frame is not None
        assert frame.shape == (480, 640, 3)

    def test_read_returns_none_when_no_frames(self):
        cap = MagicMock()
        cap.isOpened.return_value = False

        with patch('cv2.VideoCapture', return_value=cap):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('http://fake.url/video', reconnect_delay=0.1, max_reconnect_delay=0.2)
            frame = sl.read(timeout=0.3)
            sl.stop()
        assert frame is None

    def test_reconnects_after_stream_failure(self):
        call_count = [0]
        cap_ok = _make_cap()
        cap_fail = MagicMock()
        cap_fail.isOpened.return_value = False

        def cap_factory(*args, **kwargs):
            call_count[0] += 1
            return cap_fail if call_count[0] == 1 else cap_ok

        with patch('cv2.VideoCapture', side_effect=cap_factory):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('0', reconnect_delay=0.1, max_reconnect_delay=0.5)
            time.sleep(0.5)
            frame = sl.read(timeout=1.0)
            sl.stop()
        assert call_count[0] >= 2, "Should try at least 2 times (first fail, second ok)"

    def test_queue_is_bounded(self):
        """Only the latest frames are kept — old ones discarded to avoid memory growth."""
        cap = _make_cap()
        with patch('cv2.VideoCapture', return_value=cap):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('0', reconnect_delay=0.1)
            time.sleep(0.5)
            assert sl._frame_queue.maxsize == 2
            sl.stop()

    def test_connected_flag_set(self):
        cap = _make_cap()
        with patch('cv2.VideoCapture', return_value=cap):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('0', reconnect_delay=0.1)
            time.sleep(0.3)
            assert sl.connected is True
            sl.stop()

    def test_stop_sets_running_false(self):
        cap = _make_cap()
        with patch('cv2.VideoCapture', return_value=cap):
            from src.utils.stream_loader import StreamLoader
            sl = StreamLoader('0', reconnect_delay=0.1)
            sl.stop()
        assert sl.running is False
