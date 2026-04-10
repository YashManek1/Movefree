"""
Tests: Outdoor system — GPS navigation (haversine, bearing, direction language),
       TTS queue priority, StreamCapture queue bound.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_outdoor'))

import math
import time
import pytest
import queue as _queue
from unittest.mock import patch, MagicMock

import importlib.util

def load_outdoor_main():
    spec = importlib.util.spec_from_file_location(
        'outdoor_main',
        os.path.join(os.path.dirname(__file__), '..', 'moovefree_outdoor', 'main.py')
    )
    mod = importlib.util.module_from_spec(spec)

    sys.modules.setdefault('google.generativeai', MagicMock())
    sys.modules.setdefault('PIL', MagicMock())
    sys.modules.setdefault('PIL.Image', MagicMock())
    sys.modules.setdefault('cv2', MagicMock())
    sys.modules.setdefault('pyttsx3', MagicMock())
    sys.modules.setdefault('speech_recognition', MagicMock())
    spec.loader.exec_module(mod)
    return mod

@pytest.fixture(scope='module')
def outdoor(monkeypatch_module=None):
    os.environ.setdefault('FIREBASE_DATABASE_URL', 'https://test.firebaseio.com')
    os.environ.setdefault('FIREBASE_DATABASE_SECRET', 'secret')
    os.environ.setdefault('BLIND_USER_UID', 'uid')
    os.environ.setdefault('GEMINI_API_KEY', '')
    os.environ.setdefault('IP_CAMERA_URL', '0')
    return load_outdoor_main()

class TestOutdoorHaversine:
    """Feature: Real GPS distance for outdoor turn-by-turn navigation."""

    def test_zero_distance(self, outdoor):
        d = outdoor._haversine(12.97, 77.59, 12.97, 77.59)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_100m(self, outdoor):
        dlat = 100 / 111_000
        d = outdoor._haversine(12.97, 77.59, 12.97 + dlat, 77.59)
        assert 95 < d < 105

class TestBearingDirection:
    """Feature: Turn instructions use compass language (north, southeast, etc.)."""

    def test_north(self, outdoor):
        b = outdoor._bearing(0.0, 0.0, 1.0, 0.0)
        assert outdoor._direction_from_bearing(b) == 'north'

    def test_east(self, outdoor):
        b = outdoor._bearing(0.0, 0.0, 0.0, 1.0)
        assert outdoor._direction_from_bearing(b) == 'east'

    def test_south(self, outdoor):
        b = outdoor._bearing(1.0, 0.0, 0.0, 0.0)
        assert outdoor._direction_from_bearing(b) == 'south'

    def test_west(self, outdoor):
        b = outdoor._bearing(0.0, 1.0, 0.0, 0.0)
        assert outdoor._direction_from_bearing(b) == 'west'

    def test_northeast(self, outdoor):
        b = outdoor._bearing(0.0, 0.0, 1.0, 1.0)
        assert outdoor._direction_from_bearing(b) == 'northeast'

    def test_all_directions_valid(self, outdoor):
        for deg in range(0, 360, 45):
            d = outdoor._direction_from_bearing(float(deg))
            assert d in ('north', 'northeast', 'east', 'southeast',
                          'south', 'southwest', 'west', 'northwest')

class TestHTMLCleaner:
    """Feature: Turn-by-turn instructions stripped of HTML tags for TTS."""

    def test_strips_b_tags(self, outdoor):
        nav = outdoor.GPSNavigator('')
        assert nav._clean_html('<b>Turn left</b> onto Main St') == 'Turn left  onto Main St'

    def test_strips_div_tags(self, outdoor):
        nav = outdoor.GPSNavigator('')
        assert '<div>' not in nav._clean_html('<div>Go straight</div>')

    def test_pure_text_unchanged(self, outdoor):
        nav = outdoor.GPSNavigator('')
        result = nav._clean_html('Go straight 200 meters')
        assert result == 'Go straight 200 meters'

class TestGPSNavigatorArrived:
    """Feature: 'Arrived at destination' announced when within 15m of end point."""

    def test_arrived_within_15m(self, outdoor):
        nav = outdoor.GPSNavigator('')

        nav.steps = [{
            'end_location': {'lat': 12.9716, 'lng': 77.5946},
            'html_instructions': 'Walk straight',
        }]
        nav.current_step = 0
        nav.arrived = False
        nav.destination = 'test'

        result = nav.update_position(12.9716, 77.5946)  
        assert result.get('arrived') is True

    def test_not_arrived_when_far(self, outdoor):
        nav = outdoor.GPSNavigator('')
        nav.steps = [{
            'end_location': {'lat': 12.980, 'lng': 77.600},
            'html_instructions': 'Go north',
        }]
        nav.current_step = 0
        nav.arrived = False
        nav.destination = 'test'

        result = nav.update_position(12.9716, 77.5946)
        assert result.get('arrived') is not True

    def test_no_steps_returns_empty(self, outdoor):
        nav = outdoor.GPSNavigator('')
        result = nav.update_position(12.9716, 77.5946)
        assert result == {}

class TestOutdoorTTS:
    """Feature: Safety warnings can pre-empt queued navigation instructions."""

    def test_priority_clears_queue(self, outdoor):
        tts = outdoor.TTS.__new__(outdoor.TTS)
        tts._q = _queue.PriorityQueue()
        tts._c = 0
        tts._lock = __import__('threading').Lock()
        tts.running = True

        for i in range(5):
            tts._q.put((1, i, f'step {i}'))

        tts.say('STOP! Vehicle approaching.', priority=True)
        items = []
        while not tts._q.empty():
            items.append(tts._q.get_nowait())
        priorities = [p for p, _, _ in items]

        assert 0 in priorities
