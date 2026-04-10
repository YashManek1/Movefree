"""
Tests: AudioFeedback — priority queue, volume/rate control, stop semantics.
       pyttsx3 engine is mocked so no audio hardware is needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import time
import queue
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def audio():
    """AudioFeedback with pyttsx3 fully mocked."""
    mock_engine = MagicMock()
    with patch('pyttsx3.init', return_value=mock_engine):
        from src.inference.audio_feedback import AudioFeedback
        af = AudioFeedback(rate=200, volume=0.8)
        af._engine = mock_engine
    yield af
    af.stop()

class TestAudioPriority:
    """Feature: Priority messages (SOS, stop warnings) skip ahead of queued items."""

    def test_priority_message_clears_queue(self, audio):
        """When priority=True is spoken, normal messages in queue are discarded."""

        for i in range(5):
            audio._queue.put((1, i, f'normal message {i}'))

        audio.speak('STOP! Obstacle!', priority=True)

        items = []
        while not audio._queue.empty():
            items.append(audio._queue.get_nowait())

        priorities = [p for p, _, _ in items]
        assert 0 in priorities, "Priority message must be in queue with priority_val=0"

    def test_normal_message_enqueued(self, audio):
        audio.speak('Path clear.')
        assert not audio._queue.empty()
        pv, _, text = audio._queue.get_nowait()
        assert text == 'Path clear.'
        assert pv == 1  

    def test_priority_message_has_lower_priority_value(self, audio):
        audio.speak('EMERGENCY', priority=True)
        pv, _, _ = audio._queue.get_nowait()
        assert pv == 0

    def test_empty_text_not_enqueued(self, audio):
        before = audio._queue.qsize()
        audio.speak('')
        assert audio._queue.qsize() == before

    def test_none_text_not_enqueued(self, audio):
        before = audio._queue.qsize()
        audio.speak(None)
        assert audio._queue.qsize() == before

class TestAudioSettings:
    """Feature: Remote config can tune TTS volume (0–1) and rate."""

    def test_volume_clamped_min(self, audio):
        audio.set_volume(-5.0)
        assert audio._volume == 0.0

    def test_volume_clamped_max(self, audio):
        audio.set_volume(10.0)
        assert audio._volume == 1.0

    def test_volume_set_valid(self, audio):
        audio.set_volume(0.6)
        assert audio._volume == pytest.approx(0.6)

    def test_rate_set(self, audio):
        audio.set_rate(180)
        assert audio._rate == 180

class TestAudioStop:
    """Feature: Stop cleanly without leaving zombie threads."""

    def test_stop_sets_running_false(self, audio):
        audio.stop()
        assert audio.running is False

    def test_stop_puts_sentinel(self, audio):
        audio.stop()

        found_sentinel = False
        while not audio._queue.empty():
            _, _, text = audio._queue.get_nowait()
            if text is None:
                found_sentinel = True
        assert found_sentinel
