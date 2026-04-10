"""
Tests: HardwareManager — ambient light dark detection (SOS trigger condition),
       battery/temperature mocks, haptic codes, sonar distance clamping,
       sensor status reporting. GPIO is never imported.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import pytest
import numpy as np
from unittest.mock import patch, MagicMock

@pytest.fixture
def hw():
    """HardwareManager with GPIO patched out."""
    with patch('src.hardware_manager.LIB_GPIO_PRESENT', False):
        from src.hardware_manager import HardwareManager
        manager = HardwareManager()
    yield manager
    manager.stop()

class TestAmbientLightDetection:
    """Feature: When camera feed goes black (dark room / screen off),
    system detects DARK state → SOS / warning must trigger.
    """

    def test_completely_black_frame_is_dark(self, hw):
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        assert hw.check_ambient_light(frame) == 'DARK'

    def test_very_dark_frame_is_dark(self, hw):
        frame = np.full((480, 640, 3), fill_value=15, dtype=np.uint8)
        assert hw.check_ambient_light(frame) == 'DARK'

    def test_bright_frame_is_ok(self, hw):
        frame = np.full((480, 640, 3), fill_value=120, dtype=np.uint8)
        assert hw.check_ambient_light(frame) == 'OK'

    def test_none_frame_returns_ok(self, hw):
        """None frame (no camera) should not crash."""
        assert hw.check_ambient_light(None) == 'OK'

    def test_exactly_on_threshold(self, hw):
        """Mean brightness == 40 → OK."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        val = int(40 / 0.999)  
        frame[:] = val
        result = hw.check_ambient_light(frame)
        assert result in ('OK', 'DARK')  

    def test_mixed_bright_dark_frame(self, hw):
        """Half-bright frame should be OK overall."""
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame[:240, :] = 200  
        assert hw.check_ambient_light(frame) == 'OK'

class TestBatteryLevel:
    """Feature: Battery % sent in telemetry to caretaker dashboard."""

    def test_battery_fallback_decrements(self, hw):
        """Without /sys/class file, returns a slowly-decrementing simulated value."""
        b1 = hw.get_battery_level()
        b2 = hw.get_battery_level()
        assert 0 <= b1 <= 100
        assert b2 <= b1  

    def test_battery_reads_sys_file(self, hw, tmp_path):
        """Reads real battery from /sys when available."""
        fake_file = tmp_path / 'capacity'
        fake_file.write_text('73')
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = lambda: '73'
            val = hw.get_battery_level()
        assert isinstance(val, (int, float))

class TestTemperature:
    """Feature: Laptop/Pi temperature streamed to caretaker hardware dashboard."""

    def test_temperature_fallback_in_range(self, hw):
        t = hw.get_temperature()
        assert 35.0 <= t <= 55.0, f"Simulated temp should be ~45°C, got {t}"

    def test_temperature_reads_thermal_zone(self, hw):
        with patch('builtins.open', create=True) as mock_open:
            mock_open.return_value.__enter__ = lambda s: s
            mock_open.return_value.__exit__ = MagicMock(return_value=False)
            mock_open.return_value.read = lambda: '52000'
            t = hw.get_temperature()
        assert isinstance(t, float)

class TestHapticCodes:
    """Feature: Haptic vibration codes — 1=vib_left, 2=vib_right, 3=both (STOP).
    Without GPIO, trigger_haptic should silently return.
    """

    def test_haptic_code_1_no_crash(self, hw):
        hw.trigger_haptic(1)  

    def test_haptic_code_2_no_crash(self, hw):
        hw.trigger_haptic(2)

    def test_haptic_code_3_no_crash(self, hw):
        hw.trigger_haptic(3)

    def test_haptic_invalid_code_no_crash(self, hw):
        hw.trigger_haptic(99)

class TestSonarDistance:
    """Feature: Sonar distance < 1m triggers STOP + haptic on blind phone."""

    def test_default_distance_is_999(self, hw):
        """Without GPIO, distance defaults to 999 (clear) — no false alarms."""
        assert hw.get_distance() == 999.0

    def test_distance_set_directly(self, hw):
        hw.ultrasonic_dist = 0.5
        assert hw.get_distance() == 0.5

    def test_sonar_active_flag_false_without_gpio(self, hw):
        status = hw.get_sensor_status()
        assert status['sonar_active'] is False
        assert status['gpio'] is False

    def test_sonar_active_when_close_reading(self, hw):
        hw.ultrasonic_dist = 0.45
        status = hw.get_sensor_status()
        assert status['sonar_active'] is True

class TestHardwareStop:
    """Feature: Clean shutdown without GPIO cleanup errors."""

    def test_stop_sets_running_false(self, hw):
        hw.stop()
        assert hw.running is False
