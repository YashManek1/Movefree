"""
Tests: GeofenceMonitor — Haversine calculation, breach detection, cooldown,
       no alert when inside zone, alert when outside zone.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import math
import time
import pytest
from unittest.mock import patch, MagicMock

from src.api.geofence_monitor import _haversine_m, GeofenceMonitor

class TestHaversine:
    """Feature: Accurate real-world GPS distance for geofence boundary check."""

    def test_same_point_zero_distance(self):
        d = _haversine_m(12.9716, 77.5946, 12.9716, 77.5946)
        assert d == pytest.approx(0.0, abs=0.01)

    def test_known_distance_bangalore_to_mysore(self):

        d = _haversine_m(12.9716, 77.5946, 12.2958, 76.6394)
        assert 120_000 < d < 150_000, f"Expected ~128km, got {d:.0f}m"

    def test_100m_north(self):

        lat_delta = 100 / 111_000  
        d = _haversine_m(12.9716, 77.5946, 12.9716 + lat_delta, 77.5946)
        assert 95 < d < 105, f"Expected ~100m, got {d:.1f}m"

    def test_symmetry(self):
        d1 = _haversine_m(12.9716, 77.5946, 12.9800, 77.6000)
        d2 = _haversine_m(12.9800, 77.6000, 12.9716, 77.5946)
        assert d1 == pytest.approx(d2, rel=1e-6)

    def test_equatorial_crossing(self):
        d = _haversine_m(0.0, 0.0, 0.0, 1.0)

        assert 110_000 < d < 113_000

class TestGeofenceMonitorBreachDetection:
    """Feature: Caretaker receives alert when patient leaves safe zone."""

    def _make_monitor(self):
        with patch('src.api.geofence_monitor.fb') as mock_fb:
            mon = GeofenceMonitor(blind_uid='test_uid', check_interval=10.0)
            mon._fb = mock_fb
        return mon

    def test_inside_zone_no_alert(self):
        mon = GeofenceMonitor('uid')

        with patch('src.api.geofence_monitor.fb') as mock_fb:

            mock_fb.get_location.return_value = {'latitude': 12.9716, 'longitude': 77.5946}
            mock_fb.get_data.return_value = {
                'zones': [{'lat': 12.9716, 'lng': 77.5946, 'radius': 100}]
            }
            mon._check()
            mock_fb.push_geofence_alert.assert_not_called()

    def test_outside_zone_triggers_alert(self):
        mon = GeofenceMonitor('uid')
        with patch('src.api.geofence_monitor.fb') as mock_fb:

            mock_fb.get_location.return_value = {'latitude': 12.976, 'longitude': 77.5946}
            mock_fb.get_data.return_value = {
                'zones': [{'lat': 12.9716, 'lng': 77.5946, 'radius': 100}]
            }
            mon._check()
            mock_fb.push_geofence_alert.assert_called_once()
            alert = mock_fb.push_geofence_alert.call_args[0][1]
            assert alert['breach'] is True

    def test_alert_not_sent_in_cooldown(self):
        mon = GeofenceMonitor('uid')
        mon._last_alert_time = time.time()  

        with patch('src.api.geofence_monitor.fb') as mock_fb:
            mock_fb.get_location.return_value = {'latitude': 12.976, 'longitude': 77.5946}
            mock_fb.get_data.return_value = {
                'zones': [{'lat': 12.9716, 'lng': 77.5946, 'radius': 100}]
            }
            mon._check()
            mock_fb.push_geofence_alert.assert_not_called()

    def test_alert_sent_after_cooldown_expires(self):
        mon = GeofenceMonitor('uid')
        mon._last_alert_time = time.time() - 35  

        with patch('src.api.geofence_monitor.fb') as mock_fb:
            mock_fb.get_location.return_value = {'latitude': 12.976, 'longitude': 77.5946}
            mock_fb.get_data.return_value = {
                'zones': [{'lat': 12.9716, 'lng': 77.5946, 'radius': 100}]
            }
            mon._check()
            mock_fb.push_geofence_alert.assert_called_once()

    def test_no_location_data_skips_check(self):
        mon = GeofenceMonitor('uid')
        with patch('src.api.geofence_monitor.fb') as mock_fb:
            mock_fb.get_location.return_value = None
            mon._check()
            mock_fb.push_geofence_alert.assert_not_called()

    def test_no_zones_skips_check(self):
        mon = GeofenceMonitor('uid')
        with patch('src.api.geofence_monitor.fb') as mock_fb:
            mock_fb.get_location.return_value = {'latitude': 12.976, 'longitude': 77.5946}
            mock_fb.get_data.return_value = None
            mon._check()
            mock_fb.push_geofence_alert.assert_not_called()

    def test_multiple_zones_inside_any_is_safe(self):
        mon = GeofenceMonitor('uid')
        with patch('src.api.geofence_monitor.fb') as mock_fb:

            mock_fb.get_location.return_value = {'latitude': 12.9720, 'longitude': 77.5946}
            mock_fb.get_data.return_value = {
                'zones': [
                    {'lat': 0.0, 'lng': 0.0, 'radius': 50},        
                    {'lat': 12.9720, 'lng': 77.5946, 'radius': 100},  
                ]
            }
            mon._check()
            mock_fb.push_geofence_alert.assert_not_called()

    def test_start_stop_lifecycle(self):
        mon = GeofenceMonitor('uid', check_interval=100.0)
        mon.start()
        assert mon.running is True
        mon.stop()
        assert mon.running is False
