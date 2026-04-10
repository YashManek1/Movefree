"""
Tests: Firebase client — set_data, push_child, get_data, domain functions
       (push_telemetry, push_hazard, push_sos, geofence_alert, stream_url).
       HTTP calls are mocked with requests-mock.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import pytest
import time
from unittest.mock import patch, MagicMock
import json

DB_URL = 'https://moovefree-ab842-default-rtdb.asia-southeast1.firebasedatabase.app'
SECRET = 'test_secret_123'
UID = 'test_uid_456'

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv('FIREBASE_DATABASE_URL', DB_URL)
    monkeypatch.setenv('FIREBASE_DATABASE_SECRET', SECRET)

    import importlib, src.firebase_client as fc
    importlib.reload(fc)
    return fc

@pytest.fixture
def fc():
    import importlib, src.firebase_client as mod
    importlib.reload(mod)
    return mod

class TestURLBuilder:
    def test_url_includes_auth(self, fc):
        url = fc._url('sessions/uid/telemetry')
        assert SECRET in url
        assert 'sessions/uid/telemetry.json' in url

    def test_url_strips_leading_slash(self, fc):
        url1 = fc._url('/some/path')
        url2 = fc._url('some/path')
        assert url1 == url2

class TestSetData:
    def test_set_data_makes_put_request(self, fc):
        with patch('requests.put') as mock_put:
            mock_put.return_value.status_code = 200
            fc.set_data('sessions/uid/telemetry', {'battery': 85})
            mock_put.assert_called_once()
            call_url = mock_put.call_args[0][0]
            assert 'telemetry' in call_url

    def test_set_data_silent_on_error(self, fc):
        with patch('requests.put', side_effect=Exception('timeout')):
            fc.set_data('sessions/uid/test', {'x': 1})  

    def test_set_data_skips_when_no_url(self, fc, monkeypatch):
        monkeypatch.setenv('FIREBASE_DATABASE_URL', '')
        import importlib; importlib.reload(fc)
        with patch('requests.put') as mock_put:
            fc.set_data('sessions/uid/test', {})
            mock_put.assert_not_called()

class TestGetData:
    def test_get_data_returns_json(self, fc):
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = {'latitude': 12.34}
            mock_get.return_value.status_code = 200
            result = fc.get_data('sessions/uid/location')
        assert result == {'latitude': 12.34}

    def test_get_data_returns_none_on_error(self, fc):
        with patch('requests.get', side_effect=Exception('no network')):
            assert fc.get_data('sessions/uid/location') is None

class TestPushTelemetry:
    """Feature: Real-time battery/temp/signal/FPS to caretaker dashboard."""

    def test_push_telemetry_includes_timestamp(self, fc):
        with patch('requests.put') as mock_put:
            mock_put.return_value.status_code = 200
            fc.push_telemetry(UID, {'battery': 90, 'temperature': 42.0})
            sent = mock_put.call_args[1]['json']
            assert 'timestamp' in sent
            assert sent['battery'] == 90

    def test_push_telemetry_correct_path(self, fc):
        with patch('requests.put') as mock_put:
            fc.push_telemetry(UID, {})
            url = mock_put.call_args[0][0]
            assert f'sessions/{UID}/telemetry' in url

class TestPushHazard:
    """Feature: High-priority object detections logged to Firebase hazard feed."""

    def test_push_hazard_uses_post(self, fc):
        with patch('requests.post') as mock_post:
            fc.push_hazard(UID, {'label': 'stairs', 'distance': 1.5})
            mock_post.assert_called_once()
            sent = mock_post.call_args[1]['json']
            assert sent['label'] == 'stairs'
            assert 'timestamp' in sent

class TestPushSOS:
    """Feature: SOS alert pushed to Firebase — activates alarm on phone."""

    def test_sos_sets_active_true(self, fc):
        with patch('requests.put') as mock_put:
            fc.push_sos(UID, location={'latitude': 12.34, 'longitude': 77.56})
            sent = mock_put.call_args[1]['json']
            assert sent['active'] is True
            assert 'timestamp' in sent

    def test_sos_includes_location(self, fc):
        loc = {'latitude': 12.34, 'longitude': 77.56}
        with patch('requests.put') as mock_put:
            fc.push_sos(UID, location=loc)
            sent = mock_put.call_args[1]['json']
            assert sent['location'] == loc

    def test_sos_without_location(self, fc):
        with patch('requests.put') as mock_put:
            fc.push_sos(UID, location=None)
            sent = mock_put.call_args[1]['json']
            assert sent['location'] is None

class TestStreamURL:
    """Feature: Laptop stream URL published so caretaker app knows where to connect."""

    def test_stream_url_stored(self, fc):
        url = 'http://192.168.29.252:5000/video_feed'
        with patch('requests.put') as mock_put:
            fc.set_stream_url(UID, url)
            sent = mock_put.call_args[1]['json']
            assert sent == url

class TestGeofenceAlert:
    """Feature: Geofence breach notification to caretaker."""

    def test_geofence_alert_has_breach_flag(self, fc):
        with patch('requests.put') as mock_put:
            fc.push_geofence_alert(UID, {'breach': True, 'message': 'Left safe zone!'})
            sent = mock_put.call_args[1]['json']
            assert sent['breach'] is True
            assert 'timestamp' in sent

class TestRemoteConfig:
    """Feature: Caretaker can change AI sensitivity/volume/mode remotely."""

    def test_returns_empty_dict_when_null(self, fc):
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = None
            result = fc.get_remote_config(UID)
        assert result == {}

    def test_returns_config_when_set(self, fc):
        cfg = {'sensitivity': 70, 'volume': 80, 'mode': 'indoor'}
        with patch('requests.get') as mock_get:
            mock_get.return_value.json.return_value = cfg
            result = fc.get_remote_config(UID)
        assert result == cfg
