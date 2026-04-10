"""
Tests: TelemetryServer Flask API — /status, /config, /health endpoints.
       No real system or hardware needed.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'moovefree_indoor'))

import json
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture
def system():
    """Minimal mock of MooveFreeIndoorSystem."""
    sys_ref = MagicMock()
    sys_ref.running = True
    sys_ref.fps = 28.5
    sys_ref.tracking_mode = 'auto'
    sys_ref.cached_detections = [MagicMock(), MagicMock()]
    sys_ref.hw.ultrasonic_dist = 2.3
    return sys_ref

@pytest.fixture
def client(system):
    from src.api.telemetry_server import create_telemetry_app
    app = create_telemetry_app(system)
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c, system

class TestStatusEndpoint:
    """Feature: Caretaker can see real-time system status."""

    def test_status_200(self, client):
        c, _ = client
        r = c.get('/status')
        assert r.status_code == 200

    def test_status_contains_fps(self, client):
        c, _ = client
        data = json.loads(c.get('/status').data)
        assert data['fps'] == pytest.approx(28.5)

    def test_status_contains_running(self, client):
        c, _ = client
        data = json.loads(c.get('/status').data)
        assert data['running'] is True

    def test_status_contains_mode(self, client):
        c, _ = client
        data = json.loads(c.get('/status').data)
        assert data['mode'] == 'auto'

    def test_status_contains_sonar(self, client):
        c, _ = client
        data = json.loads(c.get('/status').data)
        assert data['sonar'] == pytest.approx(2.3)

    def test_status_contains_detections(self, client):
        c, _ = client
        data = json.loads(c.get('/status').data)
        assert data['detections'] == 2

class TestHealthEndpoint:
    """Feature: Heartbeat endpoint for connectivity check."""

    def test_health_200(self, client):
        c, _ = client
        r = c.get('/health')
        assert r.status_code == 200

    def test_health_alive_true(self, client):
        c, _ = client
        data = json.loads(c.get('/health').data)
        assert data['alive'] is True

    def test_health_has_timestamp(self, client):
        c, _ = client
        data = json.loads(c.get('/health').data)
        assert 'ts' in data
        assert isinstance(data['ts'], int)

class TestConfigEndpoint:
    """Feature: Caretaker can remotely tune AI sensitivity and volume."""

    def test_config_sensitivity_updates_detector(self, client):
        c, system = client
        r = c.post('/config', json={'sensitivity': 70})
        assert r.status_code == 200
        system.detector.update_conf.assert_called_once_with(0.7)

    def test_config_volume_updates_audio(self, client):
        c, system = client
        r = c.post('/config', json={'volume': 80})
        assert r.status_code == 200
        system.audio.set_volume.assert_called_once_with(0.8)

    def test_config_mode_sets_force_mode(self, client):
        c, system = client
        r = c.post('/config', json={'mode': 'outdoor'})
        assert r.status_code == 200
        assert system.force_mode == 'outdoor'

    def test_config_empty_body_no_crash(self, client):
        c, _ = client
        r = c.post('/config', data='', content_type='application/json')
        assert r.status_code == 200

    def test_config_sensitivity_boundary_100(self, client):
        c, system = client
        c.post('/config', json={'sensitivity': 100})
        system.detector.update_conf.assert_called_once_with(1.0)

    def test_config_sensitivity_boundary_0(self, client):
        c, system = client
        c.post('/config', json={'sensitivity': 0})
        system.detector.update_conf.assert_called_once_with(0.0)
