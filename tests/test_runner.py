"""
Tests: Orchestrator runner — mode fetching, process lifecycle, Firebase push.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest
from unittest.mock import MagicMock, patch, call

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv('FIREBASE_DATABASE_URL', 'https://test.firebaseio.com')
    monkeypatch.setenv('FIREBASE_DATABASE_SECRET', 'secret')
    monkeypatch.setenv('BLIND_USER_UID', 'test_uid')

def _import_orchestrator():
    """Import runner fresh with env patched."""
    import importlib
    import runner as r
    importlib.reload(r)
    return r

class TestGetRemoteMode:
    """Feature: Orchestrator reads Firebase to decide indoor vs outdoor."""

    def test_returns_indoor_when_firebase_says_indoor(self, monkeypatch):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = 'indoor'
            r = _import_orchestrator()
            assert r.get_remote_mode() == 'indoor'

    def test_returns_outdoor(self, monkeypatch):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = 'outdoor'
            r = _import_orchestrator()
            assert r.get_remote_mode() == 'outdoor'

    def test_defaults_to_indoor_on_error(self, monkeypatch):
        with patch('requests.get', side_effect=Exception('no internet')):
            r = _import_orchestrator()
            assert r.get_remote_mode() == 'indoor'

    def test_defaults_to_indoor_on_invalid_value(self, monkeypatch):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = 'spacewalk'
            r = _import_orchestrator()
            assert r.get_remote_mode() == 'indoor'

    def test_defaults_to_indoor_on_null_response(self, monkeypatch):
        with patch('requests.get') as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json.return_value = None
            r = _import_orchestrator()
            assert r.get_remote_mode() == 'indoor'

class TestOrchestratorProcessLifecycle:
    """Feature: Orchestrator seamlessly switches between indoor/outdoor pipelines."""

    def test_starts_indoor_process(self):
        r = _import_orchestrator()
        orc = r.MooveFreeOrchestrator()
        with patch('subprocess.Popen') as mock_popen,             patch('requests.put'):
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            orc.start_process('indoor')
            mock_popen.assert_called_once()
            args = mock_popen.call_args
            assert 'moovefree_indoor' in args[1].get('cwd', '')

    def test_starts_outdoor_process(self):
        r = _import_orchestrator()
        orc = r.MooveFreeOrchestrator()
        with patch('subprocess.Popen') as mock_popen,             patch('requests.put'):
            orc.start_process('outdoor')
            args = mock_popen.call_args
            assert 'moovefree_outdoor' in args[1].get('cwd', '')

    def test_terminates_previous_process_on_switch(self):
        r = _import_orchestrator()
        orc = r.MooveFreeOrchestrator()
        mock_old = MagicMock()
        orc.process = mock_old
        orc.current_mode = 'indoor'

        with patch('subprocess.Popen', return_value=MagicMock()),             patch('requests.put'):
            orc.start_process('outdoor')
            mock_old.terminate.assert_called_once()
            mock_old.wait.assert_called_once()

    def test_current_mode_updated(self):
        r = _import_orchestrator()
        orc = r.MooveFreeOrchestrator()
        with patch('subprocess.Popen', return_value=MagicMock()),             patch('requests.put'):
            orc.start_process('outdoor')
            assert orc.current_mode == 'outdoor'

    def test_mode_pushed_to_firebase_on_start(self):
        r = _import_orchestrator()
        orc = r.MooveFreeOrchestrator()
        with patch('subprocess.Popen', return_value=MagicMock()),             patch('requests.put') as mock_put:
            orc.start_process('indoor')
            mock_put.assert_called_once()
            assert mock_put.call_args[1]['json'] == 'indoor'
