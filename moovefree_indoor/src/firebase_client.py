import os
import threading
import requests
from dotenv import load_dotenv

load_dotenv()

_DATABASE_URL = os.getenv('FIREBASE_DATABASE_URL', '').rstrip('/')
_SECRET = os.getenv('FIREBASE_DATABASE_SECRET', '')

def _url(path):
    return f"{_DATABASE_URL}/{path.lstrip('/')}.json?auth={_SECRET}"

def set_data(path, data):
    if not _DATABASE_URL:
        return
    try:
        requests.put(_url(path), json=data, timeout=5)
    except Exception:
        pass

def push_child(path, data):
    if not _DATABASE_URL:
        return
    try:
        requests.post(_url(path), json=data, timeout=5)
    except Exception:
        pass

def get_data(path):
    if not _DATABASE_URL:
        return None
    try:
        r = requests.get(_url(path), timeout=5)
        return r.json()
    except Exception:
        return None

def push_telemetry(blind_uid, data):
    import time
    set_data(f'sessions/{blind_uid}/telemetry', {**data, 'timestamp': int(time.time() * 1000)})

def push_hazard(blind_uid, entry):
    import time
    entry['timestamp'] = int(time.time() * 1000)
    push_child(f'sessions/{blind_uid}/hazard_log', entry)

def push_sos(blind_uid, location=None):
    import time
    set_data(f'sessions/{blind_uid}/sos', {
        'active': True,
        'timestamp': int(time.time() * 1000),
        'location': location,
    })

def push_next_turn(blind_uid, instruction, distance=''):
    set_data(f'sessions/{blind_uid}/next_turn', {
        'instruction': instruction,
        'distance': distance,
    })

def push_nav_instruction(blind_uid, message, haptic_code=0):
    """Push an object detection audio instruction to the blind person's phone.
    The phone subscribes to this node, speaks the message, and vibrates if haptic > 0.
    haptic_code: 0=none, 1=left vib, 2=right vib, 3=both (STOP)
    """
    import time as _t
    set_data(f'sessions/{blind_uid}/nav_instruction', {
        'message': message,
        'haptic': haptic_code,
        'ts': int(_t.time() * 1000),
    })

def set_stream_url(blind_uid, url):
    set_data(f'sessions/{blind_uid}/stream_url', url)

def get_remote_config(blind_uid):
    return get_data(f'sessions/{blind_uid}/remote_config') or {}

def get_location(blind_uid):
    return get_data(f'sessions/{blind_uid}/location')

def push_geofence_alert(blind_uid, data):
    import time
    set_data(f'sessions/{blind_uid}/geofence_alert', {**data, 'timestamp': int(time.time() * 1000)})
