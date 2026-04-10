import threading
import time
import logging
import math
import os
from src import firebase_client as fb
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger('GeofenceMonitor')

EARTH_RADIUS_M = 6371000

def _haversine_m(lat1, lng1, lat2, lng2):
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng / 2) ** 2
    return EARTH_RADIUS_M * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

class GeofenceMonitor:
    def __init__(self, blind_uid: str, check_interval: float = 10.0):
        self.blind_uid = blind_uid
        self.check_interval = check_interval
        self.running = False
        self._last_alert_time = 0
        self._alert_cooldown = 30.0

    def start(self):
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()
        logger.info('Geofence monitor started.')

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            try:
                self._check()
            except Exception as e:
                logger.error(f'Geofence check error: {e}')
            time.sleep(self.check_interval)

    def _check(self):
        location = fb.get_location(self.blind_uid)
        if not location or 'latitude' not in location:
            return

        geofence = fb.get_data(f'sessions/{self.blind_uid}/geofence')
        if not geofence or not geofence.get('zones'):
            return

        user_lat = location['latitude']
        user_lng = location['longitude']
        zones = geofence['zones']

        inside_any = any(
            _haversine_m(user_lat, user_lng, z['lat'], z['lng']) <= z.get('radius', 100)
            for z in zones
        )

        if not inside_any:
            now = time.time()
            if now - self._last_alert_time > self._alert_cooldown:
                self._last_alert_time = now
                logger.warning(f'GEOFENCE BREACH: user at ({user_lat:.5f}, {user_lng:.5f})')
                fb.push_geofence_alert(self.blind_uid, {
                    'breach': True,
                    'user_lat': user_lat,
                    'user_lng': user_lng,
                    'message': 'Patient has left the safe zone!',
                })
