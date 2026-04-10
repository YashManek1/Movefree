import cv2
import time
import threading
import logging
import os
import math
import socket
from flask import Flask, Response
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from moovefree_indoor.src.hardware_manager import HardwareManager

try:
    import google.generativeai as genai
    from PIL import Image
    import pyttsx3
    import speech_recognition as sr
    DEPS_OK = True
except ImportError:
    DEPS_OK = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('ultralytics').setLevel(logging.WARNING)
logger = logging.getLogger('MooveFreeOutdoor')

_flask_app = Flask(__name__)
_latest_frame = None
_frame_lock = threading.Lock()

GEMINI_KEY = os.getenv('GEMINI_API_KEY', '')
DB_URL = os.getenv('FIREBASE_DATABASE_URL', '').rstrip('/')
DB_SECRET = os.getenv('FIREBASE_DATABASE_SECRET', '')
BLIND_UID = os.getenv('BLIND_USER_UID', '')
IP_CAM_URL = os.getenv('IP_CAMERA_URL', '0')
GOOGLE_API_KEY = os.getenv('GOOGLE_MAPS_API_KEY', '')

EARTH_RADIUS = 6371000

def _haversine(lat1, lon1, lat2, lon2):
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    return EARTH_RADIUS * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

def _bearing(lat1, lon1, lat2, lon2):
    l1, l2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(l2)
    x = math.cos(l1) * math.sin(l2) - math.sin(l1) * math.cos(l2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360

def _direction_from_bearing(b):
    dirs = ['north', 'northeast', 'east', 'southeast', 'south', 'southwest', 'west', 'northwest', 'north']
    return dirs[round(b / 45) % 8]

def fb_get(path):
    import requests
    try:
        r = requests.get(f'{DB_URL}/{path}.json?auth={DB_SECRET}', timeout=5)
        return r.json()
    except Exception:
        return None

def fb_set(path, data):
    import requests
    try:
        requests.put(f'{DB_URL}/{path}.json?auth={DB_SECRET}', json=data, timeout=5)
    except Exception:
        pass

def fb_push_child(path, data):
    import requests
    import time as _t
    try:
        data['timestamp'] = int(_t.time() * 1000)
        requests.post(f'{DB_URL}/{path}.json?auth={DB_SECRET}', json=data, timeout=5)
    except Exception:
        pass

def fb_push_nav(message, haptic=0):
    """Push nav instruction to the blind phone so it speaks + vibrates.
    haptic: 0=none, 1=left, 2=right, 3=stop/both
    """
    import time as _t
    if not BLIND_UID or not DB_URL:
        return
    fb_set(f'sessions/{BLIND_UID}/nav_instruction', {
        'message': message,
        'haptic': haptic,
        'ts': int(_t.time() * 1000),
    })

@_flask_app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with _frame_lock:
                if _latest_frame is None:
                    time.sleep(0.033)
                    continue
                ret, encoded = cv2.imencode('.jpg', _latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
            if not ret:
                continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

class TTS:
    def __init__(self, rate=145):
        import queue
        self._q = queue.PriorityQueue()
        self._c = 0
        self._lock = threading.Lock()
        self.running = True
        threading.Thread(target=self._run, args=(rate,), daemon=True).start()

    def _run(self, rate):
        engine = pyttsx3.init()
        engine.setProperty('rate', rate)
        while self.running:
            try:
                _, _, t = self._q.get(timeout=0.5)
                if t is None:
                    break
                engine.say(t)
                engine.runAndWait()
            except Exception:
                pass

    def say(self, text, priority=False):
        with self._lock:
            if priority:
                while not self._q.empty():
                    try:
                        self._q.get_nowait()
                    except Exception:
                        break
            pv = 0 if priority else 1
            self._c += 1
            self._q.put((pv, self._c, text))

class StreamCapture:
    def __init__(self, src):
        import queue
        self._frames = queue.Queue(maxsize=2)
        self.connected = False
        self.running = True
        threading.Thread(target=self._loop, args=(src,), daemon=True).start()

    def _loop(self, src):
        delay = 2.0
        while self.running:
            if isinstance(src, str) and src.startswith('http'):
                cap = cv2.VideoCapture(src, cv2.CAP_FFMPEG)
            else:
                cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                time.sleep(delay)
                delay = min(delay * 1.5, 30)
                continue
            self.connected = True
            delay = 2.0
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    self._frames.get_nowait()
                except Exception:
                    pass
                self._frames.put(frame)
            cap.release()
            self.connected = False
            time.sleep(delay)

    def read(self, timeout=1.0):
        import queue
        try:
            return self._frames.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

class GPSNavigator:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.steps = []
        self.current_step = 0
        self.destination = None
        self.arrived = False

    def route_to(self, origin_lat, origin_lng, dest_str) -> str:
        import requests
        try:
            url = 'https://maps.googleapis.com/maps/api/directions/json'
            params = {
                'origin': f'{origin_lat},{origin_lng}',
                'destination': dest_str,
                'mode': 'walking',
                'key': self.api_key,
            }
            r = requests.get(url, params=params, timeout=8)
            data = r.json()

            if data.get('status') != 'OK':
                return 'Could not find route.'

            leg = data['routes'][0]['legs'][0]
            self.steps = leg['steps']
            self.current_step = 0
            self.arrived = False
            self.destination = dest_str

            dest_name = leg['end_address']
            return (
                f'Route to {dest_name}. '
                f'{len(self.steps)} steps. '
                f'{leg["distance"]["text"]}. '
                f'{leg["duration"]["text"]}. '
                f'First: {self._clean_html(self.steps[0]["html_instructions"])}.'
            )
        except Exception as e:
            logger.error(f'Routing error: {e}')
            return 'Navigation service unavailable.'

    def update_position(self, user_lat, user_lng) -> dict:
        if not self.steps or self.arrived:
            return {}

        step = self.steps[self.current_step]
        end = step['end_location']
        dist = _haversine(user_lat, user_lng, end['lat'], end['lng'])

        if dist < 20 and self.current_step + 1 < len(self.steps):
            self.current_step += 1
            step = self.steps[self.current_step]
            end = step['end_location']

        if self.current_step == len(self.steps) - 1:
            final_dist = _haversine(user_lat, user_lng, end['lat'], end['lng'])
            if final_dist < 15:
                self.arrived = True
                return {'arrived': True, 'instruction': 'You have arrived at your destination.', 'distance': ''}

        bearing = _bearing(user_lat, user_lng, end['lat'], end['lng'])
        dir_str = _direction_from_bearing(bearing)
        instruction = self._clean_html(step['html_instructions'])
        dist_str = f'{int(dist)} meters'

        return {
            'arrived': False,
            'instruction': instruction,
            'distance': dist_str,
            'direction': dir_str,
            'dist_m': dist,
        }

    def _clean_html(self, text):
        import re
        return re.sub(r'<[^>]+>', ' ', text).strip()

class MooveFreeOutdoorSystem:
    def __init__(self):
        global _latest_frame

        self.running = False
        self.tts = TTS()
        self.hw = HardwareManager()
        self.voice_ai_available = DEPS_OK
        self.navigator = GPSNavigator(GOOGLE_API_KEY)
        self.destination_set = False
        self.last_nav_push = 0
        self.last_nav_update = 0
        self.nav_push_interval = 8
        self.last_telemetry_push = 0
        self.gemini = None

        if GEMINI_KEY and DEPS_OK:
            try:
                genai.configure(api_key=GEMINI_KEY)
                self.gemini = genai.GenerativeModel('gemini-2.5-flash')
                logger.info('Gemini connected.')
            except Exception:
                pass

        threading.Thread(target=self._run_flask_server, daemon=True).start()

        local_ip = _get_local_ip()
        self._stream_url = f'http://{local_ip}:5000/video_feed'
        logger.info(f'Stream: {self._stream_url}')
        if BLIND_UID:
            fb_set(f'sessions/{BLIND_UID}/stream_url', self._stream_url)

        self.tts.say('Outdoor navigation ready.', priority=True)

    @staticmethod
    def _run_flask_server():
        import logging as _l
        _l.getLogger('werkzeug').setLevel(_l.ERROR)
        _flask_app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False)

    def _get_user_location(self):
        if not BLIND_UID:
            return None
        data = fb_get(f'sessions/{BLIND_UID}/location')
        if data and 'latitude' in data:
            return data['latitude'], data['longitude']
        return None

    def _push_telemetry(self):
        if not BLIND_UID or time.time() - self.last_telemetry_push < 8:
            return
        self.last_telemetry_push = time.time()
        fb_set(f'sessions/{BLIND_UID}/telemetry', {
            'battery': 92,
            'temperature': 40.0,
            'signal': '4G',
            'mode': 'outdoor',
            'stream_active': True,
            'gps_active': True,
            'gemini_active': self.gemini is not None,
            'timestamp': int(time.time() * 1000),
        })

    def _do_scene_analysis(self, frame):
        if not self.gemini:
            return
        frame_copy = frame.copy()
        def task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                prompt = (
                    'You are guiding a blind pedestrian outdoors. Analyze this scene: '
                    '1) Identify immediate hazards (crossing lights, traffic, road obstructions). '
                    '2) Describe what is at 10, 12, and 2 o\'clock. '
                    '3) Rate safety: SAFE, CAUTION, or STOP. '
                    'Be very brief (2 sentences max). No asterisks or markdown.'
                )
                response = self.gemini.generate_content([prompt, Image.fromarray(rgb)])
                if response.text:
                    clean = response.text.replace('*', '').replace('#', '').strip()
                    self.tts.say(clean)

                    fb_push_nav(clean, haptic=0)
            except Exception as e:
                logger.error(f'Gemini outdoor: {e}')
        threading.Thread(target=task, daemon=True).start()

    def _do_hazard_analysis(self, frame):
        if not self.gemini:
            return
        frame_copy = frame.copy()
        def task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                prompt = (
                    'This is a pedestrian camera view. Identify: '
                    '1) Is there a crosswalk and is it safe? '
                    '2) Any approaching vehicles? '
                    '3) Any step, curb, or height difference ahead? '
                    'One sentence, safety-first tone.'
                )
                response = self.gemini.generate_content([prompt, Image.fromarray(rgb)])
                if response.text:
                    clean = response.text.replace('*', '').replace('#', '').strip()
                    self.tts.say(clean, priority=True)

                    fb_push_nav(clean, haptic=3)
            except Exception as e:
                logger.error(f'Gemini hazard: {e}')
        threading.Thread(target=task, daemon=True).start()

    def run(self):
        stream = StreamCapture(IP_CAM_URL)
        self.running = True
        frame_count = 0
        last_analysis = 0
        last_hazard = 0
        ANALYSIS_INTERVAL = 8
        HAZARD_INTERVAL = 4

        self.tts.say('Connecting to camera. Say navigate to address to start navigation.', priority=True)

        while self.running:
            frame = stream.read()
            if frame is None:
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (640, 480))

            with _frame_lock:
                globals()['_latest_frame'] = frame.copy()

            now = time.time()

            sonar_dist = self.hw.get_distance()
            if sonar_dist < 1.0:
                if now - last_hazard > 1.2:
                    msg = f'Stop! Obstacle {int(sonar_dist * 100)} centimeters ahead.'
                    self.tts.say(msg, priority=True)
                    self.hw.trigger_haptic(3)
                    fb_push_nav(msg, haptic=3)
                    last_hazard = now
            else:
                if now - last_hazard > HAZARD_INTERVAL and frame_count % 3 == 0:
                    self._do_hazard_analysis(frame)
                    last_hazard = now

            if now - last_analysis > ANALYSIS_INTERVAL:
                self._do_scene_analysis(frame)
                last_analysis = now

            location = self._get_user_location()
            if location and self.destination_set and now - self.last_nav_update > 3:
                self.last_nav_update = now
                nav = self.navigator.update_position(location[0], location[1])

                if nav.get('arrived'):
                    msg = 'You have arrived at your destination!'
                    self.tts.say(msg, priority=True)

                    fb_push_nav(msg, haptic=0)
                    if BLIND_UID:
                        fb_set(f'sessions/{BLIND_UID}/next_turn', {'instruction': 'Arrived', 'distance': ''})
                    self.destination_set = False
                elif nav.get('instruction'):
                    full_msg = f'{nav["instruction"]}. In {nav["distance"]}.'
                    if now - self.last_nav_push > self.nav_push_interval:
                        self.tts.say(full_msg)

                        fb_push_nav(full_msg, haptic=0)
                        self.last_nav_push = now

                    if BLIND_UID:
                        fb_set(f'sessions/{BLIND_UID}/next_turn', {
                            'instruction': nav['instruction'],
                            'distance': nav['distance'],
                        })

            self._push_telemetry()
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        stream.stop()
        self.hw.stop()

if __name__ == '__main__':
    system = MooveFreeOutdoorSystem()
    system.run()
