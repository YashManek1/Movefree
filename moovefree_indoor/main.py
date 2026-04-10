import cv2
import time
import logging
import numpy as np
import os
import socket
import threading
from pathlib import Path
from dotenv import load_dotenv
import yaml
from flask import Flask, Response

from src.inference.audio_feedback import AudioFeedback
from src.inference.detector import ObjectDetector
from src.inference.navigator import ZoneBasedNavigator
from src.inference.conversational_ai import ConversationalAI
from src.utils.stream_loader import StreamLoader
from src.hardware_manager import HardwareManager
from src.api.telemetry_server import start_telemetry_server
from src import firebase_client as fb

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger('werkzeug').setLevel(logging.WARNING)
logging.getLogger('ultralytics').setLevel(logging.WARNING)

logger = logging.getLogger('MooveFree')

_flask_app = Flask(__name__)
_frame_buffer = None
_buffer_lock = threading.Lock()

def _get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return '127.0.0.1'

@_flask_app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            with _buffer_lock:
                if _frame_buffer is None:
                    time.sleep(0.033)
                    continue
                ret, encoded = cv2.imencode('.jpg', _frame_buffer, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                continue
            yield b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + encoded.tobytes() + b'\r\n'
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

def _start_stream_server(port=5000):
    import logging as _log
    _log.getLogger('werkzeug').setLevel(_log.ERROR)
    _flask_app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

class MooveFreeIndoorSystem:
    def __init__(self, video_source='0', mic_index=None):
        global _frame_buffer

        self.video_source = video_source
        self.running = False
        self.current_frame = None
        self.cached_detections = []
        self.fps = 0
        self.frame_count = 0
        self.last_fps_time = time.time()
        self.SKIP_FRAMES = 2
        self.tracking_mode = None
        self.force_mode = None
        self.last_guidance_time = 0
        self.guidance_interval = 2.0
        self.last_light_warn = 0
        self.last_telemetry_push = 0
        self.last_config_check = 0
        self.navigator = None

        logger.info('=' * 60)
        logger.info('MooveFree Ultimate - Indoor Autopilot')
        logger.info('=' * 60)

        self.config = {}
        if os.path.exists('config/config.yaml'):
            with open('config/config.yaml', 'r') as f:
                self.config = yaml.safe_load(f) or {}

        self.blind_uid = os.getenv('BLIND_USER_UID', '')
        if not self.blind_uid:
            logger.warning('BLIND_USER_UID not set in .env. Firebase push disabled.')

        self.hw = HardwareManager()
        self.audio = AudioFeedback()
        self.audio.speak('Initializing system.', priority=True)

        logger.info('Loading YOLOv8n...')
        self.detector = ObjectDetector('yolov8n.pt', conf=0.45)

        logger.info('Initializing Voice AI...')
        self.voice_ai = ConversationalAI(mic_index=mic_index)
        self._register_voice_commands()

        self.gemini = None
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini = genai.GenerativeModel('gemini-2.5-flash')
                logger.info('Gemini 2.5 Flash connected.')
            except Exception as e:
                logger.error(f'Gemini init: {e}')

        threading.Thread(target=_start_stream_server, daemon=True).start()
        start_telemetry_server(self)

        local_ip = _get_local_ip()
        stream_url = f'http://{local_ip}:5000/video_feed'
        logger.info(f'Stream URL: {stream_url}')

        if self.blind_uid:
            fb.set_stream_url(self.blind_uid, stream_url)

        self.audio.speak('System ready. Walking mode active.', priority=True)

    def _register_voice_commands(self):
        cmds = {
            'stop': self._cmd_stop,
            'navigate': self._cmd_navigate,
            'find': self._cmd_find,
            'describe': self._cmd_describe,
            'read': self._cmd_read,
            'hazards': self._cmd_hazards,
            'exits': self._cmd_exits,
            'summary': self._cmd_summary,
            'calibrate': self._cmd_calibrate,
            'sos': self._cmd_sos,
            'help': self._cmd_help,
            'speak': lambda t, p=0: self.audio.speak(t, priority=bool(p)),
        }
        for kw, fn in cmds.items():
            self.voice_ai.register_callback(kw, fn)

    def _cmd_stop(self):
        self.tracking_mode = None
        self.audio.speak('Paused.', priority=True)

    def _cmd_navigate(self):
        self.tracking_mode = None
        self.audio.speak('Resuming navigation.', priority=True)

    def _cmd_find(self, target: str = ''):
        target = self.detector.normalize_label(target)
        if target in self.detector.class_names.values():
            self.tracking_mode = target
            self.audio.speak(f'Searching for {target}.', priority=True)
        else:
            self.audio.speak(f'Cannot find {target} in database.', priority=True)

    def _cmd_describe(self):
        if self.gemini:
            self.audio.speak('Analyzing scene.', priority=True)
            self._run_gemini('Describe this room. Mention furniture, people using clock positions, and exits. Be concise.')

    def _cmd_read(self):
        if self.gemini:
            self.audio.speak('Reading text.', priority=True)
            self._run_gemini('Read all visible text in this image.')

    def _cmd_hazards(self):
        if self.gemini:
            self.audio.speak('Checking for hazards.', priority=True)
            self._run_gemini('Identify trip hazards, stairs, or obstacles. If safe say: area clear.')

    def _cmd_exits(self):
        exit_msg = self.navigator.find_exit(self.cached_detections) if self.navigator else None
        if exit_msg:
            self.audio.speak(exit_msg, priority=True)
        elif self.gemini:
            self.audio.speak('Locating exits.', priority=True)
            self._run_gemini("Where are the doors or exits? Use clock positions.")

    def _cmd_summary(self):
        self._run_gemini('Give a 1-sentence safety-focused summary of what is in front of me.')

    def _cmd_calibrate(self):
        self.audio.speak('Recalibrating.', priority=True)

    def _cmd_sos(self):
        self.audio.speak('Triggering emergency SOS.', priority=True)
        if self.blind_uid:
            loc = None
            fb.push_sos(self.blind_uid, loc)

    def _cmd_help(self):
        self.audio.speak('Commands: navigate, stop, find object, describe, read text, hazards, exits, SOS.', priority=True)

    def _run_gemini(self, prompt):
        if self.current_frame is None:
            return
        frame_copy = self.current_frame.copy()

        def task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                response = self.gemini.generate_content([prompt, Image.fromarray(rgb)])
                if response.text:
                    self.audio.speak(response.text.replace('*', '').replace('#', '').strip())
            except Exception as e:
                logger.error(f'Gemini: {e}')
                self.audio.speak('AI service error.')

        threading.Thread(target=task, daemon=True).start()

    def _check_remote_config(self):
        now = time.time()
        if now - self.last_config_check < 10:
            return
        self.last_config_check = now
        if not self.blind_uid:
            return
        config = fb.get_remote_config(self.blind_uid)
        if not config:
            return
        if 'sensitivity' in config:
            self.detector.update_conf(float(config['sensitivity']) / 100.0)
        if 'volume' in config:
            self.audio.set_volume(float(config['volume']) / 100.0)
        if 'mode' in config:
            self.force_mode = config['mode']

    def _push_telemetry(self):
        now = time.time()
        if now - self.last_telemetry_push < 8:
            return
        self.last_telemetry_push = now
        if not self.blind_uid:
            return
        hw_status = self.hw.get_sensor_status()
        fb.push_telemetry(self.blind_uid, {
            'battery': self.hw.get_battery_level(),
            'temperature': self.hw.get_temperature(),
            'signal': '4G',
            'mode': self.tracking_mode or 'auto',
            'fps': round(self.fps, 1),
            'sonar': self.hw.get_distance(),
            'stream_active': True,
            'gps_active': False,
            'sonar_active': hw_status.get('sonar_active', False),
            'mic_active': True,
            'gemini_active': self.gemini is not None,
        })

    def run(self):
        logger.info(f'Connecting to: {self.video_source}')
        stream = StreamLoader(self.video_source)
        time.sleep(2)

        self.voice_ai.start_listening()
        self.running = True

        while self.running:
            frame = stream.read()
            if frame is None:
                time.sleep(0.5)
                continue

            frame = cv2.resize(frame, (640, 480))
            self.current_frame = frame
            h, w = frame.shape[:2]

            with _buffer_lock:
                globals()['_frame_buffer'] = frame.copy()
            self.voice_ai.set_frame(frame)

            if not self.navigator:
                self.navigator = ZoneBasedNavigator(w, h)

            if self.hw.check_ambient_light(frame) == 'DARK':
                if time.time() - self.last_light_warn > 20:
                    msg = 'Screen is dark. Cannot see clearly.'
                    self.audio.speak(msg, priority=True)

                    if self.blind_uid:
                        fb.push_sos(self.blind_uid, None)
                        fb.push_nav_instruction(
                            self.blind_uid,
                            'Warning: Camera feed has gone dark. Please check surroundings.',
                            haptic_code=3
                        )
                    self.last_light_warn = time.time()

            sonar_dist = self.hw.get_distance()

            if self.frame_count % (self.SKIP_FRAMES + 1) == 0:
                self.cached_detections = self.detector.detect(
                    frame, h, w,
                    self.navigator.left_boundary,
                    self.navigator.right_boundary,
                )
                self._push_hazards_to_firebase()

            if sonar_dist < 1.0:
                if time.time() - self.last_guidance_time > 1.2:
                    msg = f'Stop! Obstacle {int(sonar_dist * 100)} centimeters ahead.'
                    self.audio.speak(msg, priority=True)
                    self.hw.trigger_haptic(3)

                    if self.blind_uid:
                        fb.push_nav_instruction(self.blind_uid, msg, haptic_code=3)
                    self.last_guidance_time = time.time()
            else:
                self._process_guidance(self.cached_detections)

            vis = self._draw_ui(frame.copy(), self.cached_detections)
            cv2.imshow('MooveFree Indoor', vis)

            self.frame_count += 1
            if self.frame_count % 30 == 0:
                cur = time.time()
                self.fps = 30 / max(cur - self.last_fps_time, 0.001)
                self.last_fps_time = cur

            self._push_telemetry()
            self._check_remote_config()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False

        logger.info('Shutting down...')
        self.hw.stop()
        stream.stop()
        self.voice_ai.stop_listening()
        self.audio.stop()
        cv2.destroyAllWindows()

    def _push_hazards_to_firebase(self):
        if not self.blind_uid:
            return
        for d in self.cached_detections:
            if d.priority >= 1 and d.distance < 2.0:
                fb.push_hazard(self.blind_uid, {
                    'label': d.label,
                    'distance': d.distance,
                    'zone': d.zone,
                    'clock_dir': d.clock_dir,
                    'confidence': round(d.confidence, 2),
                })
                break

    def _process_guidance(self, detections):
        curr = time.time()

        def _push(message, haptic_code=0, priority=False):
            self.audio.speak(message, priority=priority)
            if self.blind_uid:
                fb.push_nav_instruction(self.blind_uid, message, haptic_code)

        if self.tracking_mode:
            targets = [d for d in detections if d.label == self.tracking_mode]
            if targets and curr - self.last_guidance_time > 2.5:
                best = min(targets, key=lambda x: x.distance)
                msg = f'{self.tracking_mode} at {best.clock_dir}, {best.distance:.1f} meters.'
                _push(msg, haptic_code=0, priority=True)
                self.last_guidance_time = curr
            return

        crit = self.navigator.get_critical_warning(detections)
        if crit and curr - self.last_guidance_time > 1.5:
            _push(crit, haptic_code=3, priority=True)
            self.hw.trigger_haptic(3)
            self.last_guidance_time = curr
            return

        if curr - self.last_guidance_time < self.guidance_interval:
            return

        res = self.navigator.analyze_detections(detections)
        if res['message']:
            haptic = res.get('haptic', 0)
            _push(res['message'], haptic_code=haptic)
            self.last_guidance_time = curr
        if res.get('haptic'):
            self.hw.trigger_haptic(res['haptic'])

    def _draw_ui(self, frame, detections):
        if self.navigator:
            frame = self.navigator.draw_zones(frame)

        for d in detections:
            color = (0, 229, 255) if d.priority == 0 else ((0, 165, 255) if d.priority == 1 else (0, 80, 255))
            cv2.rectangle(frame, (d.bbox[0], d.bbox[1]), (d.bbox[2], d.bbox[3]), color, 2)
            cv2.putText(frame, f'{d.label} {d.clock_dir} {d.distance:.1f}m',
                        (d.bbox[0], max(d.bbox[1] - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

        cv2.putText(frame, f'FPS:{self.fps:.1f}', (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        mode = self.tracking_mode or 'Auto-Nav'
        cv2.putText(frame, f'Mode:{mode}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 229, 255), 1)
        sonar = self.hw.get_distance()
        sonar_txt = f'Sonar:{sonar:.1f}m' if sonar < 5.0 else 'Sonar:Clear'
        cv2.putText(frame, sonar_txt, (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 200), 1)
        return frame

if __name__ == '__main__':
    src = os.getenv('IP_CAMERA_URL', '0')
    mic = os.getenv('MIC_INDEX')
    mic_idx = int(mic) if mic and mic.isdigit() else None
    MooveFreeIndoorSystem(video_source=src, mic_index=mic_idx).run()
