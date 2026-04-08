import cv2
import time
import logging
import numpy as np
import os
import threading
from pathlib import Path
from dotenv import load_dotenv
import yaml
from flask import Flask, Response

# --- IMPORT CUSTOM MODULES ---
from src.inference.audio_feedback import AudioFeedback
from src.inference.estimator import AdvancedDistanceEstimator
from src.inference.navigator import ZoneBasedNavigator, Detection
from src.inference.conversational_ai import ConversationalAI
from src.utils.stream_loader import StreamLoader
from src.hardware_manager import HardwareManager

# --- IMPORT AI ENGINES ---
from ultralytics import YOLO

try:
    import google.generativeai as genai
    from PIL import Image

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# --- CONFIGURATION ---
load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
# Silence noisy libraries
logging.getLogger("werkzeug").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)

logger = logging.getLogger("MoveFree")

# --- LIVESTREAMING SERVER ---
app = Flask(__name__)
frame_buffer = None
buffer_lock = threading.Lock()


@app.route("/video_feed")
def video_feed():
    """Video streaming route. Put this in the src tag of an img tag."""

    def generate():
        while True:
            with buffer_lock:
                if frame_buffer is None:
                    time.sleep(0.1)
                    continue
                ret, encoded = cv2.imencode(".jpg", frame_buffer)
            if not ret:
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + encoded.tobytes() + b"\r\n"
            )

    return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")


def start_stream_server():
    """Starts Flask server on port 5000 in a daemon thread"""
    try:
        app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
    except Exception as e:
        logger.error(f"Stream Server Failed: {e}")


# --- MAIN SYSTEM CLASS ---
class MoveFreeIndoorSystem:
    def __init__(self, video_source="0", mic_index=None):
        self.video_source = video_source
        self.running = False
        self.current_frame = None

        logger.info("=" * 60)
        logger.info("🚀 MoveFree Ultimate - Indoor Autopilot (Optimized)")
        logger.info("=" * 60)

        # 1. Initialize Hardware (Sensors & Haptics)
        self.hw = HardwareManager()

        # 2. Audio System
        self.audio = AudioFeedback()
        self.audio.speak("Initializing System...", priority=True)

        # 3. Load Config
        self.config = {}
        if os.path.exists("config/config.yaml"):
            with open("config/config.yaml", "r") as f:
                self.config = yaml.safe_load(f)

        # 4. AI Model - FORCING STANDARD YOLOv8n (Best for indoor robustness)
        logger.info("🧠 Loading YOLOv8n (COCO)...")
        self.model = YOLO("yolov8n.pt")
        self.class_names = self.model.names

        # Define Priority Classes
        self.critical_classes = [
            "person",
            "chair",
            "couch",
            "bed",
            "toilet",
            "refrigerator",
            "tv",
        ]

        # 5. Estimator & Navigator
        self.estimator = AdvancedDistanceEstimator(camera_height=1.5, focal_length=700)
        self.navigator = None  # Initialized in run loop when frame size is known

        # ** FIX: Initialize Calibration Variables **
        self.calibration_frames = 0
        self.max_calibration_frames = 60
        self.estimator.calibrated = False

        # 6. Conversational AI
        logger.info("🎤 Initializing Voice AI...")
        self.voice_ai = ConversationalAI(mic_index=mic_index)
        self._register_voice_commands()

        # 7. Gemini Vision AI
        self.gemini = None
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.gemini = genai.GenerativeModel("gemini-2.5-flash")
                logger.info("✅ Gemini Connected")
            except Exception as e:
                logger.error(f"Gemini Error: {e}")

        # 8. Start Livestream
        threading.Thread(target=start_stream_server, daemon=True).start()
        logger.info("📡 Livestream: http://<IP>:5000/video_feed")

        # 9. State & Optimization Variables
        self.tracking_mode = None
        self.last_guidance_time = 0
        self.guidance_interval = 2.0
        self.last_light_warn = 0

        # FPS & Frame Skipping Logic
        self.frame_count = 0
        self.SKIP_FRAMES = 2  # Process 1 out of every 3 frames (Boosts FPS)
        self.cached_detections = []
        self.fps = 0
        self.last_fps_time = time.time()

        self.audio.speak("System Ready. Walking mode active.", priority=True)

    def _register_voice_commands(self):
        """Map voice commands to internal functions"""
        cmds = {
            "stop": self._cmd_stop,
            "navigate": self._cmd_navigate,
            "find": self._cmd_find,
            "describe": self._cmd_describe,
            "read": self._cmd_read,
            "help": self._cmd_help,
            "hazards": self._cmd_hazards,
            "exits": self._cmd_exits,
            "summary": self._cmd_summary,
            "calibrate": self._cmd_calibrate,
            "speak": lambda t, p=0: self.audio.speak(t, priority=p),
        }
        for keyword, func in cmds.items():
            self.voice_ai.register_callback(keyword, func)

    # --- VOICE HANDLERS ---
    def _cmd_stop(self):
        self.audio.speak("Paused", priority=True)
        self.tracking_mode = None

    def _cmd_navigate(self):
        self.tracking_mode = None
        self.audio.speak("Resuming navigation", priority=True)

    def _cmd_find(self, target: str):
        target = target.lower().strip()
        # Common synonyms mapping
        mapping = {
            "sofa": "couch",
            "table": "dining table",
            "tv": "tv",
            "fridge": "refrigerator",
            "plant": "potted plant",
        }
        target = mapping.get(target, target)

        if target in self.class_names.values():
            self.tracking_mode = target
            self.audio.speak(f"Searching for {target}", priority=True)
        else:
            self.audio.speak(f"I cannot find {target} in my database.", priority=True)

    def _cmd_describe(self):
        if self.gemini:
            self.audio.speak("Analyzing scene...", priority=True)
            self._run_gemini(
                "Describe this room. Mention furniture, people (using clock positions), and exits. Be concise."
            )

    def _cmd_read(self):
        if self.gemini:
            self.audio.speak("Reading text...", priority=True)
            self._run_gemini("Read all visible text in this image.")

    def _cmd_hazards(self):
        if self.gemini:
            self.audio.speak("Checking safety...", priority=True)
            self._run_gemini(
                "Identify trip hazards, open cupboards, or stairs. If safe, say 'Area safe'."
            )

    def _cmd_exits(self):
        if self.gemini:
            self.audio.speak("Locating exits...", priority=True)
            self._run_gemini(
                "Where are the doors or exits? Use clock positions (e.g., 'Door at 2 o'clock')."
            )

    def _cmd_summary(self):
        self._run_gemini("Give a 1-sentence summary of what is in front of me.")

    def _cmd_calibrate(self):
        self.calibration_frames = 0
        self.estimator.calibrated = False
        self.audio.speak("Recalibrating distance...", priority=True)

    def _cmd_help(self):
        self.audio.speak(
            "Commands: Navigate, Stop, Find [Object], Describe, Read text, Hazards, Exits."
        )

    def _run_gemini(self, prompt):
        """Runs Gemini in background thread"""
        if self.current_frame is None:
            return
        frame_copy = self.current_frame.copy()

        def task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                response = self.gemini.generate_content([prompt, Image.fromarray(rgb)])
                if response.text:
                    self.audio.speak(response.text.replace("*", ""))
            except Exception as e:
                logger.error(f"Gemini Error: {e}")
                self.audio.speak("AI Service Error.")

        threading.Thread(target=task, daemon=True).start()

    # --- UTILS ---
    def get_clock_direction(self, bbox, width):
        """Returns string like '10 o'clock'"""
        cx = (bbox[0] + bbox[2]) // 2
        ratio = cx / width
        if ratio < 0.2:
            return "10 o'clock"
        if ratio < 0.4:
            return "11 o'clock"
        if ratio < 0.6:
            return "12 o'clock"
        if ratio < 0.8:
            return "1 o'clock"
        return "2 o'clock"

    # --- MAIN LOOP ---
    def run(self):
        logger.info(f"📹 Connecting to: {self.video_source}")
        stream = StreamLoader(self.video_source)
        time.sleep(2)

        self.voice_ai.start_listening()
        self.running = True

        while self.running:
            # 1. Read Frame
            frame = stream.read()
            if frame is None:
                # Robustness: Auto-reconnect
                logger.warning("Video stream lost. Reconnecting...")
                time.sleep(1)
                stream.stop()
                stream = StreamLoader(self.video_source)
                continue

            # 2. Resize (Critical for FPS)
            frame = cv2.resize(frame, (640, 480))
            self.current_frame = frame
            h, w = frame.shape[:2]

            # 3. Update Stream Buffer
            with buffer_lock:
                frame_buffer = frame.copy()
            self.voice_ai.set_frame(frame)

            # 4. Late Initialization
            if not self.navigator:
                self.navigator = ZoneBasedNavigator(w, h)
            if (
                not self.estimator.calibrated
                and self.calibration_frames < self.max_calibration_frames
            ):
                self.calibration_frames += 1

            # 5. SENSOR FUSION CHECK (Safety First)
            # A. Light Check
            if self.hw.check_ambient_light(frame) == "DARK":
                if time.time() - self.last_light_warn > 20:
                    self.audio.speak("Environment is too dark.", priority=True)
                    self.last_light_warn = time.time()

            # B. Ultrasonic Check (Invisible Walls/Glass)
            sonar_dist = self.hw.get_distance()

            # 6. VISION PROCESSING (With Frame Skipping)
            # Only run heavy AI model every SKIP_FRAMES
            if self.frame_count % (self.SKIP_FRAMES + 1) == 0:
                self.cached_detections = self._run_inference(frame, h, w)

                # Auto-calibrate estimator using people in the scene
                if self.cached_detections and not self.estimator.calibrated:
                    self.estimator.auto_calibrate(self.cached_detections, h)

            # Use cached detections for UI smoothness
            detections = self.cached_detections

            # 7. DECISION LOGIC
            # If Sonar detects something VERY close (< 1.0m), override Vision
            if sonar_dist < 1.0:
                if time.time() - self.last_guidance_time > 1.2:
                    self.audio.speak(
                        f"Stop! Obstacle {int(sonar_dist*100)} centimeters.",
                        priority=True,
                    )
                    self.hw.trigger_haptic(3)  # Both motors vibrate
                    self.last_guidance_time = time.time()
            else:
                # Standard Vision Guidance
                self._process_guidance(detections)

            # 8. VISUALIZATION
            vis_frame = self._draw_ui(frame, detections)
            cv2.imshow("MoveFree Indoor", vis_frame)

            # 9. FPS Calculation
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                cur = time.time()
                self.fps = 30 / (cur - self.last_fps_time)
                self.last_fps_time = cur

            if cv2.waitKey(1) & 0xFF == ord("q"):
                self.running = False

        # Cleanup
        logger.info("🛑 Shutting down...")
        self.hw.stop()
        stream.stop()
        self.voice_ai.stop_listening()
        self.audio.stop()
        cv2.destroyAllWindows()

    def _run_inference(self, frame, h, w):
        """Runs YOLO and returns structured Detections"""
        try:
            # Using 'predict' is faster than 'track' for indoor navigation with skipping
            results = self.model.predict(frame, conf=0.45, verbose=False)
        except:
            return []

        dets = []
        if not results or not results[0].boxes:
            return dets

        for box in results[0].boxes:
            try:
                cls = int(box.cls[0])
                label = self.class_names[cls]
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bbox = (x1, y1, x2, y2)

                # Calculations
                dist = self.estimator.estimate(bbox, h, w, label)
                zone = self.navigator.classify_zone(bbox)
                clock = self.get_clock_direction(bbox, w)

                # Priority
                prio = 0
                if label in ["person", "door"]:
                    prio = 2
                elif label in self.critical_classes:
                    prio = 1

                # Use Detection class from navigator.py
                dets.append(Detection(label, conf, bbox, dist, zone, prio, clock))
            except:
                continue
        return dets

    def _process_guidance(self, detections):
        """Decides what to speak"""
        curr = time.time()

        # A. Tracking Mode (User asked to find X)
        if self.tracking_mode:
            targets = [d for d in detections if d.label == self.tracking_mode]
            if targets:
                best = min(targets, key=lambda x: x.distance)
                if curr - self.last_guidance_time > 2.5:
                    self.audio.speak(
                        f"{self.tracking_mode} {best.clock_dir}, {best.distance:.1f} meters",
                        priority=True,
                    )
                    self.last_guidance_time = curr
            return

        # B. Critical Warnings (Navigator Logic)
        crit = self.navigator.get_critical_warning(detections)
        if crit:
            if curr - self.last_guidance_time > 1.5:
                self.audio.speak(crit, priority=True)
                self.hw.trigger_haptic(3)  # Vibration Alert
                self.last_guidance_time = curr
            return

        # C. Regular Navigation
        if curr - self.last_guidance_time < self.guidance_interval:
            return

        res = self.navigator.analyze_detections(detections)
        if res["message"]:
            self.audio.speak(res["message"])
            self.last_guidance_time = curr

        # Haptic Feedback (Left/Right pulse)
        if res.get("haptic"):
            self.hw.trigger_haptic(res["haptic"])

    def _draw_ui(self, frame, detections):
        """Draws Zones, Bounding Boxes and Info"""
        if self.navigator:
            frame = self.navigator.draw_zones(frame)

        for d in detections:
            color = (0, 255, 0)
            if d.priority == 2:
                color = (0, 165, 255)  # Orange

            cv2.rectangle(
                frame, (d.bbox[0], d.bbox[1]), (d.bbox[2], d.bbox[3]), color, 2
            )
            cv2.putText(
                frame,
                f"{d.label} {d.clock_dir} {d.distance:.1f}m",
                (d.bbox[0], d.bbox[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # HUD
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        mode = self.tracking_mode if self.tracking_mode else "Auto-Nav"
        cv2.putText(
            frame,
            f"Mode: {mode}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2,
        )

        # Sonar Data
        sonar = self.hw.get_distance()
        sonar_txt = f"Sonar: {sonar}m" if sonar < 5.0 else "Sonar: Clear"
        cv2.putText(
            frame, sonar_txt, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        return frame


if __name__ == "__main__":
    # Check for IP Camera in ENV
    src = "0"
    if os.getenv("IP_CAMERA_URL"):
        src = os.getenv("IP_CAMERA_URL")

    # Run System
    MoveFreeIndoorSystem(video_source=src).run()
