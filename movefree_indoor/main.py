import cv2
import time
import threading
import logging
import numpy as np
import os
import queue
import re
from dotenv import load_dotenv
from ultralytics import YOLO

# --- LOGGING SETUP ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("MoveFree")
load_dotenv()

# --- MODULES CONFIGURATION ---
try:
    import speech_recognition as sr
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False
    logger.warning("❌ Speech Recognition library not found. Voice commands disabled.")

try:
    import pyttsx3
    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    logger.warning("❌ pyttsx3 library not found. Audio feedback disabled.")

try:
    import google.generativeai as genai
    from PIL import Image
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("❌ Google Gemini not found. Describe/Read features disabled.")

# --- COMPONENT 1: SMART AUDIO (INTERRUPTIBLE) ---
class AudioFeedback:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.shutup_event = threading.Event()
        self.engine = None # Handle for stopping
        
        if AUDIO_AVAILABLE:
            threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', 160)
            self.engine.setProperty('volume', 1.0)
        except Exception as e:
            logger.error(f"TTS Init Error: {e}")
            return

        while self.running:
            try:
                item = self.queue.get(timeout=1)
                text, priority = item
                
                # If Stop/Warning happened, skip old low-priority msgs
                if self.shutup_event.is_set() and priority < 2:
                    self.queue.task_done()
                    continue

                clean_text = text.replace('*', '').replace('#', '').strip()
                if clean_text:
                    logger.info(f"🔊 Speaking: {clean_text}")
                    self.engine.say(clean_text)
                    self.engine.runAndWait()
                self.queue.task_done()
            except:
                pass

    def speak(self, text, priority=0):
        """
        Priority 0: Descriptions/Text Reading (Low)
        Priority 1: Navigation/Light Warnings (Medium)
        Priority 2: EMERGENCY STOP (High)
        """
        if not AUDIO_AVAILABLE:
            print(f"🔊 [TEXT]: {text}")
            return
        
        if priority == 2:
            # Signal to ignore pending queue items
            self.shutup_event.set()
            
            # Clear the queue immediately
            with self.queue.mutex:
                self.queue.queue.clear()
            
            # Reset the stop signal after a moment so new commands work
            threading.Timer(0.5, self.shutup_event.clear).start()
            
            # Try to stop current utterance (Best effort)
            if self.engine:
                try:
                    self.engine.stop() 
                except: 
                    pass 
        
        self.queue.put((text, priority))

    def stop(self):
        self.running = False
        self.queue.put(None)

# --- COMPONENT 2: BUFFERLESS STREAM ---
class StreamLoader:
    def __init__(self, source):
        self.source = source
        self.q = queue.Queue(maxsize=1) 
        self.running = True
        # Suppress ffmpeg/opencv warnings
        os.environ["OPENCV_LOG_LEVEL"] = "OFF"
        threading.Thread(target=self._update, daemon=True).start()

    def _update(self):
        while self.running:
            cap = cv2.VideoCapture(self.source)
            if not cap.isOpened():
                time.sleep(2)
                continue
            
            logger.info("✅ Camera Connected")
            while self.running and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                if not self.q.empty():
                    try: self.q.get_nowait()
                    except: pass
                self.q.put(frame)
            cap.release()

    def read(self):
        try:
            return self.q.get(timeout=1)
        except:
            return None

    def stop(self):
        self.running = False

# --- COMPONENT 3: MAIN SYSTEM ---
class MoveFreeSystem:
    def __init__(self, source, mic_index=None):
        self.source = source
        self.mic_index = mic_index
        self.running = True
        self.current_frame = None
        
        self.audio = AudioFeedback()
        self.audio.speak("Move Free Initializing", priority=1)

        # Vision
        logger.info("Loading YOLOv8n...")
        self.model = YOLO("yolov8n.pt") 

        # Gemini (1.5 Flash)
        self.gemini_model = None
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
            try:
                self.gemini_model = genai.GenerativeModel("gemini-2.5-flash")
                logger.info("✅ Gemini Connected")
            except:
                logger.error("Gemini Connection Failed")

        # Config
        self.camera_height = 1.6
        self.focal_length = 700   
        self.horizon_offset = 0   
        self.calibrated = False   
        self.last_light_warn = 0

    def auto_calibrate(self, frame_height):
        self.horizon_offset = 0 
        self.calibrated = True
        self.audio.speak("Horizon Calibrated", priority=1)

    def estimate_distance(self, bbox, frame_height):
        x1, y1, x2, y2 = bbox
        horizon_line = (frame_height / 2) + self.horizon_offset
        pixel_dist = y2 - horizon_line
        if pixel_dist <= 10: return 10.0 
        real_dist = (self.camera_height * self.focal_length) / pixel_dist
        return max(0.5, min(real_dist, 8.0))

    # --- FEATURE: LIGHT DETECTION ---
    def check_brightness(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        if avg_brightness < 40:
            curr_time = time.time()
            if curr_time - self.last_light_warn > 15.0:
                self.audio.speak("Warning: Environment is very dark.", priority=1)
                self.last_light_warn = curr_time
                return True
        return False

    # --- FEATURE: TEXT READING (FIXED CHUNKING) ---
    def read_text_scene(self):
        if not self.gemini_model or self.current_frame is None:
            self.audio.speak("Text reading unavailable", priority=1)
            return

        self.audio.speak("Reading...", priority=1)
        frame_copy = self.current_frame.copy()
        
        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                prompt = "Read visible text line by line. Ignore gibberish."
                response = self.gemini_model.generate_content([prompt, pil_img])
                
                if response.text:
                    # --- FIX: Split by newlines to allow interruption ---
                    lines = response.text.split('\n')
                    for line in lines:
                        clean_line = line.strip().replace('*', '').replace('-', '')
                        if len(clean_line) > 2:
                            # Send each line as a separate priority 0 message
                            self.audio.speak(clean_line, priority=0)
                else:
                    self.audio.speak("No text detected.", priority=0)
            except Exception as e:
                logger.error(f"Gemini Error: {e}")
                self.audio.speak("I couldn't read the text.", priority=1)

        threading.Thread(target=_task, daemon=True).start()

    def describe_scene(self):
        if not self.gemini_model or self.current_frame is None:
            self.audio.speak("Description unavailable", priority=1)
            return

        self.audio.speak("Analyzing...", priority=1)
        frame_copy = self.current_frame.copy()
        
        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                prompt = "Describe this scene concisely. No asterisks."
                response = self.gemini_model.generate_content([prompt, pil_img])
                if response.text:
                    # Split by sentences for interruption
                    sentences = response.text.replace('*', '').split('.')
                    for s in sentences:
                        if len(s.strip()) > 3:
                            self.audio.speak(s.strip(), priority=0)
            except Exception as e:
                logger.error(f"Gemini Error: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def listen_loop(self):
        if not VOICE_AVAILABLE: return
        r = sr.Recognizer()
        
        # --- HIGH SENSITIVITY MODE ---
        r.energy_threshold = 300  
        r.dynamic_energy_threshold = True
        r.pause_threshold = 0.6 

        try:
            mic = sr.Microphone(device_index=self.mic_index)
        except: return

        logger.info("🎤 Mic Ready (High Sensitivity)")
        
        while self.running:
            try:
                with mic as source:
                    audio = r.listen(source, timeout=2, phrase_time_limit=4)
                
                cmd = r.recognize_google(audio).lower()
                logger.info(f"🗣️ Command: {cmd}")
                
                if "stop" in cmd: 
                    self.audio.speak("Stopped", priority=2) # Priority 2 clears queue
                elif "calibrate" in cmd: 
                    self.calibrated = False 
                elif "describe" in cmd: 
                    self.describe_scene()
                elif "read" in cmd: 
                    self.read_text_scene() 
            
            except sr.WaitTimeoutError: pass
            except sr.UnknownValueError: pass
            except Exception as e: logger.error(f"Voice Error: {e}")

    def run(self):
        stream = StreamLoader(self.source)
        time.sleep(2) 
        if VOICE_AVAILABLE: threading.Thread(target=self.listen_loop, daemon=True).start()

        self.audio.speak("System Ready", priority=1)
        last_nav_msg_time = 0
        
        while self.running:
            frame = stream.read()
            if frame is None: continue
            
            frame = cv2.resize(frame, (640, 480))
            self.current_frame = frame
            h, w = frame.shape[:2]

            if not self.calibrated: self.auto_calibrate(h)

            # 1. CHECK LIGHT
            self.check_brightness(frame)

            # 2. DETECT OBJECTS
            try: results = self.model.track(frame, verbose=False, conf=0.3, persist=True)
            except: results = self.model(frame, verbose=False, conf=0.3)
            
            min_dist = 99.0
            closest_label = ""
            
            left_blocked = False
            center_blocked = False
            right_blocked = False
            
            for r in results:
                for box in r.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0])
                    label = self.model.names[cls]
                    dist = self.estimate_distance(xyxy, h)
                    
                    x1, y1, x2, y2 = map(int, xyxy)
                    cx = (x1 + x2) // 2
                    
                    if dist < min_dist:
                        min_dist = dist
                        closest_label = label

                    if dist < 2.5:
                        if cx < w*0.33: left_blocked = True
                        elif cx < w*0.66: center_blocked = True
                        else: right_blocked = True

                    color = (0, 0, 255) if dist < 2.0 else (0, 255, 0)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} {dist:.1f}m", (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # 3. NAVIGATION LOGIC
            curr_time = time.time()
            nav_msg = ""
            prio = 1

            if min_dist < 1.2 and center_blocked:
                nav_msg = f"STOP! {closest_label} ahead!"
                prio = 2
                cv2.putText(frame, "STOP", (250, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

            elif center_blocked and (curr_time - last_nav_msg_time > 3.0):
                if not left_blocked: nav_msg = "Turn Left"
                elif not right_blocked: nav_msg = "Turn Right"
                else: nav_msg = "Blocked. Turn Around."
            
            elif (left_blocked or right_blocked) and min_dist < 1.5 and (curr_time - last_nav_msg_time > 4.0):
                nav_msg = f"Caution. {closest_label} nearby."

            if nav_msg:
                if prio == 2 and (curr_time - last_nav_msg_time > 1.5):
                     self.audio.speak(nav_msg, priority=2)
                     last_nav_msg_time = curr_time
                elif prio == 1:
                     self.audio.speak(nav_msg, priority=1)
                     last_nav_msg_time = curr_time

            cv2.imshow("MoveFree Ultimate", frame)
            if cv2.waitKey(1) == ord('q'): self.running = False
                
        stream.stop()
        self.audio.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # Set your video source
    VIDEO_SOURCE = "http://10.136.208.80:8080/video" 
    
    # Set your microphone index (from previous test)
    MICROPHONE_INDEX = 2  

    app = MoveFreeSystem(source=VIDEO_SOURCE, mic_index=MICROPHONE_INDEX)
    app.run()