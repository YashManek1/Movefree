"""
MoveFree Ultimate Indoor Navigation System
Human Autopilot Mode - Your Eyes, Your Freedom

FIXED: Uses YOLOv8n COCO (80 classes) for comprehensive detection
- Microphone high sensitivity (energy_threshold=800)
- Confidence threshold raised to 0.45 (fewer false positives)
- Proper COCO class mapping (person=0, bed=59, chair=56, etc.)
"""

import cv2
import time
import logging
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
import os

# Import custom modules
from src.inference.audio_feedback import AudioFeedback
from src.inference.estimator import AdvancedDistanceEstimator
from src.inference.navigator import ZoneBasedNavigator, Detection
from src.inference.conversational_ai import ConversationalAI
from src.utils.stream_loader import StreamLoader

# YOLO
from ultralytics import YOLO

# Gemini
try:
    import google.generativeai as genai
    from PIL import Image

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MoveFree")

# Load environment
load_dotenv()


class MoveFreeUltimateSystem:
    """
    Complete MoveFree Indoor Navigation System
    Uses YOLOv8n COCO (80 classes) for comprehensive object detection
    """

    def __init__(
        self,
        video_source: str = "0",
        model_path: str = "yolov8n.pt",  # Generic COCO model
        mic_index: int = None,
    ):
        self.video_source = video_source
        self.model_path = model_path
        self.running = False
        self.current_frame = None

        logger.info("=" * 60)
        logger.info("🚀 MoveFree Ultimate - Human Autopilot Mode")
        logger.info("=" * 60)

        # Initialize components
        logger.info("📦 Loading components...")

        # 1. Audio Feedback
        self.audio = AudioFeedback()
        self.audio.speak("Initializing Move Free Ultimate System", priority=True)

        # 2. YOLO Model (Generic COCO)
        logger.info("🔍 Loading YOLOv8n model...")
        self.model = YOLO(model_path)

        # COCO class names (80 classes)
        self.class_names = self.model.names
        logger.info(f"✅ Model loaded: {len(self.class_names)} COCO classes")

        # Critical classes for indoor navigation (COCO IDs)
        self.critical_classes = {
            0: "person",  # COCO ID 0
            56: "chair",  # COCO ID 56
            57: "couch",  # COCO ID 57 (maps to "sofa")
            58: "potted plant",  # COCO ID 58
            59: "bed",  # COCO ID 59
            60: "dining table",  # COCO ID 60
            61: "toilet",  # COCO ID 61
            62: "tv",  # COCO ID 62
            63: "laptop",  # COCO ID 63
            72: "refrigerator",  # COCO ID 72
            73: "book",  # COCO ID 73
            # Note: COCO doesn't have stairs, door, wardrobe, cabinet, shelf
            # These require custom trained model
        }

        # 3. Distance Estimator
        self.estimator = AdvancedDistanceEstimator(camera_height=1.5, focal_length=700)
        self.calibration_frames = 0
        self.max_calibration_frames = 60

        # 4. Navigator (will be initialized after first frame)
        self.navigator = None

        # 5. Conversational AI (FIXED: High sensitivity)
        logger.info("🎤 Initializing voice assistant...")
        self.voice_ai = ConversationalAI(mic_index=mic_index)
        self._register_voice_commands()

        # 6. Gemini
        self.gemini = None
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.gemini = genai.GenerativeModel("gemini-2.5-flash")
                logger.info("✅ Gemini 2.5 Flash connected")
            except Exception as e:
                logger.warning(f"Gemini initialization failed: {e}")

        # State management
        self.tracking_mode = None
        self.last_guidance_time = 0
        self.guidance_interval = 3.0
        self.frame_count = 0

        # Performance monitoring
        self.fps = 0
        self.last_fps_time = time.time()

        logger.info("✅ All components loaded")
        self.audio.speak("System ready. Your AI eyes are active.", priority=True)

    def _register_voice_commands(self):
        """Register voice command callbacks"""
        self.voice_ai.register_callback("stop", self._cmd_stop)
        self.voice_ai.register_callback("find", self._cmd_find)
        self.voice_ai.register_callback("describe", self._cmd_describe)
        self.voice_ai.register_callback("read", self._cmd_read)
        self.voice_ai.register_callback("navigate", self._cmd_navigate)
        self.voice_ai.register_callback("calibrate", self._cmd_calibrate)
        self.voice_ai.register_callback("help", self._cmd_help)
        self.voice_ai.register_callback("distance", self._cmd_distance)
        self.voice_ai.register_callback("exits", self._cmd_exits)
        self.voice_ai.register_callback("hazards", self._cmd_hazards)
        self.voice_ai.register_callback("summary", self._cmd_summary)
        self.voice_ai.register_callback("count", self._cmd_count)
        self.voice_ai.register_callback("people", self._cmd_people)
        self.voice_ai.register_callback("identify", self._cmd_identify)
        self.voice_ai.register_callback("color", self._cmd_color)

        # Register speak callback for AI responses
        self.voice_ai.register_callback(
            "speak", lambda text, priority=0: self.audio.speak(text, priority=priority)
        )

    # === VOICE COMMAND HANDLERS ===

    def _cmd_stop(self):
        """Handle 'stop' command"""
        self.audio.speak("Stopped", priority=True)
        self.tracking_mode = None

    def _cmd_find(self, target: str):
        """Handle 'find X' command"""
        target_normalized = target.lower().strip()

        # Map common names to COCO classes
        name_mapping = {
            "person": "person",
            "chair": "chair",
            "couch": "couch",
            "sofa": "couch",
            "bed": "bed",
            "table": "dining table",
            "tv": "tv",
            "laptop": "laptop",
            "plant": "potted plant",
            "refrigerator": "refrigerator",
            "fridge": "refrigerator",
            "toilet": "toilet",
            "book": "book",
        }

        # Try to find mapped class
        search_class = name_mapping.get(target_normalized, target_normalized)
        valid_classes = list(self.class_names.values())

        if search_class in valid_classes:
            self.tracking_mode = search_class
            self.audio.speak(f"Searching for {search_class}", priority=True)
            logger.info(f"🎯 Tracking mode: {search_class}")
        else:
            self.audio.speak(
                f"Sorry, I cannot detect {target_normalized}. Try person, chair, bed, or table.",
                priority=True,
            )

    def _cmd_describe(self):
        """Handle 'describe' command"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Description unavailable", priority=True)
            return

        self.audio.speak("Analyzing scene", priority=True)
        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = """You are the eyes for a blind person. Describe this indoor scene concisely:
                - Overall environment type (room, hallway, etc.)
                - Key objects and their positions using clock positions (12 o'clock = straight ahead)
                - Any potential hazards or obstacles
                - Doors/exits if visible
                - People if present
                
Keep it brief and practical for navigation. No asterisks."""

                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    description = response.text.replace("*", "").replace("#", "")
                    sentences = description.split(".")
                    for sentence in sentences:
                        clean_sent = sentence.strip()
                        if len(clean_sent) > 5:
                            self.audio.speak(clean_sent, priority=False)
                else:
                    self.audio.speak("Unable to analyze scene", priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")
                self.audio.speak("Error analyzing scene", priority=True)

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_read(self):
        """Handle 'read' command"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Text reading unavailable", priority=True)
            return

        self.audio.speak("Reading text", priority=True)
        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = "Read all visible text in this image clearly. List each text element. If no text, say 'No text visible'."
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text and "no text" not in response.text.lower():
                    lines = response.text.split("\n")
                    for line in lines:
                        clean_line = line.strip().replace("*", "").replace("-", "")
                        if len(clean_line) > 2:
                            self.audio.speak(clean_line, priority=False)
                else:
                    self.audio.speak("No text detected", priority=False)

            except Exception as e:
                logger.error(f"Gemini error: {e}")
                self.audio.speak("Error reading text", priority=True)

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_navigate(self):
        """Handle 'navigate' / 'guide me' command"""
        self.audio.speak("Providing navigation guidance", priority=True)
        self.last_guidance_time = 0

    def _cmd_calibrate(self):
        """Handle 'calibrate' command"""
        self.calibration_frames = 0
        if self.estimator:
            self.estimator.calibrated = False
        self.audio.speak("Recalibrating distance estimation", priority=True)

    def _cmd_help(self):
        """Handle 'help' command"""
        help_text = """Available commands: 
        Say Stop to halt. 
        Say Find person, Find chair, or Find bed to search for objects. 
        Say Describe for scene analysis. 
        Say Read to hear text. 
        Say Any hazards to check for dangers.
        Say How many people to count objects.
        Say Give me a summary for quick overview."""

        self.audio.speak(help_text, priority=True)

    def _cmd_distance(self):
        """Handle distance inquiry"""
        self.audio.speak("Calculating distances to nearby objects", priority=True)

    def _cmd_exits(self):
        """Handle exit finding - uses Gemini since COCO doesn't have 'door'"""
        if self.gemini:
            self._cmd_hazards()  # Gemini will identify exits
        else:
            self.audio.speak("Exit detection requires Gemini AI", priority=True)

    def _cmd_hazards(self):
        """Handle hazard check"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Hazard detection unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = """Analyze this scene for potential hazards for a blind person:
                - Tripping hazards (stairs, objects on floor)
                - Collision hazards (low-hanging objects, sharp corners)
                - Moving hazards (people, doors)
                - Exits and doors if visible
                
Be brief and clear. If safe, say "No immediate hazards detected"."""

                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "")
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_summary(self):
        """Handle quick summary"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Summary unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = "Give a 2-sentence summary of this scene for navigation. Focus on safety and next steps."
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "")
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_count(self, target: str = None):
        """Handle count command"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Counting unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                if target:
                    prompt = f"Count the number of {target} in this image. Give just the number."
                else:
                    prompt = "Count people in this image. Give just the number."

                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_people(self):
        """Handle people detection"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("People detection unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = "Are there any people in this image? If yes, how many and where are they positioned using clock positions?"
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_identify(self):
        """Handle object identification"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Identification unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = "What is the main object in the center of this image? Describe it briefly."
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    def _cmd_color(self):
        """Handle color identification"""
        if not self.gemini or self.current_frame is None:
            self.audio.speak("Color detection unavailable", priority=True)
            return

        frame_copy = self.current_frame.copy()

        def _task():
            try:
                rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompt = (
                    "What are the dominant colors in this scene? List them briefly."
                )
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    self.audio.speak(clean_text, priority=True)

            except Exception as e:
                logger.error(f"Gemini error: {e}")

        import threading

        threading.Thread(target=_task, daemon=True).start()

    # === MAIN PROCESSING LOOP ===

    def run(self):
        """Main processing loop"""
        # Initialize video stream
        logger.info(f"📹 Connecting to video source: {self.video_source}")
        stream = StreamLoader(self.video_source)

        # Wait for connection
        time.sleep(2)

        # Start voice assistant
        self.voice_ai.start_listening()

        self.running = True
        self.audio.speak("Navigation system active. I am your eyes.", priority=True)

        logger.info("🚀 Main loop started")
        logger.info("Press Q to quit")

        try:
            while self.running:
                # Get latest frame
                frame = stream.read()

                if frame is None:
                    logger.warning("No frame received")
                    time.sleep(0.1)
                    continue

                # Resize for consistent processing
                frame = cv2.resize(frame, (640, 480))
                self.current_frame = frame.copy()
                h, w = frame.shape[:2]

                # Update frame for voice AI
                self.voice_ai.set_frame(frame)

                # Initialize navigator
                if self.navigator is None:
                    self.navigator = ZoneBasedNavigator(w, h)
                    logger.info("✅ Navigator initialized")

                # Auto-calibration phase
                if (
                    not self.estimator.calibrated
                    and self.calibration_frames < self.max_calibration_frames
                ):
                    self.calibration_frames += 1

                # Process frame
                detections = self._process_frame(frame, h, w)

                # Auto-calibrate with detections
                if detections and not self.estimator.calibrated:
                    self.estimator.auto_calibrate(detections, h)

                # Provide navigation guidance
                if detections:
                    self._provide_guidance(detections, frame)

                # Draw visualizations
                frame = self._draw_visualizations(frame, detections)

                # Display
                cv2.imshow("MoveFree Ultimate - Your AI Eyes", frame)

                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("Quit requested")
                    self.running = False

                # Update FPS
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    current_time = time.time()
                    self.fps = 30 / (current_time - self.last_fps_time)
                    self.last_fps_time = current_time

        except KeyboardInterrupt:
            logger.info("Interrupted by user")

        finally:
            # Cleanup
            logger.info("🛑 Shutting down...")
            stream.stop()
            self.voice_ai.stop_listening()
            self.audio.stop()
            cv2.destroyAllWindows()
            logger.info("✅ Shutdown complete")

    def _process_frame(self, frame, height, width) -> list:
        """Process frame and return list of Detection objects"""
        # Run YOLO detection with tracking
        try:
            results = self.model.track(
                frame,
                conf=0.45,  # Higher threshold for COCO (fewer false positives)
                persist=True,
                tracker="bytetrack.yaml",
                verbose=False,
            )
        except:
            # Fallback if tracking fails
            results = self.model(frame, conf=0.45, verbose=False)

        detections = []

        if not results or not results[0].boxes:
            return detections

        # Process each detection
        for box in results[0].boxes:
            try:
                class_id = int(box.cls[0])
                class_name = self.class_names.get(class_id, "unknown")
                confidence = float(box.conf[0])

                bbox = box.xyxy[0].cpu().numpy()
                bbox_tuple = tuple(map(int, bbox))

                # Get track ID if available
                track_id = (
                    int(box.id[0])
                    if hasattr(box, "id") and box.id is not None
                    else None
                )

                # Estimate distance
                distance = self.estimator.estimate(
                    bbox_tuple, height, width, class_name=class_name, track_id=track_id
                )

                # Classify zone
                zone = self.navigator.classify_zone(bbox_tuple)

                # Determine priority (person is always critical)
                priority = (
                    2
                    if class_name == "person"
                    else 1 if class_id in self.critical_classes else 0
                )

                detection = Detection(
                    label=class_name,
                    confidence=confidence,
                    bbox=bbox_tuple,
                    distance=distance,
                    zone=zone,
                    priority=priority,
                )

                detections.append(detection)

            except Exception as e:
                logger.debug(f"Error processing detection: {e}")
                continue

        return detections

    def _provide_guidance(self, detections: list, frame):
        """Provide audio navigation guidance"""
        current_time = time.time()

        # Filter detections if in tracking mode
        if self.tracking_mode:
            filtered = [d for d in detections if d.label == self.tracking_mode]
            if filtered:
                closest = min(filtered, key=lambda d: d.distance)
                message = f"{self.tracking_mode} found {closest.zone}, {closest.distance:.1f} meters away"
                self.audio.speak(message, priority=True)
                # Reset tracking mode after found
                self.tracking_mode = None
            return

        # Regular navigation guidance
        if current_time - self.last_guidance_time < self.guidance_interval:
            return

        # Get navigation decision
        guidance = self.navigator.analyze_detections(detections)

        # Speak guidance
        priority = guidance.get("priority", 0)
        message = guidance.get("message", "")

        if message and priority >= 1:
            self.audio.speak(message, priority=(priority == 2))
            self.last_guidance_time = current_time

    def _draw_visualizations(self, frame, detections: list):
        """Draw visualizations on frame"""
        # Draw zone boundaries
        frame = self.navigator.draw_zones(frame)

        # Draw detections with distance
        for det in detections:
            x1, y1, x2, y2 = det.bbox

            # Color based on distance
            if det.distance < 1.0:
                color = (0, 0, 255)  # Red - danger
            elif det.distance < 2.0:
                color = (0, 165, 255)  # Orange - caution
            else:
                color = (0, 255, 0)  # Green - safe

            # Draw bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw label with distance and confidence
            label = f"{det.label} {det.distance:.1f}m ({det.confidence:.2f})"
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        # Draw FPS
        cv2.putText(
            frame,
            f"FPS: {self.fps:.1f}",
            (10, frame.shape[0] - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        # Draw model info
        cv2.putText(
            frame,
            "YOLOv8n COCO (80 classes)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )

        # Draw calibration status
        if not self.estimator.calibrated:
            cv2.putText(
                frame,
                "CALIBRATING...",
                (10, frame.shape[0] - 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )

        return frame


if __name__ == "__main__":
    import sys

    # Configuration
    VIDEO_SOURCE = "http://10.92.94.244:8080/video"  # Your IP camera
    MIC_INDEX = 2  # Your microphone index
    MODEL_PATH = "yolov8n.pt"  # Generic COCO model

    # Initialize and run system
    logger.info(f"Using YOLOv8n COCO model with 80 classes")
    logger.info(
        "Detecting: person, furniture, vehicles, animals, electronics, and more"
    )

    system = MoveFreeUltimateSystem(
        video_source=VIDEO_SOURCE, model_path=MODEL_PATH, mic_index=MIC_INDEX
    )

    system.run()
