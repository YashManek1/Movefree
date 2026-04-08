import cv2
import threading
import logging
import time
import os
import sys

# Suppress standard errors from FFmpeg/OpenCV to clean output
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOG_LEVEL"] = "-8"

from dotenv import load_dotenv
from ultralytics import YOLO

# Import local modules
from src.gps_nav import GPSNavigator
from src.safety_guard import OutdoorSafetyMonitor
from src.audio_out import OutdoorAudio

load_dotenv()

# --- LOG CLEANUP ---
# Only show essential info
logging.basicConfig(level=logging.ERROR, format="%(message)s")
logger = logging.getLogger("MoveFree_Outdoor")
logger.setLevel(logging.INFO)

# Mute noisy libraries
logging.getLogger("comtypes").setLevel(logging.WARNING)
logging.getLogger("googlemaps").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("ultralytics").setLevel(logging.WARNING)


class MoveFreeOutdoor:
    def __init__(self):
        logger.info("🚀 Initializing Outdoor Mode...")

        # 1. Load Model
        print("⏳ Loading AI Model (this may take a moment)...")
        self.model = YOLO("yolov8s.pt")
        print("✅ AI Model Loaded")

        # 2. Subsystems
        self.audio = OutdoorAudio()
        self.gps = GPSNavigator(self.audio)
        self.safety = OutdoorSafetyMonitor(self.audio)

        self.running = False

    def start_navigation(self, origin, destination):
        self.audio.speak(f"Starting navigation from {origin} to {destination}")

        # Start GPS Thread
        self.gps.set_route(origin, destination)
        threading.Thread(target=self.gps.run, daemon=True).start()

        self.running = True
        self._vision_loop()

    def _vision_loop(self):
        # --- CAMERA SETUP ---
        source = "0"
        env_url = os.getenv("IP_CAMERA_URL")

        if env_url:
            # Clean up the URL if needed
            if (
                not env_url.isdigit()
                and not env_url.endswith("/video")
                and "http" in env_url
            ):
                logger.warning(
                    f"⚠️ URL '{env_url}' might be missing '/video'. Trying to append it."
                )
                # We won't auto-append in case it's a different brand, but we warn the user

            source = int(env_url) if env_url.isdigit() else env_url
            logger.info(f"📡 Connecting to Camera: {source}")
        else:
            logger.info("📷 Using Webcam")

        # Open Camera
        cap = cv2.VideoCapture(source)

        # Check connection
        if not cap.isOpened():
            logger.error(f"\n❌ ERROR: Could not open camera: {source}")
            logger.error(
                "👉 Tip: If using IP Webcam, ensure URL ends with '/video' (e.g., http://192.168.x.x:8080/video)"
            )
            self.audio.speak("Camera connection failed. Using GPS only.")
            # We don't return here so GPS can still work, but vision loop ends
            return

        logger.info("✅ Camera Active. Press 'Q' to stop.")

        while self.running:
            ret, frame = cap.read()
            if not ret:
                # If stream drops, wait and try again
                time.sleep(1)
                continue

            frame = cv2.resize(frame, (640, 480))

            # 1. Detection
            results = self.model.track(frame, persist=True, verbose=False, conf=0.5)

            # 2. Safety Analysis
            if results[0].boxes:
                self.safety.analyze_frame(results[0], frame)

            # Visualization
            cv2.imshow("MoveFree Outdoor", results[0].plot())
            if cv2.waitKey(1) == ord("q"):
                self.stop()
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
        self.gps.stop()
        self.audio.stop()
        self.audio.speak("Ending Navigation.")


if __name__ == "__main__":
    app = MoveFreeOutdoor()

    # Allow TTS engine to init
    time.sleep(1)

    print("\n" + "=" * 40)
    print("   🌐 MOVEFREE OUTDOOR NAVIGATION")
    print("=" * 40)
    print("ℹ️  Since this is a laptop, please enter your")
    print("    current location to simulate GPS.")
    print("=" * 40)

    try:
        origin = input("📍 Start Location (e.g. 'Your Current Building'): ").strip()
        dest = input("🏁 Destination   (e.g. 'Mira Road Station'): ").strip()

        if origin and dest:
            app.start_navigation(origin, dest)
        else:
            print("❌ Invalid input. Please enter both locations.")
    except KeyboardInterrupt:
        print("\nExiting...")
