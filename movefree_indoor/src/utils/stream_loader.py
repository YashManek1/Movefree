import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)


class StreamLoader:
    def __init__(self, source):
        self.source = source
        self.q = queue.Queue(maxsize=1)  # Only keep the LATEST frame
        self.running = True
        self.connected = False

        # Start connection in background
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while self.running:
            try:
                cap = cv2.VideoCapture(self.source)
                if not cap.isOpened():
                    logger.warning(f"⚠️ Connection failed: {self.source}. Retrying...")
                    time.sleep(2)
                    continue

                self.connected = True
                logger.info("✅ Camera Connected")

                while self.running and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Force empty the queue to drop old frames (Key to fixing lag)
                    if not self.q.empty():
                        try:
                            self.q.get_nowait()
                        except queue.Empty:
                            pass

                    self.q.put(frame)

                cap.release()
                self.connected = False
            except Exception as e:
                logger.error(f"Stream Error: {e}")
                time.sleep(1)

    def read(self):
        try:
            return self.q.get(timeout=1)  # Wait max 1s for a frame
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=1)
