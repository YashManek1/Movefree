import cv2
import threading
import queue
import time
import logging

logger = logging.getLogger(__name__)

class StreamLoader:
    def __init__(self, source, reconnect_delay=2.0, max_reconnect_delay=30.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self.max_reconnect_delay = max_reconnect_delay
        self._frame_queue = queue.Queue(maxsize=2)
        self.running = True
        self.connected = False
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()

    def _open_capture(self):
        if isinstance(self.source, str) and self.source.startswith('http'):
            cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FPS, 30)
        else:
            cap = cv2.VideoCapture(self.source)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def _capture_loop(self):
        delay = self.reconnect_delay
        while self.running:
            cap = self._open_capture()
            if not cap.isOpened():
                logger.warning(f'Stream unavailable: {self.source}. Retrying in {delay:.0f}s')
                time.sleep(delay)
                delay = min(delay * 1.5, self.max_reconnect_delay)
                continue

            self.connected = True
            delay = self.reconnect_delay
            logger.info('Camera connected.')

            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put(frame)

            cap.release()
            self.connected = False
            if self.running:
                logger.warning('Stream lost. Reconnecting...')
                time.sleep(delay)

    def read(self, timeout=1.0):
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False
        self._thread.join(timeout=2)
