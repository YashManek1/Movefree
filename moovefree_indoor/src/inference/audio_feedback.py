import pyttsx3
import threading
import queue
import logging

logger = logging.getLogger('Audio')

class AudioFeedback:
    def __init__(self, rate=150, volume=1.0):
        self._queue = queue.PriorityQueue()
        self.running = True
        self._rate = rate
        self._volume = volume
        self._counter = 0
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', self._rate)
            engine.setProperty('volume', self._volume)
        except Exception as e:
            logger.error(f'TTS init failed: {e}')
            return

        while self.running:
            try:
                _, _, text = self._queue.get(timeout=0.5)
                if text is None:
                    break
                engine.say(text)
                engine.runAndWait()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f'TTS error: {e}')

    def speak(self, text, priority=False):
        if not text:
            return
        with self._lock:
            if priority:
                while not self._queue.empty():
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break
            priority_val = 0 if priority else 1
            self._counter += 1
            self._queue.put((priority_val, self._counter, text))

    def set_rate(self, rate):
        self._rate = rate

    def set_volume(self, volume):
        self._volume = max(0.0, min(1.0, volume))

    def stop(self):
        self.running = False
        try:
            self._queue.put_nowait((0, 0, None))
        except Exception:
            pass
