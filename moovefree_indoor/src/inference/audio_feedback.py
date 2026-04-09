import pyttsx3
import threading
import queue
import logging

logger = logging.getLogger("Audio")


class AudioFeedback:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        # Start worker
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        """Single persistent engine instance"""
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 150)
        except Exception as e:
            logger.error(f"TTS Init Failed: {e}")
            return

        while self.running:
            text = self.queue.get()
            if text is None:
                break  # Stop signal

            try:
                # logger.info(f"🔊 Speaking: {text}")
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def speak(self, text, priority=False):
        if not text:
            return
        if priority:
            # Clear queue for emergency alerts
            with self.queue.mutex:
                self.queue.queue.clear()
        self.queue.put(text)

    def stop(self):
        self.running = False
        self.queue.put(None)
