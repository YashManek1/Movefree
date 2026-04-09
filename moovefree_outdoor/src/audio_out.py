import pyttsx3
import threading
import queue
import logging

logger = logging.getLogger("OutdoorAudio")


class OutdoorAudio:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        # Start the worker thread that handles speaking
        threading.Thread(target=self._worker, daemon=True).start()

    def speak(self, text, priority=False):
        print(f"🔊 OUTDOOR: {text}")
        if priority:
            # Clear previous non-critical messages to speak this immediately
            with self.queue.mutex:
                self.queue.queue.clear()
        self.queue.put(text)

    def _worker(self):
        """Dedicated thread for text-to-speech to prevent event loop errors"""
        # Initialize engine inside the thread (Crucial for Windows/COM)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 160)
        except Exception as e:
            logger.error(f"Failed to init TTS engine: {e}")
            return

        while self.running:
            try:
                # Wait for next message
                text = self.queue.get()
                if text is None:
                    break  # Exit signal

                engine.say(text)
                engine.runAndWait()
                self.queue.task_done()
            except Exception as e:
                logger.error(f"TTS Error: {e}")

    def stop(self):
        self.running = False
        self.queue.put(None)  # Signal worker to stop
