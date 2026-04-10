import speech_recognition as sr
import threading
import queue
import time
import logging
import re
import os
from typing import Callable
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

try:
    import google.generativeai as genai
    from PIL import Image
    import cv2
    import numpy as np
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

@dataclass
class Command:
    action: str
    target: str = None

class ConversationalAI:
    def __init__(self, mic_index: int = None, language: str = 'en-US'):
        self.mic_index = mic_index
        self.language = language
        self.recognizer = sr.Recognizer()
        self.recognizer.energy_threshold = 1000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3

        self.callbacks = {}
        self.listening = False
        self.current_frame = None

        self.gemini = None
        if GEMINI_AVAILABLE and os.getenv('GEMINI_API_KEY'):
            try:
                genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
                self.gemini = genai.GenerativeModel('gemini-2.5-flash')
            except Exception as e:
                logger.error(f'Gemini init: {e}')

        self._patterns = {
            'stop': [r'\bstop\b', r'\bhalt\b', r'\bwait\b', r'\bpause\b', r'\bfreeze\b'],
            'find': [r'find\s+(?:the\s+)?(\w+)', r'where\s+is\s+(?:the\s+)?(\w+)', r'locate\s+(?:the\s+)?(\w+)'],
            'describe': [r'\bdescribe\b', r'what\s+do\s+you\s+see', r'what.*around\s+me', r'what.*in\s+front'],
            'read': [r'\bread\b', r'what\s+does\s+it\s+say', r'read.*sign'],
            'navigate': [r'which\s+way', r'guide\s+me', r'navigate', r'where.*go', r'safe.*walk'],
            'distance': [r'how\s+far', r'how\s+close', r'how\s+near'],
            'identify': [r'what\s+is\s+this', r'identify', r'recognize'],
            'count': [r'how\s+many', r'count\s+(?:the\s+)?(\w+)'],
            'exits': [r'where.*exit', r'find.*door', r'way\s+out'],
            'hazards': [r'any\s+danger', r'any\s+hazard', r'safe\s+here', r'obstacles'],
            'summary': [r'quick\s+overview', r'brief\s+description', r'give.*summary'],
            'people': [r'anyone\s+here', r'any\s+people', r'is\s+someone\s+there'],
            'calibrate': [r'\bcalibrate\b', r'reset.*calibration'],
            'sos': [r'\bsos\b', r'emergency', r'help\s+me', r'i.*fallen', r'call.*caretaker'],
            'help': [r'\bhelp\b', r'what.*can.*you.*do', r'commands'],
        }

        self._compiled = {
            action: [re.compile(p, re.IGNORECASE) for p in patterns]
            for action, patterns in self._patterns.items()
        }

    def set_frame(self, frame):
        self.current_frame = frame

    def register_callback(self, action: str, callback: Callable):
        self.callbacks[action] = callback

    def start_listening(self):
        if self.listening:
            return
        self.listening = True
        threading.Thread(target=self._listen_loop, daemon=True).start()
        logger.info('Voice AI started.')

    def stop_listening(self):
        self.listening = False

    def _listen_loop(self):
        try:
            mic = sr.Microphone(device_index=self.mic_index)
        except Exception as e:
            logger.error(f'Mic init failed: {e}')
            return

        with mic as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        consecutive_errors = 0

        while self.listening:
            try:
                with mic as source:
                    audio = self.recognizer.listen(source, timeout=3, phrase_time_limit=8)

                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f'Heard: "{text}"')
                consecutive_errors = 0

                cmd = self._parse(text)
                if cmd:
                    self._execute(cmd, text)
                elif self.gemini and self.current_frame is not None:
                    self._gemini_contextual(text)

            except sr.WaitTimeoutError:
                consecutive_errors = 0
            except sr.UnknownValueError:
                consecutive_errors = 0
            except sr.RequestError as e:
                consecutive_errors += 1
                time.sleep(1)
            except Exception:
                consecutive_errors += 1
                time.sleep(0.5)

            if consecutive_errors >= 5:
                logger.error('Voice listener stopped after too many errors.')
                self.listening = False

    def _parse(self, text: str):
        text_lower = text.lower().strip()
        for action, patterns in self._compiled.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    target = match.group(1) if match.groups() else None
                    return Command(action=action, target=target)
        return None

    def _execute(self, cmd: Command, raw: str = ''):
        if cmd.action in self.callbacks:
            try:
                cb = self.callbacks[cmd.action]
                cb(cmd.target) if cmd.target else cb()
            except Exception as e:
                logger.error(f'Callback error ({cmd.action}): {e}')
        elif cmd.action in ('identify', 'count', 'people', 'summary'):
            self._gemini_command(cmd.action, raw)
        else:
            logger.debug(f'No handler for action: {cmd.action}')

    def _gemini_contextual(self, query: str):
        if not self.gemini or self.current_frame is None:
            return

        def task():
            try:
                rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                prompt = (
                    f'You are the eyes for a blind person. They asked: "{query}". '
                    'Analyze this image and respond concisely. Use clock positions '
                    'for directions (12 = ahead, 3 = right, 9 = left). '
                    'Focus on safety and navigation. No asterisks or markdown.'
                )
                response = self.gemini.generate_content([prompt, pil_img])
                if response.text:
                    clean = response.text.replace('*', '').replace('#', '').strip()
                    if 'speak' in self.callbacks:
                        self.callbacks['speak'](clean, 1)
            except Exception as e:
                logger.error(f'Gemini contextual: {e}')

        threading.Thread(target=task, daemon=True).start()

    def _gemini_command(self, action: str, raw: str):
        if not self.gemini or self.current_frame is None:
            if 'speak' in self.callbacks:
                self.callbacks['speak']('Visual analysis unavailable.', 1)
            return

        def task():
            try:
                rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)
                prompts = {
                    'identify': 'What object is this? Describe briefly for a blind person.',
                    'count': f'Answer: "{raw}". Count the requested objects.',
                    'people': 'How many people are in this image and where are they?',
                    'summary': 'Give a 2-sentence safety-focused summary of this scene.',
                }
                response = self.gemini.generate_content([prompts.get(action, raw), pil_img])
                if response.text:
                    clean = response.text.replace('*', '').replace('#', '').strip()
                    if 'speak' in self.callbacks:
                        self.callbacks['speak'](clean, 0)
            except Exception as e:
                logger.error(f'Gemini command ({action}): {e}')
                if 'speak' in self.callbacks:
                    self.callbacks['speak']('Could not process that request.', 1)

        threading.Thread(target=task, daemon=True).start()
