"""
Conversational AI System - HUMAN EYES MODE
Advanced voice command recognition with Gemini 2.5 Flash integration
Robust command handling for blind users
"""

import speech_recognition as sr
import threading
import queue
import time
import logging
import re
import os
from typing import Callable, Dict, List
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Gemini Integration
try:
    import google.generativeai as genai
    from PIL import Image
    import cv2
    import numpy as np

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    logger.warning("Gemini not available")


@dataclass
class Command:
    """Represents a parsed voice command"""

    action: str  # Command type
    target: str = None  # Object/target
    parameters: dict = None  # Additional params


class ConversationalAI:
    """
    Advanced conversational AI with Gemini 2.5 Flash
    Your eyes, your guide, your companion
    """

    def __init__(self, mic_index: int = None, language: str = "en-US"):
        self.mic_index = mic_index
        self.language = language
        self.recognizer = sr.Recognizer()

        # Aggressive sensitivity for better detection
        self.recognizer.energy_threshold = 1000
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8
        self.recognizer.phrase_threshold = 0.3

        # Command queue
        self.command_queue = queue.Queue()

        # Command callbacks
        self.callbacks = {}

        # State
        self.listening = False
        self.active = False
        self.current_frame = None

        # Initialize Gemini 2.5 Flash
        self.gemini = None
        if GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY"):
            try:
                genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
                self.gemini = genai.GenerativeModel("gemini-2.5-flash")
                logger.info("✅ Gemini 2.5 Flash connected")
            except Exception as e:
                logger.error(f"Gemini initialization failed: {e}")

        # Enhanced command patterns
        self.command_patterns = {
            "stop": [
                r"\bstop\b",
                r"\bhalt\b",
                r"\bwait\b",
                r"\bhold\b",
                r"\bpause\b",
                r"\bfreeze\b",
            ],
            "find": [
                r"find\s+(?:the\s+)?(\w+)",
                r"where\s+is\s+(?:the\s+)?(\w+)",
                r"locate\s+(?:the\s+)?(\w+)",
                r"search\s+for\s+(?:the\s+)?(\w+)",
                r"look\s+for\s+(?:the\s+)?(\w+)",
            ],
            "describe": [
                r"\bdescribe\b",
                r"what\s+do\s+you\s+see",
                r"tell\s+me\s+what.*see",
                r"describe\s+(?:the\s+)?scene",
                r"what.*around\s+me",
                r"what.*in\s+front",
            ],
            "read": [
                r"\bread\b",
                r"read\s+text",
                r"what\s+does\s+it\s+say",
                r"read\s+(?:the\s+)?sign",
                r"read\s+(?:the\s+)?label",
            ],
            "navigate": [
                r"which\s+way",
                r"what.*direction",
                r"guide\s+me",
                r"navigate",
                r"where\s+should\s+i\s+go",
                r"safe\s+to\s+walk",
            ],
            "distance": [
                r"how\s+far",
                r"distance\s+to",
                r"how\s+close",
                r"how\s+near",
            ],
            "identify": [
                r"what\s+is\s+this",
                r"what\s+am\s+i\s+holding",
                r"identify",
                r"recognize",
                r"tell\s+me\s+what\s+this\s+is",
            ],
            "count": [
                r"how\s+many",
                r"count\s+(?:the\s+)?(\w+)",
            ],
            "exits": [
                r"where.*exit",
                r"find.*door",
                r"how.*leave",
                r"way\s+out",
            ],
            "hazards": [
                r"any\s+danger",
                r"any\s+hazard",
                r"safe\s+here",
                r"obstacles",
            ],
            "color": [
                r"what\s+color",
                r"tell\s+me\s+the\s+color",
            ],
            "help": [
                r"\bhelp\b",
                r"what\s+can\s+you\s+do",
                r"commands",
                r"instructions",
            ],
            "calibrate": [r"\bcalibrate\b", r"reset\s+calibration"],
            # NEW: Contextual awareness
            "summary": [
                r"give\s+me\s+a\s+summary",
                r"quick\s+overview",
                r"brief\s+description",
            ],
            "people": [
                r"anyone\s+here",
                r"any\s+people",
                r"is\s+someone\s+there",
            ],
        }

        # Compile patterns
        self.compiled_patterns = {}
        for action, patterns in self.command_patterns.items():
            self.compiled_patterns[action] = [
                re.compile(pattern, re.IGNORECASE) for pattern in patterns
            ]

    def set_frame(self, frame):
        """Update current frame for Gemini analysis"""
        self.current_frame = frame

    def register_callback(self, action: str, callback: Callable):
        """Register a callback for a specific command action"""
        self.callbacks[action] = callback
        logger.info(f"Registered callback for action: {action}")

    def start_listening(self):
        """Start listening for voice commands in background thread"""
        if self.listening:
            logger.warning("Already listening")
            return

        self.listening = True
        self.active = True

        thread = threading.Thread(target=self._listen_loop, daemon=True)
        thread.start()

        logger.info("🎤 Conversational AI started - Your Eyes Are Active")

    def stop_listening(self):
        """Stop listening for voice commands"""
        self.listening = False
        self.active = False
        logger.info("🎤 Conversational AI stopped")

    def _listen_loop(self):
        """Main listening loop (runs in background thread)"""
        try:
            mic = sr.Microphone(device_index=self.mic_index)
        except Exception as e:
            logger.error(f"Failed to initialize microphone: {e}")
            return

        logger.info("🎤 Microphone initialized. Listening for commands...")

        # Adjust for ambient noise once
        with mic as source:
            logger.info("📊 Adjusting for ambient noise... (2 seconds)")
            self.recognizer.adjust_for_ambient_noise(source, duration=0.5)

        consecutive_errors = 0
        max_consecutive_errors = 5

        while self.listening:
            try:
                with mic as source:
                    # Listen with timeout
                    audio = self.recognizer.listen(
                        source, timeout=3, phrase_time_limit=8
                    )

                # Recognize speech
                text = self.recognizer.recognize_google(audio, language=self.language)
                logger.info(f"🗣️  Heard: '{text}'")

                # Parse command
                command = self._parse_command(text)

                if command:
                    logger.info(f"✅ Parsed command: {command.action}")
                    self._execute_command(command, text)
                    consecutive_errors = 0
                else:
                    # If no predefined command, use Gemini for contextual response
                    if self.gemini and self.current_frame is not None:
                        self._handle_contextual_query(text)

            except sr.WaitTimeoutError:
                consecutive_errors = 0
                continue

            except sr.UnknownValueError:
                logger.debug("Couldn't understand audio")
                consecutive_errors = 0
                continue

            except sr.RequestError as e:
                logger.error(f"Speech recognition service error: {e}")
                consecutive_errors += 1
                time.sleep(1)

            except Exception as e:
                logger.error(f"Unexpected error in listen loop: {e}")
                consecutive_errors += 1
                time.sleep(0.5)

            if consecutive_errors >= max_consecutive_errors:
                logger.error("Too many consecutive errors. Stopping listener.")
                self.listening = False

    def _parse_command(self, text: str) -> Command:
        """Parse natural language text into structured command"""
        text_lower = text.lower().strip()

        # Try each command pattern
        for action, patterns in self.compiled_patterns.items():
            for pattern in patterns:
                match = pattern.search(text_lower)
                if match:
                    # Extract target if present
                    target = None
                    if match.groups():
                        target = match.group(1)

                    return Command(action=action, target=target)

        return None

    def _execute_command(self, command: Command, original_text: str = ""):
        """Execute a parsed command"""
        if command.action in self.callbacks:
            try:
                callback = self.callbacks[command.action]

                # Call with target if applicable
                if command.target:
                    callback(command.target)
                else:
                    callback()

            except Exception as e:
                logger.error(f"Error executing command callback: {e}")
        elif command.action in ["identify", "color", "count", "people", "summary"]:
            # Use Gemini for these advanced commands
            self._handle_gemini_command(command.action, original_text)
        else:
            logger.warning(f"No callback registered for action: {command.action}")

    def _handle_contextual_query(self, query: str):
        """Handle queries using Gemini when no predefined command matches"""
        if not self.gemini or self.current_frame is None:
            return

        def _task():
            try:
                # Prepare image
                rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                # Context-aware prompt
                prompt = f"""You are the eyes for a blind person. They asked: "{query}"
                
Analyze this image and provide a helpful, concise response focused on their safety and navigation.
Be specific about:
- Object positions (use clock positions: 12 o'clock = straight ahead, 3 o'clock = right, etc.)
- Distances (rough estimates in meters/feet)
- Potential hazards
- Actionable guidance

Keep it brief and practical. No asterisks or formatting."""

                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    # Clean and speak
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    if "speak" in self.callbacks:
                        self.callbacks["speak"](clean_text, priority=1)

            except Exception as e:
                logger.error(f"Gemini contextual error: {e}")

        threading.Thread(target=_task, daemon=True).start()

    def _handle_gemini_command(self, action: str, original_text: str):
        """Handle specialized Gemini commands"""
        if not self.gemini or self.current_frame is None:
            if "speak" in self.callbacks:
                self.callbacks["speak"]("Visual analysis unavailable", priority=1)
            return

        def _task():
            try:
                rgb = cv2.cvtColor(self.current_frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(rgb)

                prompts = {
                    "identify": "What is this object? Describe it briefly for a blind person.",
                    "color": "What colors do you see? Describe the main colors in the scene.",
                    "count": f"Based on the question '{original_text}', count the requested objects and provide the number.",
                    "people": "Are there any people in this image? How many and where are they positioned?",
                    "summary": "Give a 2-sentence summary of this scene focusing on navigation and safety.",
                }

                prompt = prompts.get(action, original_text)
                response = self.gemini.generate_content([prompt, pil_img])

                if response.text:
                    clean_text = response.text.replace("*", "").replace("#", "").strip()
                    sentences = clean_text.split(". ")
                    for sentence in sentences:
                        if len(sentence.strip()) > 3:
                            if "speak" in self.callbacks:
                                self.callbacks["speak"](sentence.strip(), priority=0)

            except Exception as e:
                logger.error(f"Gemini command error: {e}")
                if "speak" in self.callbacks:
                    self.callbacks["speak"](
                        "I couldn't process that request", priority=1
                    )

        threading.Thread(target=_task, daemon=True).start()

    def test_microphone(self):
        """Test microphone and display available devices"""
        logger.info("\n🎤 AVAILABLE MICROPHONES:")
        for i, name in enumerate(sr.Microphone.list_microphone_names()):
            logger.info(f"  [{i}] {name}")

        logger.info("\n🎤 Testing microphone...")
        try:
            with sr.Microphone(device_index=self.mic_index) as source:
                self.recognizer.adjust_for_ambient_noise(source, duration=1)
                logger.info("✅ Microphone working. Speak now...")

                audio = self.recognizer.listen(source, timeout=5)
                text = self.recognizer.recognize_google(audio)

                logger.info(f"✅ Heard: '{text}'")
                return True

        except Exception as e:
            logger.error(f"❌ Microphone test failed: {e}")
            return False
