import time
import logging
import os
import googlemaps
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger("GPSNavigator")


class GPSNavigator:
    def __init__(self, audio_interface):
        self.audio = audio_interface
        self.destination = None
        self.steps = []
        self.step_index = 0
        self.running = False

        # Initialize Google Maps
        self.gmaps = None
        if os.getenv("GOOGLE_MAPS_API_KEY"):
            self.gmaps = googlemaps.Client(key=os.getenv("GOOGLE_MAPS_API_KEY"))
            logger.info("✅ Google Maps API Connected")
        else:
            logger.warning("⚠️ No Google Maps API Key found in .env")

    def set_route(self, origin_name, destination_name):
        """Calculates route from a specific Origin to Destination"""
        if not self.gmaps:
            self.audio.speak("Maps API unavailable.")
            return

        try:
            # 1. Geocode Origin (Where am I?)
            origin_geocode = self.gmaps.geocode(origin_name)
            if not origin_geocode:
                self.audio.speak(f"Could not find starting point: {origin_name}")
                return
            origin_loc = origin_geocode[0]["geometry"]["location"]

            # 2. Geocode Destination (Where am I going?)
            dest_geocode = self.gmaps.geocode(destination_name)
            if not dest_geocode:
                self.audio.speak(f"Could not find destination: {destination_name}")
                return
            dest_loc = dest_geocode[0]["geometry"]["location"]

            # 3. Get Directions (Walking Mode)
            directions = self.gmaps.directions(
                origin_loc, dest_loc, mode="walking", departure_time=datetime.now()
            )

            if directions:
                self.steps = directions[0]["legs"][0]["steps"]
                self.step_index = 0

                # Speak summary
                summary = directions[0]["summary"]
                first_instr = self._clean_html(self.steps[0]["html_instructions"])

                print(f"🗺️ ROUTE: {origin_name} -> {destination_name}")
                self.audio.speak(f"Route calculated via {summary}. {first_instr}")

                self.destination = destination_name
                self.running = True
            else:
                self.audio.speak("No walking route found.")

        except Exception as e:
            logger.error(f"Routing Error: {e}")
            self.audio.speak("Error calculating route.")

    def run(self):
        while self.running and self.destination:
            # SIMULATION: Updates every 10 seconds
            # In real hardware, we would check self.get_real_gps() vs step location
            time.sleep(10)

            if self.step_index < len(self.steps) - 1:
                self.step_index += 1
                instr = self._clean_html(
                    self.steps[self.step_index]["html_instructions"]
                )
                dist = self.steps[self.step_index]["distance"]["text"]

                # Print to console for verification
                print(f"📍 NEXT: In {dist}, {instr}")
                self.audio.speak(f"In {dist}, {instr}")
            else:
                self.audio.speak("You have arrived at your destination.")
                self.running = False

    def stop(self):
        self.running = False

    def _clean_html(self, raw_html):
        import re

        cleanr = re.compile("<.*?>")
        return re.sub(cleanr, "", raw_html)
