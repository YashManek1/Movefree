import os
import time
import subprocess
import logging
import requests
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s - [ORCHESTRATOR] %(message)s')
logger = logging.getLogger(__name__)

load_dotenv('moovefree_indoor/.env')

DB_URL = os.getenv('FIREBASE_DATABASE_URL', '').rstrip('/')
DB_SECRET = os.getenv('FIREBASE_DATABASE_SECRET', '')
UID = os.getenv('BLIND_USER_UID', '')

if not DB_URL or not UID:
    logger.error("Missing FIREBASE_DATABASE_URL or BLIND_USER_UID in moovefree_indoor/.env.")
    logger.error("Please populate them before running the master orchestrator.")
    exit(1)

def get_remote_mode():
    try:
        r = requests.get(f"{DB_URL}/sessions/{UID}/config/mode.json?auth={DB_SECRET}", timeout=3)
        if r.status_code == 200:
            val = r.json()

            if val in ['indoor', 'outdoor']:
                return val
    except Exception:
        pass
    return 'indoor'  

class MooveFreeOrchestrator:
    def __init__(self):
        self.current_mode = None
        self.process = None

    def start_process(self, mode):
        logger.info(f"========== SWITCHING TO {mode.upper()} MODE ==========")
        cwd_path = os.path.join(os.getcwd(), f'moovefree_{mode}')

        if self.process:
            logger.info("Terminating previous AI pipeline...")
            self.process.terminate()
            self.process.wait()

        logger.info(f"Launching {mode} AI pipeline...")
        self.process = subprocess.Popen(['python', 'main.py'], cwd=cwd_path)
        self.current_mode = mode

        try:
            requests.put(f"{DB_URL}/sessions/{UID}/telemetry/mode.json?auth={DB_SECRET}", json=mode)
        except:
            pass

    def run(self):
        logger.info("Starting MooveFree Master Orchestrator...")
        logger.info("Monitoring Firebase for mode changes...")

        while True:
            target_mode = get_remote_mode()

            if target_mode != self.current_mode:
                self.start_process(target_mode)

            time.sleep(3)

if __name__ == "__main__":
    try:
        MooveFreeOrchestrator().run()
    except KeyboardInterrupt:
        logger.info("Shutting down orchestrator...")
        if 'process' in locals() and process:
            process.terminate()
