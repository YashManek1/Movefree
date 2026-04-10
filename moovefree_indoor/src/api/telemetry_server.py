import threading
import time
import logging
import os
from flask import Flask, Response, jsonify, request
from flask_cors import CORS

logger = logging.getLogger('TelemetryServer')

def create_telemetry_app(system_ref):
    app = Flask('telemetry')
    CORS(app)

    @app.route('/status')
    def status():
        return jsonify({
            'running': getattr(system_ref, 'running', False),
            'fps': getattr(system_ref, 'fps', 0),
            'mode': getattr(system_ref, 'tracking_mode', 'Auto-Nav'),
            'sonar': getattr(system_ref.hw, 'ultrasonic_dist', 999),
            'detections': len(getattr(system_ref, 'cached_detections', [])),
        })

    @app.route('/config', methods=['POST'])
    def update_config():
        data = request.get_json(silent=True) or {}
        if 'sensitivity' in data:
            val = float(data['sensitivity']) / 100.0
            if hasattr(system_ref, 'detector'):
                system_ref.detector.update_conf(val)
        if 'volume' in data:
            vol = float(data['volume']) / 100.0
            if hasattr(system_ref, 'audio'):
                system_ref.audio.set_volume(vol)
        if 'mode' in data and hasattr(system_ref, 'force_mode'):
            system_ref.force_mode = data['mode']
        return jsonify({'status': 'ok'})

    @app.route('/health')
    def health():
        return jsonify({'alive': True, 'ts': int(time.time() * 1000)})

    return app

def start_telemetry_server(system_ref, port=5001):
    app = create_telemetry_app(system_ref)

    def run():
        import logging as _logging
        _logging.getLogger('werkzeug').setLevel(_logging.ERROR)
        app.run(host='0.0.0.0', port=port, debug=False, use_reloader=False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    logger.info(f'Telemetry server on :{port}')
