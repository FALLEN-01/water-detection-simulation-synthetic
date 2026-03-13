"""
WebSocket Server — Water Leak Detection Live Dashboard
"""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import time
import threading
from datetime import datetime

from live_data_streamer import LiveSyntheticDataGenerator
from live_inference import LiveInferenceEngine

app = Flask(__name__, static_folder='static', template_folder='static')
app.config['SECRET_KEY'] = 'water-detection-2025'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

with open('config.json', 'r') as f:
    config = json.load(f)

data_streamer    = LiveSyntheticDataGenerator()
inference_engine = LiveInferenceEngine()

simulation_state = {'running': False, 'paused': False, 'thread': None}


def simulation_loop():
    print("Simulation loop started")
    while simulation_state['running']:
        if not simulation_state['paused']:
            try:
                sample      = data_streamer.generate_sample()
                predictions = inference_engine.predict(sample)

                # Extract predictions from live_inference engine
                ae_pred = predictions.get('autoencoder', {}) or {}
                if_pred = predictions.get('isolation_forest', {}) or {}

                ae_triggered = (ae_pred.get('prediction') == 1)
                if_triggered = (if_pred.get('prediction') == 1)
                ensemble_triggered = (predictions.get('ensemble') == 1)

                recon_error = predictions.get('reconstruction_error', 0.0)
                if_score = if_pred.get('anomaly_score', 0.0)
                confidence = predictions.get('confidence', 0.0)

                packet = {
                    'timestamp': sample['timestamp'],
                    'sim_time':  sample.get('sim_time', ''),
                    'speed':     sample.get('speed', 1),
                    'flow':      sample['flow_rate'],
                    'level2': {
                        'triggered': if_triggered,
                        'score': if_score
                    },
                    'level3': {
                        'triggered': ae_triggered,
                        'reconstruction_error': recon_error,
                        'score': recon_error
                    },
                    'final_score': confidence,
                    'anomaly': ensemble_triggered,
                    'leak_active': sample.get('leak_active', False),
                    'leak_mode': sample.get('leak_mode', 'instant'),
                    'leak_intensity': sample.get('leak_intensity', 0),
                    'leak_remaining': sample.get('leak_remaining', 0),
                }

                socketio.emit('data_update', packet)

                if ensemble_triggered:
                    socketio.emit('leak_alert', {
                        'timestamp':            sample['timestamp'],
                        'sim_time':             sample.get('sim_time', ''),
                        'flow_rate':            sample['flow_rate'],
                        'confidence':           confidence,
                        'reconstruction_error': recon_error,
                    })

            except Exception as e:
                print(f"Simulation loop error: {e}")
                import traceback
                traceback.print_exc()

        time.sleep(1.0 / config['sampling_rate_hz'])

    print("Simulation loop stopped")


@app.route('/')
def index():
    return send_from_directory('static', 'index.html')

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)


@socketio.on('connect')
def handle_connect():
    print(f"Client connected")
    emit('connection_response', {
        'status': 'connected',
        'config': {
            'sampling_rate':  config['sampling_rate_hz'],
            'window_size':    config['window_size'],
            'max_flow_rate':  config['max_flow_rate'],
            'fast_forward_max': config.get('fast_forward_max', 120),
        }
    })

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected")


@socketio.on('start_simulation')
def handle_start():
    global simulation_state
    if not simulation_state['running']:
        data_streamer.reset()
        inference_engine.reset()
        simulation_state['running'] = True
        simulation_state['paused']  = False
        t = threading.Thread(target=simulation_loop, daemon=True)
        simulation_state['thread'] = t
        t.start()
        emit('simulation_state', {'state': 'running'}, broadcast=True)
        print("Simulation started")

@socketio.on('stop_simulation')
def handle_stop():
    global simulation_state
    if simulation_state['running']:
        simulation_state['running'] = False
        simulation_state['paused']  = False
        if simulation_state['thread']:
            simulation_state['thread'].join(timeout=2.0)
        emit('simulation_state', {'state': 'stopped'}, broadcast=True)
        print("Simulation stopped")

@socketio.on('pause_simulation')
def handle_pause():
    global simulation_state
    if simulation_state['running'] and not simulation_state['paused']:
        simulation_state['paused'] = True
        emit('simulation_state', {'state': 'paused'}, broadcast=True)

@socketio.on('resume_simulation')
def handle_resume():
    global simulation_state
    if simulation_state['running'] and simulation_state['paused']:
        simulation_state['paused'] = False
        emit('simulation_state', {'state': 'running'}, broadcast=True)

@socketio.on('set_speed')
def handle_set_speed(data):
    """Change simulation speed multiplier"""
    speed = int(data) if isinstance(data, int) else int(data.get('speed', 1))
    data_streamer.set_speed(speed)
    emit('speed_update', {'speed': data_streamer.speed}, broadcast=True)
    print(f"Speed changed to {data_streamer.speed}x")

@socketio.on('inject_leak')
def handle_inject_leak(data):
    """Manually inject a leak into the simulation"""
    intensity = data.get('intensity', 0.5)  # L/min
    duration = data.get('duration', 60)     # minutes
    mode = data.get('mode', 'instant')      # 'instant' or 'ramp'
    ramp_minutes = data.get('ramp_minutes', 5)

    data_streamer.inject_leak(
        intensity=intensity,
        duration=duration,
        mode=mode,
        ramp_minutes=ramp_minutes
    )
    emit('leak_status', {
        'active': data_streamer.leak_active,
        'mode': data_streamer.leak_mode,
        'intensity': data_streamer.leak_intensity,
        'leak_remaining': data_streamer.get_leak_remaining()
    }, broadcast=True)

@socketio.on('stop_leak')
def handle_stop_leak():
    """Manually stop the current leak"""
    data_streamer.stop_leak()
    emit('leak_status', {
        'active': data_streamer.leak_active,
        'mode': data_streamer.leak_mode,
        'intensity': 0,
        'leak_remaining': 0
    }, broadcast=True)


if __name__ == '__main__':
    print("=" * 55)
    print("Water Leak Detection — Live Simulation Dashboard")
    print("=" * 55)
    print(f"http://localhost:{config['websocket_port']}")
    print("=" * 55)
    socketio.run(app,
                 host='0.0.0.0',
                 port=config['websocket_port'],
                 debug=False,
                 allow_unsafe_werkzeug=True)
