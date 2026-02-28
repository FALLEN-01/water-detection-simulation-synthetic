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

                packet = {
                    'timestamp': sample['timestamp'],
                    'sim_time':  sample.get('sim_time', ''),
                    'speed':     sample.get('speed', 1),
                    'sensor_data': {
                        'flow_rate':            sample['flow_rate'],
                        'flow_normalized':       sample['flow_normalized'],
                        'turbidity':             sample['turbidity'],
                        'hour':                  sample['hour'],
                        'is_weekend':            sample['is_weekend'],
                    },
                    'predictions': {
                        'autoencoder':      predictions.get('autoencoder'),
                        'isolation_forest': predictions.get('isolation_forest'),
                        'ensemble':         predictions.get('ensemble'),
                        'confidence':       predictions.get('confidence', 0.0),
                        'reconstruction_error': predictions.get('reconstruction_error', 0.0),
                    },
                    'ground_truth': sample['label'],
                    'leak_active':  sample['leak_active'],
                }

                socketio.emit('sensor_data', packet)

                if predictions.get('ensemble') == 1:
                    socketio.emit('leak_alert', {
                        'timestamp':            sample['timestamp'],
                        'sim_time':             sample.get('sim_time', ''),
                        'flow_rate':            sample['flow_rate'],
                        'confidence':           predictions.get('confidence', 0.0),
                        'reconstruction_error': predictions.get('reconstruction_error', 0.0),
                        'autoencoder':          predictions.get('autoencoder'),
                        'isolation_forest':     predictions.get('isolation_forest'),
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
        emit('simulation_status', {'status': 'started'}, broadcast=True)
        print("Simulation started")

@socketio.on('stop_simulation')
def handle_stop():
    global simulation_state
    if simulation_state['running']:
        simulation_state['running'] = False
        simulation_state['paused']  = False
        if simulation_state['thread']:
            simulation_state['thread'].join(timeout=2.0)
        emit('simulation_status', {'status': 'stopped'}, broadcast=True)
        print("Simulation stopped")

@socketio.on('pause_simulation')
def handle_pause():
    global simulation_state
    if simulation_state['running'] and not simulation_state['paused']:
        simulation_state['paused'] = True
        emit('simulation_status', {'status': 'paused'}, broadcast=True)

@socketio.on('resume_simulation')
def handle_resume():
    global simulation_state
    if simulation_state['running'] and simulation_state['paused']:
        simulation_state['paused'] = False
        emit('simulation_status', {'status': 'resumed'}, broadcast=True)

@socketio.on('set_speed')
def handle_set_speed(data):
    """Change simulation speed multiplier"""
    speed = int(data.get('speed', 1))
    data_streamer.set_speed(speed)
    emit('speed_changed', {'speed': data_streamer.speed}, broadcast=True)
    print(f"Speed changed to {data_streamer.speed}x")


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
