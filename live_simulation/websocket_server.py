"""
Simplified WebSocket server for live simulation (without eventlet).
Uses threading for async operations instead.
"""

from flask import Flask, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import time
import threading
from datetime import datetime

from live_data_streamer import LiveSyntheticDataGenerator as LiveDataStreamer
from live_inference import LiveInferenceEngine


# Initialize Flask app
app = Flask(__name__, 
            static_folder='static',
            template_folder='static')
app.config['SECRET_KEY'] = 'water-detection-secret-2024'
CORS(app)

# Initialize SocketIO with threading mode
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Load configuration
with open('config.json', 'r') as f:
    config = json.load(f)

# Initialize components
data_streamer = LiveDataStreamer()
inference_engine = LiveInferenceEngine()

# Simulation state
simulation_state = {
    'running': False,
    'paused': False,
    'thread': None
}


def simulation_loop():
    """Main simulation loop that generates and broadcasts data"""
    print("Simulation loop started")
    
    while simulation_state['running']:
        if not simulation_state['paused']:
            try:
                # Generate sensor sample
                sample = data_streamer.generate_sample()
                
                # Perform inference
                predictions = inference_engine.predict(sample)
                
                # Prepare data packet
                data_packet = {
                    'timestamp': sample['timestamp'],
                    'sensor_data': {
                        'flow_rate': sample['flow_rate'],
                        'turbidity': sample['turbidity'],
                        'hour': sample['hour'],
                        'is_weekend': sample['is_weekend']
                    },
                    'predictions': predictions,
                    'ground_truth': sample['label'],
                    'leak_active': sample['leak_active']
                }
                
                # Broadcast to all connected clients
                socketio.emit('sensor_data', data_packet, namespace='/')
                
                # If leak detected, send alert
                if predictions.get('ensemble') == 1:
                    alert_data = {
                        'timestamp': sample['timestamp'],
                        'flow_rate': sample['flow_rate'],
                        'confidence': predictions.get('confidence', 0.0),
                        'autoencoder': predictions.get('autoencoder'),
                        'isolation_forest': predictions.get('isolation_forest')
                    }
                    socketio.emit('leak_alert', alert_data, namespace='/')
                
            except Exception as e:
                print(f"Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
        
        # Sleep for sampling interval
        time.sleep(1.0 / config['sampling_rate_hz'])
    
    print("Simulation loop stopped")


@app.route('/')
def index():
    """Serve the main dashboard page"""
    return send_from_directory('static', 'index.html')


@app.route('/static/<path:path>')
def serve_static(path):
    """Serve static files"""
    return send_from_directory('static', path)


@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    print(f"Client connected: {datetime.now().isoformat()}")
    emit('connection_response', {
        'status': 'connected',
        'config': {
            'sampling_rate': config['sampling_rate_hz'],
            'window_size': config['window_size'],
            'max_flow_rate': config['max_flow_rate']
        }
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    print(f"Client disconnected: {datetime.now().isoformat()}")


@socketio.on('start_simulation')
def handle_start_simulation():
    """Start the simulation"""
    global simulation_state
    
    if not simulation_state['running']:
        print("Starting simulation...")
        
        # Reset components
        data_streamer.reset()
        inference_engine.reset()
        
        # Start simulation thread
        simulation_state['running'] = True
        simulation_state['paused'] = False
        simulation_state['thread'] = threading.Thread(target=simulation_loop)
        simulation_state['thread'].daemon = True
        simulation_state['thread'].start()
        
        emit('simulation_status', {
            'status': 'started',
            'timestamp': datetime.now().isoformat()
        }, broadcast=True)
    else:
        print("Simulation already running")


@socketio.on('stop_simulation')
def handle_stop_simulation():
    """Stop the simulation"""
    global simulation_state
    
    if simulation_state['running']:
        print("Stopping simulation...")
        simulation_state['running'] = False
        simulation_state['paused'] = False
        
        # Wait for thread to finish
        if simulation_state['thread']:
            simulation_state['thread'].join(timeout=2.0)
        
        emit('simulation_status', {
            'status': 'stopped',
            'timestamp': datetime.now().isoformat()
        }, broadcast=True)
    else:
        print("Simulation not running")


@socketio.on('pause_simulation')
def handle_pause_simulation():
    """Pause the simulation"""
    global simulation_state
    
    if simulation_state['running'] and not simulation_state['paused']:
        print("Pausing simulation...")
        simulation_state['paused'] = True
        
        emit('simulation_status', {
            'status': 'paused',
            'timestamp': datetime.now().isoformat()
        }, broadcast=True)


@socketio.on('resume_simulation')
def handle_resume_simulation():
    """Resume the simulation"""
    global simulation_state
    
    if simulation_state['running'] and simulation_state['paused']:
        print("Resuming simulation...")
        simulation_state['paused'] = False
        
        emit('simulation_status', {
            'status': 'resumed',
            'timestamp': datetime.now().isoformat()
        }, broadcast=True)


if __name__ == '__main__':
    print("=" * 60)
    print("Water Leak Detection - Live Simulation Dashboard")
    print("=" * 60)
    print(f"Server starting on port {config['websocket_port']}...")
    print(f"Dashboard URL: http://localhost:{config['websocket_port']}")
    print("=" * 60)
    
    socketio.run(app, 
                 host='0.0.0.0', 
                 port=config['websocket_port'],
                 debug=False,
                 allow_unsafe_werkzeug=True)
