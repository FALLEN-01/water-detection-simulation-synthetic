"""
Live Inference Engine — Water Leak Detection
=============================================
Loads trained LSTM Autoencoder + Isolation Forest.
Runs real-time ensemble anomaly detection on streaming samples.
"""

import numpy as np
import json
import os
import pickle
from collections import deque
from typing import Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class LiveInferenceEngine:
    """Real-time anomaly detection using ensemble of trained models"""

    def __init__(self, config_path: str = "config.json"):
        with open(config_path, 'r') as f:
            self.config = json.load(f)

        self.window_size = self.config['window_size']   # 10 minutes
        self.max_flow    = self.config['max_flow_rate']

        # Sliding window buffer for LSTM autoencoder
        self.window_buffer = deque(maxlen=self.window_size)

        self.autoencoder           = None
        self.autoencoder_threshold = None
        self.isolation_forest      = None

        self._load_models()

        print("LiveInferenceEngine ready:")
        print(f"  Window size: {self.window_size}")
        print(f"  Autoencoder: {'✓' if self.autoencoder is not None else '✗'}")
        print(f"  IsolationForest: {'✓' if self.isolation_forest is not None else '✗'}")

    def _load_models(self):
        """Load pre-trained models from models/ directory"""
        models_cfg = self.config.get('models', {})

        # --- Autoencoder ---
        ae_path = models_cfg.get('autoencoder_path', 'models/autoencoder_lstm.keras')
        thr_path = models_cfg.get(
            'autoencoder_threshold_path',
            ae_path.replace('.keras', '_threshold.txt').replace('.h5', '_threshold.txt')
        )
        try:
            if os.path.exists(ae_path):
                from tensorflow import keras
                self.autoencoder = keras.models.load_model(ae_path)
                print(f"  Loaded autoencoder: {ae_path}")
                if os.path.exists(thr_path):
                    with open(thr_path) as f:
                        self.autoencoder_threshold = float(f.read().strip())
                    print(f"  Threshold: {self.autoencoder_threshold:.6f}")
                else:
                    self.autoencoder_threshold = 0.05
                    print(f"  Threshold file not found — using default {self.autoencoder_threshold}")
            else:
                print(f"  Autoencoder not found at {ae_path} — will run without it")
        except Exception as e:
            print(f"  Could not load autoencoder: {e}")

        # --- Isolation Forest ---
        if_path = models_cfg.get('isolation_forest_path', 'models/isolation_forest_model.pkl')
        try:
            if os.path.exists(if_path):
                with open(if_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                print(f"  Loaded isolation forest: {if_path}")
            else:
                print(f"  Isolation forest not found at {if_path} — will run without it")
        except Exception as e:
            print(f"  Could not load isolation forest: {e}")

        # No scaler used in this pipeline

    def _prepare_features(self, sample: Dict) -> np.ndarray:
        """Extract feature vector from a raw sensor sample"""
        # Feature order must match training: flow_normalized, turbidity,
        # flow_duration, hour, is_weekend
        return np.array([
            sample.get('flow_normalized', 0.0),
            sample.get('turbidity', 0.5),
            sample.get('flow_duration', 0.0),
            sample.get('hour', 0),
            sample.get('is_weekend', 0),
        ], dtype=np.float32)

    def predict(self, sample: Dict) -> Dict:
        """
        Run ensemble inference on a single incoming sample.
        Returns dict with autoencoder, isolation_forest, ensemble, confidence,
        and reconstruction_error for live display.
        """
        features = self._prepare_features(sample)
        self.window_buffer.append(features)

        result = {
            'autoencoder':       None,
            'isolation_forest':  None,
            'ensemble':          None,
            'confidence':        0.0,
            'reconstruction_error': 0.0,
        }

        # --- LSTM Autoencoder ---
        if self.autoencoder is not None and len(self.window_buffer) == self.window_size:
            try:
                win = np.array(list(self.window_buffer), dtype=np.float32)
                win_in = win.reshape(1, self.window_size, features.shape[0])
                recon = self.autoencoder.predict(win_in, verbose=0)
                mse = float(np.mean(np.square(win - recon[0])))
                is_anomaly = int(mse > self.autoencoder_threshold)
                result['autoencoder'] = {
                    'prediction': is_anomaly,
                    'reconstruction_error': mse,
                    'threshold': self.autoencoder_threshold,
                }
                result['reconstruction_error'] = mse
            except Exception as e:
                print(f"Autoencoder inference error: {e}")

        # --- Isolation Forest ---
        if self.isolation_forest is not None:
            try:
                feat = features.reshape(1, -1)
                pred  = self.isolation_forest.predict(feat)[0]
                score = float(self.isolation_forest.score_samples(feat)[0])
                is_anomaly = int(pred == -1)
                result['isolation_forest'] = {
                    'prediction': is_anomaly,
                    'anomaly_score': score,
                }
            except Exception as e:
                print(f"Isolation Forest inference error: {e}")

        # --- Ensemble majority vote ---
        votes = []
        if result['autoencoder'] is not None:
            votes.append(result['autoencoder']['prediction'])
        if result['isolation_forest'] is not None:
            votes.append(result['isolation_forest']['prediction'])

        if votes:
            result['ensemble']   = int(sum(votes) > len(votes) / 2)
            result['confidence'] = float(sum(votes) / len(votes))

        return result

    def reset(self):
        self.window_buffer.clear()
        print("LiveInferenceEngine reset")


if __name__ == "__main__":
    engine = LiveInferenceEngine()
    print("\nTest prediction:")
    sample = {
        'flow_normalized': 0.35, 'turbidity': 0.9,
        'flow_duration': 300, 'hour': 8, 'is_weekend': 0
    }
    for _ in range(10):
        r = engine.predict(sample)
    print(r)
