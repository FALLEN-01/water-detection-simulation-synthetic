"""
Real-time inference engine for water leak detection.
Loads pre-trained models and performs live anomaly detection.
"""

import numpy as np
import json
import os
import pickle
from collections import deque
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')


class LiveInferenceEngine:
    """Performs real-time anomaly detection on streaming data"""
    
    def __init__(self, config_path: str = "config.json"):
        """Initialize the inference engine with pre-trained models"""
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.window_size = self.config['window_size']
        self.max_flow = self.config['max_flow_rate']
        
        # Sliding window buffer for autoencoder
        self.window_buffer = deque(maxlen=self.window_size)
        
        # Load models
        self.autoencoder = None
        self.isolation_forest = None
        self.scaler = None
        self.autoencoder_threshold = None
        
        self._load_models()
        
        print(f"LiveInferenceEngine initialized:")
        print(f"  Window size: {self.window_size}")
        print(f"  Autoencoder loaded: {self.autoencoder is not None}")
        print(f"  Isolation Forest loaded: {self.isolation_forest is not None}")
    
    def _load_models(self):
        """Load pre-trained models if they exist"""
        try:
            # Try to load autoencoder
            autoencoder_path = self.config['models']['autoencoder_path']
            if os.path.exists(autoencoder_path):
                from tensorflow import keras
                self.autoencoder = keras.models.load_model(autoencoder_path)
                
                # Load threshold (stored separately)
                threshold_path = autoencoder_path.replace('.h5', '_threshold.txt')
                if os.path.exists(threshold_path):
                    with open(threshold_path, 'r') as f:
                        self.autoencoder_threshold = float(f.read().strip())
                else:
                    self.autoencoder_threshold = 0.1  # Default threshold
                
                print(f"  Loaded autoencoder from {autoencoder_path}")
                print(f"  Threshold: {self.autoencoder_threshold:.6f}")
            else:
                print(f"  Warning: Autoencoder not found at {autoencoder_path}")
        except Exception as e:
            print(f"  Warning: Could not load autoencoder: {e}")
        
        try:
            # Try to load isolation forest
            isolation_path = self.config['models']['isolation_forest_path']
            if os.path.exists(isolation_path):
                with open(isolation_path, 'rb') as f:
                    self.isolation_forest = pickle.load(f)
                print(f"  Loaded isolation forest from {isolation_path}")
            else:
                print(f"  Warning: Isolation forest not found at {isolation_path}")
        except Exception as e:
            print(f"  Warning: Could not load isolation forest: {e}")
        
        try:
            # Try to load scaler
            scaler_path = self.config['models']['scaler_path']
            if os.path.exists(scaler_path):
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"  Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"  Warning: Could not load scaler: {e}")
    
    def _prepare_features(self, sample: Dict) -> np.ndarray:
        """Prepare features from raw sample data"""
        # Features: flow_normalized, turbidity, flow_duration, hour, is_weekend
        # Must match the training data feature order
        features = np.array([
            sample['flow_normalized'],
            sample['turbidity'],
            sample['flow_duration'],
            sample['hour'],
            sample['is_weekend']
        ])
        return features
    
    def predict(self, sample: Dict) -> Dict:
        """
        Perform real-time prediction on a single sample.
        Returns predictions from both models.
        """
        # Prepare features
        features = self._prepare_features(sample)
        
        # Add to window buffer
        self.window_buffer.append(features)
        
        predictions = {
            'autoencoder': None,
            'isolation_forest': None,
            'ensemble': None,
            'confidence': 0.0
        }
        
        # Autoencoder prediction (requires full window)
        if self.autoencoder is not None and len(self.window_buffer) == self.window_size:
            try:
                # Prepare window
                window = np.array(list(self.window_buffer))
                window_input = window.reshape(1, self.window_size, features.shape[0])
                
                # Get reconstruction
                reconstruction = self.autoencoder.predict(window_input, verbose=0)
                
                # Calculate reconstruction error
                mse = np.mean(np.square(window - reconstruction[0]))
                
                # Predict anomaly
                is_anomaly = 1 if mse > self.autoencoder_threshold else 0
                predictions['autoencoder'] = {
                    'prediction': is_anomaly,
                    'reconstruction_error': float(mse),
                    'threshold': self.autoencoder_threshold
                }
            except Exception as e:
                print(f"Autoencoder prediction error: {e}")
        
        # Isolation Forest prediction (single sample)
        if self.isolation_forest is not None:
            try:
                # Prepare single sample
                sample_features = features.reshape(1, -1)
                
                # Predict
                prediction = self.isolation_forest.predict(sample_features)[0]
                anomaly_score = self.isolation_forest.score_samples(sample_features)[0]
                
                # Convert: -1 = anomaly, 1 = normal
                is_anomaly = 1 if prediction == -1 else 0
                
                predictions['isolation_forest'] = {
                    'prediction': is_anomaly,
                    'anomaly_score': float(anomaly_score)
                }
            except Exception as e:
                print(f"Isolation Forest prediction error: {e}")
        
        # Ensemble prediction (majority vote)
        votes = []
        if predictions['autoencoder'] is not None:
            votes.append(predictions['autoencoder']['prediction'])
        if predictions['isolation_forest'] is not None:
            votes.append(predictions['isolation_forest']['prediction'])
        
        if votes:
            ensemble_prediction = 1 if sum(votes) > len(votes) / 2 else 0
            confidence = sum(votes) / len(votes)
            predictions['ensemble'] = ensemble_prediction
            predictions['confidence'] = confidence
        
        return predictions
    
    def reset(self):
        """Reset the inference engine state"""
        self.window_buffer.clear()
        print("LiveInferenceEngine reset")


if __name__ == "__main__":
    # Test the inference engine
    engine = LiveInferenceEngine()
    
    # Create test samples
    print("\nTesting with sample data:")
    test_samples = [
        {
            'flow_normalized': 0.3,
            'turbidity': 1.2,
            'hour': 8,
            'is_weekend': 0
        },
        {
            'flow_normalized': 0.8,
            'turbidity': 2.5,
            'hour': 8,
            'is_weekend': 0
        }
    ]
    
    for i, sample in enumerate(test_samples):
        predictions = engine.predict(sample)
        print(f"\nSample {i+1}:")
        print(f"  Autoencoder: {predictions['autoencoder']}")
        print(f"  Isolation Forest: {predictions['isolation_forest']}")
        print(f"  Ensemble: {predictions['ensemble']} (confidence: {predictions['confidence']:.2f})")
