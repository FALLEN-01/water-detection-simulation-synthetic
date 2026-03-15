"""
Experiment runner for apartment building simulator.
Tests leak detection accuracy for multi-unit building with randomized leak injection scenarios.
"""

import sys
import json
import pickle
import random
import numpy as np
from pathlib import Path
from collections import deque
import importlib.util


# ─────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────

APARTMENT_DIR = Path(__file__).parent.parent / "apartment_simulator"
ARTIFACTS_DIR = APARTMENT_DIR / "artifacts"
RESULTS_DIR = Path(__file__).parent / "results" / "apartment"

WINDOW_MINUTES = 20
SIMULATION_MINUTES = 600  # Extended: 10 hours (need room for 100+ min leaks at minute 100+)


# ─────────────────────────────────────────────────────────────────────
# Dynamic imports to avoid sys.path conflicts
# ─────────────────────────────────────────────────────────────────────

def load_apartment_modules():
    """Dynamically import apartment-specific modules."""
    # Import apartment live_simulator
    apt_sim_spec = importlib.util.spec_from_file_location(
        "apartment_live_simulator",
        str(APARTMENT_DIR / "backend" / "live_simulator.py")
    )
    apt_sim = importlib.util.module_from_spec(apt_sim_spec)
    apt_sim_spec.loader.exec_module(apt_sim)
    
    # Import apartment model
    apt_model_spec = importlib.util.spec_from_file_location(
        "apartment_model",
        str(APARTMENT_DIR / "backend" / "model.py")
    )
    apt_model = importlib.util.module_from_spec(apt_model_spec)
    apt_model_spec.loader.exec_module(apt_model)
    
    return apt_sim.LiveApartmentBuildingDataGenerator, apt_model.HybridWaterAnomalyDetector


LiveApartmentBuildingDataGenerator, ApartmentHybridDetector = load_apartment_modules()

from metrics_viz import calculate_metrics, plot_confusion_matrix, plot_accuracy_dashboard, save_results_csv, save_metrics_summary


# ─────────────────────────────────────────────────────────────────────
# Global State (reset between runs)
# ─────────────────────────────────────────────────────────────────────

if_model = None
if_scaler = None
calibration = None
leak_time = None
detection_time = None
flow_history = []
cusum_history = []


def load_models():
    """Load pre-trained models and calibration (once at startup)."""
    global if_model, if_scaler, calibration
    
    def load_pickle(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    
    def load_json(path):
        with open(path) as f:
            return json.load(f)
    
    if_model = load_pickle(ARTIFACTS_DIR / "isolation_forest_building.pkl")
    if_scaler = load_pickle(ARTIFACTS_DIR / "scaler_building.pkl")
    calibration = load_json(ARTIFACTS_DIR / "calibration_building.json")
    
    print(f"Loaded models from {ARTIFACTS_DIR}")


def create_fresh_detector():
    """Create a brand new detector instance (fresh state every run)."""
    return ApartmentHybridDetector(
        if_model=if_model,
        if_scaler=if_scaler,
        cusum_k=calibration["cusum_k"],
        cusum_h=calibration["cusum_h"],
        if_threshold=calibration["if_threshold"],
        if_score_scale=calibration["if_score_scale"],
        appliance_flow_thresh=calibration["appliance_flow_thresh"],
    )


def reset_state(seed=None):
    """
    Reset all state variables for a fresh run.
    Creates brand new detector and generator instances.
    """
    global leak_time, detection_time, flow_history, cusum_history
    
    leak_time = None
    detection_time = None
    flow_history = []
    cusum_history = []
    
    # Create fresh instances (same as server.py startup)
    generator = LiveApartmentBuildingDataGenerator(
        ARTIFACTS_DIR / "all_appliances.json",
        seed=seed
    )
    detector = create_fresh_detector()
    
    return generator, detector


def inject_leak(at_minute, intensity_lpm, duration_minutes):
    """
    Inject a leak at building level at a specific simulation minute.
    
    Args:
        at_minute: Minute in simulation (0-SIMULATION_MINUTES)
        intensity_lpm: Leak flow in L/min (aggregate, >= 2.2 for detection)
        duration_minutes: How long leak persists
    """
    global leak_time
    leak_time = at_minute
    generator.inject_leak(duration_minutes=duration_minutes, flow_lpm=intensity_lpm)


def run_single_simulation(leak_at_minute, leak_intensity_lpm, leak_duration_minutes, leak_mode="instant"):
    """
    Run a single simulation with leak injection.
    Uses EXACT same logic as server.py: manual leak addition (not generator.inject_leak).
    Supports manual leak mode specification.
    
    Args:
        leak_at_minute: When leak starts
        leak_intensity_lpm: Leak flow intensity
        leak_duration_minutes: How long leak lasts
        leak_mode: "instant" or "ramp" (default: "instant")
    
    Returns:
        Dict with result data
    """
    global leak_time, detection_time, flow_history, cusum_history
    
    # Create fresh instances for this run
    generator, detector = reset_state(seed=random.randint(0, 1000000))
    leak_time = None
    detection_time = None
    leak_end_minute = leak_at_minute + leak_duration_minutes
    
    # Use provided leak mode
    leak_ramp_minutes = random.randint(5, 15) if leak_mode == "ramp" else 0
    
    window_buffer = deque(maxlen=WINDOW_MINUTES)
    
    # Run simulation (MATCHING server.py logic)
    for minute in range(SIMULATION_MINUTES):
        
        # Get next flow reading (aggregated from 50 apartments)
        flow = generator.next()
        
        # ── MANUAL leak addition (same as server.py) ──
        if leak_at_minute <= minute < leak_end_minute:
            leak_time = leak_at_minute  # Record leak start
            
            if leak_mode == "instant":
                # Immediate full intensity
                effective_intensity = leak_intensity_lpm
            elif leak_mode == "ramp":
                # Gradual ramp up over leak_ramp_minutes
                elapsed = minute - leak_at_minute
                progress = min(1.0, elapsed / max(1, leak_ramp_minutes))
                effective_intensity = leak_intensity_lpm * progress
            else:
                effective_intensity = leak_intensity_lpm
            
            flow += effective_intensity
            flow = min(flow, generator.BUILDING_MAX_FLOW_LPM if hasattr(generator, 'BUILDING_MAX_FLOW_LPM') else 750)
        
        flow_history.append(flow)
        window_buffer.append(float(flow))
        
        # When window is full, run detection
        if len(window_buffer) == WINDOW_MINUTES:
            result = detector.update(list(window_buffer))
            cusum_history.append(result["level2"]["score"])
            
            # Capture first detection
            if result["anomaly"] and detection_time is None:
                detection_time = minute
    
    # Calculate metrics
    delay = None
    false_alarm = False
    missed_detection = False
    
    if leak_time is not None and detection_time is not None:
        delay = detection_time - leak_time
    
    if detection_time is not None and leak_time is None:
        # Alarm before leak = false alarm
        false_alarm = True
    
    if leak_time is not None and detection_time is None:
        # Leak occurred but no detection = missed
        missed_detection = True
    
    return {
        "leak_time": leak_time,
        "detection_time": detection_time,
        "delay": delay,
        "leak_mode": leak_mode,
        "false_alarm": false_alarm,
        "missed_detection": missed_detection,
        "flow_history": flow_history.copy(),
        "cusum_history": cusum_history.copy(),
    }


def run_experiment(num_runs=20, intensity_min=15.0, intensity_max=25.0, intensity_label="medium"):
    """
    Run multiple leak detection experiments with randomized scenarios.
    
    Args:
        num_runs: Number of simulation iterations
        intensity_min: Minimum leak intensity (L/min)
        intensity_max: Maximum leak intensity (L/min)
        intensity_label: Label for intensity level (low/medium/high)
    """
    # Ensure models are loaded
    load_models()
    
    # Pre-shuffle leak modes to ensure fair 50/50 distribution
    leak_modes = ["instant", "ramp"] * (num_runs // 2)
    if num_runs % 2 == 1:
        leak_modes.append(random.choice(["instant", "ramp"]))
    random.shuffle(leak_modes)
    
    print(f"\n{'='*60}")
    print(f"APARTMENT SIMULATOR - {intensity_label.upper()} LEAK DETECTION")
    print(f"{'='*60}")
    print(f"Building: 50 apartments (aggregated)")
    print(f"Number of runs: {num_runs}")
    print(f"Simulation window: {SIMULATION_MINUTES} minutes")
    print(f"Leak intensity range: {intensity_min}-{intensity_max} L/min (building scale, {intensity_label})")
    print(f"Model tuning: CUSUM k=3.0, h=8.0 (less sensitive, baseline ~3.0 L/min)")
    print(f"Leak modes: 50/50 instant vs ramp (pre-shuffled for fairness)")
    
    results = []
    
    for run_num in range(num_runs):
        # Apartment: larger, less sensitive detector (50-unit building)
        # Leak intensity 5.0-40.0 L/min (building scale)
        # Leak START at 100+ minutes (give detector time to initialize)
        # Leak DURATION 100+ minutes (sustained leak signal)
        leak_at_minute = random.randint(100, SIMULATION_MINUTES - 150)
        leak_intensity = random.uniform(intensity_min, intensity_max)  # Use specified range
        leak_duration = random.randint(100, 200)  # 100-200 min sustained leak
        selected_leak_mode = leak_modes[run_num]  # Use pre-shuffled mode
        
        print(f"\nRun {run_num + 1}/{num_runs}:")
        print(f"  Leak at minute: {leak_at_minute}")
        print(f"  Intensity: {leak_intensity:.2f} L/min")
        print(f"  Duration: {leak_duration} minutes")
        
        result = run_single_simulation(leak_at_minute, leak_intensity, leak_duration, leak_mode=selected_leak_mode)
        results.append(result)
        
        if result["detection_time"] is not None:
            print(f"  ✓ Detected at minute {result['detection_time']} (delay: {result['delay']} min, mode: {result['leak_mode']})")
        else:
            print(f"  ✗ MISSED DETECTION (mode: {result['leak_mode']})")
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Detection Rate:        {metrics['detection_rate']:.2%}")
    print(f"False Alarm Rate:      {metrics['false_alarm_rate']:.2%}")
    print(f"Avg Detection Delay:   {metrics['avg_delay_seconds']:.2f} minutes")
    print(f"Accuracy:              {metrics['accuracy']:.2%}")
    print(f"Precision:             {metrics['precision']:.2%}")
    print(f"Recall:                {metrics['recall']:.2%}")
    print(f"F1-Score:              {metrics['f1_score']:.2f}")
    
    # Save results in intensity-level subdirectory
    intensity_dir = RESULTS_DIR / intensity_label.lower()
    intensity_dir.mkdir(parents=True, exist_ok=True)
    save_results_csv(results, intensity_dir / "results.csv")
    save_metrics_summary(metrics, intensity_dir / "metrics_summary.txt")
    plot_confusion_matrix(metrics, intensity_dir / "confusion_matrix.png")
    plot_accuracy_dashboard(metrics, intensity_dir / "accuracy_dashboard.png")
    
    print(f"\nResults saved to {RESULTS_DIR}")
    
    return results, metrics


if __name__ == "__main__":
    # Run three intensity levels: low, medium, high
    run_experiment(num_runs=10, intensity_min=5.0, intensity_max=15.0, intensity_label="LOW")
    run_experiment(num_runs=10, intensity_min=15.0, intensity_max=25.0, intensity_label="MEDIUM")
    run_experiment(num_runs=10, intensity_min=25.0, intensity_max=40.0, intensity_label="HIGH")
