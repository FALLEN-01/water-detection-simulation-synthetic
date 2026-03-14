"""
Real-Time Apartment Building Water Usage Simulation and Anomaly Detection Server

DESCRIPTION:
    Runs a FastAPI + Socket.IO server for real-time 50-apartment building water flow
    simulation and anomaly detection. Streams synthetic building-aggregate water usage data
    to a web frontend, supports leak injection, simulation controls, and exposes anomaly
    detection results in real time.

KEY FEATURES:
    - Live simulation of building water flow (50 apartments aggregated, minute-by-minute)
    - Real-time anomaly/leak detection using HybridWaterAnomalyDetector
    - Leak injection (instant/ramp), simulation speed control, pause/resume
    - WebSocket (Socket.IO) streaming to frontend dashboard
    - Serves static frontend assets and index.html

DEPENDENCIES:
    - FastAPI, socketio, uvicorn: Web server and real-time communication
    - numpy, pickle, json: Data/model loading
    - live_simulator, model: Simulation and detection logic
"""

#!/usr/bin/env python3

import asyncio
import json
import pickle
from collections import deque
from pathlib import Path

import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from live_simulator import LiveApartmentBuildingDataGenerator
from model import HybridWaterAnomalyDetector


BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"
MODELS_DIR = Path("/models")  # Pre-trained models location

# ──────────────────────────────────────────────────────────────────────────────
# Socket.IO & FastAPI Setup
# ──────────────────────────────────────────────────────────────────────────────

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static",
)


@app.get("/")
async def serve_index():
    """Serve the main frontend HTML page."""
    return FileResponse(FRONTEND_DIR / "index.html")


# ──────────────────────────────────────────────────────────────────────────────
# Model & Generator Loading
# ──────────────────────────────────────────────────────────────────────────────

def load_pickle(path):
    """Load a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    """Load a JSON file."""
    with open(path) as f:
        return json.load(f)


def find_model_path(name, alternates=None):
    """Find model in either artifacts or /models directory."""
    # Try artifacts first (local)
    artifact_path = ARTIFACTS_DIR / name
    if artifact_path.exists():
        print(f"Found {name} in artifacts/")
        return artifact_path

    # Try alternates in artifacts
    if alternates:
        for alt_name in alternates:
            alt_path = ARTIFACTS_DIR / alt_name
            if alt_path.exists():
                print(f"Found {alt_name} (alternate) in artifacts/")
                return alt_path

    # Try /models (shared)
    model_path = MODELS_DIR / name
    if model_path.exists():
        print(f"Found {name} in /models/")
        return model_path

    # Try alternates in /models
    if alternates:
        for alt_name in alternates:
            alt_path = MODELS_DIR / alt_name
            if alt_path.exists():
                print(f"Found {alt_name} (alternate) in /models/")
                return alt_path

    raise FileNotFoundError(f"{name} not found in artifacts/ or /models/")


# Load models and configuration
print("Loading models...")
# Prioritize building-scale models (trained on 50-apartment aggregate data)
try:
    if_model = load_pickle(ARTIFACTS_DIR / "isolation_forest_building.pkl")
    print("Using building-scale Isolation Forest model")
except FileNotFoundError:
    print("Building-scale model not found, falling back to household model")
    if_model = load_pickle(find_model_path("if_model.pkl", ["isolation_forest_model.pkl"]))

try:
    if_scaler = load_pickle(ARTIFACTS_DIR / "scaler_building.pkl")
    print("Using building-scale scaler")
except FileNotFoundError:
    print("Building-scale scaler not found, falling back to household scaler")
    if_scaler = load_pickle(find_model_path("if_scaler.pkl"))

# Try building calibration first
try:
    cal = load_json(ARTIFACTS_DIR / "calibration_building.json")
    print("Using building-scale calibration")
except FileNotFoundError:
    cal = load_json(ARTIFACTS_DIR / "if_calibration.json")
    print("Using household calibration")

WINDOW_MINUTES = cal["window_minutes"]

print(f"Models loaded. Window size: {WINDOW_MINUTES} minutes")

# Create data generator and detector
print("Initializing apartment building data generator...")
generator = LiveApartmentBuildingDataGenerator(
    ARTIFACTS_DIR / "all_appliances.json",
    num_apartments=50,
    seed=42,
)

print("Initializing hybrid anomaly detector...")
# v2 calibration fields (backwards-compatible: falls back to v1 defaults)
detector = HybridWaterAnomalyDetector(
    if_model=if_model,
    if_scaler=if_scaler,
    cusum_k=cal.get("cusum_k", 0.3),
    cusum_h=cal.get("cusum_h", 15.0),
    noise_floor=cal.get("noise_floor", 0.2),
    if_threshold=cal.get("if_threshold", -0.05),
    if_score_scale=cal.get("if_score_scale", 0.1),
    appliance_flow_thresh=cal.get("appliance_flow_thresh", 8.0),
    # v2: baseline stats for baseline_elev feature
    baseline_inter_mean_median=cal.get("baseline_inter_mean_median", 1.5),
    baseline_inter_mean_std=cal.get("baseline_inter_mean_std", 0.8),
    # v2: updated fusion weights
    w2=cal.get("w_cusum", 0.35),
    w3=cal.get("w_if", 0.65),
    decision_threshold=cal.get("decision_threshold", 0.55),
    persistence_windows=cal.get("persistence_windows", 2),
)

print("Setup complete!")

# ──────────────────────────────────────────────────────────────────────────────
# Global State
# ──────────────────────────────────────────────────────────────────────────────

simulation_running = False
simulation_speed = 1.0
sim_minutes = 0

window_buffer: deque = deque(maxlen=WINDOW_MINUTES)

last_result = {
    "anomaly": False,
    "final_score": 0.0,
    "level2": {"triggered": False, "score": 0.0},
    "level3": {"triggered": False, "score": 0.0, "reconstruction_error": 0.0},
}

leak_active = False
leak_intensity = 0.0
leak_end_minute = None
leak_start_minute = None
leak_mode = "instant"
leak_ramp_minutes = 5


# ──────────────────────────────────────────────────────────────────────────────
# Main Simulation Loop
# ──────────────────────────────────────────────────────────────────────────────

async def simulation_loop():
    """
    Main background loop for live simulation and anomaly detection.
    Streams flow and detection results to frontend via Socket.IO.
    """

    global simulation_running, simulation_speed, sim_minutes
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes
    global last_result

    while True:

        if simulation_running:

            # ─────────────────────────────────────────────────────
            # Generate Flow
            # ─────────────────────────────────────────────────────

            flow = generator.next()

            # ─────────────────────────────────────────────────────
            # Apply Leak (via server-side injection)
            # ─────────────────────────────────────────────────────

            if leak_active:

                if leak_end_minute is not None and sim_minutes >= leak_end_minute:

                    leak_active = False
                    leak_end_minute = None
                    leak_start_minute = None

                else:

                    if leak_mode == "instant":

                        effective_intensity = leak_intensity

                    elif leak_mode == "ramp":

                        elapsed = sim_minutes - leak_start_minute
                        progress = min(1.0, elapsed / max(1, leak_ramp_minutes))

                        effective_intensity = leak_intensity * progress

                    else:

                        effective_intensity = leak_intensity

                    flow += effective_intensity
                    flow = min(flow, generator.MAX_FLOW_LPM)

            # ─────────────────────────────────────────────────────
            # Buffer Window (20 minutes)
            # ─────────────────────────────────────────────────────

            window_buffer.append(float(flow))

            if len(window_buffer) == WINDOW_MINUTES:

                last_result = detector.update(list(window_buffer))

            result = dict(last_result)
            result["flow"] = float(flow)

            # ─────────────────────────────────────────────────────
            # Simulated Time
            # ─────────────────────────────────────────────────────

            sim_minutes += 1

            result["sim_time"] = f"{(sim_minutes // 60) % 24:02d}:{sim_minutes % 60:02d}"
            result["sim_minutes"] = sim_minutes

            # ─────────────────────────────────────────────────────
            # Leak Metadata
            # ─────────────────────────────────────────────────────

            result["leak_active"] = bool(leak_active)
            result["leak_mode"] = str(leak_mode)
            result["leak_intensity"] = float(leak_intensity if leak_active else 0.0)

            result["leak_remaining"] = (
                max(0, leak_end_minute - sim_minutes)
                if leak_active and leak_end_minute
                else 0
            )

            # ─────────────────────────────────────────────────────
            # Emit to Frontend
            # ─────────────────────────────────────────────────────

            await sio.emit("data_update", result)

            await asyncio.sleep(1.0 / simulation_speed)

        else:

            await asyncio.sleep(0.1)


# ──────────────────────────────────────────────────────────────────────────────
# Server Start Event
# ──────────────────────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Start the simulation loop on server startup."""
    asyncio.create_task(simulation_loop())


# ──────────────────────────────────────────────────────────────────────────────
# Socket.IO Event Handlers
# ──────────────────────────────────────────────────────────────────────────────

@sio.event
async def start_simulation(sid):
    """Start/resume the simulation."""
    global simulation_running
    simulation_running = True
    await sio.emit("simulation_state", {"state": "running"})


@sio.event
async def pause_simulation(sid):
    """Pause the simulation."""
    global simulation_running
    simulation_running = False
    await sio.emit("simulation_state", {"state": "paused"})


@sio.event
async def stop_simulation(sid):
    """Stop and reset the simulation and leak state."""
    global simulation_running, sim_minutes
    global leak_active, leak_end_minute, leak_start_minute

    simulation_running = False
    sim_minutes = 0

    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    window_buffer.clear()

    generator.reset()
    detector.reset()

    await sio.emit("simulation_state", {"state": "stopped"})


@sio.event
async def set_speed(sid, data):
    """Set the simulation speed (1x to 10x)."""
    global simulation_speed

    try:

        simulation_speed = max(1.0, min(float(data), 10.0))

        await sio.emit("speed_update", {"speed": simulation_speed})

    except Exception:

        pass


@sio.event
async def inject_leak(sid, data):
    """Inject a leak with specified intensity, duration, and mode (instant/ramp)."""
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes

    try:

        intensity = max(0.1, min(float(data.get("intensity", 0.5)), 20.0))
        duration = max(1, int(data.get("duration", 60)))
        mode = data.get("mode", "instant")
        ramp_minutes = max(1, int(data.get("ramp_minutes", 5)))

        leak_active = True
        leak_intensity = intensity
        leak_end_minute = sim_minutes + duration
        leak_start_minute = sim_minutes
        leak_mode = mode if mode in ["instant", "ramp"] else "instant"
        leak_ramp_minutes = ramp_minutes

        await sio.emit("leak_status", {
            "active": True,
            "mode": leak_mode,
            "intensity": leak_intensity
        })

    except Exception:

        pass


@sio.event
async def stop_leak(sid):
    """Stop any active leak in the simulation."""
    global leak_active, leak_end_minute, leak_start_minute

    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    await sio.emit("leak_status", {"active": False})


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    uvicorn.run(socket_app, host="0.0.0.0", port=5000, log_level="info")
