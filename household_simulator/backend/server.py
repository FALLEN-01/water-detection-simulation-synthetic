#!/usr/bin/env python3

import asyncio
from pathlib import Path
import numpy as np
import socketio
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from tensorflow.keras.models import load_model  # type: ignore

from live_simulator import LiveWaterFlowGenerator
from model import HybridWaterAnomalyDetector


# --------------------------------------------------
# Resolve Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
FRONTEND_DIR = BASE_DIR / "frontend"


# --------------------------------------------------
# Socket.IO + FastAPI Setup
# --------------------------------------------------

sio = socketio.AsyncServer(cors_allowed_origins="*", async_mode="asgi")
app = FastAPI()
socket_app = socketio.ASGIApp(sio, app)


# --------------------------------------------------
# Serve Frontend
# --------------------------------------------------

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR / "static"),
    name="static"
)

@app.get("/")
async def serve_index():
    return FileResponse(FRONTEND_DIR / "index.html")


# --------------------------------------------------
# Load Artifacts
# --------------------------------------------------

model = load_model(ARTIFACTS_DIR / "cnn_autoencoder.keras")
flow_mean = np.load(ARTIFACTS_DIR / "flow_mean.npy")
flow_std = np.load(ARTIFACTS_DIR / "flow_std.npy")
ae_threshold = float(np.load(ARTIFACTS_DIR / "ae_threshold.npy"))

generator = LiveWaterFlowGenerator(BASE_DIR / "all_appliances.json")

detector = HybridWaterAnomalyDetector(
    model=model,
    flow_mean=flow_mean,
    flow_std=flow_std,
    ae_threshold=ae_threshold
)


# --------------------------------------------------
# Simulation State
# --------------------------------------------------

simulation_running = False
simulation_speed = 1.0
sim_minutes = 0

# Leak State (handled ONLY here)
leak_active = False
leak_intensity = 0.0
leak_end_minute = None
leak_start_minute = None
leak_mode = "instant"
leak_ramp_minutes = 5


# --------------------------------------------------
# Background Simulation Loop
# --------------------------------------------------

async def simulation_loop():
    global simulation_running, simulation_speed, sim_minutes
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes

    while True:

        if simulation_running:

            # ----------------------------
            # Generate baseline flow
            # ----------------------------
            flow = generator.next()

            # ----------------------------
            # Apply Leak (Server Controlled)
            # ----------------------------
            if leak_active:

                # Auto-expire by duration
                if leak_end_minute is not None and sim_minutes >= leak_end_minute:
                    leak_active = False
                    leak_end_minute = None
                    leak_start_minute = None

                else:

                    if leak_mode == "instant":
                        effective_intensity = leak_intensity

                    elif leak_mode == "ramp":
                        elapsed = sim_minutes - leak_start_minute # type: ignore
                        progress = min(1.0, elapsed / max(1, leak_ramp_minutes))
                        effective_intensity = leak_intensity * progress

                    else:
                        effective_intensity = leak_intensity

                    flow += effective_intensity
                    flow = min(flow, generator.MAX_FLOW_LPM)

            # ----------------------------
            # Run Detector
            # ----------------------------
            result = detector.update(flow)
            result["flow"] = float(flow)

            # ----------------------------
            # Simulated Time
            # ----------------------------
            sim_minutes += 1
            hours = (sim_minutes // 60) % 24
            minutes = sim_minutes % 60
            result["sim_time"] = f"{hours:02d}:{minutes:02d}"

            # ----------------------------
            # Leak Metadata
            # ----------------------------
            result["leak_active"] = bool(leak_active)
            result["leak_mode"] = str(leak_mode)
            result["leak_intensity"] = float(leak_intensity if leak_active else 0.0)

            if leak_active and leak_end_minute:
                result["leak_remaining"] = max(0, leak_end_minute - sim_minutes)
            else:
                result["leak_remaining"] = 0

            # ----------------------------
            # Emit to Frontend
            # ----------------------------
            await sio.emit("data_update", result)

            delay = 1.0 / simulation_speed
            await asyncio.sleep(delay)

        else:
            await asyncio.sleep(0.1)


@app.on_event("startup")
async def startup_event():
    asyncio.create_task(simulation_loop())


# --------------------------------------------------
# Simulation Controls
# --------------------------------------------------

@sio.event
async def start_simulation(sid):
    global simulation_running
    simulation_running = True
    await sio.emit("simulation_state", {"state": "running"})


@sio.event
async def pause_simulation(sid):
    global simulation_running
    simulation_running = False
    await sio.emit("simulation_state", {"state": "paused"})


@sio.event
async def stop_simulation(sid):
    global simulation_running, sim_minutes
    global leak_active, leak_end_minute, leak_start_minute

    simulation_running = False
    sim_minutes = 0

    # Reset leak state
    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    # Reset generator
    generator.current_day = 0
    generator.current_minute = 0
    generator._generate_new_day()

    # Reset detector
    detector.reset()

    await sio.emit("simulation_state", {"state": "stopped"})


@sio.event
async def set_speed(sid, data):
    global simulation_speed
    try:
        speed = float(data)
        simulation_speed = max(1.0, min(speed, 10.0))
        await sio.emit("speed_update", {"speed": simulation_speed})
    except Exception:
        pass


# --------------------------------------------------
# Leak Controls
# --------------------------------------------------

@sio.event
async def inject_leak(sid, data):
    global leak_active, leak_intensity
    global leak_end_minute, leak_start_minute
    global leak_mode, leak_ramp_minutes
    global sim_minutes

    try:
        intensity = float(data.get("intensity", 0.5))
        duration = int(data.get("duration", 60))
        mode = data.get("mode", "instant")
        ramp_minutes = int(data.get("ramp_minutes", 5))

        intensity = max(0.1, min(intensity, 2.0))
        duration = max(1, duration)
        ramp_minutes = max(1, ramp_minutes)

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
    global leak_active, leak_end_minute, leak_start_minute

    leak_active = False
    leak_end_minute = None
    leak_start_minute = None

    await sio.emit("leak_status", {"active": False})


# --------------------------------------------------
# Run Server
# --------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(socket_app, host="0.0.0.0", port=8000, log_level="info")