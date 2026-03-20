## Household Water Leak Detection Simulator

Real-time, AI-powered water leak detection for a **single household**. Generates
minute-by-minute synthetic household flow, runs a hybrid detector (CUSUM +
Isolation Forest), and streams results to a browser dashboard.

## Quick start (Docker)

```bash
cd household_simulator
docker-compose up --build
```

Then open `http://localhost:8000`.

## Quick start (local Python)

```bash
cd household_simulator
pip install -r requirements.txt
py backend/server.py
```

Then open `http://localhost:8000`.

## Project structure

```
household_simulator/
├── backend/
│   ├── server.py          # FastAPI + Socket.IO server (streams to frontend)
│   ├── live_simulator.py  # Minute-by-minute flow generator
│   ├── model.py           # HybridWaterAnomalyDetector (CUSUM + IF)
│   └── isolation_forest.py
├── frontend/              # Dashboard UI (index.html + static/)
├── artifacts/             # all_appliances.json + trained IF/scaler + calibration JSON
├── visualization/         # Offline plots / sanity checks
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## Notes

- **Artifacts**: `backend/server.py` expects models + calibration under `artifacts/`.
- **Leak injection**: Use the dashboard controls to inject instant or ramp leaks.

