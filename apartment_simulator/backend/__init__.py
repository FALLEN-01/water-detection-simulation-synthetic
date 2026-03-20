"""
Apartment simulator backend package.

This package contains the runtime backend that powers the apartment-building
simulation demo:
- `server.py`: FastAPI + Socket.IO server that streams results to the dashboard
- `live_simulator.py`: synthetic building-aggregate flow generator
- `model.py`: hybrid anomaly detector (CUSUM + Isolation Forest)
"""
