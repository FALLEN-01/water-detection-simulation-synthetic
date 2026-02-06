@echo off
REM Quick start script for Live Water Detection Dashboard
REM Windows batch file

echo ============================================================
echo Water Leak Detection - Live Simulation Dashboard
echo ============================================================
echo.

REM Check if in correct directory
if not exist "websocket_server.py" (
    echo Error: Please run this script from the live_simulation folder
    pause
    exit /b 1
)

REM Check if dependencies are installed
echo Checking dependencies...
python -c "import flask, flask_socketio" 2>nul
if errorlevel 1 (
    echo.
    echo Dependencies not found. Installing...
    pip install -r requirements_live.txt
    if errorlevel 1 (
        echo.
        echo Error: Failed to install dependencies
        pause
        exit /b 1
    )
)

echo.
echo Dependencies OK
echo.
echo Starting WebSocket server...
echo Dashboard will be available at: http://localhost:5000
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.

python websocket_server.py
