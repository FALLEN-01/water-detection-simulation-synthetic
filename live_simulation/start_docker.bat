@echo off
REM Quick start script for Docker deployment (Windows)

echo ============================================================
echo Water Leak Detection - Live Simulation Dashboard (Docker)
echo ============================================================
echo.

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo Error: Docker is not running
    echo Please start Docker Desktop and try again
    pause
    exit /b 1
)

echo Docker is running
echo.
echo Building and starting the dashboard...
echo This may take a few minutes on first run...
echo.
echo Dashboard will be available at: http://localhost:5000
echo Press Ctrl+C to stop the container
echo ============================================================
echo.

docker-compose up --build

echo.
echo Dashboard stopped
pause
