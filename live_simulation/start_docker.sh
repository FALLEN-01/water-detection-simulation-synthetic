#!/bin/bash
# Quick start script for Docker deployment
# Linux/Mac bash script

echo "============================================================"
echo "Water Leak Detection - Live Simulation Dashboard (Docker)"
echo "============================================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Please install Docker from: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "Error: docker-compose is not installed"
    echo "Please install docker-compose from: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "Docker and docker-compose found ✓"
echo ""
echo "Building and starting the dashboard..."
echo "This may take a few minutes on first run..."
echo ""

# Build and start
docker-compose up --build

echo ""
echo "Dashboard stopped"
