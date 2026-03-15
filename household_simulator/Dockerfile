# Use TensorFlow image with Keras pre-built
FROM tensorflow/tensorflow:latest

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies (TensorFlow/Keras already included in image)
RUN pip install --upgrade --ignore-installed --no-cache-dir -r requirements.txt

# Copy application files.
COPY artifacts/ ./artifacts/
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# Expose port
EXPOSE 8000

# Set the working directory to backend for running the server
WORKDIR /app/backend

# Run the application
CMD ["python", "server.py"]
