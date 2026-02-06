FROM tensorflow/tensorflow:2.13.0

# Disable Python output buffering for Docker logs
ENV PYTHONUNBUFFERED=1
ENV TF_CPP_MIN_LOG_LEVEL=3
ENV TF_ENABLE_ONEDNN_OPTS=0

WORKDIR /app

# Upgrade pip and install dependencies
RUN pip install --upgrade pip && \
    pip install pandas==2.0.3 scikit-learn==1.3.0 matplotlib==3.7.2

# Install live simulation specific dependencies
COPY requirements_live.txt .
RUN pip install -r requirements_live.txt

# Copy application files
COPY *.py .
COPY config.json .
COPY static/ ./static/
# Note: priors_india is mounted via volume in docker-compose.yml

# Expose port for WebSocket server
EXPOSE 5000

CMD ["python", "-u", "websocket_server.py"]

