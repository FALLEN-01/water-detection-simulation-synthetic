/**
 * Water Leak Detection Dashboard - Frontend Application
 * Real-time data visualization and control
 */

// ===========================
// Global State
// ===========================

let socket = null;
let charts = {};
let simulationStartTime = null;
let timerInterval = null;

const DATA_BUFFER_SIZE = 60; // Keep last 60 seconds
const dataBuffers = {
    flowRate: [],
    turbidity: [],
    anomaly: []
};

// ===========================
// Initialization
// ===========================

document.addEventListener('DOMContentLoaded', () => {
    console.log('Dashboard initializing...');
    initializeWebSocket();
    initializeCharts();
});

// ===========================
// WebSocket Connection
// ===========================

function initializeWebSocket() {
    socket = io.connect(window.location.origin);

    socket.on('connect', () => {
        console.log('WebSocket connected');
        updateConnectionStatus(true);
    });

    socket.on('disconnect', () => {
        console.log('WebSocket disconnected');
        updateConnectionStatus(false);
    });

    socket.on('connection_response', (data) => {
        console.log('Connection response:', data);
    });

    socket.on('sensor_data', (data) => {
        handleSensorData(data);
    });

    socket.on('leak_alert', (data) => {
        showLeakAlert(data);
    });

    socket.on('simulation_status', (data) => {
        handleSimulationStatus(data);
    });
}

function updateConnectionStatus(connected) {
    const statusElement = document.getElementById('connectionStatus');
    const statusText = statusElement.querySelector('.status-text');

    if (connected) {
        statusElement.classList.remove('disconnected');
        statusElement.classList.add('connected');
        statusText.textContent = 'Connected';
    } else {
        statusElement.classList.remove('connected');
        statusElement.classList.add('disconnected');
        statusText.textContent = 'Disconnected';
    }
}

// ===========================
// Chart Initialization
// ===========================

function initializeCharts() {
    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: {
            duration: 300
        },
        plugins: {
            legend: {
                display: false
            }
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'second',
                    displayFormats: {
                        second: 'HH:mm:ss'
                    }
                },
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            },
            y: {
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8'
                }
            }
        }
    };

    // Flow Rate Chart
    const flowCtx = document.getElementById('flowRateChart').getContext('2d');
    charts.flowRate = new Chart(flowCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Flow Rate',
                data: [],
                borderColor: '#00d4ff',
                backgroundColor: 'rgba(0, 212, 255, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    min: 0,
                    max: 15,
                    title: {
                        display: true,
                        text: 'L/min',
                        color: '#94a3b8'
                    }
                }
            }
        }
    });

    // Turbidity Chart
    const turbidityCtx = document.getElementById('turbidityChart').getContext('2d');
    charts.turbidity = new Chart(turbidityCtx, {
        type: 'line',
        data: {
            datasets: [{
                label: 'Turbidity',
                data: [],
                borderColor: '#7c3aed',
                backgroundColor: 'rgba(124, 58, 237, 0.1)',
                borderWidth: 2,
                fill: true,
                tension: 0.4,
                pointRadius: 0
            }]
        },
        options: {
            ...commonOptions,
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    min: 0,
                    max: 5,
                    title: {
                        display: true,
                        text: 'NTU',
                        color: '#94a3b8'
                    }
                }
            }
        }
    });

    // Anomaly Detection Chart
    const anomalyCtx = document.getElementById('anomalyChart').getContext('2d');
    charts.anomaly = new Chart(anomalyCtx, {
        type: 'scatter',
        data: {
            datasets: [
                {
                    label: 'Normal',
                    data: [],
                    backgroundColor: '#00d4ff',
                    borderColor: '#00d4ff',
                    pointRadius: 3,
                    pointHoverRadius: 5
                },
                {
                    label: 'Leak Detected',
                    data: [],
                    backgroundColor: '#ef4444',
                    borderColor: '#ef4444',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
        },
        options: {
            ...commonOptions,
            plugins: {
                legend: {
                    display: true,
                    labels: {
                        color: '#94a3b8'
                    }
                }
            },
            scales: {
                ...commonOptions.scales,
                y: {
                    ...commonOptions.scales.y,
                    min: 0,
                    max: 15,
                    title: {
                        display: true,
                        text: 'Flow Rate (L/min)',
                        color: '#94a3b8'
                    }
                }
            }
        }
    });
}

// ===========================
// Data Handling
// ===========================

function handleSensorData(data) {
    const timestamp = new Date(data.timestamp);
    const sensorData = data.sensor_data;
    const predictions = data.predictions;

    // Update live metrics
    updateMetrics(sensorData, predictions);

    // Add to buffers
    addToBuffer(dataBuffers.flowRate, {
        x: timestamp,
        y: sensorData.flow_rate
    });

    addToBuffer(dataBuffers.turbidity, {
        x: timestamp,
        y: sensorData.turbidity
    });

    // Add to anomaly chart
    const anomalyPoint = {
        x: timestamp,
        y: sensorData.flow_rate
    };

    const isAnomaly = predictions.ensemble === 1;
    if (isAnomaly) {
        addToBuffer(dataBuffers.anomaly, anomalyPoint, 'anomaly');
    } else {
        addToBuffer(dataBuffers.anomaly, anomalyPoint, 'normal');
    }

    // Update charts
    updateCharts();
}

function addToBuffer(buffer, dataPoint, type = null) {
    if (type === 'normal') {
        if (!buffer.normal) buffer.normal = [];
        buffer.normal.push(dataPoint);
        if (buffer.normal.length > DATA_BUFFER_SIZE) {
            buffer.normal.shift();
        }
    } else if (type === 'anomaly') {
        if (!buffer.anomaly) buffer.anomaly = [];
        buffer.anomaly.push(dataPoint);
        if (buffer.anomaly.length > DATA_BUFFER_SIZE) {
            buffer.anomaly.shift();
        }
    } else {
        buffer.push(dataPoint);
        if (buffer.length > DATA_BUFFER_SIZE) {
            buffer.shift();
        }
    }
}

function updateMetrics(sensorData, predictions) {
    // Flow Rate
    document.getElementById('flowRateValue').textContent = sensorData.flow_rate.toFixed(2);

    // Turbidity
    document.getElementById('turbidityValue').textContent = sensorData.turbidity.toFixed(2);

    // Autoencoder
    if (predictions.autoencoder) {
        const autoPred = predictions.autoencoder.prediction === 1 ? 'LEAK' : 'NORMAL';
        const autoElement = document.getElementById('autoencoderPrediction');
        autoElement.textContent = autoPred;
        autoElement.style.color = autoPred === 'LEAK' ? '#ef4444' : '#10b981';

        const errorText = `Error: ${predictions.autoencoder.reconstruction_error.toFixed(6)}`;
        document.getElementById('autoencoderError').textContent = errorText;
    }

    // Isolation Forest
    if (predictions.isolation_forest) {
        const isoPred = predictions.isolation_forest.prediction === 1 ? 'LEAK' : 'NORMAL';
        const isoElement = document.getElementById('isolationPrediction');
        isoElement.textContent = isoPred;
        isoElement.style.color = isoPred === 'LEAK' ? '#ef4444' : '#10b981';

        const scoreText = `Score: ${predictions.isolation_forest.anomaly_score.toFixed(4)}`;
        document.getElementById('isolationScore').textContent = scoreText;
    }

    // Update status badge if leak detected
    if (predictions.ensemble === 1) {
        const statusBadge = document.getElementById('simulationStatus');
        statusBadge.textContent = 'LEAK DETECTED';
        statusBadge.className = 'status-badge status-leak';
    }
}

function updateCharts() {
    // Update flow rate chart
    charts.flowRate.data.datasets[0].data = [...dataBuffers.flowRate];
    charts.flowRate.update('none');

    // Update turbidity chart
    charts.turbidity.data.datasets[0].data = [...dataBuffers.turbidity];
    charts.turbidity.update('none');

    // Update anomaly chart
    if (dataBuffers.anomaly.normal) {
        charts.anomaly.data.datasets[0].data = [...dataBuffers.anomaly.normal];
    }
    if (dataBuffers.anomaly.anomaly) {
        charts.anomaly.data.datasets[1].data = [...dataBuffers.anomaly.anomaly];
    }
    charts.anomaly.update('none');
}

// ===========================
// Alert Handling
// ===========================

function showLeakAlert(data) {
    const alertBanner = document.getElementById('alertBanner');
    const alertDetails = document.getElementById('alertDetails');

    const time = new Date(data.timestamp).toLocaleTimeString();
    const details = `Flow: ${data.flow_rate.toFixed(2)} L/min | Confidence: ${(data.confidence * 100).toFixed(0)}% | Time: ${time}`;

    alertDetails.textContent = details;
    alertBanner.classList.remove('hidden');

    // Auto-hide after 5 seconds
    setTimeout(() => {
        alertBanner.classList.add('hidden');
    }, 5000);
}

function closeAlert() {
    document.getElementById('alertBanner').classList.add('hidden');
}

// ===========================
// Simulation Control
// ===========================

function startSimulation() {
    socket.emit('start_simulation');

    // Update UI
    document.getElementById('startBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = false;
    document.getElementById('stopBtn').disabled = false;

    // Start timer
    simulationStartTime = Date.now();
    timerInterval = setInterval(updateTimer, 1000);

    // Clear buffers
    dataBuffers.flowRate = [];
    dataBuffers.turbidity = [];
    dataBuffers.anomaly = { normal: [], anomaly: [] };

    // Update status
    const statusBadge = document.getElementById('simulationStatus');
    statusBadge.textContent = 'RUNNING';
    statusBadge.className = 'status-badge status-running';
}

function pauseSimulation() {
    const pauseBtn = document.getElementById('pauseBtn');

    if (pauseBtn.textContent.includes('Pause')) {
        socket.emit('pause_simulation');
        pauseBtn.innerHTML = '<span class="btn-icon">▶</span> Resume';

        const statusBadge = document.getElementById('simulationStatus');
        statusBadge.textContent = 'PAUSED';
        statusBadge.className = 'status-badge status-paused';
    } else {
        socket.emit('resume_simulation');
        pauseBtn.innerHTML = '<span class="btn-icon">⏸</span> Pause';

        const statusBadge = document.getElementById('simulationStatus');
        statusBadge.textContent = 'RUNNING';
        statusBadge.className = 'status-badge status-running';
    }
}

function stopSimulation() {
    socket.emit('stop_simulation');

    // Update UI
    document.getElementById('startBtn').disabled = false;
    document.getElementById('pauseBtn').disabled = true;
    document.getElementById('pauseBtn').innerHTML = '<span class="btn-icon">⏸</span> Pause';
    document.getElementById('stopBtn').disabled = true;

    // Stop timer
    clearInterval(timerInterval);
    document.getElementById('simulationTime').textContent = '00:00:00';

    // Update status
    const statusBadge = document.getElementById('simulationStatus');
    statusBadge.textContent = 'IDLE';
    statusBadge.className = 'status-badge status-idle';
}

function handleSimulationStatus(data) {
    console.log('Simulation status:', data.status);
}

function updateTimer() {
    if (simulationStartTime) {
        const elapsed = Date.now() - simulationStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);

        const timeString = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        document.getElementById('simulationTime').textContent = timeString;
    }
}
