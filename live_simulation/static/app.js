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

const DATA_BUFFER_SIZE = 600; // Keep last 10 minutes (600 seconds)
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
        console.log('Leak alert:', data);
        // Alert banner removed - predictions shown in control panel
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
    // Plugin to display current value on chart
    const valueDisplayPlugin = {
        id: 'valueDisplay',
        afterDatasetsDraw(chart) {
            const { ctx, chartArea: { top, right }, scales: { x, y } } = chart;
            const dataset = chart.data.datasets[0];
            const data = dataset.data;

            if (data.length > 0) {
                const lastPoint = data[data.length - 1];
                const value = lastPoint.y;
                const label = dataset.label;

                // Determine unit based on label
                let unit = '';
                if (label === 'Flow Rate') {
                    unit = ' L/min';
                } else if (label === 'Turbidity') {
                    unit = ' NTU';
                }

                ctx.save();
                ctx.font = 'bold 20px Inter';
                ctx.fillStyle = dataset.borderColor;
                ctx.textAlign = 'right';
                ctx.fillText(`${value.toFixed(2)}${unit}`, right - 10, top + 25);
                ctx.restore();
            }
        }
    };

    const commonOptions = {
        responsive: true,
        maintainAspectRatio: false,
        animation: false,  // Disable animation for smoother scrolling
        plugins: {
            legend: {
                display: false
            },
            valueDisplay: true
        },
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute',
                    displayFormats: {
                        minute: 'HH:mm',
                        second: 'HH:mm:ss'
                    }
                },
                grid: {
                    color: 'rgba(148, 163, 184, 0.1)'
                },
                ticks: {
                    color: '#94a3b8',
                    maxRotation: 0,
                    autoSkip: true,
                    maxTicksLimit: 10
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
        },
        plugins: [valueDisplayPlugin]
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
        },
        plugins: [valueDisplayPlugin]
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
                    backgroundColor: 'rgba(0, 212, 255, 0.6)',
                    borderColor: '#00d4ff',
                    pointRadius: 3,
                    pointHoverRadius: 5
                },
                {
                    label: 'Anomaly',
                    data: [],
                    backgroundColor: 'rgba(239, 68, 68, 0.8)',
                    borderColor: '#ef4444',
                    pointRadius: 5,
                    pointHoverRadius: 7
                }
            ]
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

function updateCharts() {
    // Get the latest timestamp from data
    let maxTime = null;
    let minTime = null;

    if (dataBuffers.flowRate.length > 0) {
        const timestamps = dataBuffers.flowRate.map(d => d.x.getTime());
        maxTime = new Date(Math.max(...timestamps));
        minTime = new Date(maxTime.getTime() - (10 * 60 * 1000)); // 10 minutes window
    }

    // Update Flow Rate Chart
    charts.flowRate.data.datasets[0].data = dataBuffers.flowRate;
    if (minTime && maxTime) {
        charts.flowRate.options.scales.x.min = minTime.getTime();
        charts.flowRate.options.scales.x.max = maxTime.getTime();
    }
    charts.flowRate.update('none');

    // Update Turbidity Chart
    charts.turbidity.data.datasets[0].data = dataBuffers.turbidity;
    if (minTime && maxTime) {
        charts.turbidity.options.scales.x.min = minTime.getTime();
        charts.turbidity.options.scales.x.max = maxTime.getTime();
    }
    charts.turbidity.update('none');

    // Update Anomaly Chart
    const normalPoints = dataBuffers.anomaly.normal || [];
    const anomalyPoints = dataBuffers.anomaly.anomaly || [];

    charts.anomaly.data.datasets[0].data = normalPoints;
    charts.anomaly.data.datasets[1].data = anomalyPoints;
    if (minTime && maxTime) {
        charts.anomaly.options.scales.x.min = minTime.getTime();
        charts.anomaly.options.scales.x.max = maxTime.getTime();
    }
    charts.anomaly.update('none');
}

function updateMetrics(sensorData, predictions) {
    // These elements may not exist anymore since we removed the sidebar
    const flowElement = document.getElementById('flowRateValue');
    const turbidityElement = document.getElementById('turbidityValue');

    if (flowElement) flowElement.textContent = sensorData.flow_rate.toFixed(2);
    if (turbidityElement) turbidityElement.textContent = sensorData.turbidity.toFixed(2);

    // Autoencoder
    const autoElement = document.getElementById('autoencoderPrediction');

    if (autoElement) {
        if (predictions.autoencoder) {
            const autoPred = predictions.autoencoder.prediction === 1 ? 'LEAK' : 'NORMAL';
            autoElement.textContent = autoPred;
            autoElement.style.color = autoPred === 'LEAK' ? '#ef4444' : '#10b981';
        } else {
            autoElement.textContent = '-';
            autoElement.style.color = '#64748b';
        }
    }

    // Isolation Forest
    const isoElement = document.getElementById('isolationPrediction');

    if (isoElement) {
        if (predictions.isolation_forest) {
            const isoPred = predictions.isolation_forest.prediction === 1 ? 'LEAK' : 'NORMAL';
            isoElement.textContent = isoPred;
            isoElement.style.color = isoPred === 'LEAK' ? '#ef4444' : '#10b981';
        } else {
            isoElement.textContent = '-';
            isoElement.style.color = '#64748b';
        }
    }
}

// ===========================
// Alert Handling
// ===========================

function showLeakAlert(data) {
    const banner = document.getElementById('alertBanner');
    const details = document.getElementById('alertDetails');

    // Check if elements exist (they may have been removed)
    if (!banner || !details) {
        return;
    }

    details.textContent = `Detected at ${new Date(data.timestamp).toLocaleTimeString()} | Confidence: ${(data.confidence * 100).toFixed(1)}%`;

    banner.classList.remove('hidden');

    // Auto-hide after 10 seconds
    setTimeout(() => {
        banner.classList.add('hidden');
    }, 10000);
}

function closeAlert() {
    document.getElementById('alertBanner').classList.add('hidden');
}

// ===========================
// Simulation Control
// ===========================

function startSimulation() {
    socket.emit('start_simulation');
    simulationStartTime = Date.now();
    startTimer();

    document.getElementById('startBtn').disabled = true;
    document.getElementById('pauseBtn').disabled = false;
    document.getElementById('stopBtn').disabled = false;
}

function pauseSimulation() {
    socket.emit('pause_simulation');
    stopTimer();

    document.getElementById('pauseBtn').textContent = '▶ Resume';
    document.getElementById('pauseBtn').onclick = resumeSimulation;
}

function resumeSimulation() {
    socket.emit('resume_simulation');
    startTimer();

    document.getElementById('pauseBtn').textContent = '⏸ Pause';
    document.getElementById('pauseBtn').onclick = pauseSimulation;
}

function stopSimulation() {
    socket.emit('stop_simulation');
    stopTimer();
    simulationStartTime = null;

    document.getElementById('startBtn').disabled = false;
    document.getElementById('pauseBtn').disabled = true;
    document.getElementById('stopBtn').disabled = true;
    document.getElementById('simulationTime').textContent = '00:00:00';
}

function handleSimulationStatus(data) {
    const statusBadge = document.getElementById('simulationStatus');
    statusBadge.className = 'status-badge';

    switch (data.status) {
        case 'running':
            statusBadge.classList.add('status-running');
            statusBadge.textContent = 'RUNNING';
            break;
        case 'paused':
            statusBadge.classList.add('status-paused');
            statusBadge.textContent = 'PAUSED';
            break;
        case 'stopped':
            statusBadge.classList.add('status-idle');
            statusBadge.textContent = 'IDLE';
            break;
    }
}

// ===========================
// Timer
// ===========================

function startTimer() {
    if (timerInterval) return;

    timerInterval = setInterval(() => {
        if (!simulationStartTime) return;

        const elapsed = Date.now() - simulationStartTime;
        const hours = Math.floor(elapsed / 3600000);
        const minutes = Math.floor((elapsed % 3600000) / 60000);
        const seconds = Math.floor((elapsed % 60000) / 1000);

        const timeString = `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(seconds).padStart(2, '0')}`;
        document.getElementById('simulationTime').textContent = timeString;
    }, 1000);
}

function stopTimer() {
    if (timerInterval) {
        clearInterval(timerInterval);
        timerInterval = null;
    }
}
