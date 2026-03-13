/**
 * AquaGuard — Live Dashboard JS
 * Handles simulation, charts, stat cards, speed, animations
 */

let charts = {};
let simStart = null;
let timerInt = null;
let lastLeak = false;
let leakStartTime = null;

const MAX_PTS = 600; // Increased buffer for smoother long-term visualization
const bufs = { flow: [], recon: [], stats: [], normal: [], anomaly: [] };
let socket = null;

document.addEventListener('DOMContentLoaded', () => {
  initBg();
  initCharts();
  initSimulator();
});

// =============================================
// Simulator initialization
// =============================================
function initSimulator() {
  // Initialize connection status as offline
  setConn(false);

  // Initialize system status
  setLeakState(false);

  // Set initial time display
  document.getElementById('simTimeDisplay').textContent = '--:-- --/--';

  // Initialize AI badges
  ['predStat', 'predCNN', 'predFusion'].forEach(id => setBadge(id, null, id === 'predFusion'));

  // Initialize leak status
  updateLeakStatus({ active: false });

  // Initialize socket connection to backend
  initSocket();
}

function initSocket() {
  if (typeof io === 'undefined') return console.warn('Socket.IO client not loaded');
  socket = io.connect(window.location.origin);

  socket.on('connect', () => setConn(true));
  socket.on('disconnect', () => setConn(false));

  socket.on('server_status', (d) => {
    // optional server status messages
    console.log('server_status', d);
  });

  socket.on('simulation_state', (d) => {
    const state = d.state || d.status || 'idle';
    const map = { running: 'running', paused: 'paused', stopped: 'stopped', resumed: 'running' };
    const cls = map[state] || 'idle';
    const pill = document.getElementById('simStatusPill');
    pill.className = 'spill spill-' + cls;
    pill.textContent = cls.toUpperCase();

    const btnStart = document.getElementById('btnStart');
    const btnPause = document.getElementById('btnPause');
    const btnStop = document.getElementById('btnStop');

    // toggle buttons based on state
    if (state === 'running') {
      btnStart.disabled = true;
      btnPause.disabled = false;
      btnPause.textContent = 'Pause';
      btnStop.disabled = false;
    } else if (state === 'paused') {
      btnStart.disabled = true;
      btnPause.disabled = false;
      btnPause.textContent = 'Resume'; // Change to Resume when paused
      btnStop.disabled = false;
    } else {
      btnStart.disabled = false;
      btnPause.disabled = true;
      btnPause.textContent = 'Pause'; // Reset to Pause when stopped
      btnStop.disabled = true;
    }
  });

  socket.on('speed_update', (d) => {
    document.getElementById('speedVal').textContent = d.speed || d;
    document.getElementById('speedSlider').value = d.speed || d;
  });

  socket.on('data_update', (msg) => {
    // Backend returns: flow, anomaly, final_score, level2, level3
    // level2 = Isolation Forest, level3 = Autoencoder
    const det = {
      autoencoder: {
        anomaly: (msg.level3 && msg.level3.triggered) || false,
        error: (msg.level3 && msg.level3.reconstruction_error) || 0,
        score: (msg.level3 && msg.level3.score) || 0
      },
      isolation_forest: {
        anomaly: (msg.level2 && msg.level2.triggered) || false,
        confidence: (msg.level2 && msg.level2.score) || 0
      },
      fusion: {
        score: msg.final_score || 0,
        anomaly: msg.anomaly || false
      }
    };

    const payload = {
      timestamp: Date.now(),
      sim_time: msg.sim_time || '--:-- --/--',
      sensor_data: { flow_rate: msg.flow || 0 },
      detection: det,
      leak_active: msg.anomaly || false
    };

    handleData(payload);

    // Update leak status from data_update
    if (msg.leak_active !== undefined) {
      updateLeakStatus({
        active: msg.leak_active,
        mode: msg.leak_mode || 'instant',
        intensity: msg.leak_intensity || 0,
        leak_remaining: msg.leak_remaining || 0
      });
    }
  });

  socket.on('leak_status', (data) => {
    updateLeakStatus(data);
  });
}

// =============================================
// Connection status
// =============================================
function setConn(ok) {
  const el = document.getElementById('connStatus');
  el.className = 'conn-status ' + (ok ? 'online' : 'offline');
  el.querySelector('.conn-label').textContent = ok ? 'ONLINE' : 'OFFLINE';
}

// =============================================
// Controls
// =============================================
function startSimulation() {
  if (socket) {
    socket.emit('start_simulation');
    simStart = Date.now();
    startTimer();
  }
}

function pauseSimulation() {
  if (socket) {
    const btnPause = document.getElementById('btnPause');
    if (btnPause.textContent === 'Pause') {
      socket.emit('pause_simulation');
    } else if (btnPause.textContent === 'Resume') {
      socket.emit('start_simulation'); // Resume by starting again
    }
  }
}

function stopSimulation() {
  if (socket) {
    socket.emit('stop_simulation');
    simStart = null;
    stopTimer();
    document.getElementById('elapsedTime').textContent = '00:00:00';
    // Reset leak status when simulation stops
    updateLeakStatus({ active: false });
  }
}

function onSpeedChange(v) {
  const speed = parseInt(v);
  const limitedSpeed = Math.min(10, Math.max(1, speed));
  if (socket) socket.emit('set_speed', limitedSpeed);
  document.getElementById('speedVal').textContent = limitedSpeed;
  document.getElementById('speedSlider').value = limitedSpeed;
}

// =============================================
// Timer
// =============================================
function startTimer() {
  if (timerInt) return;
  timerInt = setInterval(() => {
    if (!simStart) return;
    const e = Date.now() - simStart;
    const h = String(Math.floor(e / 3600000)).padStart(2, '0');
    const m = String(Math.floor((e % 3600000) / 60000)).padStart(2, '0');
    const s = String(Math.floor((e % 60000) / 1000)).padStart(2, '0');
    document.getElementById('elapsedTime').textContent = `${h}:${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  if (timerInt) {
    clearInterval(timerInt);
    timerInt = null;
  }
}

// =============================================
// Data Handling
// =============================================
function handleData(d) {
  const ts = new Date(d.timestamp);
  const s = d.sensor_data;
  const det = d.detection;
  const isLeak = !!d.leak_active;

  if (d.sim_time) document.getElementById('simTimeDisplay').textContent = d.sim_time;

  // Check for leak state change - MUST check before updating lastLeak
  if (isLeak !== lastLeak) {
    if (isLeak) {
      // New leak detected
      leakStartTime = new Date();
      handleAlert({
        sim_time: d.sim_time,
        fusion_score: det.fusion.score,
        start_time: d.sim_time
      });
    } else {
      // Leak cleared
      leakStartTime = null;
    }

    // Only update lastLeak AFTER handling the transition
    lastLeak = isLeak;
    setLeakState(isLeak);
  }

  // Update stat cards
  const ifScore = (det.isolation_forest.confidence || 0) * 100;
  const aeError = det.autoencoder.error || 0;
  const fusionScore = (det.fusion.score || 0) * 100;

  document.getElementById('valFlow').textContent = s.flow_rate.toFixed(2);
  document.getElementById('valStat').textContent = ifScore.toFixed(0);
  document.getElementById('valCnn').textContent = aeError < 0.001 ? aeError.toExponential(2) : aeError.toFixed(5);
  document.getElementById('valFusion').textContent = fusionScore.toFixed(0);
  document.getElementById('subThresh').textContent = '--'; // Backend handles threshold internally

  // Update progress bars
  bar('barFlow', s.flow_rate / 15 * 100);
  bar('barStat', ifScore);
  bar('barCnn', (det.autoencoder.score || 0) * 100);
  bar('barFusion', fusionScore);

  // Update alert states
  ['sc-flow', 'sc-stat', 'sc-cnn', 'sc-fusion'].forEach(id => {
    document.getElementById(id).classList.toggle('alert-state', isLeak);
  });
  document.querySelectorAll('.chart-card').forEach(el => {
    el.classList.toggle('leak-border', isLeak);
  });

  // Update AI badges
  setBadge('predStat', det.isolation_forest.anomaly ? 1 : 0);
  setBadge('predCNN', det.autoencoder.anomaly ? 1 : 0);
  setBadge('predFusion', det.fusion.anomaly ? 1 : 0, true);

  // Update chart data
  push(bufs.flow, { x: ts, y: s.flow_rate });
  push(bufs.recon, { x: ts, y: aeError });
  push(bufs.stats, { x: ts, y: det.isolation_forest.confidence || 0 });
  const pt = { x: ts, y: s.flow_rate };
  push(det.fusion.anomaly ? bufs.anomaly : bufs.normal, pt);

  updateCharts(d.timestamp);
}

function bar(id, pct) {
  document.getElementById(id).style.width = Math.min(100, Math.max(0, pct)) + '%';
}

function setBadge(id, pred, isEnsemble = false) {
  const el = document.getElementById(id);
  const base = 'ai-badge' + (isEnsemble ? ' ai-badge-ensemble' : '');
  if (pred === null || pred === undefined) {
    el.className = base + ' badge-wait'; el.textContent = '--';
  } else if (pred === 1) {
    el.className = base + ' badge-leak'; el.textContent = 'TRIGGERED';
  } else {
    el.className = base + ' badge-normal'; el.textContent = 'NORMAL';
  }
}

function setLeakState(isLeak) {
  const sys = document.getElementById('systemStatus');
  const text = document.getElementById('systemStatusText');
  sys.className = 'system-status ' + (isLeak ? 'leak' : 'normal');
  text.textContent = isLeak ? 'LEAK DETECTED' : 'NORMAL';
}

// =============================================
// Alert
// =============================================
function handleAlert(d) {
  const box = document.getElementById('alertBox');
  const meta = document.getElementById('alertMeta');
  meta.textContent = `Started: ${d.start_time || d.sim_time} | Fusion Score: ${(d.fusion_score * 100).toFixed(1)}%`;
  box.classList.remove('hidden');
  clearTimeout(box._t);
  box._t = setTimeout(closeAlert, 8000);
}

function closeAlert() {
  document.getElementById('alertBox').classList.add('hidden');
}

// =============================================
// Charts and Background
// =============================================
function initBg() {
  // Initialize background animations if needed
  // This can be extended for background visual effects
}

function initCharts() {
  // Initialize Chart.js instances
  const commonConfig = {
    type: 'line',
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: {
        intersect: false,
        mode: 'index'
      },
      elements: {
        point: {
          radius: 0, // Hide points by default for smoother lines
          hoverRadius: 4,
          hitRadius: 10
        },
        line: {
          tension: 0.2, // Increased curve for even smoother appearance
          borderWidth: 2,
          stepped: false
        }
      },
      scales: {
        x: {
          type: 'time',
          time: {
            unit: 'second',
            displayFormats: {
              second: 'HH:mm:ss'
            },
            tooltipFormat: 'HH:mm:ss'
          },
          ticks: {
            maxTicksLimit: 8, // Limit number of x-axis labels to reduce clutter
            autoSkip: true,
            maxRotation: 0 // Keep labels horizontal for readability
          },
          title: {
            display: true,
            text: 'Time'
          }
        },
        y: {
          beginAtZero: true,
          title: {
            display: true,
            text: 'Flow (L/min)'
          }
        }
      },
      plugins: {
        legend: {
          display: true
        },
        decimation: {
          enabled: true,
          algorithm: 'lttb', // Largest-Triangle-Three-Buckets algorithm for smooth downsampling
          samples: 250 // Downsample to 250 points for smooth rendering
        }
      }
    }
  };

  // Flow Rate Chart (split into normal vs anomaly datasets for clear visual distinction)
  // Note: This creates line discontinuities but provides clear anomaly highlighting
  // For continuous lines with scriptable coloring, consider a single dataset approach
  const flowCtx = document.getElementById('cFlow');
  if (flowCtx) {
    charts.flow = new Chart(flowCtx, {
      ...commonConfig,
      data: {
        datasets: [
          {
            label: 'Normal Flow',
            data: bufs.normal, // Reference buffer directly - no cloning
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.2)',
            pointRadius: 0, // Override default for this dataset
            pointHoverRadius: 4,
            tension: 0.2, // Match global smooth setting
            borderWidth: 2
          },
          {
            label: 'Anomaly',
            data: bufs.anomaly, // Reference buffer directly - no cloning
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.2)',
            pointRadius: 2, // Keep visible points for anomalies
            pointHoverRadius: 6,
            tension: 0.2, // Match global smooth setting
            borderWidth: 2
          }
        ]
      }
    });
  }

  // Autoencoder Reconstruction Error Chart
  const reconCtx = document.getElementById('cRecon');
  if (reconCtx) {
    charts.recon = new Chart(reconCtx, {
      ...commonConfig,
      data: {
        datasets: [
          {
            label: 'Autoencoder MSE',
            data: bufs.recon, // Reference buffer directly - no cloning
            borderColor: 'rgb(153, 102, 255)',
            backgroundColor: 'rgba(153, 102, 255, 0.2)',
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2, // Match global smooth setting
            borderWidth: 2
          }
        ]
      },
      options: {
        ...commonConfig.options,
        scales: {
          ...commonConfig.options.scales,
          y: {
            beginAtZero: true,
            title: {
              display: true,
              text: 'Reconstruction Error (MSE)'
            }
          }
        }
      }
    });
  }

  // Isolation Forest Anomaly Score Chart
  const statsCtx = document.getElementById('cStats');
  if (statsCtx) {
    charts.stats = new Chart(statsCtx, {
      ...commonConfig,
      data: {
        datasets: [
          {
            label: 'Isolation Forest Score',
            data: bufs.stats, // Reference buffer directly - no cloning
            borderColor: 'rgb(255, 159, 64)',
            backgroundColor: 'rgba(255, 159, 64, 0.2)',
            pointRadius: 0,
            pointHoverRadius: 4,
            tension: 0.2, // Match global smooth setting
            borderWidth: 2
          }
        ]
      },
      options: {
        ...commonConfig.options,
        scales: {
          ...commonConfig.options.scales,
          y: {
            beginAtZero: true,
            max: 1,
            title: {
              display: true,
              text: 'Anomaly Score'
            }
          }
        }
      }
    });
  }

  // Anomaly Timeline Chart (wider chart showing combined flow with anomaly indicators)
  const anomalyCtx = document.getElementById('cAnomaly');
  if (anomalyCtx) {
    charts.anomaly = new Chart(anomalyCtx, {
      ...commonConfig,
      data: {
        datasets: [
          {
            label: 'Flow Rate',
            data: bufs.flow, // Reference buffer directly - no cloning
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.1)',
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
            tension: 0.2 // Match global smooth setting
          }
        ]
      }
    });
  }

  console.log('Chart.js instances initialized:', Object.keys(charts));
}

function push(buffer, dataPoint) {
  buffer.push(dataPoint);
  if (buffer.length > MAX_PTS) {
    buffer.shift();
  }
}

function updateCharts(timestamp) {
  // Update Chart.js instances - no need to clone buffers since they're referenced directly

  if (charts.flow) {
    charts.flow.update('none'); // 'none' mode for performance - no buffer cloning needed
  }

  if (charts.recon) {
    charts.recon.update('none');
  }

  if (charts.stats) {
    charts.stats.update('none');
  }

  if (charts.anomaly) {
    charts.anomaly.update('none');
  }
}

// =============================================
// Leak Injection Controls
// =============================================
function injectLeak() {
  if (!socket) return;

  const intensity = parseFloat(document.getElementById('leakIntensity').value);
  const duration = parseInt(document.getElementById('leakDuration').value);
  const mode = document.getElementById('leakMode').value;
  const rampMinutes = parseInt(document.getElementById('leakRamp').value);

  socket.emit('inject_leak', {
    intensity: intensity,
    duration: duration,
    mode: mode,
    ramp_minutes: rampMinutes
  });
}

function stopLeak() {
  if (!socket) return;
  socket.emit('stop_leak');
}

function updateLeakStatus(data) {
  const statusPill = document.getElementById('leakStatusPill');
  const btnInject = document.getElementById('btnInjectLeak');
  const btnStop = document.getElementById('btnStopLeak');
  const remaining = document.getElementById('leakRemaining');

  if (data.active) {
    statusPill.className = 'spill spill-orange';
    statusPill.textContent = data.mode.toUpperCase();
    btnInject.disabled = true;
    btnStop.disabled = false;

    // Update remaining time if provided
    if (data.leak_remaining !== undefined) {
      remaining.textContent = `${data.leak_remaining}min left`;
    } else {
      remaining.textContent = 'ACTIVE';
    }
  } else {
    statusPill.className = 'spill spill-gray';
    statusPill.textContent = 'INACTIVE';
    btnInject.disabled = false;
    btnStop.disabled = true;
    remaining.textContent = '--';
  }
}

// Initialize leak control handlers
document.addEventListener('DOMContentLoaded', () => {
  // Update leak intensity display
  const intensitySlider = document.getElementById('leakIntensity');
  const intensityVal = document.getElementById('leakIntensityVal');
  intensitySlider.addEventListener('input', () => {
    intensityVal.textContent = intensitySlider.value;
  });

  // Update leak duration display
  const durationSlider = document.getElementById('leakDuration');
  const durationVal = document.getElementById('leakDurationVal');
  durationSlider.addEventListener('input', () => {
    durationVal.textContent = durationSlider.value;
  });

  // Update ramp duration display
  const rampSlider = document.getElementById('leakRamp');
  const rampVal = document.getElementById('leakRampVal');
  rampSlider.addEventListener('input', () => {
    rampVal.textContent = rampSlider.value;
  });

  // Show/hide ramp controls based on mode
  const modeSelect = document.getElementById('leakMode');
  const rampRow = document.getElementById('rampRow');
  modeSelect.addEventListener('change', () => {
    if (modeSelect.value === 'ramp') {
      rampRow.style.display = 'flex';
    } else {
      rampRow.style.display = 'none';
    }
  });
});
