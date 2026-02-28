/**
 * Water Leak Detection — Live Dashboard JS
 * Handles WebSocket, charts, stat cards, speed control, animations
 */

// =============================================
// State
// =============================================
let socket = null;
let charts = {};
let simStartTime  = null;
let timerInterval = null;
let maxReconError = 0.001;   // auto-scale reconstruction error chart
let lastLeakState = false;

const MAX_POINTS = 300;   // points to keep in each chart buffer

const bufs = {
  flow:     [],
  turb:     [],
  recon:    [],
  normal:   [],
  anomaly:  [],
};

// =============================================
// Init
// =============================================
document.addEventListener('DOMContentLoaded', () => {
  initBg();
  initCharts();
  initSocket();
});

// =============================================
// Particle Background
// =============================================
function initBg() {
  const canvas = document.getElementById('bgCanvas');
  const ctx    = canvas.getContext('2d');
  let W, H, particles = [];

  function resize() {
    W = canvas.width  = window.innerWidth;
    H = canvas.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  for (let i = 0; i < 55; i++) {
    particles.push({
      x: Math.random() * W, y: Math.random() * H,
      r: Math.random() * 1.5 + 0.3,
      dx: (Math.random() - 0.5) * 0.25,
      dy: (Math.random() - 0.5) * 0.25,
      alpha: Math.random() * 0.35 + 0.05,
    });
  }

  function draw() {
    ctx.clearRect(0, 0, W, H);
    particles.forEach(p => {
      p.x += p.dx; p.y += p.dy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(0,212,255,${p.alpha})`;
      ctx.fill();
    });
    requestAnimationFrame(draw);
  }
  draw();
}

// =============================================
// Charts
// =============================================
function chartDefaults(yLabel, color, yMin = null, yMax = null) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        type: 'time',
        time: { unit: 'minute', displayFormats: { minute: 'HH:mm', second: 'HH:mm:ss' } },
        grid: { color: 'rgba(255,255,255,0.04)' },
        ticks: { color: '#475569', maxTicksLimit: 8, maxRotation: 0 },
      },
      y: {
        min: yMin !== null ? yMin : undefined,
        max: yMax !== null ? yMax : undefined,
        grid: { color: 'rgba(255,255,255,0.06)' },
        ticks: { color: '#475569' },
        title: { display: !!yLabel, text: yLabel, color: '#475569', font: { size: 11 } },
      }
    }
  };
}

function initCharts() {
  // Flow Rate
  charts.flow = new Chart(
    document.getElementById('chartFlow').getContext('2d'), {
      type: 'line',
      data: { datasets: [{ label: 'Flow', data: [],
        borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.08)',
        borderWidth: 1.8, fill: true, tension: 0.4, pointRadius: 0 }] },
      options: chartDefaults('L/min', '#00d4ff', 0, 15),
    });

  // Turbidity
  charts.turb = new Chart(
    document.getElementById('chartTurb').getContext('2d'), {
      type: 'line',
      data: { datasets: [{ label: 'Turbidity', data: [],
        borderColor: '#a855f7', backgroundColor: 'rgba(168,85,247,0.08)',
        borderWidth: 1.8, fill: true, tension: 0.4, pointRadius: 0 }] },
      options: chartDefaults('NTU', '#a855f7', 0, 4),
    });

  // Reconstruction Error
  charts.recon = new Chart(
    document.getElementById('chartRecon').getContext('2d'), {
      type: 'line',
      data: { datasets: [
        { label: 'Recon Error', data: [],
          borderColor: '#38bdf8', backgroundColor: 'rgba(56,189,248,0.08)',
          borderWidth: 1.5, fill: true, tension: 0.3, pointRadius: 0 },
        { label: 'Threshold', data: [],
          borderColor: '#f59e0b', borderWidth: 1.2,
          borderDash: [5,3], fill: false, pointRadius: 0 },
      ] },
      options: chartDefaults('MSE', '#38bdf8'),
    });

  // Anomaly Scatter
  charts.anomaly = new Chart(
    document.getElementById('chartAnomaly').getContext('2d'), {
      type: 'scatter',
      data: { datasets: [
        { label: 'Normal', data: [],
          backgroundColor: 'rgba(0,212,255,0.55)', borderColor: '#00d4ff',
          pointRadius: 2.5, pointHoverRadius: 4 },
        { label: 'Leak', data: [],
          backgroundColor: 'rgba(239,68,68,0.8)', borderColor: '#ef4444',
          pointRadius: 4, pointHoverRadius: 6 },
      ] },
      options: {
        ...chartDefaults('Flow (L/min)', null, 0, 15),
        plugins: { legend: { display: false } },
      },
    });
}

function push(buf, pt) {
  buf.push(pt);
  if (buf.length > MAX_POINTS) buf.shift();
}

function updateCharts(ts) {
  const now  = new Date(ts);
  const from = new Date(now.getTime() - 120_000);

  function setWindow(ch) {
    ch.options.scales.x.min = from.getTime();
    ch.options.scales.x.max = now.getTime();
  }

  charts.flow.data.datasets[0].data = bufs.flow;
  setWindow(charts.flow); charts.flow.update('none');

  charts.turb.data.datasets[0].data = bufs.turb;
  setWindow(charts.turb); charts.turb.update('none');

  charts.recon.data.datasets[0].data = bufs.recon;
  charts.recon.data.datasets[1].data = bufs.recon.map(p => ({ x: p.x, y: p.threshold }));
  if (bufs.recon.length > 0) {
    const vals = bufs.recon.map(p => p.y);
    const mx   = Math.max(...vals, bufs.recon[0]?.threshold || 0) * 1.3;
    if (mx > maxReconError) { maxReconError = mx; charts.recon.options.scales.y.max = mx; }
  }
  setWindow(charts.recon); charts.recon.update('none');

  charts.anomaly.data.datasets[0].data = bufs.normal;
  charts.anomaly.data.datasets[1].data = bufs.anomaly;
  setWindow(charts.anomaly); charts.anomaly.update('none');
}

// =============================================
// WebSocket
// =============================================
function initSocket() {
  socket = io.connect(window.location.origin);

  socket.on('connect',    () => setConn(true));
  socket.on('disconnect', () => setConn(false));
  socket.on('connection_response', (d) => console.log('Config:', d.config));

  socket.on('sensor_data', handleData);
  socket.on('leak_alert',  handleAlert);
  socket.on('simulation_status', handleStatus);
  socket.on('speed_changed', (d) => {
    document.getElementById('speedLabel').textContent = d.speed + '×';
    document.getElementById('speedSlider').value = d.speed;
  });
}

function setConn(ok) {
  const el = document.getElementById('connectionStatus');
  el.className = 'conn-pill ' + (ok ? 'connected' : 'disconnected');
  el.querySelector('.conn-text').textContent = ok ? 'Connected' : 'Disconnected';
}

// =============================================
// Data Handling
// =============================================
function handleData(d) {
  const ts      = new Date(d.timestamp);
  const sensor  = d.sensor_data;
  const preds   = d.predictions;
  const isLeak  = d.leak_active === true;

  // Update sim-time display
  if (d.sim_time) {
    document.getElementById('simTimeDisplay').textContent = d.sim_time;
  }

  // Leak badge
  if (isLeak !== lastLeakState) {
    lastLeakState = isLeak;
    setLeakBadge(isLeak);
  }

  // Stat cards
  updateStats(sensor, preds, isLeak);

  // Buffer points
  push(bufs.flow, { x: ts, y: sensor.flow_rate });
  push(bufs.turb, { x: ts, y: sensor.turbidity });

  const reconErr = preds.reconstruction_error || 0;
  const thresh   = preds.autoencoder?.threshold || 0;
  push(bufs.recon, { x: ts, y: reconErr, threshold: thresh });

  const pt = { x: ts, y: sensor.flow_rate };
  if (preds.ensemble === 1) push(bufs.anomaly, pt);
  else                       push(bufs.normal,  pt);

  // AI predictions
  setPrediction('predAutoencoder', preds.autoencoder?.prediction);
  setPrediction('predIsolation',   preds.isolation_forest?.prediction);
  setPrediction('predEnsemble',    preds.ensemble, true);

  updateCharts(d.timestamp);
}

function updateStats(sensor, preds, isLeak) {
  const MAX_FLOW = 15;
  const MAX_TURB = 4;
  const reconErr = preds.reconstruction_error || 0;
  const conf     = (preds.confidence || 0) * 100;

  // Values
  document.getElementById('statFlow').textContent  = sensor.flow_rate.toFixed(2);
  document.getElementById('statTurb').textContent  = sensor.turbidity.toFixed(2);
  document.getElementById('statRecon').textContent = reconErr < 0.001
    ? reconErr.toExponential(2) : reconErr.toFixed(4);
  document.getElementById('statConf').textContent  = conf.toFixed(0);

  // Progress bars
  setBar('barFlow',  sensor.flow_rate / MAX_FLOW * 100);
  setBar('barTurb',  sensor.turbidity / MAX_TURB * 100);
  setBar('barRecon', Math.min(100, reconErr / (preds.autoencoder?.threshold || 0.05) * 50));
  setBar('barConf',  conf);

  // Alert highlight on cards
  ['cardFlow','cardTurb','cardRecon','cardConf'].forEach(id => {
    document.getElementById(id).classList.toggle('alert', isLeak);
  });
}

function setBar(id, pct) {
  document.getElementById(id).style.width = Math.min(100, Math.max(0, pct)) + '%';
}

function setLeakBadge(isLeak) {
  const badge = document.getElementById('leakStatusBadge');
  const text  = document.getElementById('leakStatusText');
  badge.className = 'leak-badge ' + (isLeak ? 'leak' : 'normal');
  text.textContent = isLeak ? 'LEAK DETECTED' : 'NORMAL';
}

function setPrediction(id, pred, isEnsemble = false) {
  const el = document.getElementById(id);
  if (pred === null || pred === undefined) {
    el.className = 'ai-result pending' + (isEnsemble ? ' ensemble' : '');
    el.textContent = '—';
  } else if (pred === 1) {
    el.className = 'ai-result leak' + (isEnsemble ? ' ensemble' : '');
    el.textContent = 'LEAK';
  } else {
    el.className = 'ai-result normal' + (isEnsemble ? ' ensemble' : '');
    el.textContent = 'NORMAL';
  }
}

// =============================================
// Alert
// =============================================
function handleAlert(d) {
  const banner  = document.getElementById('alertBanner');
  const detail  = document.getElementById('alertDetail');
  const simTime = d.sim_time || '';
  const conf    = ((d.confidence || 0) * 100).toFixed(1);
  const recon   = (d.reconstruction_error || 0).toFixed(5);
  detail.textContent =
    `${simTime} | Confidence: ${conf}% | LSTM Error: ${recon}`;
  banner.classList.remove('hidden');
  clearTimeout(banner._timer);
  banner._timer = setTimeout(closeAlert, 8000);
}

function closeAlert() {
  document.getElementById('alertBanner').classList.add('hidden');
}

// =============================================
// Simulation Controls
// =============================================
function startSimulation() {
  socket.emit('start_simulation');
  simStartTime = Date.now();
  startTimer();
  document.getElementById('startBtn').disabled = true;
  document.getElementById('pauseBtn').disabled = false;
  document.getElementById('stopBtn').disabled  = false;
  // Clear buffers on start
  Object.keys(bufs).forEach(k => bufs[k] = []);
  lastLeakState = false;
  setLeakBadge(false);
  setPrediction('predAutoencoder', null);
  setPrediction('predIsolation',   null);
  setPrediction('predEnsemble',    null, true);
}

function pauseSimulation() {
  socket.emit('pause_simulation');
  stopTimer();
  const btn = document.getElementById('pauseBtn');
  btn.textContent = '▶ Resume';
  btn.onclick = resumeSimulation;
}

function resumeSimulation() {
  socket.emit('resume_simulation');
  startTimer();
  const btn = document.getElementById('pauseBtn');
  btn.textContent = '⏸ Pause';
  btn.onclick = pauseSimulation;
}

function stopSimulation() {
  socket.emit('stop_simulation');
  stopTimer();
  simStartTime = null;
  document.getElementById('startBtn').disabled = false;
  document.getElementById('pauseBtn').disabled = true;
  document.getElementById('stopBtn').disabled  = true;
  document.getElementById('simElapsed').textContent = '00:00:00';
  document.getElementById('simTimeDisplay').textContent = '--:-- --/--';
  setLeakBadge(false);
}

function handleStatus(d) {
  const pill = document.getElementById('simStatus');
  const map  = { started: 'running', paused: 'paused', stopped: 'stopped', resumed: 'running' };
  const cls  = map[d.status] || 'idle';
  pill.className = 'status-pill ' + cls;
  pill.textContent = cls.toUpperCase();
}

// Speed slider
function onSpeedChange(val) {
  const n = parseInt(val);
  document.getElementById('speedLabel').textContent = n + '×';
  if (socket) socket.emit('set_speed', { speed: n });
}

// =============================================
// Timer
// =============================================
function startTimer() {
  if (timerInterval) return;
  timerInterval = setInterval(() => {
    if (!simStartTime) return;
    const e = Date.now() - simStartTime;
    const h = String(Math.floor(e / 3600000)).padStart(2,'0');
    const m = String(Math.floor((e % 3600000) / 60000)).padStart(2,'0');
    const s = String(Math.floor((e % 60000) / 1000)).padStart(2,'0');
    document.getElementById('simElapsed').textContent = `${h}:${m}:${s}`;
  }, 1000);
}

function stopTimer() {
  clearInterval(timerInterval);
  timerInterval = null;
}
