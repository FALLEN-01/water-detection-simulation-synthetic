/**
 * AquaGuard — Live Dashboard JS
 * Handles simulation, charts, stat cards, speed, animations
 */

let charts = {};
let simStart = null;
let timerInt = null;
let simInt = null;
let lastLeak = false;
let maxReconErr = 0.001;
let currentSpeed = 1;
let simRunning = false;
let simPaused = false;
let simTime = new Date('2024-01-01T06:00:00');
let leakInjected = false;
let leakStartTime = null;

const MAX_PTS = 300;
const bufs = { flow: [], recon: [], stats: [], normal: [], anomaly: [] };

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
  document.getElementById('simTimeDisplay').textContent = formatSimTime(simTime);
  
  // Initialize AI badges
  ['predStat', 'predCNN', 'predFusion'].forEach(id => setBadge(id, null, id === 'predFusion'));
}

// =============================================
// Animated background — moving grid + particles
// =============================================
function initBg() {
  const cv = document.getElementById('bgCanvas');
  const cx = cv.getContext('2d');
  let W, H;

  function resize() {
    W = cv.width = window.innerWidth;
    H = cv.height = window.innerHeight;
  }
  window.addEventListener('resize', resize);
  resize();

  // Particles
  const pts = Array.from({ length: 60 }, () => ({
    x: Math.random() * W, y: Math.random() * H,
    vx: (Math.random() - 0.5) * 0.18, vy: (Math.random() - 0.5) * 0.18,
    r: Math.random() * 1.3 + 0.2, a: Math.random() * 0.3 + 0.04,
  }));

  let t = 0;
  function draw() {
    t++;
    cx.clearRect(0, 0, W, H);

    // Grid lines (subtle depth grid)
    const STEP = 80;
    cx.strokeStyle = 'rgba(14,165,233,0.03)';
    cx.lineWidth = 1;
    for (let x = 0; x < W; x += STEP) {
      cx.beginPath(); cx.moveTo(x, 0); cx.lineTo(x, H); cx.stroke();
    }
    for (let y = 0; y < H; y += STEP) {
      cx.beginPath(); cx.moveTo(0, y); cx.lineTo(W, y); cx.stroke();
    }

    // Particles + connections
    pts.forEach(p => {
      p.x += p.vx; p.y += p.vy;
      if (p.x < 0) p.x = W; if (p.x > W) p.x = 0;
      if (p.y < 0) p.y = H; if (p.y > H) p.y = 0;
      cx.beginPath();
      cx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
      cx.fillStyle = `rgba(14,165,233,${p.a})`;
      cx.fill();
    });

    // Connect nearby particles
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].x - pts[j].x, dy = pts[i].y - pts[j].y;
        const d = Math.sqrt(dx * dx + dy * dy);
        if (d < 130) {
          cx.beginPath();
          cx.moveTo(pts[i].x, pts[i].y);
          cx.lineTo(pts[j].x, pts[j].y);
          cx.strokeStyle = `rgba(14,165,233,${0.04 * (1 - d / 130)})`;
          cx.lineWidth = 0.5;
          cx.stroke();
        }
      }
    }

    requestAnimationFrame(draw);
  }
  draw();
}

// =============================================
// Charts
// =============================================
function mkOpts(yMin = null, yMax = null) {
  return {
    responsive: true,
    maintainAspectRatio: false,
    animation: false,
    plugins: { legend: { display: false } },
    scales: {
      x: {
        type: 'time',
        time: { unit: 'minute', displayFormats: { minute: 'HH:mm', second: 'HH:mm:ss' } },
        grid: { color: 'rgba(255,255,255,0.03)' },
        ticks: { color: '#334155', maxTicksLimit: 8, maxRotation: 0 },
      },
      y: {
        min: yMin !== null ? yMin : undefined,
        max: yMax !== null ? yMax : undefined,
        grid: { color: 'rgba(255,255,255,0.05)' },
        ticks: { color: '#334155', maxTicksLimit: 5 },
      }
    }
  };
}

function initCharts() {
  charts.flow = new Chart(document.getElementById('cFlow'), {
    type: 'line',
    data: {
      datasets: [{
        data: [], borderColor: '#0ea5e9', backgroundColor: 'rgba(14,165,233,0.07)',
        borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
      }]
    },
    options: mkOpts(0, 15),
  });

  charts.recon = new Chart(document.getElementById('cRecon'), {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Reconstruction Error',
          data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.07)',
          borderWidth: 1.8, fill: true, tension: 0.3, pointRadius: 0
        },
        {
          label: 'Threshold',
          data: [], borderColor: '#f59e0b', borderWidth: 1.2, borderDash: [5, 3],
          fill: false, pointRadius: 0
        },
      ]
    },
    options: mkOpts(),
  });

  charts.stats = new Chart(document.getElementById('cStats'), {
    type: 'line',
    data: {
      datasets: [
        {
          label: 'Rolling Mean',
          data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.07)',
          borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
        },
        {
          label: 'Leak Threshold',
          data: [], borderColor: '#ef4444', borderWidth: 1.2, borderDash: [3, 3],
          fill: false, pointRadius: 0
        }
      ]
    },
    options: mkOpts(0, 2),
  });

  charts.anomaly = new Chart(document.getElementById('cAnomaly'), {
    type: 'scatter',
    data: {
      datasets: [
        { 
          label: 'Normal',
          data: [], backgroundColor: 'rgba(14,165,233,0.5)', borderColor: '#0ea5e9', pointRadius: 2.5 
        },
        { 
          label: 'Leak Detected',
          data: [], backgroundColor: 'rgba(239,68,68,0.8)', borderColor: '#ef4444', pointRadius: 4 
        },
      ]
    },
    options: { ...mkOpts(0, 15), plugins: { legend: { display: false } } },
  });
}

function push(buf, pt) {
  buf.push(pt);
  if (buf.length > MAX_PTS) buf.shift();
}

function updateCharts(ts) {
  const now = new Date(ts);
  const from = new Date(now.getTime() - 120_000);

  function setW(ch) {
    ch.options.scales.x.min = from.getTime();
    ch.options.scales.x.max = now.getTime();
  }

  charts.flow.data.datasets[0].data = bufs.flow;
  setW(charts.flow); charts.flow.update('none');

  charts.recon.data.datasets[0].data = bufs.recon;
  charts.recon.data.datasets[1].data = bufs.recon.map(p => ({ x: p.x, y: p.threshold }));
  if (bufs.recon.length > 0) {
    const mx = Math.max(...bufs.recon.map(p => p.y), bufs.recon[0]?.threshold || 0) * 1.35;
    if (mx > maxReconErr) { maxReconErr = mx; charts.recon.options.scales.y.max = mx; }
  }
  setW(charts.recon); charts.recon.update('none');

  charts.stats.data.datasets[0].data = bufs.stats;
  charts.stats.data.datasets[1].data = bufs.stats.map(p => ({ x: p.x, y: 0.4 })); // Leak threshold line
  setW(charts.stats); charts.stats.update('none');

  charts.anomaly.data.datasets[0].data = bufs.normal;
  charts.anomaly.data.datasets[1].data = bufs.anomaly;
  setW(charts.anomaly); charts.anomaly.update('none');
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
// Water flow simulation
// =============================================
function generateFlowData() {
  const hour = simTime.getHours();
  const minute = simTime.getMinutes();
  
  let baseFlow = 0;
  
  // Peak usage times
  if (hour >= 6 && hour <= 9) { // Morning
    baseFlow = 2 + Math.random() * 8;
  } else if (hour >= 17 && hour <= 22) { // Evening
    baseFlow = 1 + Math.random() * 6;
  } else if (hour >= 22 || hour <= 6) { // Night
    baseFlow = 0.1 + Math.random() * 0.5;
  } else { // Day
    baseFlow = 0.5 + Math.random() * 3;
  }
  
  // Add random spikes for appliance usage
  if (Math.random() < 0.1) {
    baseFlow += Math.random() * 5;
  }
  
  // Add leak if injected
  if (leakInjected) {
    baseFlow += 0.3 + Math.random() * 0.2; // Constant leak
  }
  
  return Math.max(0, baseFlow);
}

// =============================================
// Hybrid Detection System Simulation
// =============================================
let detectionBuffer = [];
let rollingSum = 0;
let rollingSqSum = 0;
let persistenceCounter = 0;

function runHybridDetection(flow) {
  // Buffer management
  const windowSize = 60; // 60-minute window
  detectionBuffer.push(flow);
  
  if (detectionBuffer.length > windowSize) {
    const removed = detectionBuffer.shift();
    rollingSum -= removed;
    rollingSqSum -= removed * removed;
  }
  
  rollingSum += flow;
  rollingSqSum += flow * flow;
  
  // Statistical Layer (Level 2)
  let statAnomaly = false;
  let statConfidence = 0;
  let rollingMean = 0;
  let rollingStd = 0;
  
  if (detectionBuffer.length >= windowSize) {
    const n = detectionBuffer.length;
    rollingMean = rollingSum / n;
    const variance = (rollingSqSum / n) - (rollingMean * rollingMean);
    rollingStd = Math.sqrt(Math.max(variance, 0));
    
    // Sustained leak detection
    const leakThreshold = 0.4;
    const stdThreshold = 0.5;
    
    if (rollingMean > leakThreshold && rollingStd < stdThreshold) {
      persistenceCounter++;
    } else {
      persistenceCounter = 0;
    }
    
    const persistenceMinutes = 60;
    if (persistenceCounter >= persistenceMinutes) {
      statAnomaly = true;
      statConfidence = Math.min(1.0, rollingMean / leakThreshold);
    }
  }
  
  // CNN Autoencoder Layer (Level 3)
  const baseError = 0.001 + Math.random() * 0.002;
  let reconError = baseError;
  
  if (leakInjected) {
    reconError = 0.008 + Math.random() * 0.005; // Higher error during leak
  }
  
  const threshold = 0.005;
  const cnnAnomaly = reconError > threshold;
  const cnnScore = Math.min(1.0, reconError / threshold);
  
  // Fusion Layer
  const w2 = 0.6; // Statistical weight
  const w3 = 0.4; // CNN weight
  const fusionScore = (w2 * statConfidence) + (w3 * cnnScore);
  const decisionThreshold = 0.7;
  const finalAnomaly = fusionScore > decisionThreshold;
  
  return {
    statistical: {
      anomaly: statAnomaly,
      confidence: statConfidence,
      mean: rollingMean,
      std: rollingStd
    },
    cnn: {
      anomaly: cnnAnomaly,
      error: reconError,
      threshold: threshold,
      score: cnnScore
    },
    fusion: {
      score: fusionScore,
      anomaly: finalAnomaly,
      threshold: decisionThreshold
    }
  };
}

function simulationStep() {
  if (!simRunning || simPaused) return;
  
  const flow = generateFlowData();
  const detectionResults = runHybridDetection(flow);
  
  const isLeak = detectionResults.fusion.anomaly;
  
  // Track leak start time
  if (isLeak && !lastLeak) {
    leakStartTime = new Date(simTime);
  } else if (!isLeak && lastLeak) {
    leakStartTime = null;
  }
  
  // Update display
  handleData({
    timestamp: simTime.getTime(),
    sim_time: formatSimTime(simTime),
    sensor_data: {
      flow_rate: flow
    },
    detection: detectionResults,
    leak_active: isLeak
  });
  
  // Check for new leak detection
  if (isLeak && !lastLeak) {
    handleAlert({
      sim_time: formatSimTime(simTime),
      fusion_score: detectionResults.fusion.score,
      start_time: formatSimTime(leakStartTime)
    });
  }
  
  // Advance simulation time
  simTime = new Date(simTime.getTime() + (60000 * currentSpeed)); // 1 minute per step * speed
}

function formatSimTime(time) {
  const hours = time.getHours().toString().padStart(2, '0');
  const minutes = time.getMinutes().toString().padStart(2, '0');
  const day = time.getDate().toString().padStart(2, '0');
  const month = (time.getMonth() + 1).toString().padStart(2, '0');
  return `${hours}:${minutes} ${day}/${month}`;
}

// =============================================
// Data
// =============================================
function handleData(d) {
  const ts = new Date(d.timestamp);
  const s = d.sensor_data;
  const det = d.detection;
  const isLeak = !!d.leak_active;

  if (d.sim_time) document.getElementById('simTimeDisplay').textContent = d.sim_time;

  if (isLeak !== lastLeak) { lastLeak = isLeak; setLeakState(isLeak); }

  // Update stat cards
  const statConf = (det.statistical.confidence || 0) * 100;
  const cnnError = det.cnn.error || 0;
  const fusionScore = (det.fusion.score || 0) * 100;
  const threshold = det.cnn.threshold || 0;

  document.getElementById('valFlow').textContent = s.flow_rate.toFixed(2);
  document.getElementById('valStat').textContent = statConf.toFixed(0);
  document.getElementById('valCnn').textContent = cnnError < 0.001 ? cnnError.toExponential(2) : cnnError.toFixed(5);
  document.getElementById('valFusion').textContent = fusionScore.toFixed(0);
  document.getElementById('subThresh').textContent = threshold ? threshold.toFixed(5) : '--';

  // Update progress bars
  bar('barFlow', s.flow_rate / 15 * 100);
  bar('barStat', statConf);
  bar('barCnn', Math.min(100, threshold > 0 ? cnnError / threshold * 50 : 0));
  bar('barFusion', fusionScore);

  // Update alert states
  ['sc-flow', 'sc-stat', 'sc-cnn', 'sc-fusion'].forEach(id => {
    document.getElementById(id).classList.toggle('alert-state', isLeak);
  });
  document.querySelectorAll('.chart-card').forEach(el => {
    el.classList.toggle('leak-border', isLeak);
  });

  // Update AI badges
  setBadge('predStat', det.statistical.anomaly ? 1 : 0);
  setBadge('predCNN', det.cnn.anomaly ? 1 : 0);
  setBadge('predFusion', det.fusion.anomaly ? 1 : 0, true);

  // Update chart data
  push(bufs.flow, { x: ts, y: s.flow_rate });
  push(bufs.recon, { x: ts, y: cnnError, threshold: threshold });
  push(bufs.stats, { x: ts, y: det.statistical.mean || 0 });
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
function closeAlert() { document.getElementById('alertBox').classList.add('hidden'); }

// =============================================
// Controls
// =============================================
function startSimulation() {
  simRunning = true;
  simPaused = false;
  simStart = Date.now();
  simTime = new Date('2024-01-01T06:00:00');
  
  setConn(true);
  startTimer();
  
  // Clear buffers
  Object.keys(bufs).forEach(k => bufs[k] = []);
  lastLeak = false; 
  setLeakState(false);
  
  // Reset AI badges
  ['predStat', 'predCNN', 'predFusion'].forEach(id => setBadge(id, null, id === 'predFusion'));
  
  // Update buttons
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnPause').disabled = false;
  document.getElementById('btnStop').disabled = false;
  
  // Update status
  updateSimStatus('running');
  
  // Start simulation interval
  simInt = setInterval(simulationStep, 1000 / currentSpeed);
  
  // Inject leak after 5 minutes (simulation time)
  setTimeout(() => {
    if (simRunning && !simPaused) {
      leakInjected = true;
      console.log('Leak injected at', formatSimTime(simTime));
    }
  }, (5 * 60 * 1000) / currentSpeed); // 5 sim minutes
}

function pauseSimulation() {
  simPaused = true;
  stopTimer();
  clearInterval(simInt);
  setConn(false);
  
  const b = document.getElementById('btnPause');
  b.innerHTML = '<span class="cbtn-icon">&#9654;</span> RESUME';
  b.onclick = resumeSimulation;
  
  updateSimStatus('paused');
}

function resumeSimulation() {
  simPaused = false;
  startTimer();
  setConn(true);
  
  const b = document.getElementById('btnPause');
  b.innerHTML = '<span class="cbtn-icon">&#9646;&#9646;</span> PAUSE';
  b.onclick = pauseSimulation;
  
  updateSimStatus('running');
  
  // Restart simulation interval
  simInt = setInterval(simulationStep, 1000 / currentSpeed);
}

function stopSimulation() {
  simRunning = false;
  simPaused = false;
  leakInjected = false;
  
  stopTimer(); 
  clearInterval(simInt);
  simStart = null;
  setConn(false);
  
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnPause').disabled = true;
  document.getElementById('btnStop').disabled = true;
  document.getElementById('elapsedTime').textContent = '00:00:00';
  document.getElementById('simTimeDisplay').textContent = '--:-- --/--';
  
  // Reset pause button
  const b = document.getElementById('btnPause');
  b.innerHTML = '<span class="cbtn-icon">&#9646;&#9646;</span> PAUSE';
  b.onclick = pauseSimulation;
  
  setLeakState(false);
  updateSimStatus('stopped');
}

function updateSimStatus(status) {
  const pill = document.getElementById('simStatusPill');
  const statusMap = {
    'running': 'running',
    'paused': 'paused', 
    'stopped': 'stopped'
  };
  
  const cls = statusMap[status] || 'idle';
  pill.className = 'spill spill-' + cls;
  pill.textContent = cls.toUpperCase();
}

function onSpeedChange(v) {
  const speed = parseInt(v);
  // Limit speed to 1-10x
  const limitedSpeed = Math.min(10, Math.max(1, speed));
  currentSpeed = limitedSpeed;
  document.getElementById('speedVal').textContent = limitedSpeed;
  document.getElementById('speedSlider').value = limitedSpeed;
  
  // Restart simulation interval with new speed
  if (simRunning && !simPaused) {
    clearInterval(simInt);
    simInt = setInterval(simulationStep, 1000 / currentSpeed);
  }
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
