/**
 * AquaGuard — Live Dashboard JS
 * Handles socket, charts, stat cards, speed, animations
 */

let socket = null;
let charts = {};
let simStart = null;
let timerInt = null;
let lastLeak = false;
let maxReconErr = 0.001;

const MAX_PTS = 300;
const bufs = { flow: [], turb: [], recon: [], normal: [], anomaly: [] };

document.addEventListener('DOMContentLoaded', () => {
  initBg();
  initCharts();
  initSocket();
});

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

  charts.turb = new Chart(document.getElementById('cTurb'), {
    type: 'line',
    data: {
      datasets: [{
        data: [], borderColor: '#8b5cf6', backgroundColor: 'rgba(139,92,246,0.07)',
        borderWidth: 2, fill: true, tension: 0.4, pointRadius: 0
      }]
    },
    options: mkOpts(0, 4),
  });

  charts.recon = new Chart(document.getElementById('cRecon'), {
    type: 'line',
    data: {
      datasets: [
        {
          data: [], borderColor: '#06b6d4', backgroundColor: 'rgba(6,182,212,0.07)',
          borderWidth: 1.8, fill: true, tension: 0.3, pointRadius: 0
        },
        {
          data: [], borderColor: '#f59e0b', borderWidth: 1.2, borderDash: [5, 3],
          fill: false, pointRadius: 0
        },
      ]
    },
    options: mkOpts(),
  });

  charts.anomaly = new Chart(document.getElementById('cAnomaly'), {
    type: 'scatter',
    data: {
      datasets: [
        { data: [], backgroundColor: 'rgba(14,165,233,0.5)', borderColor: '#0ea5e9', pointRadius: 2.5 },
        { data: [], backgroundColor: 'rgba(239,68,68,0.8)', borderColor: '#ef4444', pointRadius: 4 },
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

  charts.turb.data.datasets[0].data = bufs.turb;
  setW(charts.turb); charts.turb.update('none');

  charts.recon.data.datasets[0].data = bufs.recon;
  charts.recon.data.datasets[1].data = bufs.recon.map(p => ({ x: p.x, y: p.threshold }));
  if (bufs.recon.length > 0) {
    const mx = Math.max(...bufs.recon.map(p => p.y), bufs.recon[0]?.threshold || 0) * 1.35;
    if (mx > maxReconErr) { maxReconErr = mx; charts.recon.options.scales.y.max = mx; }
  }
  setW(charts.recon); charts.recon.update('none');

  charts.anomaly.data.datasets[0].data = bufs.normal;
  charts.anomaly.data.datasets[1].data = bufs.anomaly;
  setW(charts.anomaly); charts.anomaly.update('none');
}

// =============================================
// Socket
// =============================================
function initSocket() {
  socket = io.connect(window.location.origin);
  socket.on('connect', () => setConn(true));
  socket.on('disconnect', () => setConn(false));
  socket.on('sensor_data', handleData);
  socket.on('leak_alert', handleAlert);
  socket.on('simulation_status', handleStatus);
  socket.on('speed_changed', d => {
    document.getElementById('speedVal').textContent = d.speed;
    document.getElementById('speedSlider').value = d.speed;
  });
}

function setConn(ok) {
  const el = document.getElementById('connStatus');
  el.className = 'conn-status ' + (ok ? 'online' : 'offline');
  el.querySelector('.conn-label').textContent = ok ? 'ONLINE' : 'OFFLINE';
}

// =============================================
// Data
// =============================================
function handleData(d) {
  const ts = new Date(d.timestamp);
  const s = d.sensor_data;
  const p = d.predictions;
  const isLeak = !!d.leak_active;

  if (d.sim_time) document.getElementById('simTimeDisplay').textContent = d.sim_time;

  if (isLeak !== lastLeak) { lastLeak = isLeak; setLeakState(isLeak); }

  // stats
  const recon = p.reconstruction_error || 0;
  const conf = (p.confidence || 0) * 100;
  const thresh = p.autoencoder?.threshold || 0;

  document.getElementById('valFlow').textContent = s.flow_rate.toFixed(2);
  document.getElementById('valTurb').textContent = s.turbidity.toFixed(2);
  document.getElementById('valLstm').textContent = recon < 0.001 ? recon.toExponential(2) : recon.toFixed(5);
  document.getElementById('valConf').textContent = conf.toFixed(0);
  document.getElementById('subThresh').textContent = thresh ? thresh.toFixed(5) : '--';

  bar('barFlow', s.flow_rate / 15 * 100);
  bar('barTurb', s.turbidity / 4 * 100);
  bar('barLstm', Math.min(100, thresh > 0 ? recon / thresh * 50 : 0));
  bar('barConf', conf);

  ['sc-flow', 'sc-turb', 'sc-lstm', 'sc-conf'].forEach(id => {
    document.getElementById(id).classList.toggle('alert-state', isLeak);
  });
  document.querySelectorAll('.chart-card').forEach(el => {
    el.classList.toggle('leak-border', isLeak);
  });

  setBadge('predAE', p.autoencoder?.prediction);
  setBadge('predIF', p.isolation_forest?.prediction);
  setBadge('predEN', p.ensemble, true);

  push(bufs.flow, { x: ts, y: s.flow_rate });
  push(bufs.turb, { x: ts, y: s.turbidity });
  push(bufs.recon, { x: ts, y: recon, threshold: thresh });
  const pt = { x: ts, y: s.flow_rate };
  push(p.ensemble === 1 ? bufs.anomaly : bufs.normal, pt);

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
    el.className = base + ' badge-leak'; el.textContent = 'LEAK';
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
  meta.textContent = `${d.sim_time || ''} | CONF ${((d.confidence || 0) * 100).toFixed(1)}% | LSTM ${(d.reconstruction_error || 0).toFixed(5)}`;
  box.classList.remove('hidden');
  clearTimeout(box._t);
  box._t = setTimeout(closeAlert, 8000);
}
function closeAlert() { document.getElementById('alertBox').classList.add('hidden'); }

// =============================================
// Controls
// =============================================
function startSimulation() {
  socket.emit('start_simulation');
  simStart = Date.now(); startTimer();
  Object.keys(bufs).forEach(k => bufs[k] = []);
  lastLeak = false; setLeakState(false);
  ['predAE', 'predIF', 'predEN'].forEach(id => setBadge(id, null, id === 'predEN'));
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnPause').disabled = false;
  document.getElementById('btnStop').disabled = false;
}

function pauseSimulation() {
  socket.emit('pause_simulation');
  stopTimer();
  const b = document.getElementById('btnPause');
  b.textContent = ' RESUME'; b.onclick = resumeSimulation;
}

function resumeSimulation() {
  socket.emit('resume_simulation');
  startTimer();
  const b = document.getElementById('btnPause');
  b.innerHTML = '<span class="cbtn-icon">&#9646;&#9646;</span> PAUSE';
  b.onclick = pauseSimulation;
}

function stopSimulation() {
  socket.emit('stop_simulation');
  stopTimer(); simStart = null;
  document.getElementById('btnStart').disabled = false;
  document.getElementById('btnPause').disabled = true;
  document.getElementById('btnStop').disabled = true;
  document.getElementById('elapsedTime').textContent = '00:00:00';
  document.getElementById('simTimeDisplay').textContent = '--:-- --/--';
  setLeakState(false);
}

function handleStatus(d) {
  const pill = document.getElementById('simStatusPill');
  const map = { started: 'running', paused: 'paused', stopped: 'stopped', resumed: 'running' };
  const cls = map[d.status] || 'idle';
  pill.className = 'spill spill-' + cls;
  pill.textContent = cls.toUpperCase();
}

function onSpeedChange(v) {
  document.getElementById('speedVal').textContent = v;
  if (socket) socket.emit('set_speed', { speed: parseInt(v) });
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
function stopTimer() { clearInterval(timerInt); timerInt = null; }
