// Three.js ESM imports (resolved via importmap in index.html)
import * as THREE from 'three';
import { Water } from 'three/addons/objects/Water.js';
import { Sky } from 'three/addons/objects/Sky.js';

/**
 * Water Leak Detection — Dashboard JS
 * Three.js Water + Sky (realistic) + Chart.js + Socket.IO
 */

let socket = null;
let charts = {};
let simStart = null;
let timerInt = null;
let lastLeak = false;
let maxReconErr = 0.001;

const MAX_PTS = 300;
const bufs = {
  flow: [],
  recon: [],
  ae_norm: [], ae_anom: [],
  if_norm: [], if_anom: [],
};

document.addEventListener('DOMContentLoaded', () => {
  initWave();
  initCharts();
  initSocket();
});

// =============================================
// Realistic Three.js Water + Sky
// =============================================
function initWave() {
  const canvas = document.getElementById('bgCanvas');

  // Renderer
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  renderer.setSize(window.innerWidth, window.innerHeight);
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 0.8;

  const scene = new THREE.Scene();
  const camera = new THREE.PerspectiveCamera(
    60, window.innerWidth / window.innerHeight, 1, 20000
  );
  camera.position.set(30, 12, 80);
  camera.lookAt(0, 0, 0);

  // ── Physically-based Sky ────────────────────
  const sky = new Sky();
  sky.scale.setScalar(10000);
  scene.add(sky);

  const skyUniforms = sky.material.uniforms;
  skyUniforms['turbidity'].value = 10;
  skyUniforms['rayleigh'].value = 2;
  skyUniforms['mieCoefficient'].value = 0.005;
  skyUniforms['mieDirectionalG'].value = 0.8;

  // Sun position — late afternoon
  const sun = new THREE.Vector3();
  const phi = THREE.MathUtils.degToRad(80);   // elevation from zenith
  const theta = THREE.MathUtils.degToRad(200);   // azimuth
  sun.setFromSphericalCoords(1, phi, theta);
  skyUniforms['sunPosition'].value.copy(sun);

  // PMREM for IBL reflections on water
  const pmremGenerator = new THREE.PMREMGenerator(renderer);
  const sceneEnv = new THREE.Scene();
  sceneEnv.add(sky.clone());
  const renderTarget = pmremGenerator.fromScene(sceneEnv);
  scene.environment = renderTarget.texture;

  // ── Realistic Water ──────────────────────────
  // Normal map from Three.js examples CDN
  const waterNormals = new THREE.TextureLoader().load(
    'https://cdn.jsdelivr.net/gh/mrdoob/three.js@r160/examples/textures/waternormals.jpg',
    tex => { tex.wrapS = tex.wrapT = THREE.RepeatWrapping; }
  );

  const waterGeometry = new THREE.PlaneGeometry(10000, 10000);
  const water = new Water(waterGeometry, {
    textureWidth: 512,
    textureHeight: 512,
    waterNormals,
    sunDirection: sun.clone().normalize(),
    sunColor: 0xffffff,
    waterColor: 0x006994,
    distortionScale: 4.5,
    fog: false,
  });
  water.rotation.x = -Math.PI / 2;
  scene.add(water);

  // ── Fog for depth ───────────────────────────
  scene.fog = new THREE.FogExp2(0x87ceeb, 0.0006);

  // ── Resize ──────────────────────────────────
  window.addEventListener('resize', () => {
    const W = window.innerWidth, H = window.innerHeight;
    renderer.setSize(W, H);
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
  });

  // ── Animate ─────────────────────────────────
  let t = 0;
  function loop() {
    t += 0.001;
    // Gentle camera drift — feels like standing on a boat
    camera.position.x = Math.sin(t * 0.4) * 15;
    camera.position.y = 12 + Math.sin(t * 0.7) * 1.5;
    camera.lookAt(0, 0, 0);
    // Advance water shader time
    water.material.uniforms['time'].value += 1.0 / 60.0;
    renderer.render(scene, camera);
    requestAnimationFrame(loop);
  }
  loop();
}

// =============================================
// Chart.js
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
        time: {
          displayFormats: {
            millisecond: 'HH:mm:ss',
            second: 'HH:mm:ss',
            minute: 'HH:mm',
            hour: 'HH:mm dd/MM',
            day: 'dd/MM',
          },
          tooltipFormat: 'HH:mm dd/MM/yy',
        },
        grid: { color: 'rgba(2,132,199,0.08)' },
        ticks: { color: '#64748b', maxTicksLimit: 5, maxRotation: 0, autoSkip: true },
      },
      y: {
        min: yMin !== null ? yMin : undefined,
        max: yMax !== null ? yMax : undefined,
        grid: { color: 'rgba(2,132,199,0.08)' },
        ticks: { color: '#64748b', maxTicksLimit: 4 },
      },
    },
  };
}

function initCharts() {
  charts.flow = new Chart(document.getElementById('cFlow'), {
    type: 'line',
    data: {
      datasets: [{
        data: [],
        borderColor: '#0284c7', backgroundColor: 'rgba(2,132,199,0.08)',
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
          data: [], borderColor: '#0891b2', backgroundColor: 'rgba(8,145,178,0.08)',
          borderWidth: 1.8, fill: true, tension: 0.3, pointRadius: 0
        },
        {
          data: [], borderColor: '#d97706', borderWidth: 1.2,
          borderDash: [5, 3], fill: false, pointRadius: 0
        },
      ]
    },
    options: mkOpts(),
  });

  const scOpts = (mn, mx) => ({
    ...mkOpts(mn, mx), plugins: { legend: { display: false } },
  });

  charts.anomalyAE = new Chart(document.getElementById('cAnomalyAE'), {
    type: 'scatter',
    data: {
      datasets: [
        { data: [], backgroundColor: 'rgba(2,132,199,0.5)', borderColor: '#0284c7', pointRadius: 2.5 },
        { data: [], backgroundColor: 'rgba(220,38,38,0.8)', borderColor: '#dc2626', pointRadius: 4 },
      ]
    }, options: scOpts(0, 15),
  });

  charts.anomalyIF = new Chart(document.getElementById('cAnomalyIF'), {
    type: 'scatter',
    data: {
      datasets: [
        { data: [], backgroundColor: 'rgba(2,132,199,0.5)', borderColor: '#0284c7', pointRadius: 2.5 },
        { data: [], backgroundColor: 'rgba(220,38,38,0.8)', borderColor: '#dc2626', pointRadius: 4 },
      ]
    }, options: scOpts(0, 15),
  });
}

function push(buf, pt) { buf.push(pt); if (buf.length > MAX_PTS) buf.shift(); }

function setW(ch, buf) {
  if (buf.length < 2) return;
  ch.options.scales.x.min = +buf[0].x;
  ch.options.scales.x.max = +buf[buf.length - 1].x;
}

function updateCharts() {
  charts.flow.data.datasets[0].data = bufs.flow;
  setW(charts.flow, bufs.flow); charts.flow.update('none');

  charts.recon.data.datasets[0].data = bufs.recon;
  charts.recon.data.datasets[1].data = bufs.recon.map(p => ({ x: p.x, y: p.threshold }));
  if (bufs.recon.length > 0) {
    const mx = Math.max(...bufs.recon.map(p => p.y), bufs.recon[0]?.threshold || 0) * 1.35;
    if (mx > maxReconErr) { maxReconErr = mx; charts.recon.options.scales.y.max = mx; }
  }
  setW(charts.recon, bufs.recon); charts.recon.update('none');

  // Scatter: give Chart.js a fresh array copy each tick so it detects
  // the change and auto-scales the time axis from the actual data bounds.
  charts.anomalyAE.data.datasets[0].data = bufs.ae_norm.slice();
  charts.anomalyAE.data.datasets[1].data = bufs.ae_anom.slice();
  charts.anomalyAE.options.scales.x.min = undefined;
  charts.anomalyAE.options.scales.x.max = undefined;
  charts.anomalyAE.update();

  charts.anomalyIF.data.datasets[0].data = bufs.if_norm.slice();
  charts.anomalyIF.data.datasets[1].data = bufs.if_anom.slice();
  charts.anomalyIF.options.scales.x.min = undefined;
  charts.anomalyIF.options.scales.x.max = undefined;
  charts.anomalyIF.update();
}

// =============================================
// Socket.IO
// =============================================
function initSocket() {
  // io is a UMD global from the socket.io CDN script
  socket = window.io.connect(window.location.origin);
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

// =============================================
// Data Handling
// =============================================
function handleData(d) {
  const ts = new Date(d.timestamp);
  const s = d.sensor_data;
  const p = d.predictions;
  const isLeak = !!d.leak_active;

  if (d.sim_time) document.getElementById('simTimeDisplay').textContent = d.sim_time;
  if (isLeak !== lastLeak) { lastLeak = isLeak; setLeakState(isLeak); }

  const recon = p.reconstruction_error || 0;
  const conf = (p.confidence || 0) * 100;
  const thresh = p.autoencoder?.threshold || 0;

  document.getElementById('valFlow').textContent = s.flow_rate.toFixed(2);
  document.getElementById('valTurb').textContent = s.turbidity.toFixed(2);
  document.getElementById('valLstm').textContent = recon < 0.001 ? recon.toExponential(2) : recon.toFixed(5);
  document.getElementById('valConf').textContent = conf.toFixed(0);

  ['valFlow', 'valTurb', 'valLstm', 'valConf'].forEach(id => {
    document.getElementById(id).classList.toggle('alert', isLeak);
  });
  document.querySelectorAll('.chart-card').forEach(el => {
    el.classList.toggle('leak-border', isLeak);
  });

  setBadge('predAE', p.autoencoder?.prediction);
  setBadge('predIF', p.isolation_forest?.prediction);
  setBadge('predEN', p.ensemble, true);

  push(bufs.flow, { x: ts, y: s.flow_rate });
  push(bufs.recon, { x: ts, y: recon, threshold: thresh });
  const pt = { x: ts, y: s.flow_rate };
  push(p.autoencoder?.prediction === 1 ? bufs.ae_anom : bufs.ae_norm, pt);
  push(p.isolation_forest?.prediction === 1 ? bufs.if_anom : bufs.if_norm, pt);

  updateCharts();
}

function setBadge(id, pred, isLg = false) {
  const el = document.getElementById(id);
  const base = 'ai-badge' + (isLg ? ' ai-badge-lg' : '');
  if (pred === null || pred === undefined) {
    el.className = base + ' bw'; el.textContent = '--';
  } else if (pred === 1) {
    el.className = base + ' bleak'; el.textContent = 'LEAK';
  } else {
    el.className = base + ' bnorm'; el.textContent = 'NORMAL';
  }
}

function setLeakState(isLeak) {
  updateBadge(isLeak ? 'leak' : (_isConnected ? 'normal' : 'offline'));
}

let _isConnected = false;
function setConn(ok) {
  _isConnected = ok;
  updateBadge(ok ? (lastLeak ? 'leak' : 'normal') : 'offline');
}

function updateBadge(state) {
  const el = document.getElementById('statusBadge');
  const txt = document.getElementById('badgeTxt');
  if (!el) return;
  el.className = 'status-badge ' + state;
  txt.textContent = state === 'offline' ? 'OFFLINE'
    : state === 'leak' ? 'LEAK'
      : 'NORMAL';
}

// =============================================
// Alert
// =============================================
function handleAlert(d) {
  const box = document.getElementById('alertBox');
  const meta = document.getElementById('alertMeta');
  meta.textContent = `${d.sim_time || ''} | CONF ${((d.confidence || 0) * 100).toFixed(1)}% | ERR ${(d.reconstruction_error || 0).toFixed(5)}`;
  box.classList.remove('hidden');
  clearTimeout(box._t);
  box._t = setTimeout(closeAlert, 8000);
}
function closeAlert() { document.getElementById('alertBox').classList.add('hidden'); }

// =============================================
// Controls
// =============================================
function startSimulation() {
  if (!socket) return;
  socket.emit('start_simulation');
  simStart = Date.now(); startTimer();
  Object.keys(bufs).forEach(k => bufs[k] = []);
  maxReconErr = 0.001;
  lastLeak = false; setLeakState(false);
  ['predAE', 'predIF'].forEach(id => setBadge(id, null));
  setBadge('predEN', null, true);
  document.getElementById('btnStart').disabled = true;
  document.getElementById('btnPause').disabled = false;
  document.getElementById('btnStop').disabled = false;
}

function pauseSimulation() {
  if (!socket) return;
  socket.emit('pause_simulation');
  stopTimer();
  const b = document.getElementById('btnPause');
  b.textContent = '\u25ba RESUME'; b.onclick = resumeSimulation;
}

function resumeSimulation() {
  if (!socket) return;
  socket.emit('resume_simulation');
  startTimer();
  const b = document.getElementById('btnPause');
  b.innerHTML = '&#9646;&#9646; PAUSE'; b.onclick = pauseSimulation;
}

function stopSimulation() {
  if (!socket) return;
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

// =============================================
// Expose globals for HTML onclick handlers
// (required because this file is type="module")
// =============================================
window.startSimulation = startSimulation;
window.pauseSimulation = pauseSimulation;
window.stopSimulation = stopSimulation;
window.onSpeedChange = onSpeedChange;
window.closeAlert = closeAlert;
