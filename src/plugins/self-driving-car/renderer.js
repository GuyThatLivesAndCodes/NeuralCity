// Self-Driving Car plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const CW = 700, CH = 530;   // canvas dimensions

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    trackFill:  '#1c1c1c',
    trackEdge:  '#3a3a3a',
    roadCenter: '#2a2a2a',
    carAlive:   '#00e676',
    carAlive2:  '#00897b',
    carDead:    '#444',
    carBest:    '#ffd600',
    bg:         '#0d0d0d',
  };

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'self-driving-car',
    name: 'Self-Driving Car (Neuroevolution)',
    kind: 'classifier',
    pluginKind: 'self-driving-car',
    desc: 'A population of cars evolves to lap an auto-generated track without crashing. Uses Selective Reproduction (neuroevolution). No training data required.',
    arch: {
      kind: 'classifier', pluginKind: 'self-driving-car',
      inputDim: 7, outputDim: 2,
      hidden: [16, 12], activation: 'tanh', dropout: 0,
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 32, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train editor ──────────────────────────────────────────────────────────
  api.registerTrainEditor('self-driving-car', function (root) {
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#1a1a00;border:1px solid #4a4a00;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#ffd600;margin-bottom:5px;">Neuroevolution — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            This plugin runs a population of 20 neural-network-controlled cars on a procedurally generated
            track. Each generation, the fittest cars are selected, crossed over, and mutated
            (Selective Reproduction). Over generations the population learns to lap the track faster and faster.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          Architecture: <strong style="color:#aaa;">7 → [16, 12] → 2</strong><br>
          Inputs: 5 sensor raycasts + normalised speed + orientation<br>
          Outputs: steer (tanh) · throttle (sigmoid)<br><br>
          Open the <strong style="color:#ccc;">Infer</strong> tab to launch the live simulation.
        </div>
      </div>
    `;
  });

  // ── Inference renderer — the full simulation ──────────────────────────────
  api.registerInferenceRenderer('self-driving-car', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;
    let _cachedTrack = null;

    root.innerHTML = `
      <div class="panel" style="max-width:960px;">
        <h2>Self-Driving Car — Neuroevolution Demo</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          20 cars race a procedurally-generated looped track. Each generation the fastest survivors
          are selected and mutated — watch the population converge on a lap strategy over generations.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="car-canvas" width="${CW}" height="${CH}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#0d0d0d;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;">
              <button class="btn primary" id="car-start">Start</button>
              <button class="btn"         id="car-pause">Pause</button>
              <button class="btn"         id="car-reset">New evolution</button>
              <button class="btn"         id="car-newtrack">New track</button>
            </div>
            <div style="margin-top:8px;display:flex;align-items:center;gap:8px;font-size:12px;color:#666;">
              Ticks / frame:
              <input id="car-speed" type="range" min="1" max="8" value="2" style="width:80px;">
              <span id="car-speed-val">2</span>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Generation</div><div class="v" id="car-gen">0</div></div>
              <div class="kpi"><div class="k">Alive</div><div class="v" id="car-alive">0 / 20</div></div>
              <div class="kpi"><div class="k">Gen best</div><div class="v" id="car-genbest">—</div></div>
              <div class="kpi"><div class="k">All-time best</div><div class="v" id="car-best">—</div></div>
            </div>

            <div class="section">
              <h3>Best fitness per generation</h3>
              <svg id="car-chart" viewBox="0 0 260 70" preserveAspectRatio="none"
                style="width:100%;height:70px;display:block;background:#0d0d0d;border-radius:4px;"></svg>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.8;">
              <strong style="color:#666;">Legend</strong><br>
              <span style="color:#00e676;">▲</span> Car (alive) &nbsp;&nbsp;
              <span style="color:#ffd600;">▲</span> Best of gen &nbsp;&nbsp;
              <span style="color:#444;">▲</span> Crashed<br><br>
              <strong style="color:#666;">Fitness</strong> = laps + (distance / track_len)<br>
              + small speed bonus<br><br>
              Population: 20 · Elite: 4 · pMutate: 0.15<br>
              Sensors: 5 rays (±72°) · max range 180 px<br>
              Episode: max 600 frames (~20 s)
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('car-canvas');
    const ctx    = canvas.getContext('2d');

    // ── Track drawing ────────────────────────────────────────────────────────

    function drawTrack(track) {
      if (!track) return;
      const { pts, halfWidth, N } = track;

      // Filled road band
      ctx.beginPath();
      // Outer boundary
      for (let i = 0; i <= N; i++) {
        const p = pts[i % N], q = pts[(i + 1) % N];
        const nx = -(q[1] - p[1]), ny = q[0] - p[0];
        const len = Math.hypot(nx, ny) || 1;
        const ox = p[0] + nx / len * halfWidth, oy = p[1] + ny / len * halfWidth;
        i === 0 ? ctx.moveTo(ox, oy) : ctx.lineTo(ox, oy);
      }
      ctx.closePath();
      // Inner boundary (reverse winding)
      ctx.moveTo(0, 0); // break path
      for (let i = N; i >= 0; i--) {
        const p = pts[i % N], q = pts[(i + 1) % N];
        const nx = -(q[1] - p[1]), ny = q[0] - p[0];
        const len = Math.hypot(nx, ny) || 1;
        const ix = p[0] - nx / len * halfWidth, iy = p[1] - ny / len * halfWidth;
        i === N ? ctx.moveTo(ix, iy) : ctx.lineTo(ix, iy);
      }
      ctx.closePath();
      ctx.fillStyle = COL.trackFill;
      ctx.fill('evenodd');

      // Road edges
      ctx.strokeStyle = COL.trackEdge;
      ctx.lineWidth   = 2;
      for (let side = -1; side <= 1; side += 2) {
        ctx.beginPath();
        for (let i = 0; i <= N; i++) {
          const p = pts[i % N], q = pts[(i + 1) % N];
          const nx = -(q[1] - p[1]), ny = q[0] - p[0];
          const len = Math.hypot(nx, ny) || 1;
          const ex = p[0] + side * nx / len * halfWidth;
          const ey = p[1] + side * ny / len * halfWidth;
          i === 0 ? ctx.moveTo(ex, ey) : ctx.lineTo(ex, ey);
        }
        ctx.closePath();
        ctx.stroke();
      }

      // Center dashed line
      ctx.strokeStyle = '#303030';
      ctx.lineWidth   = 1;
      ctx.setLineDash([12, 12]);
      ctx.beginPath();
      pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);

      // Start/finish marker
      const p0 = pts[0], p1 = pts[1];
      const nx = -(p1[1] - p0[1]), ny = p1[0] - p0[0];
      const len = Math.hypot(nx, ny) || 1;
      ctx.strokeStyle = '#ffffff44';
      ctx.lineWidth   = 2;
      ctx.beginPath();
      ctx.moveTo(p0[0] - nx / len * halfWidth, p0[1] - ny / len * halfWidth);
      ctx.lineTo(p0[0] + nx / len * halfWidth, p0[1] + ny / len * halfWidth);
      ctx.stroke();
    }

    // ── Car drawing ──────────────────────────────────────────────────────────

    function drawCar(car, color, alpha) {
      const { x, y, angle } = car;
      ctx.save();
      ctx.globalAlpha = alpha;
      ctx.translate(x, y);
      ctx.rotate(angle + Math.PI / 2);
      ctx.fillStyle   = color;
      ctx.strokeStyle = '#000000';
      ctx.lineWidth   = 0.5;
      ctx.beginPath();
      ctx.moveTo(0, -9);
      ctx.lineTo(-5, 6);
      ctx.lineTo(5, 6);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
      ctx.restore();
    }

    function drawState(s) {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);

      const track = s ? s.track : _cachedTrack;
      if (!track) return;
      if (s && s.track) _cachedTrack = s.track;
      drawTrack(track);

      if (!s || !s.cars) return;

      // Draw dead cars faintly
      for (let i = 0; i < s.cars.length; i++) {
        const c = s.cars[i];
        if (!c.alive) drawCar(c, COL.carDead, 0.3);
      }

      // Draw alive cars
      const alive = s.cars.filter(c => c.alive);
      // Find the one with the most laps for highlighting
      let bestIdx = -1, bestLaps = -1;
      for (let i = 0; i < s.cars.length; i++) {
        if (s.cars[i].alive && s.cars[i].laps >= bestLaps) {
          bestLaps = s.cars[i].laps; bestIdx = i;
        }
      }
      for (let i = 0; i < s.cars.length; i++) {
        const c = s.cars[i];
        if (!c.alive) continue;
        const color = i === bestIdx ? COL.carBest : COL.carAlive;
        drawCar(c, color, i === bestIdx ? 1.0 : 0.75);
      }
    }

    function updateStats(s) {
      if (!s) return;
      document.getElementById('car-gen').textContent   = s.generation;
      document.getElementById('car-alive').textContent = `${s.aliveCnt} / 20`;
      document.getElementById('car-genbest').textContent = s.genBestFit > 0 ? s.genBestFit.toFixed(2) : '—';
      document.getElementById('car-best').textContent    = s.bestFit > 0 ? s.bestFit.toFixed(2) : '—';
      drawChart(s.genHistory);
    }

    function drawChart(hist) {
      const svg = document.getElementById('car-chart');
      if (!svg || !hist || hist.length < 2) { if (svg) svg.innerHTML = ''; return; }
      const W = 260, H = 70, pad = 4;
      const vals = hist.map(h => h.best);
      const min = 0, max = Math.max(...vals, 0.1);
      const pts = vals.map((v, i) => {
        const x = pad + (i / (vals.length - 1)) * (W - 2 * pad);
        const y = H - pad - ((v - min) / (max - min)) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      svg.innerHTML =
        `<polyline points="${pts}" fill="none" stroke="#ffd600" stroke-width="1.5" stroke-linejoin="round"/>`;
    }

    // ── Simulation loop ──────────────────────────────────────────────────────

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const ticks = parseInt(document.getElementById('car-speed').value) || 2;
      try {
        const s = await nb.invoke('self-driving-car:step', ticks);
        if (s) { drawState(s); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await nb.invoke('self-driving-car:init', {});
        if (r && r.error) { console.error('[self-driving-car]', r.error); return; }
        _initialized = true;
      } else {
        await nb.invoke('self-driving-car:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      nb.invoke('self-driving-car:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function resetSim() {
      pauseSim();
      await nb.invoke('self-driving-car:reset');
      _initialized = true;
      _cachedTrack = null;
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);
      ['car-gen','car-alive','car-genbest','car-best'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = id === 'car-alive' ? '0 / 20' : id === 'car-gen' ? '0' : '—';
      });
      const svg = document.getElementById('car-chart');
      if (svg) svg.innerHTML = '';
    }

    async function newTrack() {
      const wasPaused = !_running;
      pauseSim();
      await nb.invoke('self-driving-car:newTrack', null);
      if (!_initialized) { _initialized = true; }
      if (!wasPaused) {
        await nb.invoke('self-driving-car:start');
        _running = true;
        _raf = requestAnimationFrame(tick);
      } else {
        const s = await nb.invoke('self-driving-car:getState');
        if (s) { drawState(s); updateStats(s); }
      }
    }

    // ── Button wiring ────────────────────────────────────────────────────────
    document.getElementById('car-start').addEventListener('click', startSim);
    document.getElementById('car-pause').addEventListener('click', pauseSim);
    document.getElementById('car-reset').addEventListener('click', resetSim);
    document.getElementById('car-newtrack').addEventListener('click', newTrack);

    const slider = document.getElementById('car-speed');
    slider.addEventListener('input', () => {
      document.getElementById('car-speed-val').textContent = slider.value;
    });

    // Restore state if already running
    nb.invoke('self-driving-car:getState').then(s => {
      if (s) { _initialized = true; drawState(s); updateStats(s); }
      else {
        ctx.fillStyle = COL.bg;
        ctx.fillRect(0, 0, CW, CH);
      }
    }).catch(() => {
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);
    });
  });

})(api);
