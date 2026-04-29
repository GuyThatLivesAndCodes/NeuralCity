// Self-Driving Car plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const CW = 700, CH = 530;

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    trackFill:  '#1c1c1c',
    trackEdge:  '#3a3a3a',
    roadCenter: '#2a2a2a',
    carAlive:   '#00e676',
    carDead:    '#662323',
    carBest:    '#ffd600',
    carInfer:   '#64b5f6',
    bg:         '#0d0d0d',
  };

  // ── Shared drawing helpers ────────────────────────────────────────────────

  function drawTrackOn(ctx, track) {
    if (!track) return;
    const { pts, halfWidth, N } = track;

    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const p = pts[i % N], q = pts[(i + 1) % N];
      const nx = -(q[1] - p[1]), ny = q[0] - p[0];
      const len = Math.hypot(nx, ny) || 1;
      const ox = p[0] + nx / len * halfWidth, oy = p[1] + ny / len * halfWidth;
      i === 0 ? ctx.moveTo(ox, oy) : ctx.lineTo(ox, oy);
    }
    ctx.closePath();
    ctx.moveTo(0, 0);
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

    ctx.strokeStyle = '#303030';
    ctx.lineWidth   = 1;
    ctx.setLineDash([12, 12]);
    ctx.beginPath();
    pts.forEach((p, i) => i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1]));
    ctx.closePath();
    ctx.stroke();
    ctx.setLineDash([]);

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

  function drawCarOn(ctx, car, color, alpha) {
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

  function drawFitnessChart(svgEl, genHistory) {
    if (!svgEl || !genHistory || genHistory.length < 2) { if (svgEl) svgEl.innerHTML = ''; return; }
    const W = 260, H = 70, pad = 4;
    const vals = genHistory.map(h => h.best);
    const max  = Math.max(...vals, 0.1);
    const pts  = vals.map((v, i) => {
      const x = pad + (i / (vals.length - 1)) * (W - 2 * pad);
      const y = H - pad - (v / max) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    svgEl.innerHTML =
      `<polyline points="${pts}" fill="none" stroke="#ffd600" stroke-width="1.5" stroke-linejoin="round"/>`;
  }

  // ── Architecture fields ───────────────────────────────────────────────────
  api.registerArchFields('self-driving-car', {
    fields: [
      { id: 'numRays',       label: 'Sensor rays',                type: 'number',     default: 9,     min: 3,  max: 25,  step: 1, hint: 'Number of ray-cast sensors spread across the field of view. Changing this resets saved weights.' },
      { id: 'sensorFov',     label: 'Sensor FOV (°)',             type: 'number',     default: 160,   min: 30, max: 360, step: 5, hint: 'Total angular field of view covered by sensors (symmetric around heading)' },
      { id: 'maxRayDist',    label: 'Max ray range (px)',         type: 'number',     default: 40,    min: 10, max: 200, step: 5, hint: 'Maximum distance a sensor ray travels before returning 1.0 (clear)' },
      { id: 'debugRaycasts', label: 'Visualise raycasts (Infer)', type: 'boolean',    default: false,                            hint: 'Draw each sensor ray on the Infer tab canvas' },
      { id: 'hidden',        label: 'Hidden layers',              type: 'layers',     default: [64, 32, 16],                    hint: 'Comma-separated hidden layer sizes. Changing these resets saved weights.' },
      { id: 'activation',    label: 'Activation',                 type: 'activation', default: 'tanh' },
    ],
    computeDims: (a) => ({ inputDim: (a.numRays || 9) + 2, outputDim: 2 }),
  });

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'self-driving-car',
    name: 'Self-Driving Car (Neuroevolution)',
    kind: 'classifier',
    pluginKind: 'self-driving-car',
    desc: 'A population of cars evolves to lap an auto-generated track without crashing. Uses Selective Reproduction (neuroevolution). No training data required.',
    arch: {
      kind: 'classifier', pluginKind: 'self-driving-car',
      inputDim: 11, outputDim: 2,
      hidden: [64, 32, 16], activation: 'tanh', dropout: 0,
      numRays: 9, sensorFov: 160, maxRayDist: 40, debugRaycasts: false,
    },
    training: { optimizer: 'adam', learningRate: 0.05, batchSize: 20, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train settings — relabel standard training fields ─────────────────────
  api.registerTrainSettings('self-driving-car', {
    lr:        { label: 'Mutation std',    hint: 'Weight mutation standard deviation — lower = finer search (default 0.05)' },
    bs:        { label: 'Population size', hint: 'Number of cars simulated per generation (default 20)' },
    epochs:    { label: 'Generations',     hint: 'Max generations to evolve (0 = unlimited)' },
    seed:      { label: 'Track seed',      hint: 'Seed for procedural track generation' },
    workers:   { hidden: true },
    optimizer: { hidden: true },
    sectionHint: 'Neuroevolution settings — applied when the simulation starts.',
  });

  // ── Train editor (training data section) ──────────────────────────────────
  api.registerTrainEditor('self-driving-car', function (root, network) {
    const a = (network && network.architecture) || {};
    const numRays = a.numRays || 9;
    const fov     = a.sensorFov || 160;
    const hidden  = (a.hidden || [64, 32, 16]).join(', ');
    const inputDim = numRays + 2;
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#1a1a00;border:1px solid #4a4a00;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#ffd600;margin-bottom:5px;">Neuroevolution — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            A population of neural-network-controlled cars races a procedurally generated track.
            Each generation the fittest survive, cross over, and mutate (Selective Reproduction).
            Configure population size and mutation std in the <strong style="color:#ccc;">Training settings</strong> above.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          Network: <strong style="color:#aaa;">${inputDim} → [${hidden}] → 2</strong><br>
          Inputs: ${numRays} sensor rays (±${fov/2}°) · normalised speed · heading<br>
          Outputs: steer (tanh) · throttle/brake (tanh)<br><br>
          Sensor and network settings live in the <strong style="color:#ccc;">Editor</strong> tab → Architecture.<br>
          Use the <strong style="color:#ccc;">Train</strong> tab to run the live evolution.<br>
          The <strong style="color:#ccc;">Infer</strong> tab shows the best evolved car with optional raycast overlay.
        </div>
      </div>
    `;
  });

  // ── Train renderer — full neuroevolution simulation ───────────────────────
  api.registerTrainRenderer('self-driving-car', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;
    let _cachedTrack = null;

    const instanceId = (network && network.id) || 'car-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    const t             = (network && network.training) || {};
    const cfgSeed       = (t.seed    || 42) >>> 0;
    const cfgPopSize    = (t.batchSize | 0) || 20;
    const cfgMutStd     = t.learningRate  || 0.05;
    const cfgGens       = (t.epochs  | 0) || 0;
    const a             = (network && network.architecture) || {};
    const cfgHidden     = Array.isArray(a.hidden) && a.hidden.length ? a.hidden : [64, 32, 16];
    const cfgActivation = a.activation   || 'tanh';
    const cfgNumRays    = (a.numRays    | 0) || 9;
    const cfgFov        = a.sensorFov    || 160;
    const cfgRayDist    = a.maxRayDist   || 40;
    const cfgDebugRays  = !!a.debugRaycasts;

    root.innerHTML = `
      <div class="panel" style="max-width:960px;">
        <h2>Self-Driving Car — Neuroevolution</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          Evolving ${cfgPopSize} cars on a seeded track.
          Mutation std: ${cfgMutStd} · ${cfgGens > 0 ? `Max ${cfgGens} generations` : 'Unlimited generations'}.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="car-canvas" width="${CW}" height="${CH}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#0d0d0d;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;">
              <button class="btn primary" id="car-start">Start</button>
              <button class="btn"         id="car-pause">Pause</button>
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
              <div class="kpi"><div class="k">Alive</div><div class="v" id="car-alive">0 / ${cfgPopSize}</div></div>
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
              <span style="color:#662323;">▲</span> Crashed<br><br>
              <strong style="color:#666;">Fitness</strong> = laps + (segments / track_N)<br>
              + speed bonus · × survival factor<br><br>
              Population: ${cfgPopSize} · Mutation std: ${cfgMutStd}<br>
              Sensors: ${cfgNumRays} rays (±${Math.round(cfgFov/2)}°) · max range ${cfgRayDist} px<br>
              Max episode: ${30 * 18} frames (~18 s)
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('car-canvas');
    const ctx    = canvas.getContext('2d');

    function drawState(s) {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);

      const track = (s && s.track) || _cachedTrack;
      if (!track) return;
      if (s && s.track) _cachedTrack = s.track;
      drawTrackOn(ctx, track);

      if (!s || !s.cars) return;

      for (const c of s.cars) {
        if (!c.alive) drawCarOn(ctx, c, COL.carDead, 0.35);
      }

      let bestIdx = -1, bestLaps = -1;
      for (let i = 0; i < s.cars.length; i++) {
        if (s.cars[i].alive && s.cars[i].laps >= bestLaps) {
          bestLaps = s.cars[i].laps; bestIdx = i;
        }
      }
      for (let i = 0; i < s.cars.length; i++) {
        const c = s.cars[i];
        if (!c.alive) continue;
        drawCarOn(ctx, c, i === bestIdx ? COL.carBest : COL.carAlive, i === bestIdx ? 1.0 : 0.75);
      }
    }

    function updateStats(s) {
      if (!s) return;
      const ps = s.popSize || cfgPopSize;
      document.getElementById('car-gen').textContent     = s.generation;
      document.getElementById('car-alive').textContent   = `${s.aliveCnt} / ${ps}`;
      document.getElementById('car-genbest').textContent = s.genBestFit > 0 ? s.genBestFit.toFixed(2) : '—';
      document.getElementById('car-best').textContent    = s.bestFit    > 0 ? s.bestFit.toFixed(2)    : '—';
      drawFitnessChart(document.getElementById('car-chart'), s.genHistory);
    }

    function readLiveSettings() {
      const lrEl = document.getElementById('t-lr');
      const bsEl = document.getElementById('t-bs');
      const epEl = document.getElementById('t-ep');
      const mutStd  = lrEl ? Math.max(0,   parseFloat(lrEl.value) || cfgMutStd)  : cfgMutStd;
      const popSize = bsEl ? Math.max(4,   parseInt(bsEl.value)   || cfgPopSize) : cfgPopSize;
      const maxGens = epEl ? Math.max(0,   parseInt(epEl.value)   || 0)          : cfgGens;
      return { mutStd, popSize, maxGens };
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const ticks = parseInt(document.getElementById('car-speed').value) || 2;
      const live  = readLiveSettings();
      try {
        const s = await inv('self-driving-car:step', { ticks, ...live });
        if (s) { drawState(s); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await inv('self-driving-car:init', {
          seed: cfgSeed, popSize: cfgPopSize, mutStd: cfgMutStd, generations: cfgGens,
          hidden: cfgHidden, activation: cfgActivation,
          numRays: cfgNumRays, sensorFov: cfgFov, maxRayDist: cfgRayDist,
          debugRaycasts: cfgDebugRays,
        });
        if (r && r.error) { console.error('[self-driving-car]', r.error); return; }
        _initialized = true;
      } else {
        await inv('self-driving-car:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      inv('self-driving-car:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function newTrack() {
      const wasPaused = !_running;
      pauseSim();
      await inv('self-driving-car:newTrack');
      if (!_initialized) _initialized = true;
      if (!wasPaused) {
        await inv('self-driving-car:start');
        _running = true;
        _raf = requestAnimationFrame(tick);
      } else {
        const s = await inv('self-driving-car:getState');
        if (s) { drawState(s); updateStats(s); }
      }
    }

    document.getElementById('car-start').addEventListener('click', startSim);
    document.getElementById('car-pause').addEventListener('click', pauseSim);
    document.getElementById('car-newtrack').addEventListener('click', newTrack);

    const slider = document.getElementById('car-speed');
    slider.addEventListener('input', () => {
      document.getElementById('car-speed-val').textContent = slider.value;
    });

    inv('self-driving-car:getState').then(s => {
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

  // ── Inference renderer — best evolved car + noise slider ──────────────────
  api.registerInferenceRenderer('self-driving-car', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _ready       = false;
    let _cachedTrack = null;
    let _noiseStd    = 0;

    const instanceId     = (network && network.id) || 'car-default';
    const inv            = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });
    const ia             = (network && network.architecture) || {};
    const cfgDebugRays   = !!ia.debugRaycasts;

    root.innerHTML = `
      <div class="panel" style="max-width:960px;">
        <h2>Self-Driving Car — Best Model</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          The best genome found so far drives the track. Add sensor noise to stress-test robustness.${cfgDebugRays ? ' Raycast overlay is <strong style="color:#ffd600;">ON</strong> — toggle in Editor → Architecture.' : ''}
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="car-i-canvas" width="${CW}" height="${CH}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#0d0d0d;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;">
              <button class="btn primary" id="car-i-start">Start</button>
              <button class="btn"         id="car-i-pause">Pause</button>
              <button class="btn"         id="car-i-rerun">Re-run</button>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Trained gens</div><div class="v" id="car-i-gen">—</div></div>
              <div class="kpi"><div class="k">Best fitness</div><div class="v" id="car-i-fit">—</div></div>
              <div class="kpi"><div class="k">Laps (this run)</div><div class="v" id="car-i-laps">0</div></div>
              <div class="kpi"><div class="k">Speed (px/s)</div><div class="v" id="car-i-spd">0</div></div>
            </div>

            <div class="section">
              <h3>Sensor noise</h3>
              <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa;">
                <span style="min-width:18px;">0</span>
                <input id="car-i-noise" type="range" min="0" max="40" value="0" style="flex:1;">
                <span style="min-width:28px;">0.40</span>
                <span id="car-i-noise-val" style="min-width:36px;text-align:right;color:#ffd600;">0.00</span>
              </div>
              <div style="font-size:11px;color:#555;margin-top:4px;">
                Gaussian noise std added to each sensor reading. Drag right to stress-test robustness.
              </div>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.8;">
              <span style="color:#64b5f6;">▲</span> Best evolved car<br><br>
              The car auto-resets when it crashes or times out.<br>
              Switch to the <strong style="color:#666;">Train</strong> tab to keep evolving.
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('car-i-canvas');
    const ctx    = canvas.getContext('2d');

    function showPlaceholder(msg) {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);
      ctx.fillStyle   = '#444';
      ctx.font        = '13px monospace';
      ctx.textAlign   = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(msg, CW / 2, CH / 2);
      ctx.textAlign   = 'left';
      ctx.textBaseline = 'alphabetic';
    }

    function drawInferState(s) {
      ctx.clearRect(0, 0, CW, CH);
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);

      const track = (s && s.track) || _cachedTrack;
      if (!track) return;
      if (s && s.track) _cachedTrack = s.track;
      drawTrackOn(ctx, track);

      if (!s || !s.car) return;

      // Raycast overlay (when debugRaycasts is enabled)
      if (s.rays && s.rays.length) {
        for (const ray of s.rays) {
          ctx.beginPath();
          ctx.moveTo(s.car.x, s.car.y);
          ctx.lineTo(ray.x, ray.y);
          ctx.strokeStyle = ray.hit ? '#ff572299' : '#00e67644';
          ctx.lineWidth = 1;
          ctx.stroke();
          if (ray.hit) {
            ctx.fillStyle = '#ff5722cc';
            ctx.beginPath();
            ctx.arc(ray.x, ray.y, 2.5, 0, Math.PI * 2);
            ctx.fill();
          }
        }
      }

      drawCarOn(ctx, s.car, COL.carInfer, 1.0);

      // Speed bar along bottom edge
      const barW = Math.round(CW * Math.min(1, (s.car.speed || 0) / 340));
      ctx.fillStyle = '#64b5f622';
      ctx.fillRect(0, CH - 4, barW, 4);
    }

    function updateInferStats(s) {
      if (!s || !s.car) return;
      document.getElementById('car-i-laps').textContent = s.car.laps;
      document.getElementById('car-i-spd').textContent  = Math.round(s.car.speed);
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      try {
        const s = await inv('self-driving-car:inferStep', { noiseStd: _noiseStd });
        if (s) { drawInferState(s); updateInferStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startInfer() {
      if (!_ready) {
        const r = await inv('self-driving-car:inferInit');
        if (!r || r.error) {
          showPlaceholder(r ? r.error : 'No trained model — run the Train tab first.');
          return;
        }
        document.getElementById('car-i-gen').textContent = r.generation || '—';
        document.getElementById('car-i-fit').textContent = r.bestFit != null ? r.bestFit.toFixed(2) : '—';
        if (r.track) _cachedTrack = r.track;
        _ready = true;
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseInfer() {
      _running = false;
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function rerun() {
      const wasRunning = _running;
      pauseInfer();
      _ready = false;
      document.getElementById('car-i-laps').textContent = '0';
      document.getElementById('car-i-spd').textContent  = '0';
      const r = await inv('self-driving-car:inferInit');
      if (!r || r.error) {
        showPlaceholder(r ? r.error : 'No trained model — run the Train tab first.');
        return;
      }
      document.getElementById('car-i-gen').textContent = r.generation || '—';
      document.getElementById('car-i-fit').textContent = r.bestFit != null ? r.bestFit.toFixed(2) : '—';
      if (r.track) _cachedTrack = r.track;
      _ready = true;
      if (wasRunning) {
        _running = true;
        _raf = requestAnimationFrame(tick);
      }
    }

    document.getElementById('car-i-start').addEventListener('click', startInfer);
    document.getElementById('car-i-pause').addEventListener('click', pauseInfer);
    document.getElementById('car-i-rerun').addEventListener('click', rerun);

    const noiseSlider = document.getElementById('car-i-noise');
    noiseSlider.addEventListener('input', () => {
      _noiseStd = parseInt(noiseSlider.value) / 100;
      document.getElementById('car-i-noise-val').textContent = _noiseStd.toFixed(2);
    });

    showPlaceholder('Press Start to view the best evolved model.');
  });

})(api);
