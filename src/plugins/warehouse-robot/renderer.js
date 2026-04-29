// Warehouse Robot plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const CANVAS_SIZE = 432;   // fixed canvas size; cell size is computed from grid at render time

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    bg:             '#111111',
    gridLine:       '#1e1e1e',
    obstacle:       '#1a1a1a',
    obstacleHatch:  '#2e2e2e',
    target:         '#0a2a0a',
    targetRing:     '#2d6a2d',
    targetDone:     '#0d3a0d',
    targetRingDone: '#4caf50',
    box:            '#7c5522',
    boxStroke:      '#5a3a10',
    boxCarried:     '#ff8f00',
    boxCarriedStr:  '#c65c00',
    robot:          '#1976d2',
    robotHi:        '#90caf9',
    robotInfer:     '#e91e63',
    robotInfHi:     '#f8bbd0',
  };

  // ── Shared drawing ────────────────────────────────────────────────────────

  function drawGridState(ctx, s, robotColor, robotHiColor) {
    const gridN = (s && s.grid) || 8;
    const CELL  = Math.floor(CANVAS_SIZE / gridN);

    ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);

    // Floor
    for (let r = 0; r < gridN; r++) {
      for (let c = 0; c < gridN; c++) {
        ctx.fillStyle = COL.bg;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        ctx.strokeStyle = COL.gridLine;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * CELL, r * CELL, CELL, CELL);
      }
    }

    if (!s) return;

    const hatch = Math.max(4, Math.round(CELL * 0.15));

    // Obstacles
    for (const [or, oc] of (s.obstacles || [])) {
      ctx.fillStyle = COL.obstacle;
      ctx.fillRect(oc * CELL, or * CELL, CELL, CELL);
      ctx.strokeStyle = '#252525';
      ctx.lineWidth = 1;
      ctx.strokeRect(oc * CELL + 1, or * CELL + 1, CELL - 2, CELL - 2);
      ctx.strokeStyle = COL.obstacleHatch;
      ctx.lineWidth = 2;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(oc * CELL + hatch, or * CELL + hatch);
      ctx.lineTo(oc * CELL + CELL - hatch, or * CELL + CELL - hatch);
      ctx.moveTo(oc * CELL + CELL - hatch, or * CELL + hatch);
      ctx.lineTo(oc * CELL + hatch, or * CELL + CELL - hatch);
      ctx.stroke();
      ctx.lineCap = 'butt';
    }

    const deliveredMask = s.deliveredMask || [];

    // Targets
    for (let i = 0; i < (s.targets || []).length; i++) {
      const [tr, tc] = s.targets[i];
      const done = deliveredMask[i];
      const cx = tc * CELL + CELL / 2, cy = tr * CELL + CELL / 2;
      ctx.fillStyle = done ? COL.targetDone : COL.target;
      ctx.fillRect(tc * CELL, tr * CELL, CELL, CELL);
      ctx.strokeStyle = done ? COL.targetRingDone : COL.targetRing;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.arc(cx, cy, CELL * 0.32, 0, Math.PI * 2);
      ctx.stroke();
      const tick = Math.max(4, Math.round(CELL * 0.14));
      if (done) {
        ctx.strokeStyle = COL.targetRingDone;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - tick, cy);
        ctx.lineTo(cx - Math.round(tick * 0.3), cy + tick * 0.9);
        ctx.lineTo(cx + tick * 1.1, cy - tick);
        ctx.stroke();
      } else {
        ctx.strokeStyle = COL.targetRing;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - tick, cy); ctx.lineTo(cx + tick, cy);
        ctx.moveTo(cx, cy - tick); ctx.lineTo(cx, cy + tick);
        ctx.stroke();
      }
    }

    // Undelivered, non-carried boxes
    for (let i = 0; i < (s.boxes || []).length; i++) {
      if (deliveredMask[i] || s.carrying === i) continue;
      const [br, bc] = s.boxes[i];
      const pad = Math.max(3, Math.round(CELL * 0.17));
      const x = bc * CELL, y = br * CELL;
      ctx.fillStyle   = COL.box;
      ctx.strokeStyle = COL.boxStroke;
      ctx.lineWidth   = 1.5;
      ctx.fillRect  (x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      ctx.strokeRect(x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      const mx = x + CELL / 2, my = y + CELL / 2;
      const cross = Math.max(3, Math.round(CELL * 0.09));
      ctx.strokeStyle = '#ffcc80';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx - cross, my - cross); ctx.lineTo(mx + cross, my + cross);
      ctx.moveTo(mx + cross, my - cross); ctx.lineTo(mx - cross, my + cross);
      ctx.stroke();
    }

    // Robot
    const [rr, rc] = s.robot;
    const rx = rc * CELL + CELL / 2, ry = rr * CELL + CELL / 2;
    ctx.fillStyle = robotColor;
    ctx.beginPath();
    ctx.arc(rx, ry, CELL * 0.27, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = robotHiColor;
    ctx.beginPath();
    ctx.arc(rx, ry, Math.max(2, Math.round(CELL * 0.07)), 0, Math.PI * 2);
    ctx.fill();

    // Carry indicator
    if (s.carrying >= 0) {
      const pad = Math.max(3, Math.round(CELL * 0.09));
      const sz  = Math.max(8, Math.round(CELL * 0.24));
      ctx.fillStyle   = COL.boxCarried;
      ctx.strokeStyle = COL.boxCarriedStr;
      ctx.lineWidth   = 1.5;
      ctx.fillRect  (rc * CELL + CELL - pad - sz, rr * CELL + pad, sz, sz);
      ctx.strokeRect(rc * CELL + CELL - pad - sz, rr * CELL + pad, sz, sz);
    }
  }

  function drawRewardChart(svgEl, hist) {
    if (!svgEl || !hist || hist.length < 2) { if (svgEl) svgEl.innerHTML = ''; return; }
    const W = 260, H = 70, pad = 4;
    const min = Math.min(...hist), max = Math.max(...hist);
    const range = max === min ? 1 : max - min;
    const pts = hist.map((v, i) => {
      const x = pad + (i / (hist.length - 1)) * (W - 2 * pad);
      const y = H - pad - ((v - min) / range) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');
    const zy = Math.max(pad, Math.min(H - pad,
      H - pad - ((0 - min) / range) * (H - 2 * pad))).toFixed(1);
    svgEl.innerHTML =
      `<line x1="${pad}" y1="${zy}" x2="${W - pad}" y2="${zy}" stroke="#222" stroke-width="1"/>` +
      `<polyline points="${pts}" fill="none" stroke="#4caf50" stroke-width="1.5" stroke-linejoin="round"/>`;
  }

  // ── Architecture fields ───────────────────────────────────────────────────
  api.registerArchFields('warehouse-robot', {
    fields: [
      { id: 'numBoxes',     label: 'Number of boxes', type: 'number',     default: 1,   min: 1, max: 5,  step: 1, hint: 'Boxes to pick up and deliver. Changing resets saved weights.' },
      { id: 'gridSize',     label: 'Grid size (n×n)', type: 'number',     default: 8,   min: 4, max: 16, step: 1, hint: 'Size of the square grid' },
      { id: 'numObstacles', label: 'Obstacles',       type: 'number',     default: 4,   min: 0, max: 20, step: 1, hint: 'Number of obstacle cells in the grid (seeded by env seed)' },
      { id: 'hidden',       label: 'Hidden layers',   type: 'layers',     default: [128, 64], hint: 'Network hidden layer sizes. Changing resets saved weights.' },
      { id: 'activation',   label: 'Activation',      type: 'activation', default: 'relu' },
    ],
    computeDims: (a) => ({ inputDim: 5 + (a.numBoxes || 1) * 6, outputDim: 4 }),
  });

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'warehouse-robot',
    name: 'Warehouse Robot (Q-Learning)',
    kind: 'classifier',
    pluginKind: 'warehouse-robot',
    desc: 'A DQN agent navigates an obstacle-filled grid, picks up boxes, and delivers them to target squares. Grid size, box count, and obstacle count are all configurable.',
    arch: {
      kind: 'classifier', pluginKind: 'warehouse-robot',
      inputDim: 11, outputDim: 4,
      hidden: [128, 64], activation: 'relu', dropout: 0,
      numBoxes: 1, gridSize: 8, numObstacles: 4,
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 64, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train settings ────────────────────────────────────────────────────────
  api.registerTrainSettings('warehouse-robot', {
    lr:        { label: 'Learning rate', hint: 'Adam learning rate for the DQN agent (default 0.001)' },
    bs:        { label: 'Batch size',    hint: 'Replay buffer sample size per training step (default 64)' },
    epochs:    { label: 'Max episodes',  hint: 'Training episode limit (0 = unlimited)' },
    seed:      { label: 'Env seed',      hint: 'Seed for obstacle layout — same seed = same obstacle grid' },
    workers:   { hidden: true },
    optimizer: { hidden: true },
    sectionHint: 'DQN hyperparameters — applied when the simulation starts.',
  });

  // ── Train editor ──────────────────────────────────────────────────────────
  api.registerTrainEditor('warehouse-robot', function (root, network) {
    const a    = (network && network.architecture) || {};
    const gN   = a.gridSize     || 8;
    const nB   = a.numBoxes     || 1;
    const nObs = a.numObstacles != null ? a.numObstacles : 4;
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#4caf50;margin-bottom:5px;">Q-Learning (DQN) — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            A Deep Q-Network learns to navigate a ${gN}×${gN} obstacle-filled grid, pick up
            ${nB} box${nB > 1 ? 'es' : ''}, and deliver ${nB > 1 ? 'them' : 'it'} to target squares.
            Agent starts exploring randomly (ε = 1.0) and shifts to learned policy as epsilon decays.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          Grid: ${gN}×${gN} · Boxes: ${nB} · Obstacles: ${nObs}<br>
          State: robot pos · carrying · box+target positions · relative offsets<br>
          Actions: UP / DOWN / LEFT / RIGHT<br>
          Rewards: +1 pick up · +10 deliver · +50 all done · −0.5 wall · −0.3 obstacle<br><br>
          Grid and box settings live in the <strong style="color:#ccc;">Editor</strong> tab → Architecture.<br>
          Obstacle layout is seeded (same env seed = same obstacles).
        </div>
      </div>
    `;
  });

  // ── Train renderer ────────────────────────────────────────────────────────
  api.registerTrainRenderer('warehouse-robot', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;

    const instanceId = (network && network.id) || 'wh-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    const t             = (network && network.training) || {};
    const cfgLR         = t.learningRate || 0.001;
    const cfgBS         = (t.batchSize | 0) || 64;
    const cfgSeed       = (t.seed || 42) >>> 0;
    const a             = (network && network.architecture) || {};
    const cfgHidden     = Array.isArray(a.hidden) && a.hidden.length ? a.hidden : [128, 64];
    const cfgActivation = a.activation   || 'relu';
    const cfgBoxes      = Math.max(1, Math.min(5,  (a.numBoxes     | 0) || 1));
    const cfgGridSize   = Math.max(4, Math.min(16, (a.gridSize     | 0) || 8));
    const cfgObstacles  = Math.max(0, a.numObstacles != null ? (a.numObstacles | 0) : 4);

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Q-Learning</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          DQN agent picks up ${cfgBoxes} box${cfgBoxes > 1 ? 'es' : ''} on a ${cfgGridSize}×${cfgGridSize} grid.
          LR: ${cfgLR} · Batch: ${cfgBS} · ε decays 1.0 → 0.05.
        </p>

        <div style="display:grid;grid-template-columns:${CANVAS_SIZE}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-canvas" width="${CANVAS_SIZE}" height="${CANVAS_SIZE}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#111;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;">
              <button class="btn primary" id="wh-start">Start</button>
              <button class="btn"         id="wh-pause">Pause</button>
              <button class="btn"         id="wh-reset">Reset agent</button>
            </div>
            <div style="margin-top:8px;display:flex;align-items:center;gap:8px;font-size:12px;color:#666;">
              Steps / frame:
              <input id="wh-speed" type="range" min="1" max="20" value="5" style="width:80px;">
              <span id="wh-speed-val">5</span>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Episode</div><div class="v" id="wh-ep">0</div></div>
              <div class="kpi"><div class="k">Epsilon ε</div><div class="v" id="wh-eps">1.0000</div></div>
              <div class="kpi"><div class="k">Ep reward</div><div class="v" id="wh-rew">0.00</div></div>
              <div class="kpi"><div class="k">Best reward</div><div class="v" id="wh-best">—</div></div>
            </div>

            <div style="background:#0d2b0d;border:1px solid #2d4a2d;border-radius:6px;padding:10px 14px;text-align:center;">
              <div style="font-size:11px;color:#777;margin-bottom:4px;">Boxes delivered</div>
              <div id="wh-ontgt" style="font-size:28px;font-weight:700;color:#4caf50;">0 / ${cfgBoxes}</div>
            </div>

            <div class="section">
              <h3>Episode reward history</h3>
              <svg id="wh-chart" viewBox="0 0 260 70" preserveAspectRatio="none"
                style="width:100%;height:70px;display:block;background:#0d0d0d;border-radius:4px;"></svg>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.7;">
              <strong style="color:#666;">Legend</strong><br>
              <span style="color:#1976d2;">●</span> Robot &nbsp;
              <span style="color:#ff8f00;">■</span> Carried box &nbsp;
              <span style="color:#7c5522;">■</span> Box (floor)<br>
              <span style="color:#2d6a2d;">◎</span> Target &nbsp;
              <span style="color:#4caf50;">✓</span> Delivered &nbsp;
              <span style="color:#2e2e2e;">✕</span> Obstacle<br><br>
              +1 pick up · +10 deliver · +50 all done<br>
              −0.5 wall · −0.3 obstacle · −0.01/step
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('wh-canvas');
    const ctx    = canvas.getContext('2d');

    function updateStats(s) {
      const nb = s.nBoxes || cfgBoxes;
      document.getElementById('wh-ep').textContent    = s.episode;
      document.getElementById('wh-eps').textContent   = s.epsilon.toFixed(4);
      document.getElementById('wh-rew').textContent   = s.epReward.toFixed(2);
      document.getElementById('wh-best').textContent  = s.bestReward == null ? '—' : s.bestReward.toFixed(2);
      document.getElementById('wh-ontgt').textContent = `${s.delivered} / ${nb}`;
      drawRewardChart(document.getElementById('wh-chart'), s.rewardHistory);
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const n = parseInt(document.getElementById('wh-speed').value) || 5;
      try {
        const s = await inv('warehouse-robot:step', { n });
        if (s) { drawGridState(ctx, s, COL.robot, COL.robotHi); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await inv('warehouse-robot:init', { lr: cfgLR, batchSize: cfgBS, seed: cfgSeed, nBoxes: cfgBoxes, hidden: cfgHidden, activation: cfgActivation, gridSize: cfgGridSize, numObstacles: cfgObstacles });
        if (r && r.error) { console.error('[warehouse-robot]', r.error); return; }
        _initialized = true;
      } else {
        await inv('warehouse-robot:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      inv('warehouse-robot:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function resetSim() {
      pauseSim();
      await inv('warehouse-robot:reset');
      _initialized = true;
      drawGridState(ctx, null, COL.robot, COL.robotHi);
      document.getElementById('wh-ep').textContent    = '0';
      document.getElementById('wh-eps').textContent   = '1.0000';
      document.getElementById('wh-rew').textContent   = '0.00';
      document.getElementById('wh-best').textContent  = '—';
      document.getElementById('wh-ontgt').textContent = `0 / ${cfgBoxes}`;
      const svg = document.getElementById('wh-chart');
      if (svg) svg.innerHTML = '';
    }

    document.getElementById('wh-start').addEventListener('click', startSim);
    document.getElementById('wh-pause').addEventListener('click', pauseSim);
    document.getElementById('wh-reset').addEventListener('click', resetSim);

    const slider = document.getElementById('wh-speed');
    slider.addEventListener('input', () => {
      document.getElementById('wh-speed-val').textContent = slider.value;
    });

    inv('warehouse-robot:getState').then(s => {
      if (s) { _initialized = true; drawGridState(ctx, s, COL.robot, COL.robotHi); updateStats(s); }
      else    { drawGridState(ctx, null, COL.robot, COL.robotHi); }
    }).catch(() => drawGridState(ctx, null, COL.robot, COL.robotHi));
  });

  // ── Inference renderer ────────────────────────────────────────────────────
  api.registerInferenceRenderer('warehouse-robot', function (root, network, nb) {
    let _raf      = null;
    let _running  = false;
    let _ready    = false;
    let _noiseStd = 0;
    let _nBoxes   = 1;

    const instanceId = (network && network.id) || 'wh-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Greedy Policy</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          The trained DQN runs greedy (ε = 0). Add state noise to stress-test robustness.
        </p>

        <div style="display:grid;grid-template-columns:${CANVAS_SIZE}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-i-canvas" width="${CANVAS_SIZE}" height="${CANVAS_SIZE}"
              style="display:block;border:1px solid #2a2a2a;border-radius:4px;background:#111;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;">
              <button class="btn primary" id="wh-i-start">Start</button>
              <button class="btn"         id="wh-i-pause">Pause</button>
              <button class="btn"         id="wh-i-rerun">New layout</button>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Trained eps</div><div class="v" id="wh-i-ep">—</div></div>
              <div class="kpi"><div class="k">Best reward</div><div class="v" id="wh-i-best">—</div></div>
              <div class="kpi"><div class="k">Episodes done</div><div class="v" id="wh-i-done">0</div></div>
              <div class="kpi"><div class="k">Ep reward</div><div class="v" id="wh-i-rew">0.00</div></div>
            </div>

            <div style="background:#0d2b0d;border:1px solid #2d4a2d;border-radius:6px;padding:10px 14px;text-align:center;">
              <div style="font-size:11px;color:#777;margin-bottom:4px;">Boxes delivered</div>
              <div id="wh-i-ontgt" style="font-size:28px;font-weight:700;color:#4caf50;">0 / —</div>
            </div>

            <div class="section">
              <h3>State noise</h3>
              <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa;">
                <span style="min-width:18px;">0</span>
                <input id="wh-i-noise" type="range" min="0" max="30" value="0" style="flex:1;">
                <span style="min-width:28px;">0.30</span>
                <span id="wh-i-noise-val" style="min-width:36px;text-align:right;color:#4caf50;">0.00</span>
              </div>
              <div style="font-size:11px;color:#555;margin-top:4px;">
                Gaussian noise std added to all state dimensions.
              </div>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.7;">
              <span style="color:#e91e63;">●</span> Robot (greedy)<br><br>
              The agent resets to a new random layout after each episode.<br>
              Obstacle positions are fixed (same env seed as training).<br>
              Switch to the <strong style="color:#666;">Train</strong> tab to keep training.
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('wh-i-canvas');
    const ctx    = canvas.getContext('2d');

    function showPlaceholder(msg) {
      drawGridState(ctx, null, COL.robotInfer, COL.robotInfHi);
      ctx.fillStyle    = '#444';
      ctx.font         = '13px monospace';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(msg, CANVAS_SIZE / 2, CANVAS_SIZE / 2);
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'alphabetic';
    }

    function updateInferStats(s) {
      if (!s) return;
      if (s.nBoxes) _nBoxes = s.nBoxes;
      document.getElementById('wh-i-ontgt').textContent = `${s.delivered} / ${_nBoxes}`;
      document.getElementById('wh-i-rew').textContent   = s.epReward.toFixed(2);
      document.getElementById('wh-i-done').textContent  = s.episodesDone;
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      try {
        const s = await inv('warehouse-robot:inferStep', { noiseStd: _noiseStd });
        if (s) { drawGridState(ctx, s, COL.robotInfer, COL.robotInfHi); updateInferStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startInfer() {
      if (!_ready) {
        const r = await inv('warehouse-robot:inferInit');
        if (!r || r.error) {
          showPlaceholder(r ? r.error : 'No trained agent — run the Train tab first.');
          return;
        }
        _nBoxes = r.nBoxes || 1;
        document.getElementById('wh-i-ep').textContent    = r.episode || '—';
        document.getElementById('wh-i-best').textContent  = r.bestReward != null ? r.bestReward.toFixed(2) : '—';
        document.getElementById('wh-i-ontgt').textContent = `0 / ${_nBoxes}`;
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
      document.getElementById('wh-i-rew').textContent   = '0.00';
      document.getElementById('wh-i-done').textContent  = '0';
      const r = await inv('warehouse-robot:inferInit');
      if (!r || r.error) {
        showPlaceholder(r ? r.error : 'No trained agent — run the Train tab first.');
        return;
      }
      _nBoxes = r.nBoxes || _nBoxes;
      document.getElementById('wh-i-ep').textContent    = r.episode || '—';
      document.getElementById('wh-i-best').textContent  = r.bestReward != null ? r.bestReward.toFixed(2) : '—';
      document.getElementById('wh-i-ontgt').textContent = `0 / ${_nBoxes}`;
      _ready = true;
      if (wasRunning) { _running = true; _raf = requestAnimationFrame(tick); }
    }

    document.getElementById('wh-i-start').addEventListener('click', startInfer);
    document.getElementById('wh-i-pause').addEventListener('click', pauseInfer);
    document.getElementById('wh-i-rerun').addEventListener('click', rerun);

    const noiseSlider = document.getElementById('wh-i-noise');
    noiseSlider.addEventListener('input', () => {
      _noiseStd = parseInt(noiseSlider.value) / 100;
      document.getElementById('wh-i-noise-val').textContent = _noiseStd.toFixed(2);
    });

    showPlaceholder('Press Start to view the greedy policy.');
  });

})(api);
