// Warehouse Robot plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const GRID = 8;
  const CELL = 54;
  const CW   = GRID * CELL;  // 432

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
    ctx.clearRect(0, 0, CW, CW);

    // Floor
    for (let r = 0; r < GRID; r++) {
      for (let c = 0; c < GRID; c++) {
        ctx.fillStyle = COL.bg;
        ctx.fillRect(c * CELL, r * CELL, CELL, CELL);
        ctx.strokeStyle = COL.gridLine;
        ctx.lineWidth = 0.5;
        ctx.strokeRect(c * CELL, r * CELL, CELL, CELL);
      }
    }

    if (!s) return;

    // Obstacles
    for (const [or, oc] of (s.obstacles || [])) {
      ctx.fillStyle = COL.obstacle;
      ctx.fillRect(oc * CELL, or * CELL, CELL, CELL);
      ctx.strokeStyle = '#252525';
      ctx.lineWidth = 1;
      ctx.strokeRect(oc * CELL + 1, or * CELL + 1, CELL - 2, CELL - 2);
      ctx.strokeStyle = COL.obstacleHatch;
      ctx.lineWidth = 2.5;
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(oc * CELL + 8, or * CELL + 8);
      ctx.lineTo(oc * CELL + CELL - 8, or * CELL + CELL - 8);
      ctx.moveTo(oc * CELL + CELL - 8, or * CELL + 8);
      ctx.lineTo(oc * CELL + 8, or * CELL + CELL - 8);
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
      if (done) {
        ctx.strokeStyle = COL.targetRingDone;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(cx - 7, cy);
        ctx.lineTo(cx - 2, cy + 6);
        ctx.lineTo(cx + 8, cy - 7);
        ctx.stroke();
      } else {
        ctx.strokeStyle = COL.targetRing;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
        ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
        ctx.stroke();
      }
    }

    // Undelivered, non-carried boxes
    for (let i = 0; i < (s.boxes || []).length; i++) {
      if (deliveredMask[i] || s.carrying === i) continue;
      const [br, bc] = s.boxes[i];
      const x = bc * CELL, y = br * CELL, pad = 9;
      ctx.fillStyle   = COL.box;
      ctx.strokeStyle = COL.boxStroke;
      ctx.lineWidth   = 1.5;
      ctx.fillRect  (x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      ctx.strokeRect(x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
      const mx = x + CELL / 2, my = y + CELL / 2;
      ctx.strokeStyle = '#ffcc80';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5);
      ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5);
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
    ctx.arc(rx, ry, 4, 0, Math.PI * 2);
    ctx.fill();

    // Carry indicator — small box in top-right corner of robot cell
    if (s.carrying >= 0) {
      const pad = 5, sz = 13;
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

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'warehouse-robot',
    name: 'Warehouse Robot (Q-Learning)',
    kind: 'classifier',
    pluginKind: 'warehouse-robot',
    desc: 'A DQN agent navigates an obstacle-filled 8×8 grid, picks up boxes, and delivers them to target squares. Set box count in Training settings.',
    arch: {
      kind: 'classifier', pluginKind: 'warehouse-robot',
      inputDim: 11, outputDim: 4,   // default 1 box: 5 + 1×6 = 11
      hidden: [128, 64], activation: 'relu', dropout: 0,
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 64, epochs: 0, seed: 42, workers: 1 },
    trainingData: {},
  });

  // ── Train settings ────────────────────────────────────────────────────────
  api.registerTrainSettings('warehouse-robot', {
    lr:        { label: 'Learning rate', hint: 'Adam learning rate for the DQN agent (default 0.001)' },
    bs:        { label: 'Batch size',    hint: 'Replay buffer sample size per training step (default 64)' },
    epochs:    { label: 'Max episodes',  hint: 'Training episode limit (0 = unlimited)' },
    seed:      { label: 'Env seed',      hint: 'Seed for obstacle layout — same seed = same obstacle grid' },
    workers:   { label: 'Box count',     hint: 'Number of boxes to pick up and deliver (1–5, default 1)' },
    optimizer: { hidden: true },
    sectionHint: 'DQN hyperparameters — applied when the simulation starts.',
  });

  // ── Train editor ──────────────────────────────────────────────────────────
  api.registerTrainEditor('warehouse-robot', function (root) {
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#4caf50;margin-bottom:5px;">Q-Learning — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            A Deep Q-Network learns to navigate an obstacle-filled grid, pick up boxes, and deliver
            them to target squares. The agent starts exploring randomly (ε = 1.0) and shifts to a
            learned policy as epsilon decays to 0.05.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          State: robot pos · carrying flag · box positions · target positions · relative offsets<br>
          Actions: UP / DOWN / LEFT / RIGHT<br>
          Rewards: +1 pick up · +10 deliver · +50 all done<br>
          &minus;0.5 wall · &minus;0.3 obstacle · &minus;0.01/step<br><br>
          Obstacle layout is fixed per session (env seed). Box and target positions<br>
          are randomised each episode. Set <strong style="color:#ccc;">Box count</strong> in Training settings.
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

    const t        = (network && network.training) || {};
    const cfgLR    = t.learningRate || 0.001;
    const cfgBS    = (t.batchSize | 0) || 64;
    const cfgSeed  = (t.seed || 42) >>> 0;
    const cfgBoxes = Math.max(1, Math.min(5, (t.workers | 0) || 1));

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Q-Learning</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          DQN agent picks up ${cfgBoxes} box${cfgBoxes > 1 ? 'es' : ''} and delivers
          ${cfgBoxes > 1 ? 'them' : 'it'} to target squares on an obstacle-filled 8×8 grid.
          LR: ${cfgLR} · Batch: ${cfgBS} · ε decays 1.0 → 0.05.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-canvas" width="${CW}" height="${CW}"
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
        const r = await inv('warehouse-robot:init', { lr: cfgLR, batchSize: cfgBS, seed: cfgSeed, nBoxes: cfgBoxes });
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

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="wh-i-canvas" width="${CW}" height="${CW}"
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
      ctx.fillText(msg, CW / 2, CW / 2);
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
