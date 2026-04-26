// Warehouse Robot plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  const GRID = 8;
  const CELL = 54;           // px per grid cell
  const CW   = GRID * CELL;  // canvas width/height = 432

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    bg:        '#111111',
    gridLine:  '#222222',
    target:    '#1e4d1e',
    targetRing:'#3a8a3a',
    box:       '#7c5522',
    boxStroke: '#5a3a10',
    boxOnTgt:  '#388e3c',
    boxOnStr:  '#1b5e20',
    robot:     '#1976d2',
    robotHi:   '#90caf9',
  };

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id: 'warehouse-robot',
    name: 'Warehouse Robot (Q-Learning)',
    kind: 'classifier',
    pluginKind: 'warehouse-robot',
    desc: 'A DQN agent learns to push 3 boxes onto their target squares through trial and error. No training data required — launch the simulation from the Infer tab.',
    arch: {
      kind: 'classifier', pluginKind: 'warehouse-robot',
      inputDim: 14, outputDim: 4,
      hidden: [128, 64], activation: 'relu', dropout: 0,
    },
    training: { optimizer: 'adam', learningRate: 0.001, batchSize: 64, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train editor (shown in Edit tab "Training data" section) ──────────────
  api.registerTrainEditor('warehouse-robot', function (root) {
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d2b0d;border:1px solid #2d5a2d;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#4caf50;margin-bottom:5px;">Q-Learning — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            This plugin uses a Deep Q-Network (DQN) that generates its own experience by interacting with
            an 8×8 grid environment. The agent explores randomly at first (ε = 1.0) and progressively
            shifts to a learned policy as epsilon decays to 0.05.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;line-height:1.6;">
          Architecture: <strong style="color:#aaa;">14 → [128, 64] → 4</strong><br>
          State: robot pos + 3 box positions + 3 target positions (normalized)<br>
          Actions: UP / DOWN / LEFT / RIGHT<br><br>
          Open the <strong style="color:#ccc;">Infer</strong> tab to launch the live simulation.
        </div>
      </div>
    `;
  });

  // ── Inference renderer — the full simulation ──────────────────────────────
  api.registerInferenceRenderer('warehouse-robot', function (root, network, nb) {
    let _raf       = null;
    let _running   = false;
    let _initialized = false;

    root.innerHTML = `
      <div class="panel" style="max-width:860px;">
        <h2>Warehouse Robot — Q-Learning Demo</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          DQN agent explores an 8×8 grid and learns to push all 3 boxes (brown) onto the target rings (green).
          Epsilon decays from 1.0 (random) → 0.05 (policy) over ~13 000 steps.
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
              <input id="wh-speed" type="range" min="1" max="50" value="10" style="width:80px;">
              <span id="wh-speed-val">10</span>
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
              <div style="font-size:11px;color:#777;margin-bottom:4px;">Boxes on target</div>
              <div id="wh-ontgt" style="font-size:28px;font-weight:700;color:#4caf50;">0 / 3</div>
            </div>

            <div class="section">
              <h3>Episode reward history</h3>
              <svg id="wh-chart" viewBox="0 0 260 70" preserveAspectRatio="none"
                style="width:100%;height:70px;display:block;background:#0d0d0d;border-radius:4px;"></svg>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.7;">
              <strong style="color:#666;">Legend</strong><br>
              <span style="color:#7c5522;">■</span> Box &nbsp;&nbsp;
              <span style="color:#388e3c;">■</span> Box on target &nbsp;&nbsp;
              <span style="color:#1976d2;">●</span> Robot<br>
              <span style="color:#3a8a3a;">◎</span> Target ring<br><br>
              Rewards: +10 per box placed · +50 all done<br>
              −0.5 wall hit · −0.4 blocked push · −0.01/step
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('wh-canvas');
    const ctx    = canvas.getContext('2d');

    // ── Drawing ─────────────────────────────────────────────────────────────

    function drawState(s) {
      ctx.clearRect(0, 0, CW, CW);

      // Grid background
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

      // Target rings
      for (const [tr, tc] of s.targets) {
        const cx = tc * CELL + CELL / 2, cy = tr * CELL + CELL / 2;
        ctx.fillStyle = COL.target;
        ctx.fillRect(tc * CELL, tr * CELL, CELL, CELL);
        ctx.strokeStyle = COL.targetRing;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        ctx.arc(cx, cy, CELL * 0.32, 0, Math.PI * 2);
        ctx.stroke();
        ctx.strokeStyle = COL.targetRing;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(cx - 7, cy); ctx.lineTo(cx + 7, cy);
        ctx.moveTo(cx, cy - 7); ctx.lineTo(cx, cy + 7);
        ctx.stroke();
      }

      // Boxes
      for (const [br, bc] of s.boxes) {
        const onTgt = s.targets.some(t => t[0] === br && t[1] === bc);
        const x = bc * CELL, y = br * CELL;
        const pad = 7;
        ctx.fillStyle   = onTgt ? COL.boxOnTgt  : COL.box;
        ctx.strokeStyle = onTgt ? COL.boxOnStr  : COL.boxStroke;
        ctx.lineWidth   = 1.5;
        ctx.fillRect  (x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
        ctx.strokeRect(x + pad, y + pad, CELL - pad * 2, CELL - pad * 2);
        // Cross decoration
        const mx = x + CELL / 2, my = y + CELL / 2;
        ctx.strokeStyle = onTgt ? '#a5d6a7' : '#ffcc80';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(mx - 5, my - 5); ctx.lineTo(mx + 5, my + 5);
        ctx.moveTo(mx + 5, my - 5); ctx.lineTo(mx - 5, my + 5);
        ctx.stroke();
      }

      // Robot
      const [rr, rc] = s.robot;
      const rx = rc * CELL + CELL / 2, ry = rr * CELL + CELL / 2;
      ctx.fillStyle = COL.robot;
      ctx.beginPath();
      ctx.arc(rx, ry, CELL * 0.27, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = COL.robotHi;
      ctx.beginPath();
      ctx.arc(rx, ry, 4, 0, Math.PI * 2);
      ctx.fill();
    }

    function updateStats(s) {
      document.getElementById('wh-ep').textContent    = s.episode;
      document.getElementById('wh-eps').textContent   = s.epsilon.toFixed(4);
      document.getElementById('wh-rew').textContent   = s.epReward.toFixed(2);
      document.getElementById('wh-best').textContent  = s.bestReward == null ? '—' : s.bestReward.toFixed(2);
      document.getElementById('wh-ontgt').textContent = `${s.onTarget} / ${GRID === 8 ? 3 : 3}`;
      drawChart(s.rewardHistory);
    }

    function drawChart(hist) {
      const svg = document.getElementById('wh-chart');
      if (!svg || !hist || hist.length < 2) { if (svg) svg.innerHTML = ''; return; }
      const W = 260, H = 70, pad = 4;
      const min = Math.min(...hist), max = Math.max(...hist);
      const range = max === min ? 1 : max - min;
      const pts = hist.map((v, i) => {
        const x = pad + (i / (hist.length - 1)) * (W - 2 * pad);
        const y = H - pad - ((v - min) / range) * (H - 2 * pad);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      }).join(' ');
      const zy = H - pad - ((0 - min) / range) * (H - 2 * pad);
      const clampedZy = Math.max(pad, Math.min(H - pad, zy)).toFixed(1);
      svg.innerHTML =
        `<line x1="${pad}" y1="${clampedZy}" x2="${W - pad}" y2="${clampedZy}" stroke="#222" stroke-width="1"/>` +
        `<polyline points="${pts}" fill="none" stroke="#4caf50" stroke-width="1.5" stroke-linejoin="round"/>`;
    }

    // ── Simulation loop ──────────────────────────────────────────────────────

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const stepsPerFrame = parseInt(document.getElementById('wh-speed').value) || 10;
      try {
        const s = await nb.invoke('warehouse-robot:step', stepsPerFrame);
        if (s) { drawState(s); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await nb.invoke('warehouse-robot:init', {});
        if (r && r.error) { console.error('[warehouse-robot]', r.error); return; }
        _initialized = true;
      } else {
        await nb.invoke('warehouse-robot:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      nb.invoke('warehouse-robot:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function resetSim() {
      pauseSim();
      await nb.invoke('warehouse-robot:reset');
      _initialized = true;
      drawState(null);
      ['wh-ep','wh-eps','wh-rew','wh-best'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = id === 'wh-eps' ? '1.0000' : id === 'wh-rew' ? '0.00' : id === 'wh-ep' ? '0' : '—';
      });
      document.getElementById('wh-ontgt').textContent = '0 / 3';
      const svg = document.getElementById('wh-chart');
      if (svg) svg.innerHTML = '';
    }

    // ── Button wiring ────────────────────────────────────────────────────────
    document.getElementById('wh-start').addEventListener('click', startSim);
    document.getElementById('wh-pause').addEventListener('click', pauseSim);
    document.getElementById('wh-reset').addEventListener('click', resetSim);

    const slider = document.getElementById('wh-speed');
    slider.addEventListener('input', () => {
      document.getElementById('wh-speed-val').textContent = slider.value;
    });

    // Restore live state if simulation was already running
    nb.invoke('warehouse-robot:getState').then(s => {
      if (s) { _initialized = true; drawState(s); updateStats(s); }
      else    { drawState(null); }
    }).catch(() => drawState(null));
  });

})(api);
