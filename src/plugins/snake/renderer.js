// Snake Neuroevolution plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  // ── Grid / canvas constants ───────────────────────────────────────────────
  const GRID_W = 15;
  const GRID_H = 17;
  const POP_SIZE = 20;
  const GEN_DURATION_MS = 5000;

  // Training canvas: 5-column × 4-row grid of mini snake arenas
  const CW = 700, CH = 660;
  const COLS = 5, ROWS = 4;
  const MINI_CELL  = 8;
  const MINI_GW    = GRID_W * MINI_CELL;   // 120
  const MINI_GH    = GRID_H * MINI_CELL;   // 136
  const LABEL_H    = 16;
  const GAP_X      = 8;
  const GAP_Y      = 8;
  const SLOT_W     = MINI_GW + GAP_X;      // 128
  const SLOT_H     = MINI_GH + LABEL_H + GAP_Y; // 160
  const OFF_X      = Math.floor((CW - COLS * SLOT_W) / 2);  // 30
  const OFF_Y      = Math.floor((CH - ROWS * SLOT_H) / 2);  // 10

  // Inference canvas: single large snake
  const ICW = 400, ICH = 430;
  const INFER_CELL = 22;
  const IGW = GRID_W * INFER_CELL;   // 330
  const IGH = GRID_H * INFER_CELL;   // 374
  const I_OFF_X = Math.floor((ICW - IGW) / 2); // 35
  const I_OFF_Y = Math.floor((ICH - IGH) / 2); // 28

  // ── Palette ───────────────────────────────────────────────────────────────
  const COL = {
    bg:           '#0d0d0d',
    gridBg:       '#121212',
    gridBorder:   '#222',
    gridLine:     '#1a1a1a',
    headAlive:    '#00e676',
    bodyAlive:    '#00897b',
    headDead:     '#444',
    bodyDead:     '#2a2a2a',
    apple:        '#ff1744',
    appleDead:    '#5a1422',
    bestBorder:   '#ffd600',
    labelAlive:   '#888',
    labelDead:    '#3a3a3a',
    labelBest:    '#ffd600',
    inferHead:    '#64b5f6',
    inferBody:    '#1565c0',
    inferApple:   '#ff1744',
    chartLine:    '#ffd600',
    chartMean:    '#444',
    timerOk:      '#00e676',
    timerLow:     '#ffd600',
  };

  // ── Drawing helpers ───────────────────────────────────────────────────────

  // Draw a single mini snake arena at pixel offset (x0, y0)
  function drawMiniSnake(ctx, snake, x0, y0, isBest) {
    const alive  = snake.alive;
    const cell   = MINI_CELL;

    // Arena background
    ctx.fillStyle = COL.gridBg;
    ctx.fillRect(x0, y0, MINI_GW, MINI_GH);

    // Optional subtle grid lines
    ctx.strokeStyle = COL.gridLine;
    ctx.lineWidth   = 0.5;
    for (let gx = 0; gx <= GRID_W; gx++) {
      ctx.beginPath();
      ctx.moveTo(x0 + gx * cell, y0);
      ctx.lineTo(x0 + gx * cell, y0 + MINI_GH);
      ctx.stroke();
    }
    for (let gy = 0; gy <= GRID_H; gy++) {
      ctx.beginPath();
      ctx.moveTo(x0,           y0 + gy * cell);
      ctx.lineTo(x0 + MINI_GW, y0 + gy * cell);
      ctx.stroke();
    }

    // Apple
    const ax = x0 + snake.apple.x * cell + 1;
    const ay = y0 + snake.apple.y * cell + 1;
    ctx.fillStyle = alive ? COL.apple : COL.appleDead;
    ctx.fillRect(ax, ay, cell - 2, cell - 2);

    // Snake body (tail → head so head draws on top)
    for (let i = snake.body.length - 1; i >= 0; i--) {
      const seg = snake.body[i];
      const px  = x0 + seg.x * cell;
      const py  = y0 + seg.y * cell;
      if (i === 0) {
        // Head
        ctx.fillStyle = alive ? COL.headAlive : COL.headDead;
        ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        // Eyes
        if (alive) {
          ctx.fillStyle = '#000';
          const eyeSize = 2;
          ctx.fillRect(px + 2, py + 2, eyeSize, eyeSize);
          ctx.fillRect(px + cell - 4, py + 2, eyeSize, eyeSize);
        }
      } else {
        ctx.fillStyle = alive ? COL.bodyAlive : COL.bodyDead;
        ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      }
    }

    // Border — gold for best, dim for others
    ctx.strokeStyle = isBest ? COL.bestBorder : COL.gridBorder;
    ctx.lineWidth   = isBest ? 2 : 1;
    ctx.strokeRect(x0, y0, MINI_GW, MINI_GH);

    // Label: apple count beneath arena
    ctx.fillStyle = isBest ? COL.labelBest : alive ? COL.labelAlive : COL.labelDead;
    ctx.font      = `${isBest ? '600 ' : ''}10px monospace`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(
      `🍎 ${snake.applesEaten}${alive ? '' : ' ✗'}`,
      x0 + MINI_GW / 2,
      y0 + MINI_GH + LABEL_H / 2
    );
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  function drawAllSnakes(ctx, snakes) {
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, CW, CH);

    if (!snakes || snakes.length === 0) return;

    // Find the best alive (or best overall) snake by apples eaten
    let bestIdx = 0;
    for (let i = 1; i < snakes.length; i++) {
      const cur  = snakes[i].applesEaten;
      const prev = snakes[bestIdx].applesEaten;
      if (cur > prev || (cur === prev && snakes[i].alive && !snakes[bestIdx].alive)) {
        bestIdx = i;
      }
    }

    for (let i = 0; i < snakes.length; i++) {
      const col  = i % COLS;
      const row  = Math.floor(i / COLS);
      const x0   = OFF_X + col * SLOT_W;
      const y0   = OFF_Y + row * SLOT_H;
      drawMiniSnake(ctx, snakes[i], x0, y0, i === bestIdx);
    }
  }

  // Draw a single large snake for the inference view
  function drawInferSnake(ctx, snake) {
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, ICW, ICH);

    // Arena background
    ctx.fillStyle = '#111';
    ctx.fillRect(I_OFF_X, I_OFF_Y, IGW, IGH);

    const cell = INFER_CELL;

    // Grid lines
    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth   = 0.5;
    for (let gx = 0; gx <= GRID_W; gx++) {
      ctx.beginPath();
      ctx.moveTo(I_OFF_X + gx * cell, I_OFF_Y);
      ctx.lineTo(I_OFF_X + gx * cell, I_OFF_Y + IGH);
      ctx.stroke();
    }
    for (let gy = 0; gy <= GRID_H; gy++) {
      ctx.beginPath();
      ctx.moveTo(I_OFF_X,        I_OFF_Y + gy * cell);
      ctx.lineTo(I_OFF_X + IGW,  I_OFF_Y + gy * cell);
      ctx.stroke();
    }

    // Apple
    ctx.fillStyle = COL.inferApple;
    ctx.beginPath();
    const acx = I_OFF_X + (snake.apple.x + 0.5) * cell;
    const acy = I_OFF_Y + (snake.apple.y + 0.5) * cell;
    ctx.arc(acx, acy, cell * 0.38, 0, Math.PI * 2);
    ctx.fill();

    // Snake body
    for (let i = snake.body.length - 1; i >= 0; i--) {
      const seg = snake.body[i];
      const px  = I_OFF_X + seg.x * cell;
      const py  = I_OFF_Y + seg.y * cell;
      const r   = 3;
      if (i === 0) {
        ctx.fillStyle = COL.inferHead;
      } else {
        const fade = Math.max(0.25, 1 - i / snake.body.length);
        ctx.fillStyle = `rgba(21,101,192,${fade})`;
      }
      // Rounded rectangle segments
      ctx.beginPath();
      ctx.roundRect(px + 1, py + 1, cell - 2, cell - 2, r);
      ctx.fill();

      // Eyes on head
      if (i === 0) {
        ctx.fillStyle = '#0a2540';
        const eyeR = 3;
        ctx.beginPath(); ctx.arc(px + cell * 0.28, py + cell * 0.3, eyeR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(px + cell * 0.72, py + cell * 0.3, eyeR, 0, Math.PI * 2); ctx.fill();
      }
    }

    // Border
    ctx.strokeStyle = '#333';
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(I_OFF_X, I_OFF_Y, IGW, IGH);
  }

  // Fitness-over-generations SVG chart
  function drawFitnessChart(svgEl, genHistory) {
    if (!svgEl || !genHistory || genHistory.length < 2) {
      if (svgEl) svgEl.innerHTML = '';
      return;
    }
    const W = 260, H = 60, pad = 4;
    const maxVal = Math.max(...genHistory.map(h => h.best), 0.1);

    const bestPts = genHistory.map((h, i) => {
      const x = pad + (i / (genHistory.length - 1)) * (W - 2 * pad);
      const y = H - pad - (h.best / maxVal) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    const meanPts = genHistory.map((h, i) => {
      const x = pad + (i / (genHistory.length - 1)) * (W - 2 * pad);
      const y = H - pad - (h.mean / maxVal) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    svgEl.innerHTML = `
      <polyline points="${meanPts}" fill="none" stroke="#333" stroke-width="1" stroke-linejoin="round"/>
      <polyline points="${bestPts}" fill="none" stroke="${COL.chartLine}" stroke-width="1.5" stroke-linejoin="round"/>
      <text x="${W - pad}" y="${H - pad - 1}" text-anchor="end" font-size="8" fill="#555">${maxVal.toFixed(1)}</text>
    `;
  }

  // ── Template registration ─────────────────────────────────────────────────
  api.registerTemplate({
    id:         'snake-neuro',
    name:       'Snake AI (Neuroevolution)',
    kind:       'classifier',
    pluginKind: 'snake-neuro',
    desc:       'A population of 20 neural-net snakes evolves on a 15×17 grid. Each generation runs for 5 seconds; the best scorer reproduces. Fitness = apples eaten (−10% if died).',
    arch: {
      kind:       'classifier',
      pluginKind: 'snake-neuro',
      inputDim:   255,
      outputDim:  4,
      hidden:     [255, 128, 64, 32],
      activation: 'tanh',
      dropout:    0,
    },
    training: { optimizer: 'adam', learningRate: 0.08, batchSize: 20, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train settings ────────────────────────────────────────────────────────
  api.registerTrainSettings('snake-neuro', {
    lr:        { label: 'Mutation std',    hint: 'Weight mutation std — lower = finer search (default 0.08)' },
    bs:        { label: 'Population size', hint: 'Always 20 snakes (fixed for this plugin)' },
    epochs:    { label: 'Generations',     hint: 'Max generations to evolve (0 = run indefinitely)' },
    seed:      { label: 'RNG seed',        hint: 'Seed for network initialisation' },
    workers:   { hidden: true },
    optimizer: { hidden: true },
    sectionHint: 'Neuroevolution settings — applied when the simulation starts.',
  });

  // ── Train editor (data section) ───────────────────────────────────────────
  api.registerTrainEditor('snake-neuro', function (root) {
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d1a0d;border:1px solid #1a4a1a;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#00e676;margin-bottom:5px;">Snake Neuroevolution — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            20 snakes play simultaneously on independent 15×17 grids.
            After <strong style="color:#ccc;">5 seconds</strong>, the highest-scoring snake reproduces
            into the next generation via mutation. No crossover — pure selective reproduction.
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #222;border-radius:4px;padding:8px 12px;line-height:1.8;">
          <strong style="color:#aaa;">Architecture:</strong> 255 → [255, 128, 64, 32] → 4<br>
          <strong style="color:#aaa;">Inputs:</strong> full 15×17 grid (head=1.0, body=0.5, apple=−1.0, empty=0.0)<br>
          <strong style="color:#aaa;">Outputs:</strong> ↑ ↓ ← → direction logits (argmax)<br>
          <strong style="color:#aaa;">Fitness:</strong> apples eaten × (died ? 0.9 : 1.0)<br>
          <strong style="color:#aaa;">Stale limit:</strong> snake dies if no apple eaten in 200 moves<br><br>
          Use the <strong style="color:#ccc;">Train</strong> tab to run live evolution.<br>
          Use the <strong style="color:#ccc;">Infer</strong> tab to watch the best genome play.
        </div>
      </div>
    `;
  });

  // ── Train renderer ────────────────────────────────────────────────────────
  api.registerTrainRenderer('snake-neuro', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;

    const instanceId = (network && network.id) || 'snake-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    const t          = (network && network.training) || {};
    const cfgMutStd  = t.learningRate || 0.08;
    const cfgSeed    = (t.seed | 0) || 42;

    root.innerHTML = `
      <div class="panel" style="max-width:1080px;">
        <h2>Snake AI — Neuroevolution</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          20 neural-net snakes on 15×17 grids. Each generation lasts 5 seconds — the best apple-collector reproduces.
        </p>

        <div style="display:grid;grid-template-columns:${CW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="snake-canvas" width="${CW}" height="${CH}"
              style="display:block;border:1px solid #222;border-radius:4px;background:#0d0d0d;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;flex-wrap:wrap;">
              <button class="btn primary" id="sn-start">Start</button>
              <button class="btn"         id="sn-pause">Pause</button>
              <button class="btn"         id="sn-reset">New evolution</button>
            </div>
            <div style="margin-top:8px;display:flex;align-items:center;gap:8px;font-size:12px;color:#666;">
              Ticks / frame:
              <input id="sn-speed" type="range" min="1" max="15" value="5" style="width:90px;">
              <span id="sn-speed-val">5</span>
              <span style="color:#444;margin-left:8px;">· More = faster snake decisions per second</span>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Generation</div><div class="v" id="sn-gen">0</div></div>
              <div class="kpi"><div class="k">Alive</div><div class="v" id="sn-alive">0 / 20</div></div>
              <div class="kpi"><div class="k">Gen best 🍎</div><div class="v" id="sn-genbest">—</div></div>
              <div class="kpi"><div class="k">All-time best 🍎</div><div class="v" id="sn-best">—</div></div>
            </div>

            <div class="kpis" style="grid-template-columns:1fr;">
              <div class="kpi">
                <div class="k">Time left this generation</div>
                <div class="v" id="sn-timer" style="font-size:20px;">5.0s</div>
                <div style="margin-top:6px;height:4px;background:#1a1a1a;border-radius:2px;">
                  <div id="sn-timer-bar" style="height:4px;background:#00e676;border-radius:2px;width:100%;transition:width 0.1s linear;"></div>
                </div>
              </div>
            </div>

            <div class="section">
              <h3>Best apples per generation</h3>
              <svg id="sn-chart" viewBox="0 0 260 60" preserveAspectRatio="none"
                style="width:100%;height:60px;display:block;background:#0d0d0d;border-radius:4px;"></svg>
              <div style="font-size:10px;color:#333;margin-top:3px;">— mean &nbsp;&nbsp; <span style="color:${COL.chartLine}">— best</span></div>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.9;">
              <strong style="color:#555;">Legend</strong><br>
              <span style="color:${COL.headAlive};">■</span> Head &nbsp;
              <span style="color:${COL.bodyAlive};">■</span> Body &nbsp;
              <span style="color:${COL.apple};">■</span> Apple &nbsp;
              <span style="color:${COL.bestBorder};">▣</span> Best this gen<br>
              <span style="color:#444;">■</span> Dead snake<br><br>
              <strong style="color:#555;">Fitness</strong> = apples eaten &nbsp;×&nbsp; (died ? 0.9 : 1.0)<br>
              <strong style="color:#555;">Stale kill</strong>: no apple in 200 moves → death<br>
              <strong style="color:#555;">Network</strong>: 255 → [255, 128, 64, 32] → 4 &nbsp;(tanh)<br>
              <strong style="color:#555;">Mutation std</strong>: ${cfgMutStd} &nbsp;·&nbsp;
              <strong style="color:#555;">Seed</strong>: ${cfgSeed}
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('snake-canvas');
    const ctx    = canvas.getContext('2d');

    function updateStats(s) {
      if (!s) return;
      document.getElementById('sn-gen').textContent     = s.generation;
      document.getElementById('sn-alive').textContent   = `${s.aliveCnt} / 20`;
      document.getElementById('sn-genbest').textContent = s.genBestFit > 0 ? s.genBestFit.toFixed(1) : '—';
      document.getElementById('sn-best').textContent    = s.bestFit    > 0 ? s.bestFit.toFixed(1)    : '—';

      // Timer
      const pct  = Math.max(0, Math.min(1, s.timeLeft / GEN_DURATION_MS));
      const secs = (s.timeLeft / 1000).toFixed(1);
      const timerEl = document.getElementById('sn-timer');
      const barEl   = document.getElementById('sn-timer-bar');
      if (timerEl) {
        timerEl.textContent = `${secs}s`;
        timerEl.style.color = pct > 0.4 ? COL.timerOk : COL.timerLow;
      }
      if (barEl) {
        barEl.style.width      = `${pct * 100}%`;
        barEl.style.background = pct > 0.4 ? COL.timerOk : COL.timerLow;
      }

      drawFitnessChart(document.getElementById('sn-chart'), s.genHistory);
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const ticks = parseInt(document.getElementById('sn-speed').value, 10) || 5;
      try {
        const s = await inv('snake-neuro:step', { ticks });
        if (s) { drawAllSnakes(ctx, s.snakes); updateStats(s); }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await inv('snake-neuro:init', { seed: cfgSeed, mutStd: cfgMutStd });
        if (r && r.error) { console.error('[snake-neuro]', r.error); return; }
        _initialized = true;
      } else {
        await inv('snake-neuro:start');
      }
      if (_running) return;
      _running = true;
      _raf = requestAnimationFrame(tick);
    }

    function pauseSim() {
      _running = false;
      inv('snake-neuro:stop');
      if (_raf) { cancelAnimationFrame(_raf); _raf = null; }
    }

    async function resetSim() {
      pauseSim();
      await inv('snake-neuro:reset');
      _initialized = true;
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);
      ['sn-gen', 'sn-alive', 'sn-genbest', 'sn-best'].forEach(id => {
        const el = document.getElementById(id);
        if (!el) return;
        el.textContent = id === 'sn-alive' ? '0 / 20' : id === 'sn-gen' ? '0' : '—';
      });
      const timerEl = document.getElementById('sn-timer');
      const barEl   = document.getElementById('sn-timer-bar');
      if (timerEl) { timerEl.textContent = '5.0s'; timerEl.style.color = COL.timerOk; }
      if (barEl)   { barEl.style.width = '100%'; barEl.style.background = COL.timerOk; }
      const svg = document.getElementById('sn-chart');
      if (svg) svg.innerHTML = '';
    }

    document.getElementById('sn-start').addEventListener('click', startSim);
    document.getElementById('sn-pause').addEventListener('click', pauseSim);
    document.getElementById('sn-reset').addEventListener('click', resetSim);

    const speedSlider = document.getElementById('sn-speed');
    speedSlider.addEventListener('input', () => {
      document.getElementById('sn-speed-val').textContent = speedSlider.value;
    });

    // Try to restore existing state
    inv('snake-neuro:getState').then(s => {
      if (s && s.snakes && s.snakes.length > 0) {
        _initialized = true;
        drawAllSnakes(ctx, s.snakes);
        updateStats(s);
      } else {
        ctx.fillStyle = COL.bg;
        ctx.fillRect(0, 0, CW, CH);
        ctx.fillStyle    = '#333';
        ctx.font         = '13px monospace';
        ctx.textAlign    = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText('Press Start to begin evolution', CW / 2, CH / 2);
        ctx.textAlign    = 'left';
        ctx.textBaseline = 'alphabetic';
      }
    }).catch(() => {
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, CW, CH);
    });
  });

  // ── Inference renderer ────────────────────────────────────────────────────
  api.registerInferenceRenderer('snake-neuro', function (root, network, nb) {
    let _raf     = null;
    let _running = false;
    let _ready   = false;
    let _totalApplesThisRun = 0;
    let _runs    = 0;
    let _inferSpeed = 1; // ticks between renders (via setInterval trick)

    const instanceId = (network && network.id) || 'snake-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    root.innerHTML = `
      <div class="panel" style="max-width:960px;">
        <h2>Snake AI — Best Evolved Model</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          The best genome found so far plays snake. It auto-resets on death.
        </p>

        <div style="display:grid;grid-template-columns:${ICW}px 1fr;gap:20px;align-items:start;">

          <div>
            <canvas id="sn-i-canvas" width="${ICW}" height="${ICH}"
              style="display:block;border:1px solid #222;border-radius:4px;background:#0d0d0d;"></canvas>
            <div class="row" style="margin-top:10px;gap:8px;">
              <button class="btn primary" id="sn-i-start">Start</button>
              <button class="btn"         id="sn-i-pause">Pause</button>
              <button class="btn"         id="sn-i-rerun">Restart</button>
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Trained gens</div><div class="v" id="sn-i-gen">—</div></div>
              <div class="kpi"><div class="k">All-time best 🍎</div><div class="v" id="sn-i-fit">—</div></div>
              <div class="kpi"><div class="k">This game 🍎</div><div class="v" id="sn-i-apples">0</div></div>
              <div class="kpi"><div class="k">Games played</div><div class="v" id="sn-i-runs">0</div></div>
            </div>

            <div class="section">
              <h3>Playback speed</h3>
              <div style="display:flex;align-items:center;gap:8px;font-size:12px;color:#aaa;">
                <span>Slow</span>
                <input id="sn-i-speed" type="range" min="1" max="10" value="3" style="flex:1;">
                <span>Fast</span>
                <span id="sn-i-speed-val" style="min-width:36px;text-align:right;color:#64b5f6;">3×</span>
              </div>
              <div style="font-size:11px;color:#444;margin-top:4px;">Ticks per rendered frame.</div>
            </div>

            <div style="font-size:11px;color:#444;line-height:1.9;">
              <span style="color:${COL.inferHead};">■</span> Head &nbsp;
              <span style="color:${COL.inferBody};">■</span> Body &nbsp;
              <span style="color:${COL.inferApple};">●</span> Apple<br><br>
              The snake auto-resets when it dies or loops without eating.<br>
              Switch to the <strong style="color:#666;">Train</strong> tab to keep evolving.
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('sn-i-canvas');
    const ctx    = canvas.getContext('2d');

    function showPlaceholder(msg) {
      ctx.fillStyle = COL.bg;
      ctx.fillRect(0, 0, ICW, ICH);
      ctx.fillStyle    = '#444';
      ctx.font         = '13px monospace';
      ctx.textAlign    = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(msg, ICW / 2, ICH / 2);
      ctx.textAlign    = 'left';
      ctx.textBaseline = 'alphabetic';
    }

    function updateInferStats(s) {
      if (!s || !s.snake) return;
      document.getElementById('sn-i-apples').textContent = s.snake.applesEaten;
      if (s.justReset) {
        _runs++;
        document.getElementById('sn-i-runs').textContent = _runs;
      }
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const speed = parseInt(document.getElementById('sn-i-speed').value, 10) || 3;

      try {
        let s = null;
        // Step `speed` times, render only the last result
        for (let i = 0; i < speed; i++) {
          s = await inv('snake-neuro:inferStep');
          if (!s) break;
          if (s.justReset) {
            _runs++;
            document.getElementById('sn-i-runs').textContent = _runs;
          }
        }
        if (s && s.snake) {
          drawInferSnake(ctx, s.snake);
          document.getElementById('sn-i-apples').textContent = s.snake.applesEaten;
          if (s.generation != null) document.getElementById('sn-i-gen').textContent = s.generation;
          if (s.bestFit   != null) document.getElementById('sn-i-fit').textContent  = s.bestFit.toFixed(1);
        }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startInfer() {
      if (!_ready) {
        const r = await inv('snake-neuro:inferInit');
        if (!r || r.error) {
          showPlaceholder(r ? r.error : 'No trained model — run the Train tab first.');
          return;
        }
        document.getElementById('sn-i-gen').textContent = r.generation ?? '—';
        document.getElementById('sn-i-fit').textContent = r.bestFit != null ? r.bestFit.toFixed(1) : '—';
        _ready = true;
        _runs  = 0;
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
      _runs  = 0;
      document.getElementById('sn-i-apples').textContent = '0';
      document.getElementById('sn-i-runs').textContent   = '0';
      const r = await inv('snake-neuro:inferInit');
      if (!r || r.error) {
        showPlaceholder(r ? r.error : 'No trained model — run the Train tab first.');
        return;
      }
      document.getElementById('sn-i-gen').textContent = r.generation ?? '—';
      document.getElementById('sn-i-fit').textContent = r.bestFit != null ? r.bestFit.toFixed(1) : '—';
      _ready = true;
      if (wasRunning) {
        _running = true;
        _raf = requestAnimationFrame(tick);
      }
    }

    document.getElementById('sn-i-start').addEventListener('click', startInfer);
    document.getElementById('sn-i-pause').addEventListener('click', pauseInfer);
    document.getElementById('sn-i-rerun').addEventListener('click', rerun);

    const speedSlider = document.getElementById('sn-i-speed');
    speedSlider.addEventListener('input', () => {
      document.getElementById('sn-i-speed-val').textContent = `${speedSlider.value}×`;
    });

    showPlaceholder('Press Start to watch the best evolved snake.');
  });

})(api);