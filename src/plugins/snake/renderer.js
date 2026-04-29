// Snake Neuroevolution plugin — evaluated in renderer via new Function('api', code)(api)
(function (api) {
  'use strict';

  // ── Canvas dimensions (fixed) ─────────────────────────────────────────────
  const CW  = 700, CH  = 660;   // training canvas
  const ICW = 400, ICH = 430;   // inference canvas

  // ── Default grid constants (used for fallback / template) ─────────────────
  const DEF_GRID_W = 15;
  const DEF_GRID_H = 17;

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
    timerOk:      '#00e676',
    timerLow:     '#ffd600',
  };

  // ── Layout helpers ────────────────────────────────────────────────────────

  function computeTrainLayout(gridW, gridH, count) {
    const COLS    = Math.min(5, count);
    const ROWS    = Math.ceil(count / COLS);
    const LABEL_H = 14;
    const GAP     = 6;
    const cellW   = Math.floor((CW - (COLS + 1) * GAP) / (COLS * gridW));
    const cellH   = Math.floor((CH - ROWS * (LABEL_H + GAP) - GAP) / (ROWS * gridH));
    const cell    = Math.max(2, Math.min(cellW, cellH, 14));
    const mGW     = gridW * cell;
    const mGH     = gridH * cell;
    const slotW   = mGW + GAP;
    const slotH   = mGH + LABEL_H + GAP;
    const offX    = Math.floor((CW - COLS * slotW) / 2);
    const offY    = Math.floor((CH - ROWS * slotH) / 2);
    return { COLS, ROWS, cell, mGW, mGH, slotW, slotH, offX, offY, LABEL_H };
  }

  function computeInferLayout(gridW, gridH) {
    const margin = 20;
    const cellW  = Math.floor((ICW - margin) / gridW);
    const cellH  = Math.floor((ICH - margin) / gridH);
    const cell   = Math.max(4, Math.min(cellW, cellH, 28));
    const iGW    = gridW * cell;
    const iGH    = gridH * cell;
    const offX   = Math.floor((ICW - iGW) / 2);
    const offY   = Math.floor((ICH - iGH) / 2);
    return { cell, iGW, iGH, offX, offY };
  }

  // ── Drawing helpers ───────────────────────────────────────────────────────

  function drawMiniSnake(ctx, snake, x0, y0, isBest, lay, gridW, gridH) {
    const { cell, mGW, mGH, LABEL_H } = lay;
    const alive = snake.alive;

    ctx.fillStyle = COL.gridBg;
    ctx.fillRect(x0, y0, mGW, mGH);

    ctx.strokeStyle = COL.gridLine;
    ctx.lineWidth   = 0.5;
    for (let gx = 0; gx <= gridW; gx++) {
      ctx.beginPath(); ctx.moveTo(x0 + gx * cell, y0); ctx.lineTo(x0 + gx * cell, y0 + mGH); ctx.stroke();
    }
    for (let gy = 0; gy <= gridH; gy++) {
      ctx.beginPath(); ctx.moveTo(x0, y0 + gy * cell); ctx.lineTo(x0 + mGW, y0 + gy * cell); ctx.stroke();
    }

    const ax = x0 + snake.apple.x * cell + 1;
    const ay = y0 + snake.apple.y * cell + 1;
    ctx.fillStyle = alive ? COL.apple : COL.appleDead;
    ctx.fillRect(ax, ay, cell - 2, cell - 2);

    for (let i = snake.body.length - 1; i >= 0; i--) {
      const seg = snake.body[i];
      const px  = x0 + seg.x * cell;
      const py  = y0 + seg.y * cell;
      if (i === 0) {
        ctx.fillStyle = alive ? COL.headAlive : COL.headDead;
        ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
        if (alive && cell >= 5) {
          ctx.fillStyle = '#000';
          ctx.fillRect(px + 2, py + 2, 2, 2);
          ctx.fillRect(px + cell - 4, py + 2, 2, 2);
        }
      } else {
        ctx.fillStyle = alive ? COL.bodyAlive : COL.bodyDead;
        ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      }
    }

    ctx.strokeStyle = isBest ? COL.bestBorder : COL.gridBorder;
    ctx.lineWidth   = isBest ? 2 : 1;
    ctx.strokeRect(x0, y0, mGW, mGH);

    ctx.fillStyle    = isBest ? COL.labelBest : alive ? COL.labelAlive : COL.labelDead;
    ctx.font         = `${isBest ? '600 ' : ''}9px monospace`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(`🍎${snake.applesEaten}${alive ? '' : '✗'}`, x0 + mGW / 2, y0 + mGH + LABEL_H / 2);
    ctx.textAlign    = 'left';
    ctx.textBaseline = 'alphabetic';
  }

  function drawAllSnakes(ctx, snakes, gridW, gridH) {
    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, CW, CH);
    if (!snakes || snakes.length === 0) return;

    const gW  = gridW  || DEF_GRID_W;
    const gH  = gridH  || DEF_GRID_H;
    const lay = computeTrainLayout(gW, gH, snakes.length);

    let bestIdx = 0;
    for (let i = 1; i < snakes.length; i++) {
      const cur  = snakes[i].applesEaten;
      const prev = snakes[bestIdx].applesEaten;
      if (cur > prev || (cur === prev && snakes[i].alive && !snakes[bestIdx].alive)) bestIdx = i;
    }

    for (let i = 0; i < snakes.length; i++) {
      const col = i % lay.COLS;
      const row = Math.floor(i / lay.COLS);
      const x0  = lay.offX + col * lay.slotW;
      const y0  = lay.offY + row * lay.slotH;
      drawMiniSnake(ctx, snakes[i], x0, y0, i === bestIdx, lay, gW, gH);
    }
  }

  function drawInferSnake(ctx, snake, gridW, gridH) {
    const gW  = gridW  || DEF_GRID_W;
    const gH  = gridH  || DEF_GRID_H;
    const lay = computeInferLayout(gW, gH);
    const { cell, iGW, iGH, offX, offY } = lay;

    ctx.fillStyle = COL.bg;
    ctx.fillRect(0, 0, ICW, ICH);
    ctx.fillStyle = '#111';
    ctx.fillRect(offX, offY, iGW, iGH);

    ctx.strokeStyle = '#1a1a1a';
    ctx.lineWidth   = 0.5;
    for (let gx = 0; gx <= gW; gx++) {
      ctx.beginPath(); ctx.moveTo(offX + gx * cell, offY); ctx.lineTo(offX + gx * cell, offY + iGH); ctx.stroke();
    }
    for (let gy = 0; gy <= gH; gy++) {
      ctx.beginPath(); ctx.moveTo(offX, offY + gy * cell); ctx.lineTo(offX + iGW, offY + gy * cell); ctx.stroke();
    }

    ctx.fillStyle = COL.inferApple;
    ctx.beginPath();
    ctx.arc(offX + (snake.apple.x + 0.5) * cell, offY + (snake.apple.y + 0.5) * cell, cell * 0.38, 0, Math.PI * 2);
    ctx.fill();

    const r = Math.max(2, Math.round(cell * 0.2));
    for (let i = snake.body.length - 1; i >= 0; i--) {
      const seg = snake.body[i];
      const px  = offX + seg.x * cell;
      const py  = offY + seg.y * cell;
      ctx.fillStyle = (i === 0) ? COL.inferHead
        : `rgba(21,101,192,${Math.max(0.25, 1 - i / snake.body.length)})`;
      ctx.beginPath();
      ctx.roundRect(px + 1, py + 1, cell - 2, cell - 2, r);
      ctx.fill();
      if (i === 0 && cell >= 8) {
        ctx.fillStyle = '#0a2540';
        const eyeR = Math.max(1, Math.round(cell * 0.14));
        ctx.beginPath(); ctx.arc(px + cell * 0.28, py + cell * 0.3, eyeR, 0, Math.PI * 2); ctx.fill();
        ctx.beginPath(); ctx.arc(px + cell * 0.72, py + cell * 0.3, eyeR, 0, Math.PI * 2); ctx.fill();
      }
    }

    ctx.strokeStyle = '#333';
    ctx.lineWidth   = 1.5;
    ctx.strokeRect(offX, offY, iGW, iGH);
  }

  function drawFitnessChart(svgEl, genHistory) {
    if (!svgEl || !genHistory || genHistory.length < 2) { if (svgEl) svgEl.innerHTML = ''; return; }
    const W = 260, H = 60, pad = 4;
    const maxVal = Math.max(...genHistory.map(h => h.best), 0.1);

    const ptsFn = (key) => genHistory.map((h, i) => {
      const x = pad + (i / (genHistory.length - 1)) * (W - 2 * pad);
      const y = H - pad - (h[key] / maxVal) * (H - 2 * pad);
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    }).join(' ');

    svgEl.innerHTML = `
      <polyline points="${ptsFn('mean')}" fill="none" stroke="#333" stroke-width="1" stroke-linejoin="round"/>
      <polyline points="${ptsFn('best')}" fill="none" stroke="${COL.chartLine}" stroke-width="1.5" stroke-linejoin="round"/>
      <text x="${W - pad}" y="${H - pad - 1}" text-anchor="end" font-size="8" fill="#555">${maxVal.toFixed(1)}</text>
    `;
  }

  // ── Architecture fields ───────────────────────────────────────────────────
  api.registerArchFields('snake-neuro', {
    fields: [
      { id: 'gridW',      label: 'Grid width',        type: 'number',     default: 15,  min: 4,  max: 30,  step: 1,  hint: 'Width of the snake grid in cells. Changing this resets saved weights.' },
      { id: 'gridH',      label: 'Grid height',       type: 'number',     default: 17,  min: 4,  max: 30,  step: 1,  hint: 'Height of the snake grid in cells. Changing this resets saved weights.' },
      { id: 'staleLimit', label: 'Stale kill (moves)', type: 'number',    default: 150, min: 20, max: 400, step: 10, hint: 'Snake dies if it goes this many moves without eating' },
      { id: 'popSize',    label: 'Population size',   type: 'number',     default: 50,  min: 5,  max: 200, step: 5,  hint: 'Number of snakes simulated per generation' },
      { id: 'hidden',     label: 'Hidden layers',     type: 'layers',     default: [255, 128, 64], hint: 'Comma-separated hidden layer sizes. Changing these resets saved weights.' },
      { id: 'activation', label: 'Activation',        type: 'activation', default: 'tanh' },
    ],
    computeDims: (a) => ({ inputDim: (a.gridW || 15) * (a.gridH || 17), outputDim: 4 }),
  });

  // ── Template ──────────────────────────────────────────────────────────────
  api.registerTemplate({
    id:         'snake-neuro',
    name:       'Snake AI (Neuroevolution)',
    kind:       'classifier',
    pluginKind: 'snake-neuro',
    desc:       'A population of neural-net snakes evolves to eat apples on a configurable grid. Uses Selective Reproduction (neuroevolution). No training data required.',
    arch: {
      kind: 'classifier', pluginKind: 'snake-neuro',
      inputDim: 255, outputDim: 4,
      hidden: [255, 128, 64], activation: 'tanh', dropout: 0,
      gridW: 15, gridH: 17, popSize: 50, staleLimit: 150,
    },
    training: { optimizer: 'adam', learningRate: 0.08, batchSize: 50, epochs: 0, seed: 42, workers: 0 },
    trainingData: {},
  });

  // ── Train settings ────────────────────────────────────────────────────────
  api.registerTrainSettings('snake-neuro', {
    lr:        { label: 'Mutation std',    hint: 'Weight mutation std — lower = finer search (default 0.08)' },
    bs:        { label: 'Population size', hint: 'Population size override (use Architecture tab for persistent config)' },
    epochs:    { label: 'Generations',     hint: 'Max generations to evolve (0 = run indefinitely)' },
    seed:      { label: 'RNG seed',        hint: 'Seed for network initialisation' },
    workers:   { hidden: true },
    optimizer: { hidden: true },
    sectionHint: 'Neuroevolution settings — applied when the simulation starts.',
  });

  // ── Train editor (data section) ───────────────────────────────────────────
  api.registerTrainEditor('snake-neuro', function (root, network) {
    const a   = (network && network.architecture) || {};
    const gW  = a.gridW  || 15;
    const gH  = a.gridH  || 17;
    const pop = a.popSize || 50;
    const hid = (a.hidden || [255, 128, 64]).join(', ');
    root.innerHTML = `
      <div style="display:grid;gap:10px;">
        <div style="background:#0d1a0d;border:1px solid #1a4a1a;border-radius:6px;padding:10px 14px;">
          <div style="font-size:13px;font-weight:600;color:#00e676;margin-bottom:5px;">Snake Neuroevolution — no training data required</div>
          <div style="font-size:12px;color:#aaa;line-height:1.6;">
            ${pop} snakes play simultaneously on independent ${gW}×${gH} grids.
            After <strong style="color:#ccc;">8 seconds</strong>, the highest-scoring snake reproduces
            into the next generation via mutation (Selective Reproduction).
          </div>
        </div>
        <div style="font-size:12px;color:#666;background:#111;border:1px solid #222;border-radius:4px;padding:8px 12px;line-height:1.8;">
          <strong style="color:#aaa;">Network:</strong> ${gW*gH} → [${hid}] → 4<br>
          <strong style="color:#aaa;">Inputs:</strong> full ${gW}×${gH} grid (head/body/apple/empty encoded)<br>
          <strong style="color:#aaa;">Outputs:</strong> ↑ ↓ ← → direction logits (argmax)<br><br>
          Grid and network settings live in the <strong style="color:#ccc;">Editor</strong> tab → Architecture.<br>
          Use the <strong style="color:#ccc;">Train</strong> tab to run live evolution.<br>
          The <strong style="color:#ccc;">Infer</strong> tab watches the best evolved genome play.
        </div>
      </div>
    `;
  });

  // ── Train renderer ────────────────────────────────────────────────────────
  api.registerTrainRenderer('snake-neuro', function (root, network, nb) {
    let _raf         = null;
    let _running     = false;
    let _initialized = false;
    let _cachedGridW = DEF_GRID_W;
    let _cachedGridH = DEF_GRID_H;

    const instanceId = (network && network.id) || 'snake-default';
    const inv = (ch, extra = {}) => nb.invoke(ch, { instanceId, ...extra });

    const t             = (network && network.training) || {};
    const cfgMutStd     = t.learningRate || 0.08;
    const cfgSeed       = (t.seed | 0)   || 42;
    const a             = (network && network.architecture) || {};
    const cfgHidden     = Array.isArray(a.hidden) && a.hidden.length ? a.hidden : [255, 128, 64];
    const cfgActivation = a.activation   || 'tanh';
    const cfgGridW      = (a.gridW       | 0) || 15;
    const cfgGridH      = (a.gridH       | 0) || 17;
    const cfgPopSize    = (a.popSize     | 0) || 50;
    const cfgStaleLimit = (a.staleLimit  | 0) || 150;

    root.innerHTML = `
      <div class="panel" style="max-width:1080px;">
        <h2>Snake AI — Neuroevolution</h2>
        <p style="font-size:12px;color:#777;margin:-4px 0 16px;">
          ${cfgPopSize} neural-net snakes on ${cfgGridW}×${cfgGridH} grids. Mutation std: ${cfgMutStd}.
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
            </div>
          </div>

          <div style="display:grid;gap:12px;">

            <div class="kpis" style="grid-template-columns:repeat(2,1fr);">
              <div class="kpi"><div class="k">Generation</div><div class="v" id="sn-gen">0</div></div>
              <div class="kpi"><div class="k">Alive</div><div class="v" id="sn-alive">0 / ${cfgPopSize}</div></div>
              <div class="kpi"><div class="k">Gen best 🍎</div><div class="v" id="sn-genbest">—</div></div>
              <div class="kpi"><div class="k">All-time best 🍎</div><div class="v" id="sn-best">—</div></div>
            </div>

            <div class="kpis" style="grid-template-columns:1fr;">
              <div class="kpi">
                <div class="k">Time left this generation</div>
                <div class="v" id="sn-timer" style="font-size:20px;">8.0s</div>
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
              <span style="color:${COL.bestBorder};">▣</span> Best this gen<br><br>
              <strong style="color:#555;">Fitness</strong> = apples eaten × (died ? 0.9 : 1.0)<br>
              <strong style="color:#555;">Stale kill</strong>: no apple in ${cfgStaleLimit} moves → death<br>
              Grid: ${cfgGridW}×${cfgGridH} · Population: ${cfgPopSize} · Seed: ${cfgSeed}
            </div>

          </div>
        </div>
      </div>
    `;

    const canvas = document.getElementById('snake-canvas');
    const ctx    = canvas.getContext('2d');

    function updateStats(s) {
      if (!s) return;
      const ps = s.snakes ? s.snakes.length : cfgPopSize;
      document.getElementById('sn-gen').textContent     = s.generation;
      document.getElementById('sn-alive').textContent   = `${s.aliveCnt} / ${ps}`;
      document.getElementById('sn-genbest').textContent = s.genBestFit > 0 ? s.genBestFit.toFixed(1) : '—';
      document.getElementById('sn-best').textContent    = s.bestFit    > 0 ? s.bestFit.toFixed(1)    : '—';

      const pct = Math.max(0, Math.min(1, s.timeLeft / 8000));
      const secs = (s.timeLeft / 1000).toFixed(1);
      const timerEl = document.getElementById('sn-timer');
      const barEl   = document.getElementById('sn-timer-bar');
      if (timerEl) { timerEl.textContent = `${secs}s`; timerEl.style.color = pct > 0.4 ? COL.timerOk : COL.timerLow; }
      if (barEl)   { barEl.style.width = `${pct * 100}%`; barEl.style.background = pct > 0.4 ? COL.timerOk : COL.timerLow; }

      drawFitnessChart(document.getElementById('sn-chart'), s.genHistory);
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const ticks = parseInt(document.getElementById('sn-speed').value, 10) || 5;
      try {
        const s = await inv('snake-neuro:step', { ticks });
        if (s) {
          if (s.gridW) _cachedGridW = s.gridW;
          if (s.gridH) _cachedGridH = s.gridH;
          drawAllSnakes(ctx, s.snakes, _cachedGridW, _cachedGridH);
          updateStats(s);
        }
      } catch (_) {}
      _raf = requestAnimationFrame(tick);
    }

    async function startSim() {
      if (!_initialized) {
        const r = await inv('snake-neuro:init', {
          seed: cfgSeed, mutStd: cfgMutStd,
          hidden: cfgHidden, activation: cfgActivation,
          gridW: cfgGridW, gridH: cfgGridH,
          popSize: cfgPopSize, staleLimit: cfgStaleLimit,
        });
        if (r && r.error) { console.error('[snake-neuro]', r.error); return; }
        _cachedGridW = cfgGridW;
        _cachedGridH = cfgGridH;
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
        if (el) el.textContent = id === 'sn-gen' ? '0' : `—`;
      });
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

    inv('snake-neuro:getState').then(s => {
      if (s && s.snakes && s.snakes.length > 0) {
        if (s.gridW) _cachedGridW = s.gridW;
        if (s.gridH) _cachedGridH = s.gridH;
        _initialized = true;
        drawAllSnakes(ctx, s.snakes, _cachedGridW, _cachedGridH);
        updateStats(s);
      } else {
        ctx.fillStyle    = COL.bg; ctx.fillRect(0, 0, CW, CH);
        ctx.fillStyle    = '#333'; ctx.font = '13px monospace';
        ctx.textAlign    = 'center'; ctx.textBaseline = 'middle';
        ctx.fillText('Press Start to begin evolution', CW / 2, CH / 2);
        ctx.textAlign    = 'left'; ctx.textBaseline = 'alphabetic';
      }
    }).catch(() => { ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, CW, CH); });
  });

  // ── Inference renderer ────────────────────────────────────────────────────
  api.registerInferenceRenderer('snake-neuro', function (root, network, nb) {
    let _raf      = null;
    let _running  = false;
    let _ready    = false;
    let _runs     = 0;
    let _inferGridW = DEF_GRID_W;
    let _inferGridH = DEF_GRID_H;

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
      ctx.fillStyle = COL.bg; ctx.fillRect(0, 0, ICW, ICH);
      ctx.fillStyle = '#444'; ctx.font = '13px monospace';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(msg, ICW / 2, ICH / 2);
      ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    }

    async function tick() {
      if (!_running || !canvas.isConnected) return;
      const speed = parseInt(document.getElementById('sn-i-speed').value, 10) || 3;
      try {
        let s = null;
        for (let i = 0; i < speed; i++) {
          s = await inv('snake-neuro:inferStep');
          if (!s) break;
          if (s.justReset) {
            _runs++;
            document.getElementById('sn-i-runs').textContent = _runs;
          }
        }
        if (s && s.snake) {
          if (s.gridW) _inferGridW = s.gridW;
          if (s.gridH) _inferGridH = s.gridH;
          drawInferSnake(ctx, s.snake, _inferGridW, _inferGridH);
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
        if (r.gridW) _inferGridW = r.gridW;
        if (r.gridH) _inferGridH = r.gridH;
        _ready = true; _runs = 0;
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
      _ready = false; _runs = 0;
      document.getElementById('sn-i-apples').textContent = '0';
      document.getElementById('sn-i-runs').textContent   = '0';
      const r = await inv('snake-neuro:inferInit');
      if (!r || r.error) {
        showPlaceholder(r ? r.error : 'No trained model — run the Train tab first.');
        return;
      }
      document.getElementById('sn-i-gen').textContent = r.generation ?? '—';
      document.getElementById('sn-i-fit').textContent = r.bestFit != null ? r.bestFit.toFixed(1) : '—';
      if (r.gridW) _inferGridW = r.gridW;
      if (r.gridH) _inferGridH = r.gridH;
      _ready = true;
      if (wasRunning) { _running = true; _raf = requestAnimationFrame(tick); }
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
