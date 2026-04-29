'use strict';

const path = require('path');
const { app } = require('electron');

// Load neuroevolution + tensor from the app bundle.
let Population, T, buildFromSpec;
try {
  ({ Population } = require(path.join(app.getAppPath(), 'src', 'engine', 'neuroevolution')));
  T = require(path.join(app.getAppPath(), 'src', 'engine', 'tensor'));
  ({ buildFromSpec } = require(path.join(app.getAppPath(), 'src', 'engine', 'model')));
} catch (e) {
  console.error('[snake-neuro] Failed to load engines:', e.message);
}

// ── Simulation constants (defaults) ───────────────────────────────────────────
const GRID_W          = 15;
const GRID_H          = 17;
const INPUT_DIM       = GRID_W * GRID_H;   // 255
const OUTPUT_DIM      = 4;                  // up, right, down, left
const POP_SIZE        = 50;
const GEN_DURATION_MS = 8000;
const STALE_LIMIT     = 150;

// Direction vectors: 0=up, 1=right, 2=down, 3=left
const DIRS = [[0, -1], [1, 0], [0, 1], [-1, 0]];

const ARCH = {
  kind:       'classifier',
  inputDim:   INPUT_DIM,
  outputDim:  OUTPUT_DIM,
  hidden:     [255, 128, 64],
  activation: 'tanh',
  dropout:    0,
};

// ── Per-instance sessions ──────────────────────────────────────────────────────
function makeSession() {
  return {
    pop:          null,
    popSize:      POP_SIZE,
    snakes:       [],
    snakeFit:     [],
    generation:   0,
    bestFit:      0,
    genBestFit:   0,
    running:      false,
    genHistory:   [],
    genStartTime: 0,
    inferSnake:   null,
    arch:         null,
    gridW:        GRID_W,
    gridH:        GRID_H,
    inputDim:     INPUT_DIM,
    staleLimit:   STALE_LIMIT,
  };
}

const _sessions = new Map();
function getSession(id) {
  const key = id || 'default';
  if (!_sessions.has(key)) _sessions.set(key, makeSession());
  return _sessions.get(key);
}

// ── Snake helpers ──────────────────────────────────────────────────────────────

function randomPos(body, gridW, gridH) {
  const gW = gridW || GRID_W, gH = gridH || GRID_H;
  const total       = gW * gH;
  const occupiedSet = new Set(body.map(p => p.y * gW + p.x));
  if (occupiedSet.size >= total) return { x: 0, y: 0 };
  let idx;
  do { idx = Math.floor(Math.random() * total); } while (occupiedSet.has(idx));
  return { x: idx % gW, y: Math.floor(idx / gW) };
}

function spawnSnake(gridW, gridH) {
  const gW = gridW || GRID_W, gH = gridH || GRID_H;
  const cx = Math.floor(gW / 2);
  const cy = Math.floor(gH / 2);
  const body = [
    { x: cx, y: cy },
    { x: cx, y: cy + 1 },
    { x: cx, y: cy + 2 },
  ];
  return {
    body,
    dir:             0,
    apple:           randomPos(body, gW, gH),
    alive:           true,
    applesEaten:     0,
    steps:           0,
    stepsSinceApple: 0,
  };
}

/**
 * 5-class grid encoding:
 *  0.0  = Blank   1.0 = Apple   0.5 = Head   0.25 = Body   -0.5 = Tail
 */
function buildGrid(snake, gridW, gridH) {
  const gW  = gridW  || GRID_W;
  const gH  = gridH  || GRID_H;
  const dim = gW * gH;
  const grid = new Float32Array(dim);
  for (let i = 0; i < snake.body.length; i++) {
    const { x, y } = snake.body[i];
    const idx = y * gW + x;
    if      (i === 0)                       grid[idx] = 0.5;
    else if (i === snake.body.length - 1)   grid[idx] = -0.5;
    else                                    grid[idx] = 0.25;
  }
  grid[snake.apple.y * gW + snake.apple.x] = 1.0;
  return grid;
}

function netForward(model, inputArr, inputDim) {
  const dim = inputDim || INPUT_DIM;
  const x   = new T.Tensor([1, dim], inputArr, false);
  const out = model.forward(x, { training: false });
  return out.data;
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < OUTPUT_DIM; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function stepSnake(snake, model, s) {
  if (!snake.alive) return;
  const gridW      = (s && s.gridW)      || GRID_W;
  const gridH      = (s && s.gridH)      || GRID_H;
  const inputDim   = (s && s.inputDim)   || INPUT_DIM;
  const staleLimit = (s && s.staleLimit) || STALE_LIMIT;

  const grid   = buildGrid(snake, gridW, gridH);
  const outs   = netForward(model, grid, inputDim);
  const rawDir = argmax(outs);

  const opposite = (snake.dir + 2) % 4;
  snake.dir = (rawDir === opposite) ? snake.dir : rawDir;

  const [dx, dy] = DIRS[snake.dir];
  const nx = snake.body[0].x + dx;
  const ny = snake.body[0].y + dy;

  if (nx < 0 || nx >= gridW || ny < 0 || ny >= gridH) {
    snake.alive = false;
    return;
  }

  for (let i = 0; i < snake.body.length; i++) {
    if (snake.body[i].x === nx && snake.body[i].y === ny) {
      snake.alive = false;
      return;
    }
  }

  snake.body.unshift({ x: nx, y: ny });

  if (nx === snake.apple.x && ny === snake.apple.y) {
    snake.applesEaten++;
    snake.stepsSinceApple = 0;
    snake.apple = randomPos(snake.body, gridW, gridH);
  } else {
    snake.body.pop();
  }

  snake.steps++;
  snake.stepsSinceApple++;

  if (snake.stepsSinceApple > staleLimit) {
    snake.alive = false;
  }
}

function manhattanDistance(p1, p2) {
  return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
}

function snakeFitness(snake, gridW, gridH) {
  const gW = gridW || GRID_W, gH = gridH || GRID_H;
  const maxDist = gW + gH - 2;
  let fitness = snake.applesEaten;
  if (snake.alive) {
    const distToApple = manhattanDistance(snake.body[0], snake.apple);
    fitness -= (distToApple / maxDist);
  }
  if (!snake.alive) fitness *= 0.9;
  return Math.max(0, fitness);
}

function stepGeneration(s, opts) {
  if (!s.pop || !s.running) return null;

  const ticks   = Math.max(1, Math.min((opts.ticks || 10), 50));
  const now     = Date.now();
  const elapsed = now - s.genStartTime;
  const allDead = s.snakes.every(sn => !sn.alive);
  const popSize = s.popSize || POP_SIZE;

  if (elapsed >= GEN_DURATION_MS || allDead) {
    s.snakes.forEach((sn, i) => { s.snakeFit[i] = snakeFitness(sn, s.gridW, s.gridH); });
    s.pop.evaluate((_, idx) => s.snakeFit[idx]);

    const stats = s.pop.evolve();
    s.generation++;

    const best = Math.max(...s.snakeFit);
    s.genHistory.push({
      gen:  s.generation,
      best: +best.toFixed(3),
      mean: +((stats && stats.mean) || 0).toFixed(3),
    });
    if (s.genHistory.length > 100) s.genHistory.shift();
    if (best > s.bestFit) s.bestFit = best;
    s.genBestFit = best;

    s.snakes       = Array.from({ length: popSize }, () => spawnSnake(s.gridW, s.gridH));
    s.snakeFit     = new Array(popSize).fill(0);
    s.genStartTime = Date.now();

  } else {
    for (let t = 0; t < ticks; t++) {
      for (let i = 0; i < popSize; i++) {
        if (s.snakes[i].alive) stepSnake(s.snakes[i], s.pop.individuals[i], s);
      }
    }
  }

  return buildVisualState(s);
}

function buildVisualState(s) {
  const timeLeft = Math.max(0, GEN_DURATION_MS - (Date.now() - s.genStartTime));
  return {
    snakes: s.snakes.map(sn => ({
      body:         sn.body,
      apple:        sn.apple,
      alive:        sn.alive,
      applesEaten:  sn.applesEaten,
      dir:          sn.dir,
    })),
    generation:    s.generation,
    aliveCnt:      s.snakes.filter(sn => sn.alive).length,
    bestFit:       +s.bestFit.toFixed(3),
    genBestFit:    +s.genBestFit.toFixed(3),
    genHistory:    s.genHistory.slice(-60),
    timeLeft,
    hasBestGenome: s.pop !== null && s.pop.bestIndividual !== null,
    gridW:         s.gridW || GRID_W,
    gridH:         s.gridH || GRID_H,
  };
}

// ── Storage helpers ────────────────────────────────────────────────────────────

function savePopToStorage(storage, id, s) {
  if (!storage || !s.pop) return;
  try {
    const arch = { ...(s.arch || ARCH), pluginKind: 'snake-neuro' };
    storage.saveTrainedState(id, { state: s.pop.toJSON(), architecture: arch });
  } catch (e) {
    console.warn('[snake-neuro] Could not save state:', e.message);
  }
}

function loadPopFromStorage(storage, id) {
  if (!storage) return null;
  try {
    const net = storage.getNetwork(id);
    if (!net || !net.state || net.stateLocked) return null;
    const pop = Population.fromJSON(net.state);
    if (!pop.bestIndividual && pop.individuals.length > 0) {
      pop.bestIndividual = pop.individuals[0];
    }
    return pop;
  } catch (e) {
    console.warn('[snake-neuro] Could not load state:', e.message);
    return null;
  }
}

// ── IPC handlers (factory — receives storage from plugin-loader) ───────────────
module.exports = function ({ storage } = {}) {
  return {
    mainHandlers: {

      'snake-neuro:init': (_, opts = {}) => {
        if (!Population || !T) return { error: 'Neuroevolution engine unavailable.' };
        const id     = opts.instanceId || 'default';
        const s      = getSession(id);
        const mutStd = opts.mutStd || 0.05;

        // Grid / population config from opts
        const gridW      = Math.max(4,  Math.min(30,  (opts.gridW      | 0) || GRID_W));
        const gridH      = Math.max(4,  Math.min(30,  (opts.gridH      | 0) || GRID_H));
        const popSize    = Math.max(5,  Math.min(200, (opts.popSize    | 0) || POP_SIZE));
        const staleLimit = Math.max(20, Math.min(400, (opts.staleLimit | 0) || STALE_LIMIT));
        const inputDim   = gridW * gridH;

        s.gridW      = gridW;
        s.gridH      = gridH;
        s.popSize    = popSize;
        s.staleLimit = staleLimit;
        s.inputDim   = inputDim;

        s.arch = {
          ...ARCH,
          inputDim,
          hidden:     Array.isArray(opts.hidden) && opts.hidden.length ? opts.hidden : ARCH.hidden,
          activation: opts.activation || ARCH.activation,
          gridW, gridH, popSize, staleLimit,
        };

        // Resume from saved state when available; fall back to fresh population.
        const savedPop      = loadPopFromStorage(storage, id);
        const savedInputDim = savedPop && savedPop.arch && savedPop.arch.inputDim;
        if (savedPop && (!savedInputDim || savedInputDim === inputDim)) {
          s.pop = savedPop;
          s.pop.mutationStd = mutStd;
          s.generation = s.pop.generation || 0;
          s.bestFit    = Math.max(0, isFinite(s.pop.bestFitness) ? s.pop.bestFitness : 0);
          s.genHistory = (s.pop.history || []).slice(-100).map((h, i) => ({
            gen:  h.generation ?? i,
            best: +(h.stats?.max ?? 0).toFixed(3),
            mean: +(h.stats?.mean ?? 0).toFixed(3),
          }));
        } else {
          if (savedPop) console.log(`[snake-neuro] inputDim changed (${savedInputDim}→${inputDim}), resetting population.`);
          s.pop = new Population({
            architecture: s.arch,
            size:         popSize,
            eliteCount:   2,
            pMutate:      0.15,
            mutationStd:  mutStd,
            tournamentK:  3,
            seed:         opts.seed || 42,
          });
          s.generation = 0;
          s.bestFit    = 0;
          s.genHistory = [];
        }

        s.snakes       = Array.from({ length: popSize }, () => spawnSnake(gridW, gridH));
        s.snakeFit     = new Array(popSize).fill(0);
        s.genBestFit   = 0;
        s.running      = true;
        s.genStartTime = Date.now();

        return { ok: true, resumed: !!savedPop, generation: s.generation };
      },

      'snake-neuro:step': (_, opts = {}) => {
        const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        return stepGeneration(getSession(id), opts);
      },

      'snake-neuro:start': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        s.running      = true;
        s.genStartTime = Date.now();
        return { ok: true };
      },

      'snake-neuro:stop': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        s.running = false;
        savePopToStorage(storage, id, s);
        return { ok: true };
      },

      'snake-neuro:reset': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        if (!Population || !T) return { error: 'Engine unavailable.' };
        const popSize = s.popSize || POP_SIZE;

        s.pop = new Population({
          architecture: s.arch || ARCH,
          size:         popSize,
          eliteCount:   2,
          pMutate:      0.15,
          mutationStd:  0.05,
          tournamentK:  3,
          seed:         Math.floor(Math.random() * 0xFFFFFF),
        });

        s.snakes       = Array.from({ length: popSize }, () => spawnSnake(s.gridW, s.gridH));
        s.snakeFit     = new Array(popSize).fill(0);
        s.generation   = 0;
        s.bestFit      = 0;
        s.genBestFit   = 0;
        s.running      = true;
        s.genHistory   = [];
        s.genStartTime = Date.now();

        return { ok: true };
      },

      'snake-neuro:clearSession': (_, opts = {}) => {
        const key = ((typeof opts === 'string' ? opts : opts.instanceId) || 'default');
        _sessions.delete(key);
        return { ok: true };
      },

      'snake-neuro:getState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        return buildVisualState(getSession(id));
      },

      // ── Save / load handlers ──────────────────────────────────────────────

      'snake-neuro:saveState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        if (!s.pop) return { error: 'No trained model — run training first.' };
        savePopToStorage(storage, id, s);
        return { ok: true, generation: s.generation, bestFit: +s.bestFit.toFixed(3) };
      },

      'snake-neuro:loadState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        const pop = loadPopFromStorage(storage, id);
        if (!pop) return { error: 'No saved state found for this network.' };
        const wasRunning = s.running;
        s.running    = false;
        s.pop        = pop;
        s.generation = pop.generation || 0;
        s.bestFit    = Math.max(0, isFinite(pop.bestFitness) ? pop.bestFitness : 0);
        s.genHistory = (pop.history || []).slice(-100).map((h, i) => ({
          gen:  h.generation ?? i,
          best: +(h.stats?.max ?? 0).toFixed(3),
          mean: +(h.stats?.mean ?? 0).toFixed(3),
        }));
        const popSize = s.popSize || POP_SIZE;
        s.snakes       = Array.from({ length: popSize }, () => spawnSnake(s.gridW, s.gridH));
        s.snakeFit     = new Array(popSize).fill(0);
        s.genStartTime = Date.now();
        if (wasRunning) s.running = true;
        return { ok: true, generation: s.generation, bestFit: +s.bestFit.toFixed(3) };
      },

      // ── Inference handlers ────────────────────────────────────────────────

      'snake-neuro:inferInit': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);

        // If no live session, try loading from storage for infer-only use.
        if (!s.pop && storage) {
          const pop = loadPopFromStorage(storage, id);
          if (pop) {
            s.pop        = pop;
            s.generation = pop.generation || 0;
            s.bestFit    = Math.max(0, isFinite(pop.bestFitness) ? pop.bestFitness : 0);
            // Restore grid config from saved arch
            if (!s.arch) {
              try {
                const net = storage.getNetwork(id);
                if (net && net.architecture) {
                  const a = net.architecture;
                  s.gridW      = Math.max(4,  Math.min(30,  (a.gridW      | 0) || GRID_W));
                  s.gridH      = Math.max(4,  Math.min(30,  (a.gridH      | 0) || GRID_H));
                  s.staleLimit = Math.max(20, Math.min(400, (a.staleLimit | 0) || STALE_LIMIT));
                  s.popSize    = Math.max(5,  Math.min(200, (a.popSize    | 0) || POP_SIZE));
                  s.inputDim   = s.gridW * s.gridH;
                }
              } catch (_e) {}
            }
          }
        }

        if (!s.pop) return { error: 'No trained model — run training first.' };
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return { error: 'Population not ready.' };
        s.inferSnake = spawnSnake(s.gridW, s.gridH);
        return {
          ok: true,
          generation: s.generation,
          bestFit: +s.bestFit.toFixed(3),
          gridW: s.gridW || GRID_W,
          gridH: s.gridH || GRID_H,
        };
      },

      'snake-neuro:inferStep': (_, opts = {}) => {
        const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        const s  = getSession(id);
        if (!s.inferSnake || !s.pop) return null;
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return null;

        stepSnake(s.inferSnake, genome, s);
        let justReset = false;
        if (!s.inferSnake.alive) {
          s.inferSnake = spawnSnake(s.gridW, s.gridH);
          justReset = true;
        }

        return {
          snake: {
            body:         s.inferSnake.body,
            apple:        s.inferSnake.apple,
            alive:        s.inferSnake.alive,
            applesEaten:  s.inferSnake.applesEaten,
            dir:          s.inferSnake.dir,
          },
          justReset,
          generation: s.generation,
          bestFit:    +s.bestFit.toFixed(3),
          gridW:      s.gridW || GRID_W,
          gridH:      s.gridH || GRID_H,
        };
      },
    },
  };
};
