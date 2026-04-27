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

// ── Simulation constants ───────────────────────────────────────────────────────
const GRID_W          = 15;
const GRID_H          = 17;
const INPUT_DIM       = GRID_W * GRID_H;   // 255 — full grid occupancy map
const OUTPUT_DIM      = 4;                  // up, right, down, left
const POP_SIZE        = 50;                 // Optimized: Increased population for better exploration
const GEN_DURATION_MS = 8000;               // Optimized: Slightly longer generation for better evaluation
const STALE_LIMIT     = 150;                // Optimized: Tighter limit to force efficient pathfinding

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
    snakes:       [],
    snakeFit:     [],
    generation:   0,
    bestFit:      0,
    genBestFit:   0,
    running:      false,
    genHistory:   [],
    genStartTime: 0,
    inferSnake:   null,
  };
}

const _sessions = new Map();
function getSession(id) {
  const key = id || 'default';
  if (!_sessions.has(key)) _sessions.set(key, makeSession());
  return _sessions.get(key);
}

// ── Snake helpers ──────────────────────────────────────────────────────────────

function randomPos(body) {
  const total      = GRID_W * GRID_H;
  const occupiedSet = new Set(body.map(p => p.y * GRID_W + p.x));
  if (occupiedSet.size >= total) return { x: 0, y: 0 };
  let idx;
  do { idx = Math.floor(Math.random() * total); } while (occupiedSet.has(idx));
  return { x: idx % GRID_W, y: Math.floor(idx / GRID_W) };
}

function spawnSnake() {
  const cx = Math.floor(GRID_W / 2);
  const cy = Math.floor(GRID_H / 2);
  const body = [
    { x: cx, y: cy },
    { x: cx, y: cy + 1 },
    { x: cx, y: cy + 2 },
  ];
  return {
    body,               // body[0] = head, body[last] = tail
    dir:              0, // 0=up initially
    apple:            randomPos(body),
    alive:            true,
    applesEaten:      0,
    steps:            0,
    stepsSinceApple:  0,
  };
}

/**
 * Optimized Grid Encoding with 5-class labels:
 * 0.0  = Blank
 * 1.0  = Apple
 * 0.5  = Snake Head
 * 0.25 = Snake Body
 * -0.5 = Snake End (Tail)
 *
 * Using distinct values helps the neural network distinguish between functional parts.
 */
function buildGrid(snake) {
  const grid = new Float32Array(INPUT_DIM);
  // Fill snake parts
  for (let i = 0; i < snake.body.length; i++) {
    const { x, y } = snake.body[i];
    const idx = y * GRID_W + x;
    if (i === 0) {
      grid[idx] = 0.5; // Snake Head
    } else if (i === snake.body.length - 1) {
      grid[idx] = -0.5; // Snake End (Tail)
    } else {
      grid[idx] = 0.25; // Snake Body
    }
  }
  // Fill apple
  grid[snake.apple.y * GRID_W + snake.apple.x] = 1.0; // Apple
  return grid;
}

function netForward(model, inputArr) {
  const x   = new T.Tensor([1, INPUT_DIM], inputArr, false);
  const out = model.forward(x, { training: false });
  return out.data;
}

function argmax(arr) {
  let best = 0;
  for (let i = 1; i < OUTPUT_DIM; i++) if (arr[i] > arr[best]) best = i;
  return best;
}

function stepSnake(snake, model) {
  if (!snake.alive) return;

  const grid   = buildGrid(snake);
  const outs   = netForward(model, grid);
  const rawDir = argmax(outs);

  const opposite = (snake.dir + 2) % 4;
  snake.dir = (rawDir === opposite) ? snake.dir : rawDir;

  const [dx, dy] = DIRS[snake.dir];
  const nx = snake.body[0].x + dx;
  const ny = snake.body[0].y + dy;

  if (nx < 0 || nx >= GRID_W || ny < 0 || ny >= GRID_H) {
    snake.alive = false;
    return;
  }

  // Optimized self-collision: check all segments
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
    snake.apple = randomPos(snake.body);
  } else {
    snake.body.pop();
  }

  snake.steps++;
  snake.stepsSinceApple++;

  if (snake.stepsSinceApple > STALE_LIMIT) {
    snake.alive = false;
  }
}

/**
 * Optimized Fitness Function:
 * Encourages eating apples while providing small rewards for surviving and
 * moving towards apples to avoid random wandering.
 */
function manhattanDistance(p1, p2) {
  return Math.abs(p1.x - p2.x) + Math.abs(p1.y - p2.y);
}

const MAX_DISTANCE = GRID_W + GRID_H - 2; // Max Manhattan distance on the grid

function snakeFitness(snake) {
  let fitness = snake.applesEaten;

  // Incorporate distance to apple
  if (snake.alive) {
    const head = snake.body[0];
    const distToApple = manhattanDistance(head, snake.apple);
    fitness -= (distToApple / MAX_DISTANCE);
  }

  // Apply death penalty
  if (!snake.alive) {
    fitness *= 0.9;
  }

  // Ensure fitness is not negative
  return Math.max(0, fitness);
}

function stepGeneration(s, opts) {
  if (!s.pop || !s.running) return null;

  const ticks   = Math.max(1, Math.min((opts.ticks || 10), 50)); // Optimized: Higher tick cap for faster simulation
  const now     = Date.now();
  const elapsed = now - s.genStartTime;
  const allDead = s.snakes.every(sn => !sn.alive);

  if (elapsed >= GEN_DURATION_MS || allDead) {
    s.snakes.forEach((sn, i) => { s.snakeFit[i] = snakeFitness(sn); });
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

    s.snakes       = Array.from({ length: POP_SIZE }, () => spawnSnake());
    s.snakeFit     = new Array(POP_SIZE).fill(0);
    s.genStartTime = Date.now();

  } else {
    for (let t = 0; t < ticks; t++) {
      for (let i = 0; i < POP_SIZE; i++) {
        if (s.snakes[i].alive) stepSnake(s.snakes[i], s.pop.individuals[i]);
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
    generation:   s.generation,
    aliveCnt:     s.snakes.filter(sn => sn.alive).length,
    bestFit:      +s.bestFit.toFixed(3),
    genBestFit:   +s.genBestFit.toFixed(3),
    genHistory:   s.genHistory.slice(-60),
    timeLeft,
    hasBestGenome: s.pop !== null && s.pop.bestIndividual !== null,
  };
}

// ── Storage helpers ────────────────────────────────────────────────────────────

function savePopToStorage(storage, id, s) {
  if (!storage || !s.pop) return;
  try {
    storage.saveTrainedState(id, { state: s.pop.toJSON(), architecture: ARCH });
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
        const id = opts.instanceId || 'default';
        const s  = getSession(id);
        const mutStd = opts.mutStd || 0.05;

        // Resume from saved state when available; fall back to fresh population.
        const savedPop = loadPopFromStorage(storage, id);
        if (savedPop) {
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
          s.pop = new Population({
            architecture: ARCH,
            size:         POP_SIZE,
            eliteCount:   2,
            pMutate:      0.15,
            mutationStd:  mutStd,
            tournamentK:  3,
            seed:         opts.seed || 42,
          });
          s.generation   = 0;
          s.bestFit      = 0;
          s.genHistory   = [];
        }

        s.snakes       = Array.from({ length: POP_SIZE }, () => spawnSnake());
        s.snakeFit     = new Array(POP_SIZE).fill(0);
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

        s.pop = new Population({
          architecture: ARCH,
          size:        POP_SIZE,
          eliteCount:  2,
          pMutate:     0.15,
          mutationStd: 0.05,
          tournamentK: 3,
          seed:        Math.floor(Math.random() * 0xFFFFFF),
        });

        s.snakes       = Array.from({ length: POP_SIZE }, () => spawnSnake());
        s.snakeFit     = new Array(POP_SIZE).fill(0);
        s.generation   = 0;
        s.bestFit      = 0;
        s.genBestFit   = 0;
        s.running      = true;
        s.genHistory   = [];
        s.genStartTime = Date.now();

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
        s.snakes       = Array.from({ length: POP_SIZE }, () => spawnSnake());
        s.snakeFit     = new Array(POP_SIZE).fill(0);
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
          }
        }

        if (!s.pop) return { error: 'No trained model — run training first.' };
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return { error: 'Population not ready.' };
        s.inferSnake = spawnSnake();
        return { ok: true, generation: s.generation, bestFit: +s.bestFit.toFixed(3) };
      },

      'snake-neuro:inferStep': (_, opts = {}) => {
        const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        const s  = getSession(id);
        if (!s.inferSnake || !s.pop) return null;
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return null;

        stepSnake(s.inferSnake, genome);
        let justReset = false;
        if (!s.inferSnake.alive) {
          s.inferSnake = spawnSnake();
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
        };
      },
    },
  };
};
