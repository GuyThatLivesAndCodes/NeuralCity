'use strict';

const path = require('path');
const { app } = require('electron');

// Load DQNAgent from the app bundle so this works from the userData plugin copy.
let _DQNAgent;
try {
  _DQNAgent = require(path.join(app.getAppPath(), 'src', 'engine', 'rl')).DQNAgent;
} catch (e) {
  console.error('[warehouse-robot] Failed to load DQNAgent:', e.message);
}

// ── Environment constants ─────────────────────────────────────────────────────
const GRID = 8;
const N_BOXES = 3;
const MAX_STEPS = 200;
// State: robot(r,c) + 3×box(r,c) + 3×target(r,c) = 14 floats, all / (GRID-1)
const STATE_DIM = 2 + N_BOXES * 2 + N_BOXES * 2;
const N_ACTIONS = 4;
const DIRS = [[-1, 0], [1, 0], [0, -1], [0, 1]]; // UP DOWN LEFT RIGHT

// ── Module-level state (persists across IPC calls) ────────────────────────────
let _agent   = null;
let _env     = null;
let _running = false;
let _episode = 0;
let _epReward   = 0;
let _totalSteps = 0;
let _bestReward = -Infinity;
let _rewardHistory = [];

// ── Tiny LCG for seeded random positions ─────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }

// ── Grid helpers ──────────────────────────────────────────────────────────────

function resetEnv(seed) {
  let rng = (seed || (Math.floor(Math.random() * 0x7FFFFFFF) + 1)) >>> 0;
  const used = new Set();

  function randCell() {
    let r, c;
    do {
      rng = lcg(rng); r = rng % GRID;
      rng = lcg(rng); c = rng % GRID;
    } while (used.has(r * GRID + c));
    used.add(r * GRID + c);
    return [r, c];
  }

  const robot   = randCell();
  const boxes   = [randCell(), randCell(), randCell()];
  const targets = [randCell(), randCell(), randCell()];
  return { robot, boxes, targets, step: 0, onTarget: countOnTarget(boxes, targets) };
}

function countOnTarget(boxes, targets) {
  let n = 0;
  for (const t of targets)
    if (boxes.some(b => b[0] === t[0] && b[1] === t[1])) n++;
  return n;
}

function encodeState(env) {
  const G = GRID - 1;
  const v = new Float32Array(STATE_DIM);
  v[0] = env.robot[0] / G;
  v[1] = env.robot[1] / G;
  for (let i = 0; i < N_BOXES; i++) {
    v[2 + i * 2]     = env.boxes[i][0] / G;
    v[2 + i * 2 + 1] = env.boxes[i][1] / G;
  }
  for (let i = 0; i < N_BOXES; i++) {
    v[2 + N_BOXES * 2 + i * 2]     = env.targets[i][0] / G;
    v[2 + N_BOXES * 2 + i * 2 + 1] = env.targets[i][1] / G;
  }
  return v;
}

function stepEnv(env, action) {
  const [dr, dc] = DIRS[action];
  const [r, c]   = env.robot;
  const nr = r + dr, nc = c + dc;

  // Hit wall
  if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) {
    return { env: { ...env, step: env.step + 1 }, reward: -0.5, done: env.step + 1 >= MAX_STEPS };
  }

  let boxes = env.boxes;
  const bi = boxes.findIndex(b => b[0] === nr && b[1] === nc);

  if (bi >= 0) {
    const bnr = nr + dr, bnc = nc + dc;
    // Box push is blocked by wall or another box
    if (bnr < 0 || bnr >= GRID || bnc < 0 || bnc >= GRID ||
        boxes.some((b, i) => i !== bi && b[0] === bnr && b[1] === bnc)) {
      return { env: { ...env, step: env.step + 1 }, reward: -0.4, done: env.step + 1 >= MAX_STEPS };
    }
    boxes = boxes.map((b, i) => i === bi ? [bnr, bnc] : b);
  }

  const newEnv = { ...env, robot: [nr, nc], boxes, step: env.step + 1 };
  const onNow  = countOnTarget(newEnv.boxes, newEnv.targets);
  const onPrev = env.onTarget;
  newEnv.onTarget = onNow;

  let reward = -0.01; // small time penalty
  if (onNow > onPrev) reward += 10 * (onNow - onPrev);  // box placed on target
  if (onNow < onPrev) reward -= 3  * (onPrev - onNow);  // box knocked off target

  const done = (onNow === N_BOXES) || (newEnv.step >= MAX_STEPS);
  if (onNow === N_BOXES) reward += 50; // all boxes placed

  return { env: newEnv, reward, done };
}

function currentVisualState() {
  if (!_env) return null;
  return {
    grid: GRID, nBoxes: N_BOXES,
    robot: _env.robot,
    boxes: _env.boxes,
    targets: _env.targets,
    onTarget: _env.onTarget,
    episode: _episode,
    stepInEp: _env.step,
    totalSteps: _totalSteps,
    epReward: +_epReward.toFixed(2),
    bestReward: _bestReward === -Infinity ? null : +_bestReward.toFixed(2),
    epsilon: _agent ? +_agent.epsilon.toFixed(4) : 1.0,
    rewardHistory: _rewardHistory.slice(-80),
  };
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'warehouse-robot:init': (_, opts = {}) => {
      if (!_DQNAgent) return { error: 'DQNAgent module unavailable — check app bundle.' };
      _agent = new _DQNAgent({
        architecture: {
          kind: 'classifier',
          inputDim: STATE_DIM, outputDim: N_ACTIONS,
          hidden: [128, 64], activation: 'relu', dropout: 0,
        },
        gamma: 0.95, lr: opts.lr || 1e-3,
        batchSize: 64, bufferCapacity: 10000,
        epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.9995,
        targetUpdateFreq: 200, seed: opts.seed || 42, optimizer: 'adam',
      });
      _env       = resetEnv();
      _running   = true;
      _episode   = 0; _epReward = 0; _totalSteps = 0;
      _bestReward = -Infinity; _rewardHistory = [];
      return { ok: true, grid: GRID, nBoxes: N_BOXES };
    },

    'warehouse-robot:getState': () => currentVisualState(),

    'warehouse-robot:step': (_, n = 4) => {
      if (!_agent || !_env || !_running) return currentVisualState();
      n = Math.max(1, Math.min(n | 0, 40));

      for (let i = 0; i < n; i++) {
        const s      = encodeState(_env);
        const a      = _agent.selectAction(s);
        const { env: next, reward, done } = stepEnv(_env, a);
        const ns     = encodeState(next);
        _agent.observe(s, a, reward, ns, done);
        _agent.trainStep();
        _epReward   += reward;
        _totalSteps += 1;
        _env         = next;

        if (done) {
          _rewardHistory.push(+_epReward.toFixed(2));
          if (_epReward > _bestReward) _bestReward = _epReward;
          _episode++;
          _epReward = 0;
          _env = resetEnv();
        }
      }

      return currentVisualState();
    },

    'warehouse-robot:start': () => { _running = true;  return { ok: true }; },
    'warehouse-robot:stop':  () => { _running = false; return { ok: true }; },

    'warehouse-robot:reset': () => {
      if (!_agent) return { error: 'Not initialized — call init first.' };
      _env = resetEnv();
      _episode = 0; _epReward = 0; _totalSteps = 0;
      _bestReward = -Infinity; _rewardHistory = [];
      _agent.epsilon = _agent.epsilonStart;
      _running = true;
      return { ok: true };
    },
  },
};
