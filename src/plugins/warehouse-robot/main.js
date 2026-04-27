'use strict';

const path = require('path');
const { app } = require('electron');

let _DQNAgent;
try {
  _DQNAgent = require(path.join(app.getAppPath(), 'src', 'engine', 'rl')).DQNAgent;
} catch (e) {
  console.error('[warehouse-robot] Failed to load DQNAgent:', e.message);
}

// ── Environment constants ─────────────────────────────────────────────────────
const GRID       = 8;
const MAX_STEPS  = 200;
const N_ACTIONS  = 4;
const N_OBSTACLES = 4;
const DIRS       = [[-1, 0], [1, 0], [0, -1], [0, 1]];

// State: robot(2) + carrying(1) + boxes(nBoxes×2) + targets(nBoxes×2)
//        + box→target relative offsets(nBoxes×2) + robot→goal relative(2)
// = 5 + nBoxes×6
function computeStateDim(nBoxes) { return 5 + nBoxes * 6; }

// ── Per-instance sessions ─────────────────────────────────────────────────────
function makeSession() {
  return {
    agent: null, env: null, running: false,
    episode: 0, epReward: 0, totalSteps: 0,
    bestReward: -Infinity, rewardHistory: [],
    inferEnv: null, inferEpReward: 0, inferLapsDone: 0,
    nBoxes: 1,
  };
}
const _sessions = new Map();
function getSession(id) {
  const key = id || 'default';
  if (!_sessions.has(key)) _sessions.set(key, makeSession());
  return _sessions.get(key);
}

// ── LCG for seeded obstacle generation ───────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }

function generateObstacles(seed) {
  let rng = (seed || 0xABCD1234) >>> 0;
  const used = new Set();
  function randCell() {
    let r, c, key;
    do {
      rng = lcg(rng); r = Math.floor((rng / 4294967296) * GRID);
      rng = lcg(rng); c = Math.floor((rng / 4294967296) * GRID);
      key = r * GRID + c;
    } while (used.has(key));
    used.add(key);
    return [r, c];
  }
  return Array.from({ length: N_OBSTACLES }, randCell);
}

// ── Per-episode reset ─────────────────────────────────────────────────────────
function resetEpisode(nBoxes, obstacles) {
  const used = new Set(obstacles.map(([r, c]) => r * GRID + c));
  function randCell() {
    let r, c, key;
    do {
      r = Math.floor(Math.random() * GRID);
      c = Math.floor(Math.random() * GRID);
      key = r * GRID + c;
    } while (used.has(key));
    used.add(key);
    return [r, c];
  }
  const robot   = randCell();
  const boxes   = Array.from({ length: nBoxes }, randCell);
  const targets = Array.from({ length: nBoxes }, randCell);
  return {
    robot, boxes, targets, obstacles,
    carrying: -1,
    deliveredMask: new Array(nBoxes).fill(false),
    delivered: 0,
    step: 0,
  };
}

// ── State encoding ────────────────────────────────────────────────────────────
function encodeState(env, nBoxes) {
  const G = GRID - 1;
  const v = new Float32Array(computeStateDim(nBoxes));
  let idx = 0;

  v[idx++] = env.robot[0] / G;
  v[idx++] = env.robot[1] / G;
  v[idx++] = env.carrying >= 0 ? 1.0 : 0.0;

  // Absolute box positions (at target if delivered, at robot if carried)
  for (let i = 0; i < nBoxes; i++) {
    let br, bc;
    if (env.deliveredMask[i])    { [br, bc] = env.targets[i]; }
    else if (env.carrying === i) { [br, bc] = env.robot; }
    else                         { [br, bc] = env.boxes[i]; }
    v[idx++] = br / G;
    v[idx++] = bc / G;
  }

  // Absolute target positions
  for (let i = 0; i < nBoxes; i++) {
    v[idx++] = env.targets[i][0] / G;
    v[idx++] = env.targets[i][1] / G;
  }

  // Relative box→target offset for each box (direct "error vector")
  for (let i = 0; i < nBoxes; i++) {
    if (env.deliveredMask[i]) {
      v[idx++] = 0; v[idx++] = 0;
    } else {
      const [br, bc] = env.carrying === i ? env.robot : env.boxes[i];
      v[idx++] = (env.targets[i][0] - br) / G;
      v[idx++] = (env.targets[i][1] - bc) / G;
    }
  }

  // Relative robot→immediate goal (nearest undelivered box if not carrying; target if carrying)
  if (env.carrying >= 0) {
    const [tr, tc] = env.targets[env.carrying];
    v[idx++] = (tr - env.robot[0]) / G;
    v[idx++] = (tc - env.robot[1]) / G;
  } else {
    let bestDr = 0, bestDc = 0, bestD = Infinity;
    for (let i = 0; i < nBoxes; i++) {
      if (!env.deliveredMask[i]) {
        const d = Math.abs(env.robot[0] - env.boxes[i][0]) + Math.abs(env.robot[1] - env.boxes[i][1]);
        if (d < bestD) {
          bestD = d;
          bestDr = env.boxes[i][0] - env.robot[0];
          bestDc = env.boxes[i][1] - env.robot[1];
        }
      }
    }
    v[idx++] = bestDr / G;
    v[idx++] = bestDc / G;
  }

  return v;
}

// ── Shaping potential ─────────────────────────────────────────────────────────
function shapingPotential(env, nBoxes) {
  if (env.carrying >= 0) {
    const [tr, tc] = env.targets[env.carrying];
    return -(Math.abs(env.robot[0] - tr) + Math.abs(env.robot[1] - tc));
  }
  let best = Infinity;
  for (let i = 0; i < nBoxes; i++) {
    if (!env.deliveredMask[i]) {
      const d = Math.abs(env.robot[0] - env.boxes[i][0]) + Math.abs(env.robot[1] - env.boxes[i][1]);
      best = Math.min(best, d);
    }
  }
  return best === Infinity ? 0 : -best;
}

// ── Environment step ──────────────────────────────────────────────────────────
function stepEnv(env, action, nBoxes) {
  const [dr, dc] = DIRS[action];
  const [r, c]   = env.robot;
  const nr = r + dr, nc = c + dc;
  const nextStep = env.step + 1;

  if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID) {
    return { env: { ...env, step: nextStep }, reward: -0.5, done: nextStep >= MAX_STEPS };
  }

  if (env.obstacles.some(o => o[0] === nr && o[1] === nc)) {
    return { env: { ...env, step: nextStep }, reward: -0.3, done: nextStep >= MAX_STEPS };
  }

  const oldPotential = shapingPotential(env, nBoxes);
  let newEnv = { ...env, robot: [nr, nc], step: nextStep };
  let reward = -0.01;

  // Pick up undelivered box at new cell (only if not already carrying)
  if (newEnv.carrying < 0) {
    for (let i = 0; i < nBoxes; i++) {
      if (!newEnv.deliveredMask[i] && newEnv.boxes[i][0] === nr && newEnv.boxes[i][1] === nc) {
        newEnv = { ...newEnv, carrying: i };
        reward += 1.0;
        break;
      }
    }
  }

  // Deliver carried box to its target
  if (newEnv.carrying >= 0) {
    const bi = newEnv.carrying;
    if (newEnv.targets[bi][0] === nr && newEnv.targets[bi][1] === nc) {
      const deliveredMask = [...newEnv.deliveredMask];
      deliveredMask[bi] = true;
      const delivered = newEnv.delivered + 1;
      newEnv = { ...newEnv, carrying: -1, deliveredMask, delivered };
      reward += 10.0;
      if (delivered === nBoxes) reward += 50.0;
    }
  }

  // Potential-based dense shaping (scaled small to not overpower event rewards)
  reward += 0.1 * (shapingPotential(newEnv, nBoxes) - oldPotential);

  const done = (newEnv.delivered === nBoxes) || (nextStep >= MAX_STEPS);
  return { env: newEnv, reward, done };
}

// ── Visual state ──────────────────────────────────────────────────────────────
function buildVisualState(s) {
  if (!s.env) return null;
  const env = s.env;
  return {
    grid: GRID, nBoxes: s.nBoxes,
    robot:         env.robot,
    carrying:      env.carrying,
    boxes:         env.boxes,
    targets:       env.targets,
    obstacles:     env.obstacles,
    deliveredMask: env.deliveredMask,
    delivered:     env.delivered,
    episode:       s.episode,
    stepInEp:      env.step,
    totalSteps:    s.totalSteps,
    epReward:      +s.epReward.toFixed(2),
    bestReward:    s.bestReward === -Infinity ? null : +s.bestReward.toFixed(2),
    epsilon:       s.agent ? +s.agent.epsilon.toFixed(4) : 1.0,
    rewardHistory: s.rewardHistory.slice(-80),
  };
}

// ── Gaussian noise ────────────────────────────────────────────────────────────
function gaussRand() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'warehouse-robot:init': (_, opts = {}) => {
      if (!_DQNAgent) return { error: 'DQNAgent module unavailable — check app bundle.' };
      const id = opts.instanceId || 'default';
      const s  = getSession(id);

      s.nBoxes = Math.max(1, Math.min(5, (opts.nBoxes | 0) || 1));
      const obstacles = generateObstacles(opts.seed || 42);

      s.agent = new _DQNAgent({
        architecture: {
          kind: 'classifier',
          inputDim: computeStateDim(s.nBoxes), outputDim: N_ACTIONS,
          hidden: [128, 64], activation: 'relu', dropout: 0,
        },
        gamma: 0.95,
        lr: opts.lr || opts.learningRate || 1e-3,
        batchSize:      opts.batchSize || 64,
        bufferCapacity: 5000,
        epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.999,
        targetUpdateFreq: 200,
        seed: opts.seed || 42,
        optimizer: 'adam',
      });

      s.env          = resetEpisode(s.nBoxes, obstacles);
      s.running      = true;
      s.episode      = 0;
      s.epReward     = 0;
      s.totalSteps   = 0;
      s.bestReward   = -Infinity;
      s.rewardHistory = [];
      return { ok: true, grid: GRID, nBoxes: s.nBoxes };
    },

    'warehouse-robot:getState': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      return buildVisualState(getSession(id));
    },

    'warehouse-robot:step': (_, opts = {}) => {
      const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
      const s  = getSession(id);
      if (!s.agent || !s.env || !s.running) return buildVisualState(s);
      const n = Math.max(1, Math.min((opts.n || (typeof opts === 'number' ? opts : 4)) | 0, 20));

      for (let i = 0; i < n; i++) {
        const state  = encodeState(s.env, s.nBoxes);
        const a      = s.agent.selectAction(state);
        const { env: next, reward, done } = stepEnv(s.env, a, s.nBoxes);
        const ns     = encodeState(next, s.nBoxes);
        s.agent.observe(state, a, reward, ns, done);
        s.epReward   += reward;
        s.totalSteps += 1;
        s.env         = next;

        if (done) {
          s.rewardHistory.push(+s.epReward.toFixed(2));
          if (s.rewardHistory.length > 200) s.rewardHistory.shift();
          if (s.epReward > s.bestReward) s.bestReward = s.epReward;
          s.episode++;
          s.epReward = 0;
          s.env = resetEpisode(s.nBoxes, s.env.obstacles);
        }
      }

      if (s.agent.buffer.ready) s.agent.trainStep();
      return buildVisualState(s);
    },

    'warehouse-robot:start': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      getSession(id).running = true;
      return { ok: true };
    },

    'warehouse-robot:stop': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      getSession(id).running = false;
      return { ok: true };
    },

    'warehouse-robot:reset': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      const s  = getSession(id);
      if (!s.agent) return { error: 'Not initialized — call init first.' };
      s.env           = resetEpisode(s.nBoxes, s.env.obstacles);
      s.episode       = 0;
      s.epReward      = 0;
      s.totalSteps    = 0;
      s.bestReward    = -Infinity;
      s.rewardHistory = [];
      s.agent.epsilon = s.agent.epsilonStart;
      s.running       = true;
      return { ok: true };
    },

    'warehouse-robot:inferInit': (_, opts = {}) => {
      const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
      const s  = getSession(id);
      if (!s.agent) return { error: 'No trained agent — run training first.' };
      s.inferEnv      = resetEpisode(s.nBoxes, s.env.obstacles);
      s.inferEpReward = 0;
      s.inferLapsDone = 0;
      return {
        ok: true, grid: GRID, nBoxes: s.nBoxes,
        epsilon:    +s.agent.epsilon.toFixed(4),
        totalSteps: s.totalSteps,
        episode:    s.episode,
        bestReward: s.bestReward === -Infinity ? null : +s.bestReward.toFixed(2),
      };
    },

    'warehouse-robot:inferStep': (_, opts = {}) => {
      const id       = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
      const noiseStd = (typeof opts === 'object' ? opts.noiseStd : 0) || 0;
      const s        = getSession(id);
      if (!s.agent || !s.inferEnv) return null;

      const rawState = encodeState(s.inferEnv, s.nBoxes);
      let state = rawState;
      if (noiseStd > 0) {
        state = new Float32Array(rawState.length);
        for (let i = 0; i < rawState.length; i++) state[i] = rawState[i] + gaussRand() * noiseStd;
      }

      const savedEps = s.agent.epsilon;
      s.agent.epsilon = 0;
      const action = s.agent.selectAction(state);
      s.agent.epsilon = savedEps;

      const { env: next, reward, done } = stepEnv(s.inferEnv, action, s.nBoxes);
      s.inferEpReward += reward;
      s.inferEnv       = next;

      let justReset = false;
      if (done) {
        s.inferLapsDone++;
        s.inferEpReward = 0;
        s.inferEnv      = resetEpisode(s.nBoxes, s.env.obstacles);
        justReset       = true;
      }

      return {
        grid: GRID, nBoxes: s.nBoxes,
        robot:         s.inferEnv.robot,
        carrying:      s.inferEnv.carrying,
        boxes:         s.inferEnv.boxes,
        targets:       s.inferEnv.targets,
        obstacles:     s.inferEnv.obstacles,
        deliveredMask: s.inferEnv.deliveredMask,
        delivered:     s.inferEnv.delivered,
        epReward:      +s.inferEpReward.toFixed(2),
        episodesDone:  s.inferLapsDone,
        justReset,
      };
    },
  },
};
