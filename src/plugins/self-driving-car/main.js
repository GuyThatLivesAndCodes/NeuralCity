'use strict';

const path = require('path');
const { app } = require('electron');

// Load neuroevolution + tensor from the app bundle.
let Population, T;
try {
  ({ Population } = require(path.join(app.getAppPath(), 'src', 'engine', 'neuroevolution')));
  T = require(path.join(app.getAppPath(), 'src', 'engine', 'tensor'));
} catch (e) {
  console.error('[self-driving-car] Failed to load engines:', e.message);
}

// ── Simulation constants ──────────────────────────────────────────────────────
const POP_SIZE    = 20;
const INPUT_DIM   = 7;   // 5 sensor rays + normalised speed + steer_memory
const OUTPUT_DIM  = 2;   // steer, throttle_raw
const TRACK_PTS   = 16;
const HALF_W      = 38;  // track half-width in pixels
const MAX_FRAMES  = 600; // ~20 s at 30 fps before forced death
const RAY_ANGLES  = [-1.2, -0.6, 0, 0.6, 1.2]; // 5 sensors in radians offset from heading
const RAY_MAX     = 180; // max sensor range px
const RAY_STEP    = 8;   // ray march step px
const DT          = 1 / 30;
const MAX_SPEED   = 130;
const ACCEL       = 170;
const FRICTION    = 25;
const STEER_RATE  = 2.3;
const CANVAS_CX   = 350;
const CANVAS_CY   = 265;

// ── Module-level state ────────────────────────────────────────────────────────
let _pop        = null;
let _track      = null;
let _cars       = [];
let _carFit     = [];      // fitness accumulated per car this generation
let _generation = 0;
let _bestFit    = 0;
let _genBestFit = 0;
let _running    = false;
let _genHistory = [];      // [{gen, best, mean}]

// ── LCG seeded random ─────────────────────────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }
function lcgFloat(s) { return [(lcg(s) >>> 0) / 4294967296, lcg(s)]; }

// ── Track generation ──────────────────────────────────────────────────────────

function generateTrack(seed) {
  let rng = (seed || 0xDEADBEEF) >>> 0;
  const N = TRACK_PTS;

  // Control points: random radii around a base circle
  const base = 155, jitter = 0.45;
  const angles = Array.from({ length: N }, (_, i) => (i / N) * 2 * Math.PI);
  const radii  = angles.map(() => {
    let f; [f, rng] = lcgFloat(rng);
    return base * (1 - jitter / 2 + f * jitter);
  });
  let pts = angles.map((a, i) => [
    CANVAS_CX + Math.cos(a) * radii[i],
    CANVAS_CY + Math.sin(a) * radii[i],
  ]);

  // Smooth 4 times — Chaikin-style averaging
  for (let iter = 0; iter < 4; iter++) {
    pts = pts.map((p, i) => {
      const prev = pts[(i - 1 + N) % N];
      const next = pts[(i + 1) % N];
      return [(prev[0] + p[0] * 2 + next[0]) / 4, (prev[1] + p[1] * 2 + next[1]) / 4];
    });
  }

  // Arc lengths for progress calculation
  const segLens = pts.map((p, i) => {
    const q = pts[(i + 1) % N];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
  const totalLen = segLens.reduce((a, b) => a + b, 0);

  return { pts, segLens, totalLen, halfWidth: HALF_W, N };
}

// ── Track geometry helpers ────────────────────────────────────────────────────

function distToSeg(px, py, ax, ay, bx, by) {
  const dx = bx - ax, dy = by - ay;
  const lenSq = dx * dx + dy * dy;
  if (lenSq < 1e-8) return Math.hypot(px - ax, py - ay);
  const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lenSq));
  return Math.hypot(px - (ax + t * dx), py - (ay + t * dy));
}

function isOnTrack(x, y, track) {
  const { pts, halfWidth, N } = track;
  for (let i = 0; i < N; i++) {
    const a = pts[i], b = pts[(i + 1) % N];
    if (distToSeg(x, y, a[0], a[1], b[0], b[1]) <= halfWidth) return true;
  }
  return false;
}

function castRay(cx, cy, angle, track) {
  const dx = Math.cos(angle), dy = Math.sin(angle);
  for (let d = RAY_STEP; d <= RAY_MAX; d += RAY_STEP) {
    if (!isOnTrack(cx + dx * d, cy + dy * d, track)) return (d - RAY_STEP) / RAY_MAX;
  }
  return 1.0;
}

function nearestSegIdx(x, y, track) {
  let best = 0, bestD = Infinity;
  for (let i = 0; i < track.N; i++) {
    const d = Math.hypot(x - track.pts[i][0], y - track.pts[i][1]);
    if (d < bestD) { bestD = d; best = i; }
  }
  return best;
}

// ── Car helpers ───────────────────────────────────────────────────────────────

function spawnCar(track) {
  const p0 = track.pts[0], p1 = track.pts[1];
  const angle = Math.atan2(p1[1] - p0[1], p1[0] - p0[0]);
  return {
    x: p0[0], y: p0[1],
    angle, speed: 0,
    alive: true,
    segIdx: 0,
    laps: 0,
    segProgress: 0,   // cumulative segment index
    frames: 0,
    totalDist: 0,     // total metres (proxy via segments)
  };
}

function senseCar(car, track) {
  const sensors = RAY_ANGLES.map(da => castRay(car.x, car.y, car.angle + da, track));
  sensors.push(car.speed / MAX_SPEED);         // speed normalised
  sensors.push(car.angle / (2 * Math.PI));     // steer memory (orientation)
  return new Float32Array(sensors);            // length INPUT_DIM = 7
}

function stepCar(car, steer, throttle) {
  car.angle += steer * STEER_RATE * DT;
  car.speed  = Math.max(0, Math.min(MAX_SPEED,
    car.speed + throttle * ACCEL * DT - FRICTION * DT));
  car.x += Math.cos(car.angle) * car.speed * DT;
  car.y += Math.sin(car.angle) * car.speed * DT;
  car.frames++;
}

function updateCarProgress(car, track) {
  const N     = track.N;
  const newSeg = nearestSegIdx(car.x, car.y, track);
  let delta   = newSeg - car.segIdx;
  if (delta >  N / 2) delta -= N;  // wrapped backward → big negative
  if (delta < -N / 2) delta += N;  // wrapped forward  → positive
  if (delta > 0) {
    car.segProgress += delta;
    if (car.segProgress >= N) { car.laps++; car.segProgress -= N; }
    car.totalDist += delta;
  }
  car.segIdx = newSeg;
}

function carFitness(car) {
  // Primary: distance traveled. Tie-break: avg speed.
  const progress   = car.laps + car.totalDist / Math.max(1, car.frames) * (car.frames / _track.N);
  const speedBonus = (car.frames > 0 ? car.speed / MAX_SPEED : 0) * 0.15;
  return progress + speedBonus;
}

// ── Neural net inference ──────────────────────────────────────────────────────

function netForward(model, inputArr) {
  const x   = new T.Tensor([1, INPUT_DIM], inputArr, false);
  const out = model.forward(x, { training: false });
  return out.data;
}

// ── Population step (advances all cars one physics tick) ─────────────────────

function stepGeneration(ticks) {
  if (!_pop || !_track || !_running) return null;
  ticks = Math.max(1, Math.min(ticks | 0, 6));

  for (let t = 0; t < ticks; t++) {
    let allDead = true;

    for (let i = 0; i < POP_SIZE; i++) {
      const car = _cars[i];
      if (!car.alive) continue;
      allDead = false;

      const inputs = senseCar(car, _track);
      const outs   = netForward(_pop.individuals[i], inputs);
      // Interpret outputs: tanh activation so both in [-1,1]
      const steer    = Math.max(-1, Math.min(1, outs[0]));
      const throttle = (outs[1] + 1) / 2;  // map [-1,1] → [0,1]

      stepCar(car, steer, throttle);
      updateCarProgress(car, _track);

      // Kill if off-track or timed out
      if (!isOnTrack(car.x, car.y, _track) || car.frames >= MAX_FRAMES) {
        car.alive = false;
        _carFit[i] = carFitness(car);
      }
    }

    if (allDead) {
      // All cars done — evaluate fitness (use pre-computed values) then evolve
      _pop.evaluate((_, idx) => _carFit[idx]);
      const stats = _pop.evolve();
      _generation++;
      const gen = { gen: _generation, best: +_pop.bestFitness.toFixed(3), mean: +stats.mean.toFixed(3) };
      _genHistory.push(gen);
      if (_genHistory.length > 100) _genHistory.shift();
      if (_pop.bestFitness > _bestFit) _bestFit = _pop.bestFitness;
      _genBestFit = stats.max;

      // Respawn all cars for the new generation
      _cars   = Array.from({ length: POP_SIZE }, () => spawnCar(_track));
      _carFit = new Array(POP_SIZE).fill(0);
      break; // Renderer will see new state on next call
    }
  }

  return buildVisualState();
}

function buildVisualState() {
  if (!_cars || !_track) return null;
  const aliveCnt = _cars.filter(c => c.alive).length;
  return {
    track: _track,
    cars: _cars.map(c => ({
      x: c.x, y: c.y, angle: c.angle, alive: c.alive,
      speed: c.speed, laps: c.laps,
    })),
    generation: _generation,
    aliveCnt,
    bestFit: +_bestFit.toFixed(3),
    genBestFit: +_genBestFit.toFixed(3),
    genHistory: _genHistory.slice(-60),
  };
}

// ── IPC handlers ──────────────────────────────────────────────────────────────
module.exports = {
  mainHandlers: {
    'self-driving-car:init': (_, opts = {}) => {
      if (!Population || !T) return { error: 'Neuroevolution engine unavailable.' };

      const seed = (opts.seed || 0) >>> 0;
      _track = generateTrack(seed || 0xC0FFEE);

      _pop = new Population({
        architecture: {
          kind: 'classifier',
          inputDim: INPUT_DIM, outputDim: OUTPUT_DIM,
          hidden: [16, 12], activation: 'tanh', dropout: 0,
        },
        size: POP_SIZE, eliteCount: 4,
        pMutate: 0.15, mutationStd: 0.05,
        tournamentK: 3, seed: opts.seed || 42,
      });

      _cars       = Array.from({ length: POP_SIZE }, () => spawnCar(_track));
      _carFit     = new Array(POP_SIZE).fill(0);
      _generation = 0;
      _bestFit    = 0;
      _genBestFit = 0;
      _running    = true;
      _genHistory = [];

      return { ok: true, track: _track };
    },

    'self-driving-car:getState': () => buildVisualState(),

    'self-driving-car:step': (_, ticks = 2) => stepGeneration(ticks),

    'self-driving-car:start': () => { _running = true;  return { ok: true }; },
    'self-driving-car:stop':  () => { _running = false; return { ok: true }; },

    'self-driving-car:newTrack': (_, seed) => {
      if (!Population) return { error: 'Not initialized.' };
      _track = generateTrack((seed || Math.floor(Math.random() * 0xFFFFFF)) >>> 0);
      _cars  = Array.from({ length: POP_SIZE }, () => spawnCar(_track));
      _carFit = new Array(POP_SIZE).fill(0);
      return { ok: true, track: _track };
    },

    'self-driving-car:reset': () => {
      if (!Population || !T) return { error: 'Not initialized.' };
      const seed = Math.floor(Math.random() * 0xFFFFFF);
      _track = generateTrack(seed);
      _pop   = new Population({
        architecture: {
          kind: 'classifier',
          inputDim: INPUT_DIM, outputDim: OUTPUT_DIM,
          hidden: [16, 12], activation: 'tanh', dropout: 0,
        },
        size: POP_SIZE, eliteCount: 4,
        pMutate: 0.15, mutationStd: 0.05,
        tournamentK: 3, seed: seed,
      });
      _cars       = Array.from({ length: POP_SIZE }, () => spawnCar(_track));
      _carFit     = new Array(POP_SIZE).fill(0);
      _generation = 0;
      _bestFit    = 0;
      _genBestFit = 0;
      _running    = true;
      _genHistory = [];
      return { ok: true };
    },
  },
};
