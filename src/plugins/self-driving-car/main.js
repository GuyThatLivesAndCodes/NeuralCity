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
  console.error('[self-driving-car] Failed to load engines:', e.message);
}

// ── Simulation constants ──────────────────────────────────────────────────────
const DEFAULT_POP_SIZE = 20;
const INPUT_DIM   = 11;   // 9 sensor rays + normalised speed + steer_memory
const OUTPUT_DIM  = 2;    // steer (-1 to 1), throttle/brake (-1 to 1)
const TRACK_PTS   = 8;    // Base control points before Chaikin smoothing
const MAX_FRAMES  = (30 * 18);  // ~18 s episode limit

const RAY_ANGLES  = Array.from({length: 9}, (_, i) => -1.4 + (i * 2.8 / 14));
const RAY_MAX     = 40;
const RAY_STEP    = 5;

const DT          = 1 / 30;
const MAX_SPEED   = 340;
const ACCEL       = 230;
const BRAKE_FORCE = 350;    // Stronger than acceleration for realistic braking
const FRICTION    = 0.93;   // Multiplicative drag applied every frame
const STEER_RATE  = 2.4;
const GRIP_LOSS   = 0.4;    // Speed lost during hard cornering (proportional to steer × speed)

const CANVAS_CX   = 350;
const CANVAS_CY   = 285;

const ARCH = {
  kind: 'classifier',
  inputDim: INPUT_DIM, outputDim: OUTPUT_DIM,
  hidden: [64, 32, 16], activation: 'tanh', dropout: 0,
};

// ── Per-instance sessions ─────────────────────────────────────────────────────
function makeSession() {
  return {
    pop: null, popSize: DEFAULT_POP_SIZE, maxGens: 0,
    track: null, cars: [], carFit: [],
    generation: 0, bestFit: 0, genBestFit: 0,
    running: false, genHistory: [],
    inferCar: null, inferTrack: null,
    arch: null,
  };
}
const _sessions = new Map();
function getSession(id) {
  const key = id || 'default';
  if (!_sessions.has(key)) _sessions.set(key, makeSession());
  return _sessions.get(key);
}

// ── LCG seeded random ─────────────────────────────────────────────────────────
function lcg(s) { return (Math.imul(s | 0, 1664525) + 1013904223) >>> 0; }
function lcgFloat(s) { return [(lcg(s) >>> 0) / 4294967296, lcg(s)]; }

// ── Track generation ──────────────────────────────────────────────────────────

function generateTrack(seed) {
  let rng = (seed || 0xDEADBEEF) >>> 0;

  function nextRand() {
    let f; [f, rng] = lcgFloat(rng);
    return f;
  }

  const initialN  = TRACK_PTS * 2;
  const baseRadius = 200;
  const complexity = Math.floor(nextRand() * 3) + 2;
  const phase      = nextRand() * Math.PI * 2;
  const lobeDepth  = 40 + nextRand() * 60;
  const jitter     = 0.15;

  const angles = Array.from({ length: initialN }, (_, i) => (i / initialN) * 2 * Math.PI);

  const radii = angles.map((a) => {
    let r = baseRadius + Math.sin(a * complexity + phase) * lobeDepth;
    r += (nextRand() - 0.5) * baseRadius * jitter;
    return Math.max(50, Math.min(230, r));
  });

  let pts = angles.map((a, i) => [
    CANVAS_CX + Math.cos(a) * radii[i],
    CANVAS_CY + Math.sin(a) * radii[i],
  ]);

  // True Chaikin corner-cutting — 4 passes doubles point count each time
  for (let iter = 0; iter < 4; iter++) {
    const newPts = [];
    for (let i = 0; i < pts.length; i++) {
      const p1 = pts[i], p2 = pts[(i + 1) % pts.length];
      newPts.push([p1[0] * 0.75 + p2[0] * 0.25, p1[1] * 0.75 + p2[1] * 0.25]);
      newPts.push([p1[0] * 0.25 + p2[0] * 0.75, p1[1] * 0.25 + p2[1] * 0.75]);
    }
    pts = newPts;
  }

  const segLens = pts.map((p, i) => {
    const q = pts[(i + 1) % pts.length];
    return Math.hypot(q[0] - p[0], q[1] - p[1]);
  });
  const totalLen = segLens.reduce((a, b) => a + b, 0);

  return { pts, segLens, totalLen, halfWidth: 22, N: pts.length };
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
    angle, speed: 10,
    alive: true,
    segIdx: 0,
    laps: 0,
    segProgress: 0,
    frames: 0,
    totalDist: 0,
  };
}

function senseCar(car, track) {
  const sensors = RAY_ANGLES.map(da => castRay(car.x, car.y, car.angle + da, track));
  sensors.push(car.speed / MAX_SPEED);
  sensors.push(car.angle / (2 * Math.PI));
  return new Float32Array(sensors);  // length INPUT_DIM = 11
}

function stepCar(car, steerOut, throttleOut) {
  const gripLoss = GRIP_LOSS * Math.abs(steerOut) * (car.speed / MAX_SPEED);
  car.angle += steerOut * STEER_RATE * DT;
  const accel = throttleOut > 0 ?  throttleOut * ACCEL       * DT : 0;
  const brake = throttleOut < 0 ? -throttleOut * BRAKE_FORCE * DT : 0;
  car.speed = Math.max(0, Math.min(MAX_SPEED,
    (car.speed + accel - brake) * FRICTION * (1 - gripLoss)));
  car.x += Math.cos(car.angle) * car.speed * DT;
  car.y += Math.sin(car.angle) * car.speed * DT;
  car.frames++;
}

function updateCarProgress(car, track) {
  const N      = track.N;
  const newSeg = nearestSegIdx(car.x, car.y, track);
  let delta    = newSeg - car.segIdx;
  if (delta >  N / 2) delta -= N;
  if (delta < -N / 2) delta += N;
  if (delta > 0) {
    car.segProgress += delta;
    if (car.segProgress >= N) { car.laps++; car.segProgress -= N; }
    car.totalDist += delta;
  }
  car.segIdx = newSeg;
}

function carFitness(car, track) {
  const progress    = car.laps + car.totalDist / Math.max(1, car.frames) * (car.frames / track.N);
  const speedBonus  = (car.frames > 0 ? car.speed / MAX_SPEED : 0) * 0.01;
  const deathFactor = car.alive ? 1 : 0.8;
  return (progress + speedBonus) * deathFactor;
}

// ── Neural net forward pass ───────────────────────────────────────────────────

function netForward(model, inputArr) {
  const x   = new T.Tensor([1, INPUT_DIM], inputArr, false);
  const out = model.forward(x, { training: false });
  return out.data;
}

// ── Box-Muller Gaussian noise ─────────────────────────────────────────────────

function gaussRand() {
  const u = Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u + 1e-9)) * Math.cos(2 * Math.PI * v);
}

// ── Population step ───────────────────────────────────────────────────────────

function stepGeneration(s, opts) {
  if (!s.pop || !s.track || !s.running) return null;
  const ticks = Math.max(1, Math.min((typeof opts === 'number' ? opts : (opts.ticks || 2)) | 0, 6));
  const liveSettings = typeof opts === 'object' ? opts : {};

  for (let t = 0; t < ticks; t++) {
    let allDead = true;

    for (let i = 0; i < s.popSize; i++) {
      const car = s.cars[i];
      if (!car.alive) continue;
      allDead = false;

      const inputs   = senseCar(car, s.track);
      const outs     = netForward(s.pop.individuals[i], inputs);
      const steer    = Math.max(-1, Math.min(1, outs[0]));
      const throttle = Math.max(-1, Math.min(1, outs[1]));

      stepCar(car, steer, throttle);
      updateCarProgress(car, s.track);

      if (!isOnTrack(car.x, car.y, s.track) || car.frames >= MAX_FRAMES) {
        car.alive    = false;
        s.carFit[i]  = carFitness(car, s.track);
      }
    }

    if (allDead) {
      if (liveSettings.mutStd  != null) s.pop.mutationStd = Math.max(0, liveSettings.mutStd);
      if (liveSettings.maxGens != null) s.maxGens = Math.max(0, liveSettings.maxGens | 0);

      s.pop.evaluate((_, idx) => s.carFit[idx]);
      const stats = s.pop.evolve();
      s.generation++;
      s.genHistory.push({ gen: s.generation, best: +s.pop.bestFitness.toFixed(3), mean: +stats.mean.toFixed(3) });
      if (s.genHistory.length > 100) s.genHistory.shift();
      if (s.pop.bestFitness > s.bestFit) s.bestFit = s.pop.bestFitness;
      s.genBestFit = stats.max;

      if (s.maxGens > 0 && s.generation >= s.maxGens) s.running = false;

      const newSize = liveSettings.popSize ? Math.max(4, Math.min(100, liveSettings.popSize | 0)) : s.popSize;
      if (newSize !== s.popSize) {
        s.popSize           = newSize;
        s.pop.size          = newSize;
        s.pop.eliteCount    = Math.max(1, Math.floor(newSize * 0.2));
        s.pop.fitnesses     = new Float32Array(newSize);
        while (s.pop.individuals.length > newSize) s.pop.individuals.pop();
        if (buildFromSpec) {
          while (s.pop.individuals.length < newSize) {
            const rng = T.rngFromSeed((Math.random() * 0xFFFFFF) | 0);
            s.pop.individuals.push(buildFromSpec(s.pop.arch, rng));
          }
        }
      }

      s.cars   = Array.from({ length: s.popSize }, () => spawnCar(s.track));
      s.carFit = new Array(s.popSize).fill(0);
      break;
    }
  }

  return buildVisualState(s);
}

function buildVisualState(s) {
  if (!s.cars || !s.track) return null;
  return {
    track: s.track,
    cars: s.cars.map(c => ({
      x: c.x, y: c.y, angle: c.angle, alive: c.alive,
      speed: c.speed, laps: c.laps,
    })),
    generation: s.generation,
    aliveCnt:   s.cars.filter(c => c.alive).length,
    popSize:    s.popSize,
    bestFit:    +s.bestFit.toFixed(3),
    genBestFit: +s.genBestFit.toFixed(3),
    genHistory: s.genHistory.slice(-60),
    hasBestGenome: s.pop !== null && s.pop.bestIndividual !== null,
  };
}

// ── Storage helpers ───────────────────────────────────────────────────────────

function savePopToStorage(storage, id, s) {
  if (!storage || !s.pop) return;
  try {
    storage.saveTrainedState(id, { state: s.pop.toJSON(), architecture: ARCH });
  } catch (e) {
    console.warn('[self-driving-car] Could not save state:', e.message);
  }
}

function loadPopFromStorage(storage, id) {
  if (!storage) return null;
  try {
    const net = storage.getNetwork(id);
    if (!net || !net.state || net.stateLocked) return null;
    const pop = Population.fromJSON(net.state);
    // fromJSON doesn't restore bestIndividual pointer; elites land at index 0.
    if (!pop.bestIndividual && pop.individuals.length > 0) {
      pop.bestIndividual = pop.individuals[0];
    }
    return pop;
  } catch (e) {
    console.warn('[self-driving-car] Could not load state:', e.message);
    return null;
  }
}

// ── IPC handlers (factory — receives storage from plugin-loader) ──────────────
module.exports = function ({ storage } = {}) {
  return {
    mainHandlers: {
      'self-driving-car:init': (_, opts = {}) => {
        if (!Population || !T) return { error: 'Neuroevolution engine unavailable.' };
        const id = opts.instanceId || 'default';
        const s  = getSession(id);

        s.popSize = Math.max(4, Math.min(100, (opts.popSize | 0) || DEFAULT_POP_SIZE));
        s.maxGens = Math.max(0, (opts.generations | 0) || 0);
        const seed   = (opts.seed || 0) >>> 0;
        const mutStd = Math.max(0.001, opts.mutStd || 0.05);

        s.arch = {
          ...ARCH,
          hidden:     Array.isArray(opts.hidden) && opts.hidden.length ? opts.hidden : ARCH.hidden,
          activation: opts.activation || ARCH.activation,
        };

        s.track = generateTrack(seed || 0xC0FFEE);

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
            architecture: s.arch,
            size: s.popSize,
            eliteCount: Math.max(1, Math.floor(s.popSize * 0.2)),
            pMutate: 0.15, mutationStd: mutStd,
            tournamentK: 3, seed: opts.seed || 42,
          });
          s.generation = 0;
          s.bestFit    = 0;
          s.genHistory = [];
        }

        s.cars       = Array.from({ length: s.popSize }, () => spawnCar(s.track));
        s.carFit     = new Array(s.popSize).fill(0);
        s.genBestFit = 0;
        s.running    = true;

        return { ok: true, track: s.track, resumed: !!savedPop, generation: s.generation };
      },

      'self-driving-car:getState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        return buildVisualState(getSession(id));
      },

      'self-driving-car:step': (_, opts = {}) => {
        const id = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        return stepGeneration(getSession(id), opts);
      },

      'self-driving-car:start': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        getSession(id).running = true;
        return { ok: true };
      },

      'self-driving-car:stop': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        s.running = false;
        savePopToStorage(storage, id, s);
        return { ok: true };
      },

      'self-driving-car:newTrack': (_, opts = {}) => {
        const id   = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        const seed = (typeof opts === 'object' ? opts.seed : opts) || null;
        const s    = getSession(id);
        if (!s.pop) return { error: 'Not initialized.' };
        s.track  = generateTrack((seed || Math.floor(Math.random() * 0xFFFFFF)) >>> 0);
        s.cars   = Array.from({ length: s.popSize }, () => spawnCar(s.track));
        s.carFit = new Array(s.popSize).fill(0);
        return { ok: true, track: s.track };
      },

      'self-driving-car:reset': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        if (!Population || !T) return { error: 'Not initialized.' };
        const seed = Math.floor(Math.random() * 0xFFFFFF);
        s.track = generateTrack(seed);
        s.pop   = new Population({
          architecture: s.arch || ARCH,
          size: s.popSize,
          eliteCount: Math.max(1, Math.floor(s.popSize * 0.2)),
          pMutate: 0.15, mutationStd: 0.05,
          tournamentK: 3, seed: seed,
        });
        s.cars       = Array.from({ length: s.popSize }, () => spawnCar(s.track));
        s.carFit     = new Array(s.popSize).fill(0);
        s.generation = 0;
        s.bestFit    = 0;
        s.genBestFit = 0;
        s.running    = true;
        s.genHistory = [];
        return { ok: true };
      },

      // ── Save / load handlers ────────────────────────────────────────────────

      'self-driving-car:saveState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        if (!s.pop) return { error: 'No trained model — run training first.' };
        savePopToStorage(storage, id, s);
        return { ok: true, generation: s.generation, bestFit: +s.bestFit.toFixed(3) };
      },

      'self-driving-car:loadState': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);
        const pop = loadPopFromStorage(storage, id);
        if (!pop) return { error: 'No saved state found for this network.' };
        const wasRunning = s.running;
        s.running    = false;
        s.pop        = pop;
        s.popSize    = pop.size;
        s.generation = pop.generation || 0;
        s.bestFit    = Math.max(0, isFinite(pop.bestFitness) ? pop.bestFitness : 0);
        s.genHistory = (pop.history || []).slice(-100).map((h, i) => ({
          gen:  h.generation ?? i,
          best: +(h.stats?.max ?? 0).toFixed(3),
          mean: +(h.stats?.mean ?? 0).toFixed(3),
        }));
        if (s.track) {
          s.cars   = Array.from({ length: s.popSize }, () => spawnCar(s.track));
          s.carFit = new Array(s.popSize).fill(0);
        }
        if (wasRunning) s.running = true;
        return { ok: true, generation: s.generation, bestFit: +s.bestFit.toFixed(3) };
      },

      // ── Inference handlers ──────────────────────────────────────────────────

      'self-driving-car:inferInit': (_, opts = {}) => {
        const id = (typeof opts === 'string' ? opts : opts.instanceId) || 'default';
        const s  = getSession(id);

        // If no live session, try to load from storage for infer-only use.
        if (!s.pop && storage) {
          const pop = loadPopFromStorage(storage, id);
          if (pop) {
            s.pop        = pop;
            s.popSize    = pop.size;
            s.generation = pop.generation || 0;
            s.bestFit    = Math.max(0, isFinite(pop.bestFitness) ? pop.bestFitness : 0);
          }
        }

        if (!s.pop || !s.track) return { error: 'No trained model — run training first.' };
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return { error: 'Population not ready.' };
        s.inferCar   = spawnCar(s.track);
        s.inferTrack = s.track;
        return {
          ok: true,
          track: s.inferTrack,
          generation: s.generation,
          bestFit: +s.bestFit.toFixed(3),
        };
      },

      'self-driving-car:inferStep': (_, opts = {}) => {
        const id       = (typeof opts === 'object' ? opts.instanceId : null) || 'default';
        const noiseStd = (typeof opts === 'object' ? opts.noiseStd : opts) || 0;
        const s        = getSession(id);
        if (!s.inferCar || !s.inferTrack || !s.pop) return null;
        const genome = s.pop.bestIndividual || s.pop.individuals[0];
        if (!genome) return null;

        const rawInputs = senseCar(s.inferCar, s.inferTrack);
        let inputs = rawInputs;
        if (noiseStd > 0) {
          inputs = new Float32Array(rawInputs.length);
          for (let i = 0; i < rawInputs.length; i++) {
            inputs[i] = rawInputs[i] + gaussRand() * noiseStd;
          }
        }

        const outs     = netForward(genome, inputs);
        const steer    = Math.max(-1, Math.min(1, outs[0]));
        const throttle = Math.max(-1, Math.min(1, outs[1]));
        stepCar(s.inferCar, steer, throttle);
        updateCarProgress(s.inferCar, s.inferTrack);

        const dead = !isOnTrack(s.inferCar.x, s.inferCar.y, s.inferTrack) || s.inferCar.frames >= MAX_FRAMES;
        if (dead) s.inferCar = spawnCar(s.inferTrack);

        return {
          track: s.inferTrack,
          car: {
            x: s.inferCar.x, y: s.inferCar.y,
            angle: s.inferCar.angle, speed: s.inferCar.speed, laps: s.inferCar.laps,
          },
          justReset: dead,
        };
      },
    },
  };
};
