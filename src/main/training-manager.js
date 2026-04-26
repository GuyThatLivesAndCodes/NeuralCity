'use strict';

const path = require('path');
const { Worker } = require('worker_threads');
const { EventEmitter } = require('events');
const { infer, inferStream, buildInferenceCache } = require('../engine/trainer');
const { runScript } = require('../dsl/interpreter');

const WORKER_PATH = path.join(__dirname, '..', 'engine', 'training-session-worker.js');
// After a cooperative stop request, wait this long before force-terminating the
// worker. Keeps "Stop" responsive without sacrificing the partial-state save in
// the common case where the worker finishes its current batch quickly.
const FORCE_TERMINATE_DELAY_MS = 5000;

// ── Structured log helper ─────────────────────────────────────────────────────

function makeLog(emit, id) {
  return (line, level = 'info') => {
    emit('log', { id, line, level, ts: Date.now() });
  };
}

// ── Backend telemetry ─────────────────────────────────────────────────────────

let _backendMeta = null;
function getBackendMeta() {
  if (_backendMeta) return _backendMeta;
  try {
    const tensor = require('../engine/tensor');
    _backendMeta = tensor.__backend || { mode: 'js', reason: 'not-checked' };
  } catch (_) {
    _backendMeta = { mode: 'unknown' };
  }
  return _backendMeta;
}

// ── TrainingManager ───────────────────────────────────────────────────────────

class TrainingManager extends EventEmitter {
  constructor(storage) {
    super();
    this.storage = storage;
    this.active = new Map();      // id -> { worker, forceTimer, startedAt, lastProgress }
    this._modelCache = new Map(); // id -> { updatedAt, cache }
  }

  _getOrBuildModel(net) {
    const stored = this._modelCache.get(net.id);
    if (stored && stored.updatedAt === net.updatedAt) return stored.cache;
    const cache = buildInferenceCache(net);
    this._modelCache.set(net.id, { updatedAt: net.updatedAt, cache });
    return cache;
  }

  status(id) {
    const a = this.active.get(id);
    if (!a) return { running: false };
    return { running: true, startedAt: a.startedAt, lastProgress: a.lastProgress };
  }

  backendInfo() {
    const meta = getBackendMeta();
    let extraInfo = {};
    if (meta.mode === 'rust') {
      try {
        const native = require('../../native/rust-engine/neuralcabin-node');
        extraInfo = native.api.backendInfo();
      } catch (_) {}
    }
    return { ...meta, ...extraInfo };
  }

  // opts: { fromScratch?: boolean, overrides?: object }
  async start(id, opts) {
    if (this.active.has(id)) throw new Error('Training already running');

    let net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');

    opts = opts || {};
    const fromScratch = !!opts.fromScratch;
    if (opts.overrides) {
      net = { ...net, training: { ...net.training, ...opts.overrides } };
    }

    const log = makeLog(this.emit.bind(this), id);
    const backend = getBackendMeta();
    log(`training started (backend=${backend.mode}${backend.reason ? '/' + backend.reason : ''})`);

    const worker = new Worker(WORKER_PATH, { workerData: { network: net, fromScratch } });

    const record = {
      worker,
      forceTimer: null,
      startedAt: Date.now(),
      lastProgress: null,
    };
    this.active.set(id, record);

    const cleanup = () => {
      if (record.forceTimer) { clearTimeout(record.forceTimer); record.forceTimer = null; }
      this.active.delete(id);
    };

    worker.on('message', (msg) => {
      switch (msg.type) {
        case 'progress':
          record.lastProgress = msg;
          this.emit('progress', { id, epoch: msg.epoch, totalEpochs: msg.totalEpochs,
            step: msg.step, totalSteps: msg.totalSteps, loss: msg.loss, elapsedMs: msg.elapsedMs });
          break;

        case 'log':
          this.emit('log', { id, line: msg.line, level: msg.level || 'info', ts: Date.now() });
          break;

        case 'done': {
          const result = msg.result;
          const elapsed = ((Date.now() - record.startedAt) / 1000).toFixed(1);
          log(`training finished in ${elapsed}s (${result.metrics?.length ?? 0} epochs)`);
          this.storage.saveTrainedState(id, {
            state: result.state,
            optimizerState: result.optimizerState,
            tokenizer: result.tokenizer,
            architecture: result.architecture,
            metrics: result.metrics,
          });
          this._modelCache.delete(id);
          this.emit('done', { id, stopped: !!result.stopped, metrics: result.metrics });
          cleanup();
          break;
        }

        case 'error':
          log(`training error: ${msg.message}`, 'error');
          this.emit('error', { id, message: msg.message, stack: msg.stack });
          cleanup();
          break;
      }
    });

    worker.on('error', (e) => {
      log(`worker error: ${e.message}`, 'error');
      this.emit('error', { id, message: e.message, stack: e.stack });
      cleanup();
    });

    // Fires when the worker exits for any reason — including worker.terminate().
    // If cleanup() has already run (normal done/error path) this is a no-op.
    worker.on('exit', () => {
      if (!this.active.has(id)) return;
      log('training stopped (worker terminated)', 'warn');
      this.emit('done', { id, stopped: true, metrics: record.lastProgress
        ? [{ epoch: record.lastProgress.epoch, loss: record.lastProgress.loss }]
        : [] });
      cleanup();
    });

    return { started: true, fromScratch };
  }

  stop(id) {
    const a = this.active.get(id);
    if (!a) return { running: false };

    // Cooperative stop: worker's shouldStop() returns true on the next check.
    // The worker finishes its current batch, saves partial state, then posts 'done'.
    a.worker.postMessage({ type: 'stop' });
    makeLog(this.emit.bind(this), id)('stop requested');

    // Safety net: if the worker hasn't responded within the timeout, kill it.
    // This handles hangs and pathologically slow single batches.
    a.forceTimer = setTimeout(() => {
      if (this.active.has(id)) {
        makeLog(this.emit.bind(this), id)('force-terminating worker after timeout', 'warn');
        a.worker.terminate();
      }
    }, FORCE_TERMINATE_DELAY_MS);

    return { stopping: true };
  }

  stopAll() { for (const [id] of this.active) this.stop(id); }

  async infer(id, input) {
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');
    const cache = this._getOrBuildModel(net);
    return infer(net, input, cache);
  }

  async inferStream(id, input, onToken, cancelRef) {
    const net = this.storage.getNetwork(id);
    if (!net) throw new Error('Network not found');
    if (net.stateLocked) throw new Error('Network state is encrypted; decrypt it first.');
    const cache = this._getOrBuildModel(net);
    return inferStream(net, input, onToken, cancelRef, cache);
  }

  async runScript(id, code) {
    const net = id ? this.storage.getNetwork(id) : null;
    const ctx = {
      network: net,
      saveTrainedState: (state) => net && this.storage.saveTrainedState(net.id, state),
      storage: this.storage,
    };
    return runScript(code, ctx);
  }
}

module.exports = { TrainingManager };
