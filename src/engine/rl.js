'use strict';
// Q-Learning / DQN training interface.
// Rust handles all hot arithmetic; this module owns the agent lifecycle,
// episode loop, and model-weight bridging.

const T = require('./tensor');
const { buildFromSpec, restoreFromState } = require('./model');
const { buildOptim } = require('./optim');

// Attempt to get the Rust rust.* surface from the loaded backend.
function getRust() {
  const backend = T.__backend;
  if (backend && backend.mode === 'rust') {
    try { return require('../../native/rust-engine/neuralcabin-node').api.rust; } catch (_) {}
  }
  return null;
}

// ── Replay Buffer ─────────────────────────────────────────────────────────────

class ReplayBuffer {
  constructor(capacity, stateDim, actionDim = 1) {
    this.capacity = capacity;
    this.stateDim = stateDim;
    this.actionDim = actionDim;
    this.size = 0;
    this.pos = 0;
    // Flat typed arrays — mirrors the Rust ReplayBuffer layout so we can hand
    // off sampling to Rust without an extra copy.
    this.states = new Float32Array(capacity * stateDim);
    this.actions = new Float32Array(capacity * actionDim);
    this.rewards = new Float32Array(capacity);
    this.nextStates = new Float32Array(capacity * stateDim);
    this.dones = new Float32Array(capacity);
  }

  push(state, action, reward, nextState, done) {
    const p = this.pos;
    this.states.set(state, p * this.stateDim);
    if (typeof action === 'number') {
      this.actions[p * this.actionDim] = action;
    } else {
      this.actions.set(action, p * this.actionDim);
    }
    this.rewards[p] = reward;
    this.nextStates.set(nextState, p * this.stateDim);
    this.dones[p] = done ? 1 : 0;
    this.pos = (this.pos + 1) % this.capacity;
    this.size = Math.min(this.size + 1, this.capacity);
  }

  // Sample a random mini-batch. Returns { states, actions, rewards, nextStates, dones }
  // each as Float32Array.
  // NOTE: always uses typed-array subarray copies (O(batchSize)), never Array.from on the
  // full buffer. Passing Array.from(Float32Array(capacity × stateDim)) to NAPI boxes every
  // element as a heap Number, causing GC pressure that freezes the main process.
  sample(batchSize) {
    const n = this.size;
    const sd = this.stateDim, ad = this.actionDim;
    const states = new Float32Array(batchSize * sd);
    const actions = new Float32Array(batchSize * ad);
    const rewards = new Float32Array(batchSize);
    const nextStates = new Float32Array(batchSize * sd);
    const dones = new Float32Array(batchSize);
    for (let i = 0; i < batchSize; i++) {
      const idx = Math.floor(Math.random() * n);
      states.set(this.states.subarray(idx * sd, (idx + 1) * sd), i * sd);
      actions.set(this.actions.subarray(idx * ad, (idx + 1) * ad), i * ad);
      rewards[i] = this.rewards[idx];
      nextStates.set(this.nextStates.subarray(idx * sd, (idx + 1) * sd), i * sd);
      dones[i] = this.dones[idx];
    }
    return { states, actions, rewards, nextStates, dones };
  }

  get ready() { return this.size >= 64; }
}

// ── DQN Agent ─────────────────────────────────────────────────────────────────

class DQNAgent {
  /**
   * opts = {
   *   architecture: { kind:'classifier', inputDim, outputDim (=nActions), hidden, … }
   *   gamma: 0.99, lr: 1e-3, batchSize: 64, bufferCapacity: 10000,
   *   epsilonStart: 1.0, epsilonEnd: 0.05, epsilonDecay: 0.995,
   *   targetUpdateFreq: 100, seed: 42
   * }
   */
  constructor(opts = {}) {
    this.gamma = opts.gamma ?? 0.99;
    this.batchSize = opts.batchSize ?? 64;
    this.epsilonStart = opts.epsilonStart ?? 1.0;
    this.epsilonEnd = opts.epsilonEnd ?? 0.05;
    this.epsilonDecay = opts.epsilonDecay ?? 0.995;
    this.targetUpdateFreq = opts.targetUpdateFreq ?? 100;
    this.seed = opts.seed ?? 42;

    const arch = opts.architecture ?? { kind: 'classifier', inputDim: 4, outputDim: 2, hidden: [64, 64] };
    this.nActions = arch.outputDim;
    this.stateDim = arch.inputDim;
    this.arch = arch;

    const rng = T.rngFromSeed(this.seed);
    this.onlineNet = buildFromSpec(arch, rng);
    this.targetNet = buildFromSpec(arch, rng); // starts with same weights
    this._syncTarget();

    this.optim = buildOptim(opts.optimizer ?? 'adam', this.onlineNet.params, { lr: opts.lr ?? 1e-3 });
    this.buffer = new ReplayBuffer(opts.bufferCapacity ?? 10000, this.stateDim);
    this.epsilon = this.epsilonStart;
    this.steps = 0;
    this.losses = [];
  }

  // ── Action selection ────────────────────────────────────────────────────────

  selectAction(state) {
    const rust = getRust();
    const seed = (Math.random() * 0xFFFFFFFF) >>> 0;
    if (rust && this.epsilon < 1.0) {
      const x = new T.Tensor([1, this.stateDim], new Float32Array(state), false);
      const logits = this.onlineNet.forward(x, { training: false });
      return rust.epsilonGreedy(Array.from(logits.data), this.epsilon, seed);
    }
    // JS path
    if (Math.random() < this.epsilon) {
      return Math.floor(Math.random() * this.nActions);
    }
    const x = new T.Tensor([1, this.stateDim], new Float32Array(state), false);
    const logits = this.onlineNet.forward(x, { training: false });
    let best = 0;
    for (let i = 1; i < logits.size; i++) if (logits.data[i] > logits.data[best]) best = i;
    return best;
  }

  // ── Store transition ────────────────────────────────────────────────────────

  observe(state, action, reward, nextState, done) {
    this.buffer.push(state, action, reward, nextState, done);
  }

  // ── Training step ───────────────────────────────────────────────────────────

  trainStep(useHuber = true, huberDelta = 1.0) {
    if (!this.buffer.ready) return null;

    const { states, actions, rewards, nextStates, dones } = this.buffer.sample(this.batchSize);
    const rust = getRust();

    // Compute TD targets using the target network.
    const nsX = new T.Tensor([this.batchSize, this.stateDim], nextStates, false);
    const nextLogits = this.targetNet.forward(nsX, { training: false });
    const nextQ = Array.from(nextLogits.data);
    const rewardsArr = Array.from(rewards);
    const donesArr = Array.from(dones);

    let tdTargets;
    if (rust) {
      tdTargets = rust.computeTdTargets(rewardsArr, nextQ, donesArr, this.gamma, this.nActions);
    } else {
      tdTargets = rewardsArr.map((r, i) => {
        const row = nextQ.slice(i * this.nActions, (i + 1) * this.nActions);
        const maxQ = Math.max(...row);
        return r + this.gamma * maxQ * (1 - donesArr[i]);
      });
    }

    // Forward pass on online network.
    const sX = new T.Tensor([this.batchSize, this.stateDim], states, true);
    this.optim.zeroGrad();
    const qLogits = this.onlineNet.forward(sX, { training: true });
    const actionIdxs = actions.slice(0, this.batchSize).map(a => Math.round(a));

    let loss;
    if (rust) {
      const { loss: lVal, grad } = useHuber
        ? rust.dqnHuberLoss(Array.from(qLogits.data), actionIdxs, tdTargets, this.nActions, huberDelta)
        : rust.dqnLoss(Array.from(qLogits.data), actionIdxs, tdTargets, this.nActions);
      // Seed gradient back into the tensor graph.
      qLogits.ensureGrad();
      const g = qLogits.grad;
      for (let i = 0; i < g.length; i++) g[i] += grad[i];
      qLogits._backward && qLogits._backward();
      loss = lVal;
    } else {
      // Pure JS DQN loss (MSE over the chosen action's Q-value).
      const B = this.batchSize, C = this.nActions;
      const qChosen = new Float32Array(B);
      const tChosen = new Float32Array(B);
      for (let i = 0; i < B; i++) {
        qChosen[i] = qLogits.data[i * C + actionIdxs[i]];
        tChosen[i] = tdTargets[i];
      }
      const qT = new T.Tensor([B, 1], qChosen, false);
      const tT = new T.Tensor([B, 1], tChosen, false);
      const l = T.mseLoss(qLogits, new T.Tensor(qLogits.shape, qLogits.data, false));
      loss = l.data[0];
      l.backward();
    }

    this.optim.step();
    this.losses.push(loss);

    // Decay epsilon and sync target network.
    this.epsilon = Math.max(this.epsilonEnd, this.epsilon * this.epsilonDecay);
    this.steps++;
    if (this.steps % this.targetUpdateFreq === 0) this._syncTarget();

    return loss;
  }

  // Hard copy online → target weights.
  _syncTarget() {
    const onState = this.onlineNet.toJSON();
    this.targetNet = restoreFromState(onState, this.arch, T.rngFromSeed(this.seed));
  }

  // Polyak soft update (tau ≪ 1 for stability).
  softSyncTarget(tau = 0.005) {
    const rust = getRust();
    if (rust) {
      for (let i = 0; i < this.onlineNet.params.length; i++) {
        const op = this.onlineNet.params[i];
        const tp = this.targetNet.params[i];
        const newData = rust.softUpdateTarget(Array.from(tp.data), Array.from(op.data), tau);
        tp.data.set(newData);
      }
    } else {
      for (let i = 0; i < this.onlineNet.params.length; i++) {
        const op = this.onlineNet.params[i], tp = this.targetNet.params[i];
        for (let j = 0; j < op.size; j++) tp.data[j] = (1 - tau) * tp.data[j] + tau * op.data[j];
      }
    }
  }

  // Serialize for persistence.
  toJSON() {
    return {
      arch: this.arch,
      onlineState: this.onlineNet.toJSON(),
      targetState: this.targetNet.toJSON(),
      optimState: this.optim.toJSON(),
      epsilon: this.epsilon,
      steps: this.steps,
    };
  }

  static fromJSON(obj, opts = {}) {
    const agent = new DQNAgent({ ...opts, architecture: obj.arch });
    const rng = T.rngFromSeed(opts.seed ?? 42);
    agent.onlineNet = restoreFromState(obj.onlineState, obj.arch, rng);
    agent.targetNet = restoreFromState(obj.targetState, obj.arch, rng);
    agent.optim = buildOptim('adam', agent.onlineNet.params, { lr: opts.lr ?? 1e-3 });
    agent.optim.loadFromJSON(obj.optimState);
    agent.epsilon = obj.epsilon ?? agent.epsilonStart;
    agent.steps = obj.steps ?? 0;
    return agent;
  }
}

module.exports = { DQNAgent, ReplayBuffer };
