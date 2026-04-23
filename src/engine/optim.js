'use strict';

// ─────────────────────────────────────────────────────────────────────────────
// SGD — stochastic gradient descent with optional momentum
// ─────────────────────────────────────────────────────────────────────────────
class SGD {
  constructor(params, { lr = 0.01, momentum = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    this.momentum = momentum;
    this.v = params.map(p => new Float32Array(p.size));
  }
  step() {
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const v = this.v[i];
      for (let j = 0; j < p.size; j++) {
        v[j] = this.momentum * v[j] + p.grad[j];
        p.data[j] -= this.lr * v[j];
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }

  toJSON() {
    return { type: 'sgd', lr: this.lr, momentum: this.momentum,
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'sgd') return false;
    if (!Array.isArray(o.v) || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) this.v[i] = new Float32Array(o.v[i]);
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adam — adaptive moment estimation (Kingma & Ba, 2015)
// ─────────────────────────────────────────────────────────────────────────────
class Adam {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    const invBc1 = 1 / bc1, invSqrtBc2 = 1 / Math.sqrt(bc2);
    const hasWD = weightDecay !== 0;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      for (let j = 0; j < sz; j++) {
        const g = hasWD ? pg[j] + weightDecay * pd[j] : pg[j];
        const mj = beta1 * m[j] + ombeta1 * g;
        const vj = beta2 * v[j] + ombeta2 * g * g;
        m[j] = mj; v[j] = vj;
        pd[j] -= lr * (mj * invBc1) / (Math.sqrt(vj) * invSqrtBc2 + eps);
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }

  toJSON() {
    return { type: 'adam', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay, t: this.t,
             m: this.m.map(buf => Array.from(buf)),
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'adam') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v)) return false;
    if (o.m.length !== this.params.length || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = new Float32Array(o.m[i]);
      this.v[i] = new Float32Array(o.v[i]);
    }
    this.t = o.t || 0;
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// AdamW — Adam with decoupled weight decay (Loshchilov & Hutter, 2019)
//
// Unlike Adam, weight decay is applied directly to the weights rather than
// being folded into the gradient. This prevents the adaptive learning rate
// from scaling the effective weight decay, making the decay truly L2-like.
// ─────────────────────────────────────────────────────────────────────────────
class AdamW {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    const invBc1 = 1 / bc1, invSqrtBc2 = 1 / Math.sqrt(bc2);
    const decayFactor = 1 - lr * weightDecay;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      for (let j = 0; j < sz; j++) {
        // Update moments using raw gradient (no WD mixed in)
        const mj = beta1 * m[j] + ombeta1 * pg[j];
        const vj = beta2 * v[j] + ombeta2 * pg[j] * pg[j];
        m[j] = mj; v[j] = vj;
        // Decoupled weight decay applied directly to weights
        pd[j] = decayFactor * pd[j] - lr * (mj * invBc1) / (Math.sqrt(vj) * invSqrtBc2 + eps);
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'adamw', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay, t: this.t,
             m: this.m.map(buf => Array.from(buf)),
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'adamw') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v)) return false;
    if (o.m.length !== this.params.length || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = new Float32Array(o.m[i]);
      this.v[i] = new Float32Array(o.v[i]);
    }
    this.t = o.t || 0;
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// RAdam — Rectified Adam (Liu et al., 2019)
//
// Computes the maximum length of the approximated SMA (rho) each step. When
// rho_t > 5 the variance is "tractable" and a rectification factor is applied
// to correct the adaptive step. Below that threshold the optimizer falls back
// to SGD-with-momentum to avoid the bad variance at the start of training.
// ─────────────────────────────────────────────────────────────────────────────
class RAdam {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
    this.rhoInf = 2 / (1 - beta2) - 1; // maximum length of the SMA
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr, weightDecay, rhoInf } = this;
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    const b2t = Math.pow(beta2, this.t);
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - b2t;
    // Approximate length of the SMA at step t
    const rhoT = rhoInf - 2 * this.t * b2t / bc2;
    const warmup = rhoT > 5;
    const rect = warmup
      ? Math.sqrt(((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT))
      : 1;
    const hasWD = weightDecay !== 0;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      for (let j = 0; j < sz; j++) {
        const g = hasWD ? pg[j] + weightDecay * pd[j] : pg[j];
        m[j] = beta1 * m[j] + ombeta1 * g;
        v[j] = beta2 * v[j] + ombeta2 * g * g;
        const mHat = m[j] / bc1;
        if (warmup) {
          pd[j] -= lr * rect * mHat / (Math.sqrt(v[j] / bc2) + eps);
        } else {
          pd[j] -= lr * mHat; // SGD-like: variance not yet reliable
        }
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'radam', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay, t: this.t,
             m: this.m.map(buf => Array.from(buf)),
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'radam') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v)) return false;
    if (o.m.length !== this.params.length || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = new Float32Array(o.m[i]);
      this.v[i] = new Float32Array(o.v[i]);
    }
    this.t = o.t || 0;
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Lion — EvoLved Sign Optimizer (Chen et al., 2023)
//
// Uses only the sign of the momentum interpolation — no second moment needed.
// This halves optimizer state vs Adam. Requires a smaller lr (3–10× less than
// Adam). Default betas are beta1=0.9 (update signal), beta2=0.99 (momentum).
// ─────────────────────────────────────────────────────────────────────────────
class Lion {
  constructor(params, { lr = 1e-4, beta1 = 0.9, beta2 = 0.99, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
  }
  step() {
    const { beta1, beta2, lr, weightDecay } = this;
    const decayFactor = 1 - lr * weightDecay;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      for (let j = 0; j < sz; j++) {
        // Compute update signal (interpolation of momentum and gradient)
        const c = beta1 * m[j] + (1 - beta1) * pg[j];
        // Decoupled weight decay + sign update
        pd[j] = decayFactor * pd[j] - lr * Math.sign(c);
        // Update momentum (separate EMA of raw gradient)
        m[j] = beta2 * m[j] + (1 - beta2) * pg[j];
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'lion', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             weightDecay: this.weightDecay,
             m: this.m.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'lion') return false;
    if (!Array.isArray(o.m) || o.m.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) this.m[i] = new Float32Array(o.m[i]);
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Adafactor — memory-efficient adaptive optimizer (Shazeer & Stern, 2018)
//
// For 2D parameters (weight matrices) the second moment V is factored into
// row and column statistics: V̂[i,j] ≈ Vr[i]·Vc[j]/mean(Vr).
// This reduces memory from O(r·c) to O(r+c) per layer. Bias/1D params use
// a standard scalar second moment.
// ─────────────────────────────────────────────────────────────────────────────
class Adafactor {
  constructor(params, { lr = 1e-3, beta2Decay = -0.8, eps1 = 1e-30, clipThreshold = 1.0, weightDecay = 0 } = {}) {
    this.params = params;
    this.lr = lr;
    // beta2_t = 1 - t^beta2Decay  (approaches 1 slowly, giving long memory)
    this.beta2Decay = beta2Decay;
    this.eps1 = eps1;
    this.clipThreshold = clipThreshold;
    this.weightDecay = weightDecay;
    this.t = 0;
    this._vr  = []; // row factors for 2D params
    this._vc  = []; // col factors for 2D params
    this._v1d = []; // scalar second moment for 1D params
    for (const p of params) {
      if (p.shape && p.shape.length >= 2) {
        this._vr.push(new Float32Array(p.shape[0]));
        this._vc.push(new Float32Array(p.shape[1]));
        this._v1d.push(null);
      } else {
        this._vr.push(null);
        this._vc.push(null);
        this._v1d.push(new Float32Array(p.size));
      }
    }
  }
  step() {
    this.t++;
    const { lr, eps1, clipThreshold, weightDecay } = this;
    const beta2 = 1 - Math.pow(this.t, this.beta2Decay);
    const rho   = 1 - beta2;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const pd = p.data, pg = p.grad;
      if (this._vr[i] !== null) {
        // 2D factored second moment
        const [rows, cols] = p.shape;
        const vr = this._vr[i], vc = this._vc[i];
        // Accumulate row/col sums of g²
        const rowSums = new Float32Array(rows);
        const colSums = new Float32Array(cols);
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const g2 = pg[r * cols + c] * pg[r * cols + c];
            rowSums[r] += g2;
            colSums[c] += g2;
          }
        }
        for (let r = 0; r < rows; r++) rowSums[r] /= cols; // row means
        for (let c = 0; c < cols; c++) colSums[c] /= rows; // col means
        // Update factors
        for (let r = 0; r < rows; r++) vr[r] = beta2 * vr[r] + rho * (rowSums[r] + eps1);
        for (let c = 0; c < cols; c++) vc[c] = beta2 * vc[c] + rho * (colSums[c] + eps1);
        // mean(vr) used to normalize V̂ so its scale is consistent
        let vrMean = 0;
        for (let r = 0; r < rows; r++) vrMean += vr[r];
        vrMean /= rows;
        // Compute unclipped update and its RMS
        const update = new Float32Array(rows * cols);
        let rmsU = 0;
        for (let r = 0; r < rows; r++) {
          for (let c = 0; c < cols; c++) {
            const vHat = vr[r] * vc[c] / vrMean;
            const u = pg[r * cols + c] / Math.sqrt(Math.max(vHat, eps1));
            update[r * cols + c] = u;
            rmsU += u * u;
          }
        }
        rmsU = Math.sqrt(rmsU / (rows * cols));
        const effLr = lr / Math.max(1.0, rmsU / clipThreshold);
        for (let j = 0; j < pd.length; j++) {
          pd[j] = (1 - lr * weightDecay) * pd[j] - effLr * update[j];
        }
      } else {
        // 1D scalar second moment
        const v = this._v1d[i];
        let rmsU = 0;
        const update = new Float32Array(p.size);
        for (let j = 0; j < p.size; j++) {
          v[j] = beta2 * v[j] + rho * (pg[j] * pg[j] + eps1);
          const u = pg[j] / Math.sqrt(v[j]);
          update[j] = u;
          rmsU += u * u;
        }
        rmsU = Math.sqrt(rmsU / p.size);
        const effLr = lr / Math.max(1.0, rmsU / clipThreshold);
        for (let j = 0; j < p.size; j++) {
          pd[j] = (1 - lr * weightDecay) * pd[j] - effLr * update[j];
        }
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'adafactor', lr: this.lr, beta2Decay: this.beta2Decay,
             eps1: this.eps1, clipThreshold: this.clipThreshold,
             weightDecay: this.weightDecay, t: this.t,
             vr:  this._vr.map(b  => b ? Array.from(b) : null),
             vc:  this._vc.map(b  => b ? Array.from(b) : null),
             v1d: this._v1d.map(b => b ? Array.from(b) : null) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'adafactor') return false;
    if (!Array.isArray(o.vr) || o.vr.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (this._vr[i] !== null) {
        if (!o.vr[i] || o.vr[i].length !== this._vr[i].length) return false;
        if (!o.vc[i] || o.vc[i].length !== this._vc[i].length) return false;
      } else {
        if (!o.v1d[i] || o.v1d[i].length !== this._v1d[i].length) return false;
      }
    }
    this.t = o.t || 0;
    for (let i = 0; i < this.params.length; i++) {
      if (this._vr[i] !== null) {
        this._vr[i] = new Float32Array(o.vr[i]);
        this._vc[i] = new Float32Array(o.vc[i]);
      } else {
        this._v1d[i] = new Float32Array(o.v1d[i]);
      }
    }
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers for 8-bit block-wise quantization
// ─────────────────────────────────────────────────────────────────────────────
const Q8_BLOCK = 256;
const Q8_MAX   = 127;

function _q8encode(buf) {
  const n       = buf.length;
  const nBlocks = Math.ceil(n / Q8_BLOCK);
  const q       = new Int8Array(n);
  const scales  = new Float32Array(nBlocks);
  for (let b = 0; b < nBlocks; b++) {
    const s = b * Q8_BLOCK, e = Math.min(s + Q8_BLOCK, n);
    let maxAbs = 0;
    for (let i = s; i < e; i++) { const a = Math.abs(buf[i]); if (a > maxAbs) maxAbs = a; }
    const scale = maxAbs > 0 ? maxAbs / Q8_MAX : 1;
    scales[b] = scale;
    const inv = 1 / scale;
    for (let i = s; i < e; i++) q[i] = Math.round(buf[i] * inv);
  }
  return { q, scales };
}

function _q8decode(q, scales) {
  const n = q.length;
  const buf = new Float32Array(n);
  const nBlocks = Math.ceil(n / Q8_BLOCK);
  for (let b = 0; b < nBlocks; b++) {
    const s = b * Q8_BLOCK, e = Math.min(s + Q8_BLOCK, n);
    const sc = scales[b];
    for (let i = s; i < e; i++) buf[i] = q[i] * sc;
  }
  return buf;
}

// ─────────────────────────────────────────────────────────────────────────────
// AdamW 8-bit — AdamW with 8-bit block-wise quantized optimizer states
//
// Stores m and v as Int8 arrays with per-block float32 scales, cutting
// optimizer state memory ~4× vs full-precision AdamW. Dequantizes at the
// start of each step, runs the exact same AdamW update, then re-quantizes.
// ─────────────────────────────────────────────────────────────────────────────
class AdamW8bit {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.t = 0;
    this._mq = params.map(() => null); // Int8Array | null
    this._ms = params.map(() => null); // Float32Array scales | null
    this._vq = params.map(() => null);
    this._vs = params.map(() => null);
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    const invBc1 = 1 / bc1, invSqrtBc2 = 1 / Math.sqrt(bc2);
    const decayFactor = 1 - lr * weightDecay;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const sz = p.size;
      const m = this._mq[i] ? _q8decode(this._mq[i], this._ms[i]) : new Float32Array(sz);
      const v = this._vq[i] ? _q8decode(this._vq[i], this._vs[i]) : new Float32Array(sz);
      const pd = p.data, pg = p.grad;
      for (let j = 0; j < sz; j++) {
        const mj = beta1 * m[j] + ombeta1 * pg[j];
        const vj = beta2 * v[j] + ombeta2 * pg[j] * pg[j];
        m[j] = mj; v[j] = vj;
        pd[j] = decayFactor * pd[j] - lr * (mj * invBc1) / (Math.sqrt(vj) * invSqrtBc2 + eps);
      }
      const em = _q8encode(m); this._mq[i] = em.q; this._ms[i] = em.scales;
      const ev = _q8encode(v); this._vq[i] = ev.q; this._vs[i] = ev.scales;
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'adamw8bit', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay, t: this.t,
             mq: this._mq.map(q => q ? Array.from(q) : null),
             ms: this._ms.map(s => s ? Array.from(s) : null),
             vq: this._vq.map(q => q ? Array.from(q) : null),
             vs: this._vs.map(s => s ? Array.from(s) : null) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'adamw8bit') return false;
    if (!Array.isArray(o.mq) || o.mq.length !== this.params.length) return false;
    this.t = o.t || 0;
    for (let i = 0; i < this.params.length; i++) {
      this._mq[i] = o.mq[i] ? new Int8Array(o.mq[i]) : null;
      this._ms[i] = o.ms[i] ? new Float32Array(o.ms[i]) : null;
      this._vq[i] = o.vq[i] ? new Int8Array(o.vq[i]) : null;
      this._vs[i] = o.vs[i] ? new Float32Array(o.vs[i]) : null;
    }
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LAMB — Layer-wise Adaptive Moments (You et al., 2019)
//
// Extends Adam with a per-parameter trust ratio: lr_eff = lr · ‖w‖/‖r‖,
// where r is the Adam update + L2 term. This keeps the update scale
// proportional to the weight norm and enables very large batch sizes.
// ─────────────────────────────────────────────────────────────────────────────
class LAMB {
  constructor(params, { lr = 1e-3, beta1 = 0.9, beta2 = 0.999, eps = 1e-6, weightDecay = 0.01 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps; this.weightDecay = weightDecay;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
  }
  step() {
    this.t++;
    const { beta1, beta2, eps, lr, weightDecay } = this;
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - Math.pow(beta2, this.t);
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      let pNormSq = 0, rNormSq = 0;
      const r = new Float32Array(sz);
      for (let j = 0; j < sz; j++) {
        m[j] = beta1 * m[j] + ombeta1 * pg[j];
        v[j] = beta2 * v[j] + ombeta2 * pg[j] * pg[j];
        // Adam update with L2 regularization (weight decay inside the ratio)
        r[j] = (m[j] / bc1) / (Math.sqrt(v[j] / bc2) + eps) + weightDecay * pd[j];
        pNormSq += pd[j] * pd[j];
        rNormSq += r[j] * r[j];
      }
      const pNorm = Math.sqrt(pNormSq), rNorm = Math.sqrt(rNormSq);
      const trust = (pNorm > 0 && rNorm > 0) ? pNorm / rNorm : 1;
      for (let j = 0; j < sz; j++) pd[j] -= lr * trust * r[j];
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'lamb', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay, t: this.t,
             m: this.m.map(buf => Array.from(buf)),
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'lamb') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v)) return false;
    if (o.m.length !== this.params.length || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) {
      this.m[i] = new Float32Array(o.m[i]);
      this.v[i] = new Float32Array(o.v[i]);
    }
    this.t = o.t || 0;
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// LARS — Layer-wise Adaptive Rate Scaling (You et al., 2017)
//
// Extends SGD with momentum by computing a per-parameter local learning rate
// via a trust ratio: lr_local = lr · eta · ‖w‖ / (‖g‖ + wd·‖w‖).
// This prevents small layers from being overwhelmed by large gradient norms.
// ─────────────────────────────────────────────────────────────────────────────
class LARS {
  constructor(params, { lr = 0.1, momentum = 0.9, weightDecay = 1e-4, eta = 1e-3 } = {}) {
    this.params = params;
    this.lr = lr; this.momentum = momentum; this.weightDecay = weightDecay;
    this.eta = eta; // trust coefficient
    this.v = params.map(p => new Float32Array(p.size));
  }
  step() {
    const { lr, momentum, weightDecay, eta } = this;
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      // Per-parameter norms (treated as one "layer" per param tensor)
      let pNormSq = 0, gNormSq = 0;
      for (let j = 0; j < sz; j++) { pNormSq += pd[j] * pd[j]; gNormSq += pg[j] * pg[j]; }
      const pNorm = Math.sqrt(pNormSq), gNorm = Math.sqrt(gNormSq);
      // Trust ratio: scale global lr by eta*‖w‖/(‖g‖ + wd*‖w‖)
      const localLr = (pNorm > 0 && gNorm > 0)
        ? lr * eta * pNorm / (gNorm + weightDecay * pNorm)
        : lr * eta;
      // SGD with momentum using the local learning rate
      for (let j = 0; j < sz; j++) {
        v[j] = momentum * v[j] + localLr * (pg[j] + weightDecay * pd[j]);
        pd[j] -= v[j];
      }
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'lars', lr: this.lr, momentum: this.momentum,
             weightDecay: this.weightDecay, eta: this.eta,
             v: this.v.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'lars') return false;
    if (!Array.isArray(o.v) || o.v.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
    }
    for (let i = 0; i < this.params.length; i++) this.v[i] = new Float32Array(o.v[i]);
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ranger — RAdam + Lookahead (Wright, 2019)
//
// Runs RAdam as the inner optimizer for k steps, then Lookahead interpolates
// slow weights toward the fast weights: slow_w += alpha*(fast_w - slow_w),
// then resets fast_w = slow_w. This flattens the loss landscape by exploring
// local minima and averaging across them without extra gradient evaluations.
// ─────────────────────────────────────────────────────────────────────────────
class Ranger {
  constructor(params, { lr = 1e-3, beta1 = 0.95, beta2 = 0.999, eps = 1e-5,
                        weightDecay = 0, lookaheadK = 6, lookaheadAlpha = 0.5 } = {}) {
    this.params = params;
    this.lr = lr; this.beta1 = beta1; this.beta2 = beta2; this.eps = eps;
    this.weightDecay = weightDecay;
    this.lookaheadK = lookaheadK;
    this.lookaheadAlpha = lookaheadAlpha;
    this.m = params.map(p => new Float32Array(p.size));
    this.v = params.map(p => new Float32Array(p.size));
    this.t = 0;
    this.rhoInf = 2 / (1 - beta2) - 1;
    // Lookahead slow weights initialised to the current fast weights
    this.slow = params.map(p => Float32Array.from(p.data));
    this._innerSteps = 0;
  }
  step() {
    this.t++;
    this._innerSteps++;
    const { beta1, beta2, eps, lr, weightDecay, rhoInf } = this;
    const ombeta1 = 1 - beta1, ombeta2 = 1 - beta2;
    const b2t = Math.pow(beta2, this.t);
    const bc1 = 1 - Math.pow(beta1, this.t);
    const bc2 = 1 - b2t;
    const rhoT = rhoInf - 2 * this.t * b2t / bc2;
    const warmup = rhoT > 5;
    const rect = warmup
      ? Math.sqrt(((rhoT - 4) * (rhoT - 2) * rhoInf) / ((rhoInf - 4) * (rhoInf - 2) * rhoT))
      : 1;
    const hasWD = weightDecay !== 0;
    // Inner RAdam step
    for (let i = 0; i < this.params.length; i++) {
      const p = this.params[i];
      if (!p.grad) continue;
      const m = this.m[i], v = this.v[i];
      const pd = p.data, pg = p.grad, sz = p.size;
      for (let j = 0; j < sz; j++) {
        const g = hasWD ? pg[j] + weightDecay * pd[j] : pg[j];
        m[j] = beta1 * m[j] + ombeta1 * g;
        v[j] = beta2 * v[j] + ombeta2 * g * g;
        const mHat = m[j] / bc1;
        pd[j] -= warmup
          ? lr * rect * mHat / (Math.sqrt(v[j] / bc2) + eps)
          : lr * mHat;
      }
    }
    // Lookahead sync every k inner steps
    if (this._innerSteps >= this.lookaheadK) {
      const { lookaheadAlpha } = this;
      for (let i = 0; i < this.params.length; i++) {
        const fast = this.params[i].data, slow = this.slow[i];
        for (let j = 0; j < fast.length; j++) {
          slow[j] += lookaheadAlpha * (fast[j] - slow[j]);
          fast[j] = slow[j];
        }
      }
      this._innerSteps = 0;
    }
  }
  zeroGrad() { for (const p of this.params) p.zeroGrad(); }
  toJSON() {
    return { type: 'ranger', lr: this.lr, beta1: this.beta1, beta2: this.beta2,
             eps: this.eps, weightDecay: this.weightDecay,
             lookaheadK: this.lookaheadK, lookaheadAlpha: this.lookaheadAlpha,
             t: this.t, innerSteps: this._innerSteps,
             m: this.m.map(buf => Array.from(buf)),
             v: this.v.map(buf => Array.from(buf)),
             slow: this.slow.map(buf => Array.from(buf)) };
  }
  loadFromJSON(o) {
    if (!o || o.type !== 'ranger') return false;
    if (!Array.isArray(o.m) || !Array.isArray(o.v) || !Array.isArray(o.slow)) return false;
    if (o.m.length !== this.params.length) return false;
    for (let i = 0; i < this.params.length; i++) {
      if (!o.m[i] || o.m[i].length !== this.params[i].size) return false;
      if (!o.v[i] || o.v[i].length !== this.params[i].size) return false;
      if (!o.slow[i] || o.slow[i].length !== this.params[i].size) return false;
    }
    this.t = o.t || 0;
    this._innerSteps = o.innerSteps || 0;
    for (let i = 0; i < this.params.length; i++) {
      this.m[i]    = new Float32Array(o.m[i]);
      this.v[i]    = new Float32Array(o.v[i]);
      this.slow[i] = new Float32Array(o.slow[i]);
    }
    return true;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Factory
// ─────────────────────────────────────────────────────────────────────────────
function buildOptim(name, params, opts) {
  switch (name) {
    case 'sgd':       return new SGD(params, opts);
    case 'adam':      return new Adam(params, opts);
    case 'adamw':     return new AdamW(params, opts);
    case 'radam':     return new RAdam(params, opts);
    case 'lion':      return new Lion(params, opts);
    case 'adafactor': return new Adafactor(params, opts);
    case 'adamw8bit': return new AdamW8bit(params, opts);
    case 'lamb':      return new LAMB(params, opts);
    case 'lars':      return new LARS(params, opts);
    case 'ranger':    return new Ranger(params, opts);
    default: throw new Error('unknown optimizer: ' + name);
  }
}

module.exports = { SGD, Adam, AdamW, RAdam, Lion, Adafactor, AdamW8bit, LAMB, LARS, Ranger, buildOptim };
