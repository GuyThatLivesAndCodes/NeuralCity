'use strict';

// Data-parallel trainer using worker_threads. The math is "synchronous SGD":
// at each global step, every worker processes ONE batch, gradients are summed
// (mean-reduced) across workers, and a single optimizer step updates the
// shared weights. Effective batch = workers × per-worker batch.
//
// Why this layout:
//
//   - One "epoch" still costs the same number of WEIGHT UPDATES as the
//     single-thread version, but each update sees `workers`× more data, so
//     gradient quality is better and the loss curve typically converges in
//     fewer epochs of wall time. From the user's perspective, "an epoch" is
//     the same conceptual unit (one pass over the corpus).
//
//   - SharedArrayBuffer for both weights and per-worker gradient slots means
//     zero-copy: the main thread doesn't ship weights every step (which would
//     swamp the speedup with IPC bytes), and workers don't ship gradients
//     back — they write into a slice of shared memory and just pingback "done".
//
//   - The dispatch loop fires all N workers in parallel, then awaits all N
//     "done" messages, so each step is bounded by the slowest worker. This
//     matters: don't let one worker pull a giant slice while others starve.
//
// Falls back to single-thread training if worker_threads are unavailable
// (e.g. running under a sandbox), or if `workers` is set to 0/1.

const os = require('os');
const path = require('path');
const { Worker } = require('worker_threads');

const T = require('./tensor');
const { buildFromSpec, restoreFromState, CharLM } = require('./model');
const { buildOptim } = require('./optim');
const { CharTokenizer, WordPartTokenizer, buildTokenizer, tokenizerFromJSON } = require('./tokenizer');
const ChatFormat = require('./chat-format');

// Lay out all model params end-to-end in a single SharedArrayBuffer for the
// weights, plus per-worker SAB slots for gradients. Returns:
//   {
//     weightSAB,                // SharedArrayBuffer for weights
//     weightView,               // Float32Array view (length = total params)
//     paramSpecs: [{offset,size}],
//     totalParams,
//     gradSABs: SharedArrayBuffer[] (one per worker),
//     gradViews: Float32Array[],
//     gradOffsets: int[]        // same per worker, mirrors paramSpecs
//   }
function layoutParams(params, numWorkers) {
  const paramSpecs = [];
  let totalParams = 0;
  for (const p of params) {
    paramSpecs.push({ offset: totalParams, size: p.size });
    totalParams += p.size;
  }
  const weightSAB = new SharedArrayBuffer(totalParams * 4);
  const weightView = new Float32Array(weightSAB);
  // Copy current weights into the SAB so subsequent forward passes (in workers
  // OR main thread on the fallback path) see the right initial values.
  let off = 0;
  for (const p of params) {
    weightView.set(p.data, off);
    off += p.size;
  }
  // Re-alias each param's .data to the SAB view so optimizer updates land
  // in shared memory automatically (no explicit broadcast needed).
  off = 0;
  for (const p of params) {
    p.data = weightView.subarray(off, off + p.size);
    off += p.size;
  }
  const gradSABs = [];
  const gradViews = [];
  const gradOffsets = paramSpecs.map(s => s.offset); // same layout per worker
  for (let w = 0; w < numWorkers; w++) {
    const sab = new SharedArrayBuffer(totalParams * 4);
    gradSABs.push(sab);
    gradViews.push(new Float32Array(sab));
  }
  return { weightSAB, weightView, paramSpecs, totalParams, gradSABs, gradViews, gradOffsets };
}

function spawnWorker(workerPath, archSpec, modelState, paramSpecs, gradOffsets, weightSAB, gradSAB) {
  return new Worker(workerPath, {
    workerData: { archSpec, modelState, paramSpecs, gradOffsets, weightSAB, gradSAB }
  });
}

// Main entry. Falls back gracefully if worker_threads isn't available or
// fails for any reason — caller (trainNetwork) has another fallback layer.
async function trainNetworkParallel(network, hooks = {}) {
  const onProgress = hooks.onProgress || (() => {});
  const shouldStop = hooks.shouldStop || (() => false);
  const log = hooks.log || (() => {});
  const fromScratch = !!hooks.fromScratch;

  const cpuCount = os.cpus().length;
  // Cap at logical CPUs - 1 so the UI thread + IPC have a core to breathe on.
  // The user's `workers` value is treated as a request, capped to what's safe.
  const requested = (network.training && network.training.workers) | 0;
  const numWorkers = Math.max(2, Math.min(requested, Math.max(1, cpuCount - 1)));

  // Mirror the seed handling in the single-thread trainer so reproducibility
  // properties don't change wildly between modes.
  const seed = network.training?.seed ?? 42;
  const rng = T.rngFromSeed(seed);

  // ============== Build / restore the model in the main thread ==============
  let model;
  let resumed = false;
  if (network.state && !fromScratch) {
    model = restoreFromState(network.state, network.architecture, rng);
    resumed = true;
    if (network.state.arch) {
      const savedArch = network.state.arch;
      if (savedArch.vocabSize && savedArch.vocabSize !== network.architecture.vocabSize) {
        network.architecture = { ...network.architecture, vocabSize: savedArch.vocabSize };
      }
    }
  } else {
    model = buildFromSpec(network.architecture, rng);
  }

  // ============== Tokenizer / corpus prep (charLM only path here) ==========
  const arch = network.architecture;
  const data = network.trainingData || {};
  const batchSize = network.training?.batchSize || 32;
  const epochs = network.training?.epochs || 20;

  let tokenizer = null;
  let ids = null;
  let L = 0;
  let N = 0;

  if (arch.kind === 'charLM') {
    const corpus = ChatFormat.buildCorpus(data);
    const text = corpus.text;
    if (text.length < arch.contextLen + 2) {
      throw new Error(corpus.isChat
        ? `Chat corpus too short for contextLen=${arch.contextLen}.`
        : `Training text too short for contextLen=${arch.contextLen}.`);
    }
    arch.isChat = corpus.isChat;
    const tokKind = arch.tokenizerKind || 'char';
    if (network.tokenizer && !fromScratch) {
      tokenizer = tokenizerFromJSON(network.tokenizer);
      let needsRebuild = false;
      if (tokenizer.kind === 'char') {
        const known = new Set(tokenizer.chars);
        const novel = [];
        for (const ch of text) if (!known.has(ch)) { known.add(ch); novel.push(ch); }
        if (novel.length > 0) {
          tokenizer = new CharTokenizer(tokenizer.chars.concat(novel));
          const preview = novel.slice(0, 8).map(c => JSON.stringify(c)).join(', ');
          log(`vocab expanded by ${novel.length} new char(s) [${preview}${novel.length > 8 ? ', …' : ''}] — rebuilding model`);
          needsRebuild = true;
        }
      } else if (tokenizer.kind === 'word') {
        const tokens = text.match(/\S+|\s+/g) || [];
        const known = new Set(tokenizer.words);
        const novel = [];
        for (const t of tokens) if (!known.has(t)) { known.add(t); novel.push(t); }
        if (novel.length > 0) {
          const { WordTokenizer } = require('./tokenizer');
          tokenizer = new WordTokenizer(tokenizer.words.concat(novel));
          log(`word vocab expanded by ${novel.length} new token(s) — rebuilding model`);
          needsRebuild = true;
        }
      } else {
        const knownChars = new Set(Object.keys(tokenizer.vocab).flatMap(t => Array.from(t)));
        const hasNew = Array.from(new Set(Array.from(text))).some(ch => !knownChars.has(ch));
        if (hasNew) {
          tokenizer = buildTokenizer(text, 'wordpart', { vocabSize: 512 });
          log(`wordpart vocab rebuilt from scratch for new corpus characters`);
          needsRebuild = true;
        }
      }
      if (needsRebuild) {
        arch.vocabSize = tokenizer.vocabSize;
        model = buildFromSpec(arch, rng);
        resumed = false;
      }
    } else {
      // trainNetworkParallel is already async so we can await the BPE build
      // directly — no separate pre-build step needed.
      const _yield = () => new Promise(r => setImmediate(r));
      if (tokKind === 'wordpart') {
        tokenizer = await WordPartTokenizer.fromCorpusAsync(text, 512, _yield);
      } else {
        tokenizer = buildTokenizer(text, tokKind, { vocabSize: 512 });
      }
    }
    if (!resumed && tokenizer.vocabSize !== arch.vocabSize) {
      arch.vocabSize = tokenizer.vocabSize;
      model = buildFromSpec(arch, rng);
    }
    const _yieldFn = () => new Promise(r => setImmediate(r));
    ids = tokenizer.kind === 'wordpart'
      ? await tokenizer.encodeAsync(text, _yieldFn)
      : tokenizer.encode(text);
    L = arch.contextLen;
    N = ids.length - L;
    if (N <= 0) throw new Error('not enough tokens');
  } else if (arch.kind === 'regressor' || arch.kind === 'classifier' || arch.kind === 'mlp') {
    // Tabular data path is supported by the worker; we set this up below in
    // the dispatch loop. Nothing to pre-tokenize.
  } else {
    throw new Error('parallel trainer: unknown arch kind ' + arch.kind);
  }

  // ============== Optimizer (lives only in main thread) ====================
  const optName = network.training?.optimizer || 'adam';
  const lr = network.training?.learningRate ?? 1e-3;
  const optim = buildOptim(optName, model.params, { lr });
  if (resumed && network.optimizerState) {
    const ok = optim.loadFromJSON(network.optimizerState);
    if (ok) log(`continuing from saved weights (optimizer state restored, ${optim.t ?? 0} prior steps)`);
    else log('continuing from saved weights (optimizer state mismatched — starting optimizer fresh)');
  } else if (resumed) {
    log('continuing from saved weights (no optimizer state — starting optimizer fresh)');
  } else if (fromScratch && network.state) {
    log('training from scratch (existing weights discarded)');
  }

  // ============== Lay out shared memory and spawn workers =================
  const layout = layoutParams(model.params, numWorkers);
  const workerPath = path.join(__dirname, 'trainer-worker.js');
  // Strip rng (functions don't survive postMessage / structuredClone).
  const archForWorker = { ...arch, seed };
  // We pass NO modelState — workers will build a fresh model from the spec
  // and immediately swap their .data views to the shared weight buffer (which
  // we already populated with the current weights). This avoids serializing
  // the entire model JSON to each worker.
  const workers = [];
  for (let w = 0; w < numWorkers; w++) {
    const wkr = spawnWorker(workerPath, archForWorker, null, layout.paramSpecs,
                            layout.gradOffsets, layout.weightSAB, layout.gradSABs[w]);
    workers.push(wkr);
  }
  log(`data-parallel training: ${numWorkers} workers, effective batch = ${numWorkers * batchSize}`);

  // ============== Dispatch helpers ==========================================
  // Send one step to one worker, return a Promise that resolves to its loss.
  function dispatchStep(workerIdx, msg) {
    return new Promise((resolve, reject) => {
      const wkr = workers[workerIdx];
      const onMsg = (m) => {
        if (m.type === 'done') { wkr.off('message', onMsg); wkr.off('error', onErr); resolve(m.loss); }
        else if (m.type === 'error') { wkr.off('message', onMsg); wkr.off('error', onErr); reject(new Error(m.message)); }
      };
      const onErr = (e) => { wkr.off('message', onMsg); wkr.off('error', onErr); reject(e); };
      wkr.on('message', onMsg);
      wkr.on('error', onErr);
      wkr.postMessage(msg);
    });
  }

  // Average per-worker gradient slots back into model.params[i].grad. We sum
  // across workers, then divide by numWorkers to keep loss-scale invariant
  // (it's effectively one larger batch, so the gradient magnitude should not
  // grow with worker count).
  function reduceGradients() {
    const inv = 1 / numWorkers;
    for (let i = 0; i < model.params.length; i++) {
      const p = model.params[i];
      const sz = p.size;
      const off = layout.gradOffsets[i];
      // Re-alias .grad to a fresh accumulator so the optimizer reads from it.
      // We can write straight into the first worker's slot then add the rest;
      // we copy first so we don't mutate that worker's view (it's about to
      // be re-zeroed by the worker on its next step anyway, but explicit > implicit).
      const acc = new Float32Array(sz);
      for (let w = 0; w < numWorkers; w++) {
        const view = layout.gradViews[w].subarray(off, off + sz);
        for (let j = 0; j < sz; j++) acc[j] += view[j];
      }
      for (let j = 0; j < sz; j++) acc[j] *= inv;
      p.grad = acc;
    }
  }

  function shutdown() {
    for (const w of workers) {
      try { w.postMessage({ type: 'shutdown' }); } catch (_) {}
      try { w.terminate(); } catch (_) {}
    }
  }

  // ============== Training loop ============================================
  const start = Date.now();
  const metrics = [];

  if (arch.kind === 'charLM') {
    const stepsPerEpoch = Math.max(1, Math.floor(N / (batchSize * numWorkers)));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;
    const reportEvery = Math.max(1, Math.floor(totalSteps / 200));

    for (let ep = 0; ep < epochs; ep++) {
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) { shutdown(); return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), architecture: arch, metrics, stopped: true }; }
        // Build N batches in the main thread (cheap — just integer slicing).
        const promises = [];
        for (let w = 0; w < numWorkers; w++) {
          const batch = new Array(batchSize);
          const labels = new Array(batchSize);
          for (let b = 0; b < batchSize; b++) {
            const startIdx = Math.floor(rng() * N);
            batch[b] = ids.slice(startIdx, startIdx + L);
            labels[b] = ids[startIdx + L];
          }
          promises.push(dispatchStep(w, { type: 'step', batch, labels }));
        }
        let losses;
        try { losses = await Promise.all(promises); }
        catch (e) { shutdown(); throw e; }
        const meanLoss = losses.reduce((a, b) => a + b, 0) / numWorkers;
        reduceGradients();
        optim.step();
        epLoss += meanLoss;
        globalStep++;
        if (globalStep % reportEvery === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: meanLoss,
            elapsedMs: Date.now() - start
          });
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }

    shutdown();
    return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), architecture: arch, metrics };
  }

  if (arch.kind === 'classifier' || arch.kind === 'mlp' || arch.kind === 'regressor') {
    const samples = data.samples || [];
    if (samples.length === 0) throw new Error('No training samples provided');
    const isRegression = arch.kind === 'regressor';

    const Nrows = samples.length;
    const X = new Float32Array(Nrows * arch.inputDim);
    let Y;
    if (isRegression) Y = new Float32Array(Nrows * arch.outputDim);
    else Y = new Array(Nrows);
    for (let i = 0; i < Nrows; i++) {
      const sample = samples[i];
      for (let j = 0; j < arch.inputDim; j++) X[i * arch.inputDim + j] = sample.input[j];
      if (isRegression) {
        for (let j = 0; j < arch.outputDim; j++) Y[i * arch.outputDim + j] = sample.output[j];
      } else {
        const label = typeof sample.label === 'number' ? sample.label : sample.output;
        Y[i] = label;
      }
    }

    const idx = new Int32Array(Nrows);
    for (let i = 0; i < Nrows; i++) idx[i] = i;
    const stepsPerEpoch = Math.max(1, Math.floor(Nrows / (batchSize * numWorkers)));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;
    const reportEvery = Math.max(1, Math.floor(totalSteps / 200));

    for (let ep = 0; ep < epochs; ep++) {
      for (let i = Nrows - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) { shutdown(); return { state: model.toJSON(), optimizerState: optim.toJSON(), architecture: arch, metrics, stopped: true }; }
        const promises = [];
        for (let w = 0; w < numWorkers; w++) {
          const batchData = new Float32Array(batchSize * arch.inputDim);
          const labelData = isRegression ? new Float32Array(batchSize * arch.outputDim) : new Array(batchSize);
          for (let b = 0; b < batchSize; b++) {
            const sIdx = idx[((s * numWorkers + w) * batchSize + b) % Nrows];
            for (let j = 0; j < arch.inputDim; j++) batchData[b * arch.inputDim + j] = X[sIdx * arch.inputDim + j];
            if (isRegression) {
              for (let j = 0; j < arch.outputDim; j++) labelData[b * arch.outputDim + j] = Y[sIdx * arch.outputDim + j];
            } else {
              labelData[b] = Y[sIdx];
            }
          }
          promises.push(dispatchStep(w, {
            type: 'step',
            batchSize,
            batchData: Array.from(batchData), // postMessage doesn't keep Float32Array as Float32Array unless transferred; just copy
            labelData: isRegression ? Array.from(labelData) : labelData,
            labels: isRegression ? null : labelData
          }));
        }
        let losses;
        try { losses = await Promise.all(promises); }
        catch (e) { shutdown(); throw e; }
        const meanLoss = losses.reduce((a, b) => a + b, 0) / numWorkers;
        reduceGradients();
        optim.step();
        epLoss += meanLoss;
        globalStep++;
        if (globalStep % reportEvery === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: meanLoss,
            elapsedMs: Date.now() - start
          });
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }
    shutdown();
    return { state: model.toJSON(), optimizerState: optim.toJSON(), architecture: arch, metrics };
  }

  shutdown();
  throw new Error('parallel trainer: unsupported arch kind ' + arch.kind);
}

module.exports = { trainNetworkParallel };
