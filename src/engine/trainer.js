'use strict';

const T = require('./tensor');
const { buildFromSpec, restoreFromState, CharLM } = require('./model');
const { buildOptim } = require('./optim');
const { CharTokenizer, buildTokenizer, tokenizerFromJSON } = require('./tokenizer');
const ChatFormat = require('./chat-format');

// Top-level entry: run one training session for a network config.
// network: the full stored network object.
// hooks: { onProgress({epoch, step, totalSteps, loss, lr, elapsedMs}), shouldStop() => bool }
// Returns: { state, metrics, tokenizer? }
//
// On the multi-worker path (workers > 0 in network.training), we delegate to
// data-parallel training instead of the single-thread loop — this lets us
// actually use multiple cores instead of pinning one at 100%.
async function trainNetwork(network, hooks = {}) {
  const requestedWorkers = (network.training && network.training.workers) | 0;
  if (requestedWorkers > 1) {
    const { trainNetworkParallel } = require('./trainer-parallel');
    try {
      return await trainNetworkParallel(network, hooks);
    } catch (e) {
      // Workers can fail for environment reasons (sandbox, missing modules,
      // etc.). Fall through to the single-thread path with a logged warning
      // rather than failing the whole training run.
      if (hooks.log) hooks.log(`worker pool failed (${e.message}); falling back to single-thread training`);
    }
  }
  // Single-thread path. Yields occasionally so the UI thread stays responsive
  // and the user can still hit "Stop training" — but only every ~50ms of work,
  // not every step, since each yield costs ~0.5-1ms of event-loop overhead.
  const gen = _trainCoreGen(network, hooks);
  let result;
  let lastYield = Date.now();
  while (true) {
    const step = gen.next();
    if (step.done) { result = step.value; break; }
    // Yield only when enough wall time has passed. The generator yields at
    // its progress checkpoints, but we batch them into a single setImmediate
    // round-trip when they fire close together.
    const now = Date.now();
    if (now - lastYield >= 16) {
      await new Promise(r => setImmediate(r));
      lastYield = Date.now();
    }
  }
  return result;
}

function trainNetworkSync(network, hooks = {}) {
  // Runs the generator to completion without yielding.
  const gen = _trainCoreGen(network, hooks);
  let step;
  while (!(step = gen.next()).done) {}
  return step.value;
}

function* _trainCoreGen(network, hooks) {
  const onProgress = hooks.onProgress || (() => {});
  const shouldStop = hooks.shouldStop || (() => false);
  const log = hooks.log || (() => {});
  const fromScratch = !!hooks.fromScratch;

  const seed = network.training?.seed ?? 42;
  const rng = T.rngFromSeed(seed);

  // Build or restore model.
  // - If fromScratch is set, ignore any saved state and rebuild fresh.
  // - Otherwise, load saved weights (continuation training).
  let model;
  let resumed = false;
  if (network.state && !fromScratch) {
    model = restoreFromState(network.state, network.architecture, rng);
    resumed = true;
    // The saved state's arch is the source of truth for shape-bearing fields
    // (vocabSize specifically — set by the trainer after the tokenizer is
    // built from corpus). If the on-disk `network.architecture` drifted (e.g.
    // before we persisted arch from training, the editor's vocabSize=0 was
    // saved), align it now so downstream sanity checks don't wipe the model.
    if (network.state.arch) {
      const savedArch = network.state.arch;
      if (savedArch.vocabSize && savedArch.vocabSize !== network.architecture.vocabSize) {
        network.architecture = { ...network.architecture, vocabSize: savedArch.vocabSize };
      }
    }
  } else {
    model = buildFromSpec(network.architecture, rng);
  }

  // Build optimizer. Restore momentum/variance buffers if continuing — this
  // prevents the loss spike you'd otherwise see on the first few steps after
  // resume, because Adam would restart its running statistics from zero.
  const optName = network.training?.optimizer || 'adam';
  const lr = network.training?.learningRate ?? 1e-3;
  const optim = buildOptim(optName, model.params, { lr });
  let restoredOptim = false;
  if (resumed && network.optimizerState) {
    restoredOptim = optim.loadFromJSON(network.optimizerState);
    if (restoredOptim) log(`continuing from saved weights (optimizer state restored, ${optim.t ?? 0} prior steps)`);
    else log('continuing from saved weights (optimizer state mismatched — starting optimizer fresh)');
  } else if (resumed) {
    log('continuing from saved weights (no optimizer state — starting optimizer fresh)');
  } else if (fromScratch && network.state) {
    log('training from scratch (existing weights discarded)');
  }

  // Prepare data based on kind
  const arch = network.architecture;
  const data = network.trainingData || {};
  const batchSize = network.training?.batchSize || 32;
  const epochs = network.training?.epochs || 20;

  const start = Date.now();
  const metrics = [];

  if (arch.kind === 'classifier' || arch.kind === 'mlp' || arch.kind === 'regressor') {
    const samples = data.samples || [];
    if (samples.length === 0) throw new Error('No training samples provided');
    const isRegression = arch.kind === 'regressor';

    // Build X and y arrays
    const N = samples.length;
    const X = new Float32Array(N * arch.inputDim);
    let Y; // either Float32Array (regression) or int[] (classification)
    if (isRegression) Y = new Float32Array(N * arch.outputDim);
    else Y = new Array(N);

    for (let i = 0; i < N; i++) {
      const sample = samples[i];
      if (!Array.isArray(sample.input) || sample.input.length !== arch.inputDim) {
        throw new Error(`sample ${i} input length ${sample.input?.length} != inputDim ${arch.inputDim}`);
      }
      for (let j = 0; j < arch.inputDim; j++) X[i * arch.inputDim + j] = sample.input[j];
      if (isRegression) {
        if (!Array.isArray(sample.output) || sample.output.length !== arch.outputDim) {
          throw new Error(`sample ${i} output length invalid`);
        }
        for (let j = 0; j < arch.outputDim; j++) Y[i * arch.outputDim + j] = sample.output[j];
      } else {
        const label = typeof sample.label === 'number' ? sample.label : sample.output;
        if (typeof label !== 'number') throw new Error(`sample ${i} missing numeric label`);
        Y[i] = label;
      }
    }

    // Index shuffle buffer
    const idx = new Int32Array(N);
    for (let i = 0; i < N; i++) idx[i] = i;

    const stepsPerEpoch = Math.max(1, Math.floor(N / batchSize));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;

    for (let ep = 0; ep < epochs; ep++) {
      // shuffle
      for (let i = N - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        const tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
      }
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) return { state: model.toJSON(), optimizerState: optim.toJSON(), architecture: arch, metrics, stopped: true };
        const xbData = new Float32Array(batchSize * arch.inputDim);
        const ybLabels = isRegression ? new Float32Array(batchSize * arch.outputDim) : new Array(batchSize);
        for (let b = 0; b < batchSize; b++) {
          const sIdx = idx[(s * batchSize + b) % N];
          for (let j = 0; j < arch.inputDim; j++) xbData[b * arch.inputDim + j] = X[sIdx * arch.inputDim + j];
          if (isRegression) {
            for (let j = 0; j < arch.outputDim; j++) ybLabels[b * arch.outputDim + j] = Y[sIdx * arch.outputDim + j];
          } else {
            ybLabels[b] = Y[sIdx];
          }
        }
        const xb = new T.Tensor([batchSize, arch.inputDim], xbData, false);
        optim.zeroGrad();
        const logits = model.forward(xb, { training: true, rng });
        let loss;
        if (isRegression) {
          const yb = new T.Tensor([batchSize, arch.outputDim], ybLabels, false);
          loss = T.mseLoss(logits, yb);
        } else {
          loss = T.softmaxCrossEntropy(logits, ybLabels);
        }
        loss.backward();
        optim.step();
        epLoss += loss.data[0];
        globalStep++;

        if (globalStep % Math.max(1, Math.floor(totalSteps / 200)) === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: loss.data[0],
            elapsedMs: Date.now() - start
          });
          yield;
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }

    return { state: model.toJSON(), optimizerState: optim.toJSON(), architecture: arch, metrics };
  }

  if (arch.kind === 'charLM') {
    // text-based: data.text OR data.samples:[{text}] OR chat-shaped samples
    const corpus = ChatFormat.buildCorpus(data);
    const text = corpus.text;
    if (text.length < arch.contextLen + 2) {
      throw new Error(corpus.isChat
        ? `Chat corpus too short for contextLen=${arch.contextLen}. Add more sample pairs or shorten contextLen.`
        : `Training text too short for contextLen=${arch.contextLen}.`);
    }
    // Mark on the architecture so inference knows to use chat formatting.
    arch.isChat = corpus.isChat;

    // Build/keep tokenizer.
    //
    // Three cases:
    //   1. fromScratch: rebuild vocab from the *current* corpus. The saved
    //      tokenizer might be stale (e.g. user added a sample with new chars
    //      and clicked "Train from scratch") — reusing it here would crash
    //      with "char not in vocab" when we encode the corpus below.
    //   2. continue with saved tokenizer that already covers the corpus:
    //      keep the saved tokenizer as-is so token IDs stay aligned with
    //      the embedding/output rows the trained weights expect.
    //   3. continue but the corpus contains new chars: extend the vocab
    //      (append-only so existing IDs don't shift), then rebuild the model
    //      from scratch — the embedding and output-projection dims have to
    //      grow to fit the new vocab, and there's no clean way to splice
    //      new rows into the saved weights. We log this so the user knows
    //      why the loss restarts high in this specific case.
    const tokKind = arch.tokenizerKind || 'char';
    let tokenizer;
    if (network.tokenizer && !fromScratch) {
      tokenizer = tokenizerFromJSON(network.tokenizer);
      let needsRebuild = false;
      if (tokenizer.kind === 'char') {
        // Char: append-only extension preserves existing token IDs
        const known = new Set(tokenizer.chars);
        const novel = [];
        for (const ch of text) if (!known.has(ch)) { known.add(ch); novel.push(ch); }
        if (novel.length > 0) {
          tokenizer = new CharTokenizer(tokenizer.chars.concat(novel));
          const preview = novel.slice(0, 8).map(c => JSON.stringify(c)).join(', ');
          log(`vocab expanded by ${novel.length} new char(s) [${preview}${novel.length > 8 ? ', …' : ''}] — rebuilding model to fit new vocab size`);
          needsRebuild = true;
        }
      } else if (tokenizer.kind === 'word') {
        // Word: append new unseen words/whitespace tokens
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
        // WordPart (BPE): cannot extend merges incrementally — rebuild if new chars appear
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
        const newOpt = buildOptim(optName, model.params, { lr });
        Object.assign(optim, newOpt);
        resumed = false;
      }
    } else {
      tokenizer = buildTokenizer(text, tokKind, { vocabSize: 512 });
    }
    // First-training-run alignment: when arch.vocabSize was 0 (from a freshly
    // created network), set it to whatever the tokenizer produced and rebuild
    // the model now that we know the real vocab size. We only do this when
    // NOT resumed — on a resume path, `model` was built from the saved state
    // which already matches `tokenizer.vocabSize` (we aligned arch above).
    // Wiping the model on resume here was the source of the "Continue training
    // loss spiked back to fresh-init levels" bug.
    if (!resumed && tokenizer.vocabSize !== arch.vocabSize) {
      arch.vocabSize = tokenizer.vocabSize;
      model = buildFromSpec(arch, rng);
      const newOpt = buildOptim(optName, model.params, { lr });
      Object.assign(optim, newOpt);
    }

    const ids = tokenizer.encode(text);
    const L = arch.contextLen;
    const N = ids.length - L;
    if (N <= 0) throw new Error('not enough tokens');

    const stepsPerEpoch = Math.max(1, Math.floor(N / batchSize));
    const totalSteps = epochs * stepsPerEpoch;
    let globalStep = 0;

    for (let ep = 0; ep < epochs; ep++) {
      let epLoss = 0;
      for (let s = 0; s < stepsPerEpoch; s++) {
        if (shouldStop()) return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), architecture: arch, metrics, stopped: true };
        const idsBatch = [];
        const labels = [];
        for (let b = 0; b < batchSize; b++) {
          const start = Math.floor(rng() * N);
          const ctx = ids.slice(start, start + L);
          const nxt = ids[start + L];
          idsBatch.push(ctx);
          labels.push(nxt);
        }
        optim.zeroGrad();
        const logits = model.forward(idsBatch, { training: true, rng });
        const loss = T.softmaxCrossEntropy(logits, labels);
        loss.backward();
        optim.step();
        epLoss += loss.data[0];
        globalStep++;
        if (globalStep % Math.max(1, Math.floor(totalSteps / 200)) === 0 || globalStep === totalSteps) {
          onProgress({
            epoch: ep + 1, totalEpochs: epochs,
            step: globalStep, totalSteps,
            loss: loss.data[0],
            elapsedMs: Date.now() - start
          });
          yield;
        }
      }
      metrics.push({ epoch: ep + 1, loss: epLoss / stepsPerEpoch });
      log(`epoch ${ep + 1}/${epochs}  loss=${(epLoss / stepsPerEpoch).toFixed(6)}`);
    }
    return { state: model.toJSON(), optimizerState: optim.toJSON(), tokenizer: tokenizer.toJSON(), architecture: arch, metrics };
  }

  throw new Error('unknown arch kind ' + arch.kind);
}

// Inference entry. Returns a JSON-able result.
function infer(network, input) {
  const rng = T.rngFromSeed(network.training?.seed ?? 42);
  if (!network.state) throw new Error('Network has no trained state');
  const model = restoreFromState(network.state, network.architecture, rng);
  const arch = network.architecture;

  if (arch.kind === 'classifier' || arch.kind === 'mlp') {
    const vec = input.input || input;
    if (!Array.isArray(vec) || vec.length !== arch.inputDim) throw new Error('input must be array of length ' + arch.inputDim);
    const x = new T.Tensor([1, arch.inputDim], new Float32Array(vec), false);
    const logits = model.forward(x, { training: false });
    const probs = T.softmax(logits);
    const arr = Array.from(probs.data);
    let best = 0;
    for (let i = 1; i < arr.length; i++) if (arr[i] > arr[best]) best = i;
    return {
      kind: 'classification',
      probs: arr,
      predictedClass: best,
      label: (arch.classes || [])[best] || String(best)
    };
  }

  if (arch.kind === 'regressor') {
    const vec = input.input || input;
    const x = new T.Tensor([1, arch.inputDim], new Float32Array(vec), false);
    const out = model.forward(x, { training: false });
    return { kind: 'regression', output: Array.from(out.data) };
  }

  if (arch.kind === 'charLM') {
    if (!network.tokenizer) throw new Error('tokenizer missing');
    const tokenizer = CharTokenizer.fromJSON(network.tokenizer);
    const maxNew = input.maxTokens ?? 120;
    const temperature = input.temperature ?? 1.0;
    const topK = input.topK ?? 0;

    // If the model was trained on chat data, wrap the prompt in role tags
    // and stop generation at the assistant <|end|> tag.
    const isChat = !!arch.isChat || input.chat === true;

    // Three input shapes for chat-mode:
    //   1. { prompt }                    — single-turn (legacy, still supported)
    //   2. { history: [{role,content}] } — full conversation, last message is the new user turn
    //   3. { messages: [...] }           — alias for history
    //   4. { history, prompt }           — explicit running history + new user turn
    // For non-chat models we just use prompt/string.
    const userPrompt = String(input.prompt ?? input ?? '');
    const history = Array.isArray(input.history) ? input.history
                  : Array.isArray(input.messages) ? input.messages
                  : null;

    let promptText;
    if (isChat) {
      if (history) {
        // If the caller passed both history and prompt, treat prompt as the new user turn.
        // If only history, the last user message in history is the new turn (don't double it).
        const explicitPrompt = (typeof input.prompt === 'string' && input.prompt.length > 0);
        promptText = ChatFormat.wrapHistoryForChat(history, {
          system: input.system || '',
          userPrompt: explicitPrompt ? userPrompt : ''
        });
      } else {
        promptText = ChatFormat.wrapPromptForChat(userPrompt, input.system || '');
      }
    } else {
      promptText = userPrompt;
    }

    // Encode (silently dropping unseen chars).
    const L = arch.contextLen;
    // For chat with history, drop oldest turns to fit the context window
    // (keeps the system anchor + the trailing <|assistant|> intact).
    if (isChat && history) {
      promptText = ChatFormat.truncateWrappedToFit(
        promptText,
        (s) => tokenizer.encodeSafe(s),
        L
      );
    }
    let ids = tokenizer.encodeSafe(promptText);
    if (ids.length === 0) ids = [0];
    // Left-pad to fill the context window. Padding choice matters a lot for
    // chat models: the previous "pad with the first prompt char" produced long
    // runs of '<' in front of '<|user|>...', which the model never saw in
    // training (the training corpus joins conversations with '\n'), and the
    // model would respond with confused tag-fragment garbage. Using '\n' as
    // the pad token mirrors the training-time conversation separator, so the
    // model sees a familiar lead-in: '\n\n\n...<|user|>actual prompt<|end|><|assistant|>'.
    if (ids.length < L) {
      // Prefer newline (conversation separator at training time). If the model
      // has never seen one, fall back to space, then to the first prompt char.
      let padChar = '\n';
      if (tokenizer.stoi && !tokenizer.stoi.has(padChar)) padChar = ' ';
      if (tokenizer.stoi && !tokenizer.stoi.has(padChar)) padChar = null;
      const padId = padChar != null && tokenizer.stoi.has(padChar) ? tokenizer.stoi.get(padChar) : ids[0];
      const pad = new Array(L - ids.length).fill(padId);
      ids = pad.concat(ids);
    }

    // Pre-encode the END tag so we can detect it byte-for-byte in the output.
    const endTag = ChatFormat.TAGS.END;
    const endIds = tokenizer.encodeSafe(endTag);
    const stopOnEnd = isChat && endIds.length > 0;
    // For chat models we ALSO stop on the '<|' bigram. Small charLMs frequently
    // emit corrupted end-of-turn markers ('<aend|>', '<|en|>', etc.) and an
    // exact-match-only stop would let the model drift into the next imagined
    // conversation, burning tokens and producing visible role-tag fragments
    // even if extractAssistantReply() trims them after the fact. Stopping at
    // '<|' the moment it appears keeps generation tight.
    const tagOpenBigram = '<|';

    const out = [];
    // Window of recently decoded characters used for the '<|' early-stop check.
    // We don't want to decode the whole output every step, so we keep a small
    // tail buffer (a couple of decoded chars is enough to span the bigram).
    let tailDecoded = '';
    for (let step = 0; step < maxNew; step++) {
      const ctx = ids.slice(ids.length - L);
      const logits = model.forward([ctx], { training: false });
      const row = Array.from(logits.data);
      for (let i = 0; i < row.length; i++) row[i] /= Math.max(temperature, 1e-6);
      let indices = row.map((v, i) => i);
      if (topK > 0 && topK < row.length) {
        indices.sort((a, b) => row[b] - row[a]);
        indices = indices.slice(0, topK);
      }
      const maxv = Math.max(...indices.map(i => row[i]));
      let sum = 0;
      const ex = indices.map(i => { const e = Math.exp(row[i] - maxv); sum += e; return e; });
      const probs = ex.map(e => e / sum);
      let r = Math.random();
      let pick = indices[indices.length - 1];
      for (let i = 0; i < indices.length; i++) {
        r -= probs[i];
        if (r <= 0) { pick = indices[i]; break; }
      }
      ids.push(pick);
      out.push(pick);
      // Stop if we just generated the END tag.
      if (stopOnEnd && out.length >= endIds.length) {
        let match = true;
        for (let k = 0; k < endIds.length; k++) {
          if (out[out.length - endIds.length + k] !== endIds[k]) { match = false; break; }
        }
        if (match) {
          out.length -= endIds.length; // strip the tag from output
          break;
        }
      }
      // Chat-mode early-stop on the '<|' bigram (any role-tag attempt, even
      // corrupted). Decode just the new char and keep the last few in a tail.
      if (isChat) {
        tailDecoded += tokenizer.decode([pick]);
        if (tailDecoded.length > 8) tailDecoded = tailDecoded.slice(-8);
        const bigramAt = tailDecoded.indexOf(tagOpenBigram);
        if (bigramAt !== -1) {
          // We need to trim `out` so the decoded output ends just before '<|'.
          // The simplest robust approach: decode the entire `out`, find the
          // first '<|', then re-encode the prefix and keep only those tokens.
          // This costs one full decode at stop time, not every step.
          const fullDecoded = tokenizer.decode(out);
          const cut = fullDecoded.indexOf(tagOpenBigram);
          if (cut !== -1) {
            const keepText = fullDecoded.slice(0, cut);
            // Re-encode (encodeSafe drops unknowns) — for charLM this is exact.
            const keepIds = tokenizer.encodeSafe(keepText);
            out.length = 0;
            for (const t of keepIds) out.push(t);
          }
          break;
        }
      }
    }
    const generated = tokenizer.decode(out);
    if (isChat) {
      // Final safety: strip any leftover tag content the model produced.
      const reply = ChatFormat.extractAssistantReply(generated);
      return { kind: 'generation', text: reply, raw: generated, tokens: out, chat: true };
    }
    return { kind: 'generation', text: generated, tokens: out };
  }

  throw new Error('unknown arch kind ' + arch.kind);
}

module.exports = { trainNetwork, trainNetworkSync, infer };
