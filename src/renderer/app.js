'use strict';

// NeuralCity renderer. Single-file UI controller that reads from window.nc.

const state = {
  networks: [],
  selectedId: null,
  current: null,
  activeTab: 'networks',
  training: { running: false, history: [], logs: [] },
  apiServers: new Map(), // id -> {port, url}
  apiLogs: [],
  currentDocId: 'welcome',
  // Per-network chat sessions for the in-app Inference tab. Keyed by network id
  // so switching nets doesn't trample running conversations.
  chatSessions: new Map() // id -> { history: [{role, content}], system: string, busy: bool }
};

function getChatSession(id) {
  let s = state.chatSessions.get(id);
  if (!s) {
    // editingIndex: null normally; set to a turn index when the user taps
    // [Edit] on one of their messages, and the chat-log redraws with an
    // inline textarea in place of that turn's content bubble.
    s = { history: [], system: '', busy: false, editingIndex: null };
    state.chatSessions.set(id, s);
  }
  return s;
}

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => document.querySelectorAll(sel);

// ---------- boot ----------

window.addEventListener('DOMContentLoaded', async () => {
  bindChrome();
  bindModals();
  bindTrainingStreams();
  bindApiStreams();
  await refreshNetworks();
  await fillSystemInfo();
  renderActiveTab();
});

function bindChrome() {
  $$('.tab').forEach(b => b.addEventListener('click', () => {
    state.activeTab = b.dataset.tab;
    $$('.tab').forEach(x => x.classList.toggle('active', x === b));
    renderActiveTab();
  }));
  $('#btn-new').addEventListener('click', openNewNetworkModal);
  $('#btn-import').addEventListener('click', async () => {
    try {
      const net = await window.nc.networks.importNet();
      if (net) { await refreshNetworks(); selectNetwork(net.id); toast('Imported.'); }
    } catch (e) { toast('Import failed: ' + e.message); }
  });
  $('#btn-export').addEventListener('click', async () => {
    if (!state.selectedId) return toast('Select a network first.');
    try {
      const p = await window.nc.networks.exportNet(state.selectedId);
      if (p) toast('Saved to ' + p);
    } catch (e) { toast('Export failed: ' + e.message); }
  });
}

function bindModals() {
  $$('[data-close-modal]').forEach(b => b.addEventListener('click', () => {
    $('#' + b.dataset.closeModal).hidden = true;
  }));
  $('#btn-create').addEventListener('click', createNetworkFromModal);
  $('#pass-cancel').addEventListener('click', () => $('#modal-pass').hidden = true);
  $('#confirm-no').addEventListener('click', () => $('#modal-confirm').hidden = true);
}

function bindTrainingStreams() {
  window.nc.training.onProgress(p => {
    if (p.id !== state.selectedId) return;
    if (p.log) {
      state.training.logs.push(p.log);
    } else {
      state.training.running = true;
      state.training.lastProgress = p;
      state.training.history.push({ step: p.step, loss: p.loss });
      if (state.training.history.length > 2000) state.training.history.shift();
      state.training.logs.push(`step ${p.step}/${p.totalSteps}  ep ${p.epoch}/${p.totalEpochs}  loss=${p.loss.toFixed(6)}`);
    }
    if (state.activeTab === 'train') updateTrainLive();
  });
  window.nc.training.onDone(async (p) => {
    state.training.running = false;
    state.training.logs.push(p.stopped ? '-- stopped --' : '-- done --');
    await refreshNetworks();
    if (p.id === state.selectedId) { await loadCurrent(state.selectedId); renderActiveTab(); }
  });
  window.nc.training.onError(e => {
    state.training.running = false;
    state.training.logs.push('ERROR: ' + e.message);
    if (state.activeTab === 'train') updateTrainLive();
  });
}

function bindApiStreams() {
  window.nc.api.onLog(line => {
    state.apiLogs.push(`[${new Date().toLocaleTimeString()}] ${line.id?.slice?.(0, 6) || ''} ${line.line}`);
    if (state.apiLogs.length > 500) state.apiLogs.shift();
    if (state.activeTab === 'api') renderApiPanel();
  });
}

// ---------- networks list ----------

async function refreshNetworks() {
  state.networks = await window.nc.networks.list();
  renderNetworkList();
}

function renderNetworkList() {
  const root = $('#network-list');
  if (!state.networks.length) {
    root.innerHTML = '<div class="empty" style="height:auto;padding:24px;"><div class="big">No networks yet</div><div>Click + New to create one.</div></div>';
    return;
  }
  root.innerHTML = state.networks.map(n => {
    const active = n.id === state.selectedId ? 'active' : '';
    const encBadge = n.encrypted ? '<span class="badge">encrypted</span>' : '';
    const trainBadge = n.trained ? '<span class="badge">trained</span>' : '<span class="badge">new</span>';
    return `
      <div class="net-item ${active}" data-id="${n.id}">
        <div class="n-name">${escapeHtml(n.name)} ${encBadge}</div>
        <div class="n-meta">${n.kind || '—'} ${trainBadge}</div>
      </div>`;
  }).join('');
  root.querySelectorAll('.net-item').forEach(el => {
    el.addEventListener('click', () => selectNetwork(el.dataset.id));
  });
}

async function selectNetwork(id) {
  state.selectedId = id;
  await loadCurrent(id);
  // Stay on whatever tab the user is on. They picked it; respect it.
  renderNetworkList();
  renderActiveTab();
}

async function loadCurrent(id) {
  try { state.current = await window.nc.networks.get(id); }
  catch (e) { state.current = null; toast('Failed to load: ' + e.message); }
  state.training.history = [];
  state.training.logs = [];
}

// ---------- tabs ----------

function renderActiveTab() {
  const content = $('#content');
  switch (state.activeTab) {
    case 'networks': return renderNetworksTab(content);
    case 'train': return renderTrainTab(content);
    case 'infer': return renderInferenceTab(content);
    case 'api': return renderApiPanel();
    case 'script': return renderScriptTab(content);
    case 'docs': return renderDocsTab(content);
  }
}

// Networks / editor tab
function renderNetworksTab(root) {
  if (!state.current) {
    root.innerHTML = `<div class="empty"><div class="big">No network selected</div><div>Pick one from the sidebar or click + New.</div></div>`;
    return;
  }
  const n = state.current;
  const a = n.architecture;
  root.innerHTML = `
    <div class="panel">
      <h2>Editor — ${escapeHtml(n.name)}</h2>
      <p class="hint">${n.kind || a.kind} · ${n.encrypted ? 'encrypted at rest' : 'stored as plaintext'} · updated ${new Date(n.updatedAt).toLocaleString()}</p>

      <div class="section">
        <h3>Identity</h3>
        <div class="grid-2">
          <label class="field"><span>Name</span><input id="edit-name" type="text" value="${escapeHtml(n.name)}"></label>
          <label class="field"><span>Type</span><input readonly value="${a.kind}"></label>
        </div>
        <label class="field"><span>Description</span>
          <textarea id="edit-desc" rows="2">${escapeHtml(n.description || '')}</textarea>
        </label>
      </div>

      <div class="section">
        <h3>Architecture</h3>
        ${archEditor(a)}
      </div>

      <div class="section">
        <h3>Training defaults</h3>
        <div class="grid-3">
          <label class="field"><span>Optimizer</span>
            <select id="t-optimizer">
              <option value="adam" ${n.training.optimizer === 'adam' ? 'selected' : ''}>Adam</option>
              <option value="sgd" ${n.training.optimizer === 'sgd' ? 'selected' : ''}>SGD</option>
            </select>
          </label>
          <label class="field"><span>Learning rate</span><input id="t-lr" type="number" step="0.0001" value="${n.training.learningRate}"></label>
          <label class="field"><span>Batch size</span><input id="t-bs" type="number" value="${n.training.batchSize}"></label>
          <label class="field"><span>Epochs</span><input id="t-ep" type="number" value="${n.training.epochs}"></label>
          <label class="field"><span>Seed</span><input id="t-seed" type="number" value="${n.training.seed ?? 42}"></label>
          <label class="field"><span>Workers (parallelism)</span><input id="t-workers" type="number" min="1" value="${n.training.workers ?? 0}"></label>
        </div>
        <div class="hint" style="margin-top:6px;">
          <b>Workers</b> = number of CPU cores to use in parallel. <code>0</code> or <code>1</code> = single-threaded (legacy). Set to your core count for ${n.architecture.kind === 'charLM' ? 'large charLMs' : 'larger models'} — effective batch becomes <code>workers × batch size</code>, gradients are averaged across workers per step.
        </div>
      </div>

      <div class="section">
        <h3>Training data</h3>
        <p class="hint">${dataFormatHint(a)}</p>
        <div class="row" style="margin-bottom:8px;">
          <button class="btn sm" id="btn-import-data">Import from file</button>
          <div class="spacer"></div>
          <span class="inline-tag" id="data-stats"></span>
        </div>
        <textarea class="code-editor" id="data-json" style="min-height:220px;"></textarea>
      </div>

      <div class="section">
        <h3>Encryption</h3>
        <p class="hint">Protect weights at rest with AES-256-GCM. You will need the passphrase to train or run inference after enabling.</p>
        <div class="row">
          ${n.encrypted
            ? `<button class="btn" id="btn-decrypt">Decrypt & disable</button>`
            : (n.state ? `<button class="btn primary" id="btn-encrypt">Encrypt now</button>` : `<div class="hint">Train the network first, then enable encryption.</div>`)
          }
        </div>
      </div>

      <div class="row">
        <button class="btn primary" id="btn-save">Save changes</button>
        <button class="btn" id="btn-dup">Duplicate</button>
        <div class="spacer"></div>
        <button class="btn danger" id="btn-delete">Delete</button>
      </div>
    </div>
  `;

  // populate training data textarea
  const dataArea = $('#data-json');
  dataArea.value = JSON.stringify(n.trainingData || {}, null, 2);
  updateDataStats();
  dataArea.addEventListener('input', updateDataStats);

  $('#btn-import-data').addEventListener('click', async () => {
    const res = await window.nc.dialog.readTextFile({
      properties: ['openFile'],
      filters: [{ name: 'Text / JSON', extensions: ['json', 'txt', 'jsonl'] }]
    });
    if (!res) return;
    const ext = (res.path.split('.').pop() || '').toLowerCase();
    if (ext === 'json') {
      dataArea.value = res.content;
    } else if (a.kind === 'charLM') {
      dataArea.value = JSON.stringify({ text: res.content }, null, 2);
    } else {
      // try jsonl -> array
      try {
        const lines = res.content.split('\n').filter(Boolean).map(l => JSON.parse(l));
        dataArea.value = JSON.stringify({ samples: lines }, null, 2);
      } catch (e) {
        dataArea.value = res.content;
      }
    }
    updateDataStats();
  });

  $('#btn-save').addEventListener('click', saveEditor);
  $('#btn-dup').addEventListener('click', async () => {
    const dup = await window.nc.networks.duplicate(n.id);
    await refreshNetworks(); selectNetwork(dup.id);
  });
  $('#btn-delete').addEventListener('click', () => {
    confirmModal('Delete network?', `This will permanently remove "${n.name}" and its weights.`, async () => {
      await window.nc.networks.delete(n.id);
      state.selectedId = null; state.current = null;
      await refreshNetworks(); state.activeTab = 'networks'; renderActiveTab();
    });
  });

  const encBtn = $('#btn-encrypt');
  if (encBtn) encBtn.addEventListener('click', () => {
    passphraseModal('Set passphrase', 'Create a passphrase to encrypt this network', async (pass) => {
      if (!pass) return;
      await window.nc.networks.update(n.id, { encryptionIntent: 'enable', passphrase: pass });
      await loadCurrent(n.id); renderActiveTab(); toast('Network encrypted.');
    });
  });
  const decBtn = $('#btn-decrypt');
  if (decBtn) decBtn.addEventListener('click', () => {
    passphraseModal('Decrypt network', 'Enter the passphrase to disable encryption', async (pass) => {
      if (!pass) return;
      try {
        await window.nc.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
        await loadCurrent(n.id); renderActiveTab(); toast('Network decrypted.');
      } catch (e) { toast('Decrypt failed: ' + e.message); }
    });
  });

  // layer add/remove
  bindArchEditor(a);
}

function updateDataStats() {
  const area = $('#data-json');
  const tag = $('#data-stats');
  if (!area || !tag) return;
  try {
    const parsed = JSON.parse(area.value);
    if (Array.isArray(parsed.samples)) {
      let chatCount = 0;
      for (const s of parsed.samples) {
        if (s && typeof s === 'object' && (
            (typeof s.user === 'string' && typeof s.assistant === 'string') ||
            Array.isArray(s.messages) || Array.isArray(s.conversation)
        )) chatCount++;
      }
      tag.textContent = chatCount > 0
        ? `${parsed.samples.length} samples · ${chatCount} chat pairs`
        : `${parsed.samples.length} samples`;
    } else if (typeof parsed.text === 'string') {
      tag.textContent = `${parsed.text.length} chars`;
    } else tag.textContent = '—';
  } catch { tag.textContent = 'invalid JSON'; }
}

function dataFormatHint(arch) {
  if (arch.kind === 'classifier') return 'Format: <code>{"samples":[{"input":[…],"label":0}, …]}</code>. Input length must equal Input dim.';
  if (arch.kind === 'regressor')  return 'Format: <code>{"samples":[{"input":[…],"output":[…]}, …]}</code>.';
  if (arch.kind === 'charLM')     return 'Either <code>{"text":"…raw text…"}</code> for free text, or <code>{"samples":[{"user":"…","assistant":"…"}, …]}</code> for chat pairs. The app auto-wraps each turn with <code>&lt;|user|&gt;</code> / <code>&lt;|assistant|&gt;</code> tags so the model learns to respond.';
  return 'See Docs → Training data for the expected format.';
}

function archEditor(a) {
  if (a.kind === 'classifier' || a.kind === 'mlp' || a.kind === 'regressor') {
    return `
      <div class="grid-3">
        <label class="field"><span>Input dim</span><input id="a-in" type="number" value="${a.inputDim}"></label>
        <label class="field"><span>Output dim</span><input id="a-out" type="number" value="${a.outputDim}"></label>
        <label class="field"><span>Activation</span>
          <select id="a-act">
            ${['relu', 'leakyRelu', 'tanh', 'sigmoid', 'gelu'].map(k => `<option ${a.activation === k ? 'selected' : ''} value="${k}">${k}</option>`).join('')}
          </select>
        </label>
        <label class="field"><span>Dropout</span><input id="a-drop" type="number" step="0.05" min="0" max="0.9" value="${a.dropout || 0}"></label>
        <label class="field"><span>Hidden layers (comma-separated)</span><input id="a-hidden" type="text" value="${(a.hidden || []).join(',')}"></label>
        ${a.kind === 'classifier' ? `<label class="field"><span>Class labels (comma-separated)</span><input id="a-classes" type="text" value="${(a.classes || []).join(',')}"></label>` : ''}
      </div>
    `;
  }
  if (a.kind === 'charLM') {
    return `
      <div class="grid-3">
        <label class="field"><span>Vocab size</span><input id="a-vocab" type="number" readonly value="${a.vocabSize || 0}" title="Auto-inferred from training corpus"></label>
        <label class="field"><span>Embed dim</span><input id="a-embdim" type="number" value="${a.embDim}"></label>
        <label class="field"><span>Context length</span><input id="a-ctx" type="number" value="${a.contextLen}"></label>
        <label class="field"><span>Activation</span>
          <select id="a-act">${['relu', 'leakyRelu', 'tanh', 'sigmoid', 'gelu'].map(k => `<option ${a.activation === k ? 'selected' : ''} value="${k}">${k}</option>`).join('')}</select>
        </label>
        <label class="field"><span>Dropout</span><input id="a-drop" type="number" step="0.05" min="0" max="0.9" value="${a.dropout || 0}"></label>
        <label class="field"><span>Hidden layers</span><input id="a-hidden" type="text" value="${(a.hidden || []).join(',')}"></label>
      </div>
    `;
  }
  return '<p>Unknown kind.</p>';
}

function bindArchEditor() {/* no-op: reads happen at save time */}

async function saveEditor() {
  const n = state.current;
  if (!n) return;
  const patch = {
    name: $('#edit-name').value.trim() || 'Untitled',
    description: $('#edit-desc').value,
    training: {
      optimizer: $('#t-optimizer').value,
      learningRate: parseFloat($('#t-lr').value),
      batchSize: parseInt($('#t-bs').value),
      epochs: parseInt($('#t-ep').value),
      seed: parseInt($('#t-seed').value),
      workers: Math.max(0, parseInt($('#t-workers').value) || 0)
    }
  };
  const a = { ...n.architecture };
  if (a.kind === 'classifier' || a.kind === 'mlp' || a.kind === 'regressor') {
    a.inputDim = parseInt($('#a-in').value);
    a.outputDim = parseInt($('#a-out').value);
    a.activation = $('#a-act').value;
    a.dropout = parseFloat($('#a-drop').value) || 0;
    a.hidden = $('#a-hidden').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
    if (a.kind === 'classifier') a.classes = $('#a-classes').value.split(',').map(s => s.trim()).filter(Boolean);
  } else if (a.kind === 'charLM') {
    a.embDim = parseInt($('#a-embdim').value);
    a.contextLen = parseInt($('#a-ctx').value);
    a.activation = $('#a-act').value;
    a.dropout = parseFloat($('#a-drop').value) || 0;
    a.hidden = $('#a-hidden').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  }
  patch.architecture = a;
  let data;
  try { data = JSON.parse($('#data-json').value); }
  catch (e) { toast('Training data JSON invalid: ' + e.message); return; }
  patch.trainingData = data;
  // If arch shape changed, invalidate everything tied to the old shape:
  // weights, optimizer momentum/variance buffers, tokenizer, history.
  const shapeChanged = JSON.stringify(n.architecture) !== JSON.stringify(a);
  if (shapeChanged && n.state) {
    patch.state = null;
    patch.optimizerState = null;
    patch.tokenizer = null;
    patch.metrics = [];
  }
  try {
    await window.nc.networks.update(n.id, patch);
    await refreshNetworks();
    await loadCurrent(n.id);
    renderActiveTab();
    toast('Saved.');
  } catch (e) { toast('Save failed: ' + e.message); }
}

// Train tab
function renderTrainTab(root) {
  if (!state.current) {
    root.innerHTML = `<div class="empty"><div class="big">Select or create a network</div></div>`;
    return;
  }
  const n = state.current;
  const hasWeights = !!n.state || n.stateLocked;
  const primaryLabel = hasWeights ? 'Continue training' : 'Start training';
  const hint = hasWeights
    ? 'Continues from saved weights and optimizer state. Use "Train from scratch" to reset.'
    : 'Weights auto-save on completion.';
  root.innerHTML = `
    <div class="panel">
      <h2>Train — ${escapeHtml(n.name)}</h2>
      <div class="kpis">
        <div class="kpi"><div class="k">Status</div><div class="v" id="kpi-status">${state.training.running ? 'running' : 'idle'}</div></div>
        <div class="kpi"><div class="k">Epoch</div><div class="v" id="kpi-epoch">—</div></div>
        <div class="kpi"><div class="k">Loss</div><div class="v" id="kpi-loss">—</div></div>
        <div class="kpi"><div class="k">Elapsed</div><div class="v" id="kpi-elapsed">—</div></div>
      </div>

      <div class="section">
        <h3>Progress</h3>
        <div class="progress-wrap"><div class="progress-bar" id="progress-bar"></div></div>
        <svg class="chart-svg" id="loss-chart" viewBox="0 0 600 140" preserveAspectRatio="none"></svg>
      </div>

      <div class="row">
        <button class="btn primary" id="btn-train">${primaryLabel}</button>
        ${hasWeights ? '<button class="btn" id="btn-train-scratch">Train from scratch</button>' : ''}
        <button class="btn" id="btn-stop" disabled>Stop</button>
        <div class="spacer"></div>
        <div class="hint">${hint}</div>
      </div>

      <div class="section">
        <h3>Log</h3>
        <div class="log" id="log"></div>
      </div>
    </div>
  `;

  const launch = (fromScratch) => {
    if (n.stateLocked) {
      passphraseModal('Decrypt to train', 'Passphrase required to train an encrypted network', async (pass) => {
        if (!pass) return;
        await window.nc.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
        await loadCurrent(n.id);
        startTraining({ fromScratch });
      });
    } else startTraining({ fromScratch });
  };

  $('#btn-train').addEventListener('click', () => launch(false));
  const scratchBtn = $('#btn-train-scratch');
  if (scratchBtn) {
    scratchBtn.addEventListener('click', () => {
      confirmModal(
        'Train from scratch?',
        'This discards the saved weights and optimizer state for "' + n.name + '" and starts training over from random initialization.',
        () => launch(true)
      );
    });
  }
  $('#btn-stop').addEventListener('click', async () => {
    await window.nc.training.stop(n.id);
  });
  updateTrainLive();
}

async function startTraining(opts) {
  opts = opts || {};
  state.training.running = true;
  state.training.history = [];
  state.training.logs = [];
  state.training.lastProgress = null;
  state.training.startedAt = Date.now();
  updateTrainLive();
  try {
    await window.nc.training.start(state.current.id, { fromScratch: !!opts.fromScratch });
    const trainBtn = $('#btn-train'); if (trainBtn) trainBtn.disabled = true;
    const scratchBtn = $('#btn-train-scratch'); if (scratchBtn) scratchBtn.disabled = true;
    const stopBtn = $('#btn-stop'); if (stopBtn) stopBtn.disabled = false;
  } catch (e) {
    state.training.running = false;
    state.training.logs.push('ERROR: ' + e.message);
    updateTrainLive();
  }
}

function updateTrainLive() {
  if (state.activeTab !== 'train') return;
  const bar = $('#progress-bar');
  const epKpi = $('#kpi-epoch');
  const loKpi = $('#kpi-loss');
  const stKpi = $('#kpi-status');
  const elKpi = $('#kpi-elapsed');
  const p = state.training.lastProgress;
  if (stKpi) stKpi.textContent = state.training.running ? 'running' : 'idle';
  if (p && bar) {
    const pct = Math.min(100, (p.step / p.totalSteps) * 100);
    bar.style.width = pct.toFixed(1) + '%';
    epKpi.textContent = `${p.epoch}/${p.totalEpochs}`;
    loKpi.textContent = p.loss.toFixed(4);
    elKpi.textContent = humanMs(p.elapsedMs);
  }
  const log = $('#log');
  if (log) {
    log.textContent = state.training.logs.slice(-300).join('\n');
    log.scrollTop = log.scrollHeight;
  }
  drawLossChart();
  const btnTrain = $('#btn-train'); const btnStop = $('#btn-stop');
  if (btnTrain) btnTrain.disabled = state.training.running;
  if (btnStop) btnStop.disabled = !state.training.running;
}

function drawLossChart() {
  const svg = $('#loss-chart');
  if (!svg) return;
  const hist = state.training.history;
  if (!hist.length) { svg.innerHTML = ''; return; }
  const W = 600, H = 140, pad = 8;
  let min = Infinity, max = -Infinity;
  for (const h of hist) { if (h.loss < min) min = h.loss; if (h.loss > max) max = h.loss; }
  if (min === max) { max = min + 1; }
  const n = hist.length;
  const pts = hist.map((h, i) => {
    const x = pad + (i / Math.max(1, n - 1)) * (W - 2 * pad);
    const y = H - pad - ((h.loss - min) / (max - min)) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  svg.innerHTML = `
    <polyline fill="none" stroke="#ffffff" stroke-width="1.5" points="${pts}" />
    <text x="${W - pad}" y="${pad + 10}" text-anchor="end" font-size="10" fill="#8a8a8a">min ${min.toFixed(4)} · max ${max.toFixed(4)}</text>
  `;
}

// Inference tab
function renderInferenceTab(root) {
  if (!state.current) {
    root.innerHTML = `<div class="empty"><div class="big">No network selected</div></div>`;
    return;
  }
  const n = state.current;
  if (!n.state && !n.stateLocked) {
    root.innerHTML = `<div class="empty"><div class="big">Network is untrained</div><div>Train it on the Train tab first.</div></div>`;
    return;
  }
  const a = n.architecture;
  // Chat models get a real conversational UI with running history, not a
  // single-shot prompt box. Everything else still goes through the simple form.
  if (a.kind === 'charLM' && a.isChat) {
    renderChatTab(root, n);
    return;
  }
  root.innerHTML = `
    <div class="panel">
      <h2>Inference — ${escapeHtml(n.name)}</h2>
      <div class="section">
        <h3>Input</h3>
        ${inferenceInputUI(a)}
        <div class="row" style="margin-top:10px;">
          <button class="btn primary" id="btn-run">Run prediction</button>
          <div class="spacer"></div>
        </div>
      </div>
      <div class="section">
        <h3>Output</h3>
        <div id="infer-output" class="log" style="height:auto; min-height:60px;">—</div>
      </div>
    </div>
  `;
  $('#btn-run').addEventListener('click', runInference);
}

function renderChatTab(root, n) {
  const s = getChatSession(n.id);
  root.innerHTML = `
    <div class="panel chat-panel">
      <h2>Chat — ${escapeHtml(n.name)}</h2>
      <div class="row" style="gap:8px; align-items:flex-end;">
        <label class="field" style="flex:1;"><span>System prompt (optional, applied to every turn)</span>
          <input id="chat-system" type="text" value="${escapeHtml(s.system || '')}" placeholder="e.g. You are concise and friendly."></label>
        <button class="btn" id="chat-reset">Reset chat</button>
      </div>
      <div class="section">
        <h3>Conversation <span class="hint" id="chat-turn-count">${s.history.length} turn(s)</span></h3>
        <div class="chat-log" id="chat-log"></div>
      </div>
      <div class="section">
        <div class="row" style="gap:8px; align-items:flex-end;">
          <label class="field" style="flex:1;"><span>Your message</span>
            <textarea id="chat-input" rows="2" placeholder="Type a message and press Enter (Shift+Enter for newline)…"></textarea></label>
          <button class="btn primary" id="chat-send">Send</button>
        </div>
        <div class="grid-3" style="margin-top:10px;">
          <label class="field"><span>Max new tokens</span><input id="chat-max" type="number" value="200"></label>
          <label class="field"><span>Temperature</span><input id="chat-temp" type="number" step="0.1" value="0.8"></label>
          <label class="field"><span>Top-K (0 = off)</span><input id="chat-topk" type="number" value="0"></label>
        </div>
        <div class="hint" style="margin-top:6px;">Each turn includes the running history so the model can keep context. History is truncated from the oldest turn when it would overflow <code>contextLen=${n.architecture.contextLen}</code>.</div>
      </div>
    </div>
  `;
  redrawChatLog(n.id);

  $('#chat-system').addEventListener('input', (e) => { s.system = e.target.value; });
  $('#chat-reset').addEventListener('click', () => {
    confirmModal('Reset chat?', 'Clears the running conversation for this network. The model itself is unchanged.', () => {
      s.history = [];
      s.editingIndex = null;
      redrawChatLog(n.id);
    });
  });
  const send = () => sendChatMessage(n.id);
  $('#chat-send').addEventListener('click', send);
  $('#chat-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
}

function redrawChatLog(id) {
  const s = getChatSession(id);
  const log = $('#chat-log');
  if (!log) return;
  if (s.history.length === 0 && !s.busy) {
    log.innerHTML = `<div class="chat-empty">No messages yet — say hi.</div>`;
  } else {
    log.innerHTML = s.history.map((t, i) => {
      // Per-turn action row. Hidden while busy so the user can't queue up
      // overlapping regenerate/edit requests against an in-flight inference.
      // - assistant turns get [Regenerate]
      // - user turns get [Edit]
      // (We render the buttons even when editingIndex === i; the editor
      // markup below replaces the content area instead of the buttons.)
      const actions = (!s.busy && s.editingIndex !== i) ? `
        <div class="chat-actions">
          ${t.role === 'assistant' ? `<button class="chat-action-btn" data-act="regenerate" data-idx="${i}" title="Regenerate this reply">↻ Regenerate</button>` : ''}
          ${t.role === 'user' ? `<button class="chat-action-btn" data-act="edit" data-idx="${i}" title="Edit this message">✎ Edit</button>` : ''}
        </div>` : '';
      const isEditing = s.editingIndex === i && t.role === 'user';
      const contentMarkup = isEditing ? `
        <div class="chat-edit">
          <textarea class="chat-edit-input" data-idx="${i}" rows="${Math.min(8, Math.max(2, t.content.split('\n').length))}">${escapeHtml(t.content)}</textarea>
          <div class="chat-edit-row">
            <button class="btn primary" data-act="edit-save" data-idx="${i}">Save & regenerate</button>
            <button class="btn" data-act="edit-cancel" data-idx="${i}">Cancel</button>
            <span class="hint">Saving truncates later turns and regenerates the assistant reply.</span>
          </div>
        </div>
      ` : `<div class="chat-content">${escapeHtml(t.content)}</div>`;
      return `
        <div class="chat-msg chat-msg-${t.role}">
          <div class="chat-role">${t.role}</div>
          ${contentMarkup}
          ${actions}
        </div>
      `;
    }).join('') + (s.busy ? `<div class="chat-msg chat-msg-assistant chat-msg-pending"><div class="chat-role">assistant</div><div class="chat-content">…</div></div>` : '');
  }
  // Wire up the per-turn action buttons (event delegation would also work,
  // but the log is small and we redraw on every change anyway). Single
  // selector — every actionable button has [data-act], including the
  // chat-action icons and the inline editor's Save/Cancel.
  log.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', () => {
      const act = btn.dataset.act;
      const idx = parseInt(btn.dataset.idx, 10);
      if (act === 'regenerate') regenerateAssistantTurn(id, idx);
      else if (act === 'edit') beginEditUserTurn(id, idx);
      else if (act === 'edit-save') commitEditUserTurn(id, idx);
      else if (act === 'edit-cancel') cancelEditUserTurn(id);
    });
  });
  // Auto-focus the editor textarea when entering edit mode.
  const activeEditor = log.querySelector('.chat-edit-input');
  if (activeEditor) {
    activeEditor.focus();
    // Place cursor at end of text rather than highlighting the whole thing.
    const v = activeEditor.value; activeEditor.value = ''; activeEditor.value = v;
  }
  log.scrollTop = log.scrollHeight;
  const counter = $('#chat-turn-count');
  if (counter) counter.textContent = `${s.history.length} turn(s)`;
}

// Read the current generation knobs from the chat tab. Pulled into a helper
// so regenerate/edit-save use the *current* slider values, not the values
// captured when the original turn ran — that's the whole point of regenerate.
function readChatGenOpts() {
  return {
    maxTokens: parseInt($('#chat-max')?.value) || 200,
    temperature: parseFloat($('#chat-temp')?.value) || 0.8,
    topK: parseInt($('#chat-topk')?.value) || 0
  };
}

// Regenerate the assistant turn at `idx`. Uses the conversation up to but
// NOT including that turn as history, and the user message immediately
// before it as the new prompt. Replaces the existing assistant content
// in place; later turns (if any) are left untouched — the user might be
// just sampling a single alternative reply mid-conversation.
async function regenerateAssistantTurn(id, idx) {
  const n = state.current;
  if (!n || n.id !== id) return;
  const s = getChatSession(id);
  if (s.busy) return;
  const turn = s.history[idx];
  if (!turn || turn.role !== 'assistant') return;
  // Find the user prompt that produced this assistant turn — normally it's
  // the immediately preceding turn. If there is none (e.g. assistant-first
  // conversation), bail with a hint; nothing to regenerate from.
  let userIdx = -1;
  for (let k = idx - 1; k >= 0; k--) {
    if (s.history[k].role === 'user') { userIdx = k; break; }
  }
  if (userIdx === -1) { toast('No preceding user message to regenerate from.'); return; }
  const userPrompt = s.history[userIdx].content;
  const historyBefore = s.history.slice(0, userIdx);
  s.busy = true;
  // Show the pending bubble in place: temporarily blank the content so the
  // user sees it's being recomputed without losing their place in the log.
  const original = turn.content;
  turn.content = '…';
  redrawChatLog(id);
  try {
    const result = await window.nc.inference.run(id, {
      history: historyBefore,
      prompt: userPrompt,
      system: s.system || '',
      ...readChatGenOpts()
    });
    const reply = (result && typeof result.text === 'string') ? result.text : '';
    s.history[idx].content = reply;
  } catch (e) {
    s.history[idx].content = original; // restore on failure
    toast('Regenerate failed: ' + e.message);
  } finally {
    s.busy = false;
    redrawChatLog(id);
  }
}

function beginEditUserTurn(id, idx) {
  const s = getChatSession(id);
  if (s.busy) return;
  if (!s.history[idx] || s.history[idx].role !== 'user') return;
  s.editingIndex = idx;
  redrawChatLog(id);
}

function cancelEditUserTurn(id) {
  const s = getChatSession(id);
  s.editingIndex = null;
  redrawChatLog(id);
}

// Commit an edit to a user turn. The standard chat UX here (which the user
// will expect) is: replace the user's message, drop everything that came
// AFTER it, and regenerate the next assistant reply. Keeping later turns
// would leave them referencing a question that no longer exists, which
// is more confusing than helpful.
async function commitEditUserTurn(id, idx) {
  const n = state.current;
  if (!n || n.id !== id) return;
  const s = getChatSession(id);
  if (s.busy) return;
  const editor = document.querySelector(`.chat-edit-input[data-idx="${idx}"]`);
  if (!editor) return;
  const newText = (editor.value || '').trim();
  if (!newText) { toast('Message cannot be empty.'); return; }
  // Truncate to and including the edited turn, with the new content.
  const historyBefore = s.history.slice(0, idx);
  s.history = [...historyBefore, { role: 'user', content: newText }];
  s.editingIndex = null;
  s.busy = true;
  redrawChatLog(id);
  try {
    const result = await window.nc.inference.run(id, {
      history: historyBefore,
      prompt: newText,
      system: s.system || '',
      ...readChatGenOpts()
    });
    const reply = (result && typeof result.text === 'string') ? result.text : '';
    s.history.push({ role: 'assistant', content: reply });
  } catch (e) {
    toast('Edit-regenerate failed: ' + e.message);
  } finally {
    s.busy = false;
    redrawChatLog(id);
  }
}

async function sendChatMessage(id) {
  const n = state.current;
  if (n.id !== id) return;
  if (n.stateLocked) {
    passphraseModal('Decrypt to chat', 'Passphrase required to chat with an encrypted network', async (pass) => {
      if (!pass) return;
      await window.nc.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
      await loadCurrent(n.id); sendChatMessage(id);
    });
    return;
  }
  const s = getChatSession(id);
  if (s.busy) return;
  const inp = $('#chat-input');
  const message = (inp?.value || '').trim();
  if (!message) return;
  inp.value = '';
  s.busy = true;
  // Optimistically render the user turn so the UI feels responsive.
  s.history.push({ role: 'user', content: message });
  redrawChatLog(id);
  try {
    const payload = {
      history: s.history.slice(0, -1), // history *before* this turn; prompt carries the new one
      prompt: message,
      system: s.system || '',
      maxTokens: parseInt($('#chat-max').value) || 200,
      temperature: parseFloat($('#chat-temp').value) || 0.8,
      topK: parseInt($('#chat-topk').value) || 0
    };
    const result = await window.nc.inference.run(id, payload);
    const reply = (result && typeof result.text === 'string') ? result.text : '';
    s.history.push({ role: 'assistant', content: reply });
  } catch (e) {
    // Surface the failure inline; pop the optimistic user turn so retry is clean.
    s.history.pop();
    toast('Chat error: ' + e.message);
  } finally {
    s.busy = false;
    redrawChatLog(id);
  }
}

function inferenceInputUI(a) {
  if (a.kind === 'classifier' || a.kind === 'mlp') {
    return `<label class="field"><span>Input vector (comma-separated, length ${a.inputDim})</span>
      <input id="inp-vec" type="text" value="${new Array(a.inputDim).fill(0).join(',')}"></label>`;
  }
  if (a.kind === 'regressor') {
    return `<label class="field"><span>Input vector (comma-separated, length ${a.inputDim})</span>
      <input id="inp-vec" type="text" value="${new Array(a.inputDim).fill(0).join(',')}"></label>`;
  }
  if (a.kind === 'charLM') {
    const isChat = !!a.isChat;
    return `
      ${isChat ? `<label class="field"><span>System (optional)</span><input id="inp-system" type="text" value=""></label>` : ''}
      <label class="field"><span>${isChat ? 'Your message' : 'Prompt'}</span><textarea id="inp-prompt" rows="3">${isChat ? 'Hello, I need help' : 'the '}</textarea></label>
      <div class="grid-3">
        <label class="field"><span>Max new tokens</span><input id="inp-max" type="number" value="${isChat ? 200 : 120}"></label>
        <label class="field"><span>Temperature</span><input id="inp-temp" type="number" step="0.1" value="1.0"></label>
        <label class="field"><span>Top-K (0 = off)</span><input id="inp-topk" type="number" value="0"></label>
      </div>
    `;
  }
  return '<p>Unknown input type.</p>';
}

async function runInference() {
  const n = state.current;
  if (n.stateLocked) {
    passphraseModal('Decrypt to run', 'Passphrase required to run this network', async (pass) => {
      if (!pass) return;
      await window.nc.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
      await loadCurrent(n.id); runInference();
    });
    return;
  }
  const a = n.architecture;
  let payload;
  try {
    if (a.kind === 'charLM') {
      payload = {
        prompt: $('#inp-prompt').value,
        maxTokens: parseInt($('#inp-max').value) || 80,
        temperature: parseFloat($('#inp-temp').value) || 1.0,
        topK: parseInt($('#inp-topk').value) || 0
      };
      const sys = document.getElementById('inp-system');
      if (sys) payload.system = sys.value;
    } else {
      const vec = $('#inp-vec').value.split(',').map(v => parseFloat(v.trim()));
      if (vec.some(isNaN)) throw new Error('Vector has non-numeric value');
      payload = { input: vec };
    }
    const result = await window.nc.inference.run(n.id, payload);
    renderInferenceResult(result);
  } catch (e) { $('#infer-output').textContent = 'ERROR: ' + e.message; }
}

function renderInferenceResult(r) {
  const out = $('#infer-output');
  if (!out) return;
  if (r.kind === 'classification') {
    out.innerHTML = `
      <div><b>Predicted:</b> ${escapeHtml(r.label)} (class ${r.predictedClass})</div>
      <div style="margin-top:6px;">
        ${r.probs.map((p, i) => `class ${i}: ${(p * 100).toFixed(2)}%`).join('<br/>')}
      </div>
    `;
  } else if (r.kind === 'regression') {
    out.textContent = 'Output: [' + r.output.map(x => x.toFixed(4)).join(', ') + ']';
  } else if (r.kind === 'generation') {
    out.textContent = r.text;
  } else {
    out.textContent = JSON.stringify(r, null, 2);
  }
}

// API tab
async function renderApiPanel() {
  const root = $('#content');
  const allActive = await window.nc.api.list();
  state.apiServers = new Map(allActive.map(s => [s.id, s]));
  if (!state.current) {
    root.innerHTML = `<div class="empty"><div class="big">No network selected</div></div>`;
    return;
  }
  const n = state.current;
  const info = await window.nc.system.info();
  const running = state.apiServers.get(n.id);
  root.innerHTML = `
    <div class="panel">
      <h2>API — ${escapeHtml(n.name)}</h2>
      <p class="hint">Expose this model on your local network. Other devices can call it at the URL below.</p>
      <div class="section">
        <h3>Status</h3>
        <div class="kpis">
          <div class="kpi"><div class="k">State</div><div class="v">${running ? 'running' : 'stopped'}</div></div>
          <div class="kpi"><div class="k">Port</div><div class="v">${running ? running.port : '—'}</div></div>
          <div class="kpi"><div class="k">URL</div><div class="v" style="font-size:13px; font-family: var(--mono);">${running ? running.url : '—'}</div></div>
          <div class="kpi"><div class="k">Host IP</div><div class="v" style="font-size:13px; font-family: var(--mono);">${info.hostIp}</div></div>
        </div>
        <div class="row" style="margin-top:12px;">
          <label class="field" style="margin:0;"><span>Port (0 = auto)</span><input id="api-port" type="number" value="${running ? running.port : 0}" style="width:120px;"></label>
          <div class="spacer"></div>
          ${running
            ? `<button class="btn danger" id="btn-stop-api">Stop server</button>`
            : `<button class="btn primary" id="btn-start-api">Start server</button>`
          }
        </div>
      </div>
      <div class="section">
        <h3>Endpoints</h3>
        <div class="kv-table">
          <div class="k">GET</div><div>/info — network metadata</div>
          <div class="k">POST</div><div>/predict — body is the same shape as the Inference tab</div>
        </div>
      </div>
      <div class="section">
        <h3>Log</h3>
        <div class="log" id="api-log"></div>
      </div>
    </div>
  `;

  const startBtn = $('#btn-start-api');
  if (startBtn) startBtn.addEventListener('click', async () => {
    if (n.stateLocked) { toast('Decrypt the network first.'); return; }
    const port = parseInt($('#api-port').value) || 0;
    try {
      const r = await window.nc.api.start(n.id, port);
      toast('API running at ' + r.url);
      renderApiPanel();
    } catch (e) { toast('Failed: ' + e.message); }
  });
  const stopBtn = $('#btn-stop-api');
  if (stopBtn) stopBtn.addEventListener('click', async () => {
    await window.nc.api.stop(n.id);
    renderApiPanel();
  });
  const logEl = $('#api-log');
  if (logEl) { logEl.textContent = state.apiLogs.slice(-200).join('\n'); logEl.scrollTop = logEl.scrollHeight; }
}

// Script tab
function renderScriptTab(root) {
  const code = state.current?.script || `# NeuralScript — create a tiny XOR classifier inline.
let spec = {
  kind: "classifier",
  inputDim: 2, outputDim: 2,
  hidden: [8], activation: "relu",
  classes: ["false", "true"]
}
let data = { samples: [
  { input: [0,0], label: 0 },
  { input: [0,1], label: 1 },
  { input: [1,0], label: 1 },
  { input: [1,1], label: 0 }
] }
let opts = { optimizer: "adam", learningRate: 0.05, batchSize: 4, epochs: 150, seed: 42 }
print "training..."
let result = await(train(spec, data, opts))
print "final loss:"
print result.metrics[len(result.metrics) - 1].loss
`;
  root.innerHTML = `
    <div class="panel">
      <h2>NeuralScript</h2>
      <p class="hint">Write small programs that build and train models. See Docs → NeuralScript for the full reference.</p>
      <textarea class="code-editor" id="script-area" spellcheck="false">${escapeHtml(code)}</textarea>
      <div class="row">
        <button class="btn primary" id="btn-run-script">Run</button>
        <button class="btn" id="btn-save-script">Save to network</button>
        <div class="spacer"></div>
      </div>
      <div class="section">
        <h3>Output</h3>
        <div class="log" id="script-out"></div>
      </div>
    </div>
  `;
  $('#btn-run-script').addEventListener('click', async () => {
    const out = $('#script-out');
    out.textContent = 'running...';
    try {
      const r = await window.nc.script.run(state.current?.id || null, $('#script-area').value);
      out.textContent = (r.output || '') + (r.ok ? '' : '\n[error] ' + r.error);
    } catch (e) { out.textContent = 'ERROR: ' + e.message; }
  });
  $('#btn-save-script').addEventListener('click', async () => {
    if (!state.current) { toast('No network selected.'); return; }
    await window.nc.networks.update(state.current.id, { script: $('#script-area').value });
    await loadCurrent(state.current.id);
    toast('Saved.');
  });
}

// Docs tab
function renderDocsTab(root) {
  const doc = window.NC_DOCS.find(d => d.id === state.currentDocId) || window.NC_DOCS[0];
  root.innerHTML = `
    <div class="panel">
      <div class="docs-layout">
        <div class="docs-nav">
          ${window.NC_DOCS.map(d => `<button data-id="${d.id}" class="${d.id === doc.id ? 'active' : ''}">${d.title}</button>`).join('')}
        </div>
        <div class="docs-body" style="user-select: text;">${doc.body}</div>
      </div>
    </div>
  `;
  root.querySelectorAll('.docs-nav button').forEach(b => b.addEventListener('click', () => {
    state.currentDocId = b.dataset.id;
    renderDocsTab(root);
  }));
}

// ---------- new network modal ----------

let selectedTemplate = null;
function openNewNetworkModal() {
  $('#modal-new').hidden = false;
  $('#new-name').value = '';
  selectedTemplate = null;
  populateTemplates();
  $('#new-kind').addEventListener('change', populateTemplates);
}

function populateTemplates() {
  const kind = $('#new-kind').value;
  const matches = window.NC_TEMPLATES.filter(t => {
    if (kind === 'chat') return t.kind === 'charLM' && t.arch && t.arch.isChat;
    if (kind === 'charLM') return t.kind === 'charLM' && !(t.arch && t.arch.isChat);
    return t.kind === kind;
  });
  const grid = $('#template-grid');
  grid.innerHTML = matches.map(t => `
    <div class="template-card" data-id="${t.id}">
      <div class="t-name">${escapeHtml(t.name)}</div>
      <div class="t-desc">${escapeHtml(t.desc)}</div>
    </div>
  `).join('') + `
    <div class="template-card" data-id="__blank__">
      <div class="t-name">Blank ${kind}</div>
      <div class="t-desc">Start with a minimal default architecture and edit manually.</div>
    </div>
  `;
  grid.querySelectorAll('.template-card').forEach(c => c.addEventListener('click', () => {
    grid.querySelectorAll('.template-card').forEach(x => x.classList.remove('active'));
    c.classList.add('active');
    selectedTemplate = c.dataset.id;
  }));
}

async function createNetworkFromModal() {
  const name = $('#new-name').value.trim() || 'Untitled Network';
  const kind = $('#new-kind').value;
  let payload;
  if (selectedTemplate && selectedTemplate !== '__blank__') {
    const t = window.NC_TEMPLATES.find(x => x.id === selectedTemplate);
    payload = {
      name, description: t.desc,
      architecture: JSON.parse(JSON.stringify(t.arch)),
      training: JSON.parse(JSON.stringify(t.training)),
      trainingData: JSON.parse(JSON.stringify(t.trainingData))
    };
  } else {
    payload = {
      name,
      architecture: defaultArch(kind),
      training: { optimizer: 'adam', learningRate: 0.01, batchSize: 16, epochs: 20, seed: 42 },
      trainingData: kind === 'chat'
        ? { samples: [{ user: 'Hello', assistant: 'Hi! How can I help?' }] }
        : (kind === 'charLM' ? { text: '' } : { samples: [] })
    };
  }
  const net = await window.nc.networks.create(payload);
  $('#modal-new').hidden = true;
  await refreshNetworks();
  selectNetwork(net.id);
}

function defaultArch(kind) {
  if (kind === 'chat') return { kind: 'charLM', vocabSize: 0, embDim: 32, contextLen: 128, hidden: [96, 96], activation: 'gelu', dropout: 0.1, isChat: true };
  if (kind === 'charLM') return { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 16, hidden: [64], activation: 'gelu', dropout: 0 };
  if (kind === 'regressor') return { kind: 'regressor', inputDim: 1, outputDim: 1, hidden: [16], activation: 'tanh', dropout: 0 };
  return { kind: 'classifier', inputDim: 2, outputDim: 2, hidden: [8], activation: 'relu', dropout: 0, classes: ['A', 'B'] };
}

// ---------- modals ----------

function confirmModal(title, body, onYes) {
  $('#confirm-title').textContent = title;
  $('#confirm-body').textContent = body;
  const modal = $('#modal-confirm');
  modal.hidden = false;
  const yes = $('#confirm-yes');
  const clone = yes.cloneNode(true);
  yes.replaceWith(clone);
  clone.addEventListener('click', async () => { modal.hidden = true; try { await onYes(); } catch (e) { toast(e.message); } });
}

function passphraseModal(title, prompt, onSubmit) {
  $('#pass-title').textContent = title;
  $('#pass-prompt').textContent = prompt;
  const input = $('#pass-input'); input.value = '';
  const modal = $('#modal-pass');
  modal.hidden = false; input.focus();
  const ok = $('#pass-ok');
  const clone = ok.cloneNode(true);
  ok.replaceWith(clone);
  clone.addEventListener('click', async () => {
    modal.hidden = true;
    try { await onSubmit(input.value); } catch (e) { toast(e.message); }
  });
}

// ---------- misc ----------

async function fillSystemInfo() {
  try {
    const info = await window.nc.system.info();
    $('#sys-info').innerHTML = `
      <div>v${info.version} · ${info.platform} ${info.arch}</div>
      <div>${info.cpus} CPUs · ${Math.round(info.mem / 1024 / 1024 / 1024)} GB RAM</div>
      <div>Host: ${info.hostIp}</div>
    `;
    $('#status-right').textContent = `Host: ${info.hostIp}  ·  v${info.version}`;
  } catch (e) { /* ignore */ }
}

function toast(msg) {
  const el = $('#status-left');
  el.textContent = msg;
  setTimeout(() => { el.textContent = 'Ready'; }, 3000);
}

function humanMs(ms) {
  const s = Math.floor(ms / 1000);
  if (s < 60) return s + 's';
  const m = Math.floor(s / 60);
  return `${m}m ${s % 60}s`;
}

function escapeHtml(s) {
  return (s + '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]);
}
