'use strict';

// NeuralCabin renderer. Single-file UI controller that reads from window.nb.

const state = {
  networks: [],
  selectedId: null,
  current: null,
  activeTab: 'networks',
  training: { running: false, history: [], logs: [] },
  apiServers: new Map(), // id -> {port, url}
  apiLogs: [],
  currentDocId: 'welcome',
  chatSessions: new Map(), // id -> { history, system, busy, editingIndex }
  gptEditor: { docs: [], modality: 'corpus', pairs: [], editingDocIdx: null },
  enginePreset: 'js'      // 'js' | 'wasm' | 'webgpu'
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

// ---------- plugin kind labels ----------
const PLUGIN_KIND_LABELS = {
  'self-driving-car': 'Self-Driving Car',
  'snake-neuro':      'Snake (Neuroevolution)',
  'warehouse-robot':  'Warehouse Robot',
};
function pluginKindLabel(k) { return PLUGIN_KIND_LABELS[k] || k; }

// ---------- plugin registry ----------

const pluginRegistry = {
  templates: [],
  inferenceRenderers: {}, // pluginKind → { fn(root, net, nb), pluginId }
  trainRenderers:     {}, // pluginKind → { fn(root, net, nb), pluginId }  — takes over the Train tab
  trainEditors:       {}, // pluginKind → { fn(root, net, nb), pluginId }  — Edit tab data section
  trainSettings:      {}, // pluginKind → field-label/visibility overrides for the training defaults section
  archFields:         {}, // pluginKind → { fields[], computeDims(arch) → {inputDim,outputDim} }
};

async function initPlugins() {
  let plugins;
  try { plugins = await window.nb.plugins.list(); }
  catch (e) { console.warn('Plugin system unavailable:', e.message); return; }
  for (const p of plugins) {
    if (!p.rendererCode) continue;
    const api = {
      registerTemplate:         t       => pluginRegistry.templates.push(t),
      registerInferenceRenderer:(kind, fn)  => { pluginRegistry.inferenceRenderers[kind] = { fn, pluginId: p.id }; },
      registerTrainRenderer:    (kind, fn)  => { pluginRegistry.trainRenderers[kind]     = { fn, pluginId: p.id }; },
      registerTrainEditor:      (kind, fn)  => { pluginRegistry.trainEditors[kind]       = { fn, pluginId: p.id }; },
      // cfg: { lr, bs, epochs, seed, workers, optimizer } each { label?, hint?, hidden? }
      // plus optional cfg.sectionHint to replace the bottom Workers hint text
      registerTrainSettings:    (kind, cfg)  => { pluginRegistry.trainSettings[kind] = cfg; },
      registerArchFields:       (kind, spec) => { pluginRegistry.archFields[kind] = spec; },
      invoke:                   (ch, ...a)   => window.nb.plugins.invoke(p.id, ch, ...a)
    };
    try {
      // eslint-disable-next-line no-new-func
      new Function('api', p.rendererCode)(api);
    } catch (e) { console.error(`Plugin "${p.id}" renderer failed:`, e); }
  }
  for (const t of pluginRegistry.templates) {
    if (!window.NB_TEMPLATES.find(x => x.id === t.id)) window.NB_TEMPLATES.push(t);
  }
}

// ---------- boot ----------

window.addEventListener('DOMContentLoaded', async () => {
  bindChrome();
  bindModals();
  bindTrainingStreams();
  bindApiStreams();
  await refreshNetworks();
  await fillSystemInfo();
  await initPlugins();
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
      const net = await window.nb.networks.importNet();
      if (net) { await refreshNetworks(); selectNetwork(net.id); toast('Imported.'); }
    } catch (e) { toast('Import failed: ' + e.message); }
  });
  $('#btn-export').addEventListener('click', async () => {
    if (!state.selectedId) return toast('Select a network first.');
    try {
      const p = await window.nb.networks.exportNet(state.selectedId);
      if (p) toast('Saved to ' + p);
    } catch (e) { toast('Export failed: ' + e.message); }
  });
}

function bindModals() {
  $$('[data-close-modal]').forEach(b => b.addEventListener('click', () => {
    const el = $('#' + b.dataset.closeModal);
    if (el) el.hidden = true;
  }));
  $('#btn-create').addEventListener('click', createNetworkFromModal);
  $('#pass-cancel').addEventListener('click', () => $('#modal-pass').hidden = true);
  $('#confirm-no').addEventListener('click', () => $('#modal-confirm').hidden = true);
}

function bindTrainingStreams() {
  window.nb.training.onProgress(p => {
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
  window.nb.training.onDone(async (p) => {
    state.training.running = false;
    state.training.logs.push(p.stopped ? '-- stopped --' : '-- done --');
    await refreshNetworks();
    if (p.id === state.selectedId) { await loadCurrent(state.selectedId); renderActiveTab(); }
  });
  window.nb.training.onError(e => {
    state.training.running = false;
    state.training.logs.push('ERROR: ' + e.message);
    if (state.activeTab === 'train') updateTrainLive();
  });
}

function bindApiStreams() {
  window.nb.api.onLog(line => {
    state.apiLogs.push(`[${new Date().toLocaleTimeString()}] ${line.id?.slice?.(0, 6) || ''} ${line.line}`);
    if (state.apiLogs.length > 500) state.apiLogs.shift();
    if (state.activeTab === 'api') renderApiPanel();
  });
}

// ---------- networks list ----------

async function refreshNetworks() {
  state.networks = await window.nb.networks.list();
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
    const kindLabel = n.pluginKind ? pluginKindLabel(n.pluginKind) : (n.kind || '—');
    return `
      <div class="net-item ${active}" data-id="${n.id}">
        <div class="n-name">${escapeHtml(n.name)} ${encBadge}</div>
        <div class="n-meta">${kindLabel} ${trainBadge}</div>
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
  try { state.current = await window.nb.networks.get(id); }
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
    case 'plugins': return renderPluginsTab(content);
  }
}

// Apply plugin-defined label/visibility overrides to the Training defaults fields.
// cfg keys: lr | bs | epochs | seed | workers | optimizer — each { label?, hint?, hidden? }
// cfg.sectionHint replaces the bottom Workers hint paragraph.
function applyPluginTrainSettings(cfg) {
  const fieldMap = { optimizer: 't-optimizer', lr: 't-lr', bs: 't-bs', epochs: 't-ep', seed: 't-seed', workers: 't-workers' };
  for (const [key, settings] of Object.entries(cfg)) {
    if (key === 'sectionHint') continue;
    const inputId = fieldMap[key];
    if (!inputId) continue;
    const inputEl = document.getElementById(inputId);
    if (!inputEl) continue;
    const fieldEl = inputEl.closest('.field');
    if (!fieldEl) continue;
    if (settings.hidden) {
      fieldEl.style.display = 'none';
    } else {
      if (settings.label) {
        const spanEl = fieldEl.querySelector('span');
        if (spanEl) spanEl.textContent = settings.label;
      }
      if (settings.hint) inputEl.title = settings.hint;
    }
  }
  if (cfg.sectionHint !== undefined) {
    const hintEl = document.getElementById('t-training-hint');
    if (hintEl) hintEl.textContent = cfg.sectionHint;
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
  // Init GPT editor state BEFORE the template runs so gptEditorHtml() reads
  // the correct modality/docs/pairs from the saved network.
  if (a.kind === 'gpt' && !a.pluginKind) {
    const td = n.trainingData || {};
    const hasSamples = Array.isArray(td.samples) && td.samples.length > 0;
    const inferredModality = td.modality
      || (hasSamples && !Array.isArray(td.documents) ? 'sft' : 'corpus');
    state.gptEditor = {
      docs: Array.isArray(td.documents) ? td.documents.map(d => ({ ...d })) : [],
      modality: inferredModality,
      pairs: hasSamples
        ? td.samples.map(s => ({ user: s.user || '', assistant: s.assistant || s.output || '' }))
        : [],
      editingDocIdx: null
    };
  }

  root.innerHTML = `
    <div class="panel">
      <h2>Editor — ${escapeHtml(n.name)}</h2>
      <p class="hint">${a.pluginKind ? pluginKindLabel(a.pluginKind) : (a.kind || '—')} · ${n.encrypted ? 'encrypted at rest' : 'stored as plaintext'} · updated ${new Date(n.updatedAt).toLocaleString()}</p>

      <div class="section">
        <h3>Identity</h3>
        <div class="grid-2">
          <label class="field"><span>Name</span><input id="edit-name" type="text" value="${escapeHtml(n.name)}"></label>
          <label class="field"><span>Type</span><input readonly value="${a.pluginKind ? pluginKindLabel(a.pluginKind) : a.kind}"></label>
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
              <option value="adam"      ${n.training.optimizer === 'adam'      ? 'selected' : ''}>Adam</option>
              <option value="adamw"     ${n.training.optimizer === 'adamw'     ? 'selected' : ''}>AdamW</option>
              <option value="adamw8bit" ${n.training.optimizer === 'adamw8bit' ? 'selected' : ''}>AdamW 8-bit</option>
              <option value="radam"     ${n.training.optimizer === 'radam'     ? 'selected' : ''}>RAdam</option>
              <option value="ranger"    ${n.training.optimizer === 'ranger'    ? 'selected' : ''}>Ranger (RAdam+Lookahead)</option>
              <option value="lion"      ${n.training.optimizer === 'lion'      ? 'selected' : ''}>Lion</option>
              <option value="adafactor" ${n.training.optimizer === 'adafactor' ? 'selected' : ''}>Adafactor</option>
              <option value="lamb"      ${n.training.optimizer === 'lamb'      ? 'selected' : ''}>LAMB</option>
              <option value="lars"      ${n.training.optimizer === 'lars'      ? 'selected' : ''}>LARS</option>
              <option value="sgd"       ${n.training.optimizer === 'sgd'       ? 'selected' : ''}>SGD</option>
            </select>
          </label>
          <label class="field"><span>Learning rate</span><input id="t-lr" type="number" step="0.0001" value="${n.training.learningRate}"></label>
          <label class="field"><span>Batch size</span><input id="t-bs" type="number" value="${n.training.batchSize}"></label>
          <label class="field"><span>Epochs</span><input id="t-ep" type="number" value="${n.training.epochs}"></label>
          <label class="field"><span>Seed</span><input id="t-seed" type="number" value="${n.training.seed ?? 42}"></label>
          <label class="field"><span>Workers (parallelism)</span><input id="t-workers" type="number" min="1" value="${n.training.workers ?? 0}"></label>
        </div>
        <div id="t-training-hint" class="hint" style="margin-top:6px;">
          <b>Workers</b> = number of CPU cores to use in parallel. <code>0</code> or <code>1</code> = single-threaded (legacy). Set to your core count for ${(n.architecture.kind === 'charLM' || n.architecture.kind === 'gpt') ? 'large models' : 'larger models'} — effective batch becomes <code>workers × batch size</code>, gradients are averaged across workers per step.
        </div>
      </div>

      <div class="section">
        <h3>Training data</h3>
        ${a.pluginKind && pluginRegistry.trainEditors[a.pluginKind]
          ? `<div id="plugin-train-editor"></div>`
          : a.kind === 'gpt' ? gptEditorHtml() : `
            <p class="hint">${dataFormatHint(a)}</p>
            <div class="row" style="margin-bottom:8px;">
              <button class="btn sm" id="btn-import-data">Import from file</button>
              <div class="spacer"></div>
              <span class="inline-tag" id="data-stats"></span>
            </div>
            <textarea class="code-editor" id="data-json" style="min-height:220px;"></textarea>
          `}
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
        <button class="btn" id="btn-reset-weights">Reset</button>
        <div class="spacer"></div>
        <button class="btn" id="btn-backup" style="background:#3a3a3a;color:#b5b5b5;">Backup</button>
        <button class="btn danger" id="btn-delete">Delete</button>
      </div>
    </div>
  `;

  // Apply plugin-specific labels/visibility to the training defaults fields.
  if (a.pluginKind && pluginRegistry.trainSettings[a.pluginKind]) {
    applyPluginTrainSettings(pluginRegistry.trainSettings[a.pluginKind]);
  }

  if (a.pluginKind && pluginRegistry.trainEditors[a.pluginKind]) {
    const { fn, pluginId } = pluginRegistry.trainEditors[a.pluginKind];
    const nb = { invoke: (ch, ...args) => window.nb.plugins.invoke(pluginId, ch, ...args) };
    fn(document.getElementById('plugin-train-editor'), n, nb);
  } else if (a.kind === 'gpt') {
    bindGptEditor();
  } else {
    // populate training data textarea
    const dataArea = $('#data-json');
    dataArea.value = JSON.stringify(n.trainingData || {}, null, 2);
    updateDataStats();
    refreshHints();
    dataArea.addEventListener('input', () => { updateDataStats(); refreshHints(); });

    $('#btn-import-data').addEventListener('click', async () => {
      const res = await window.nb.dialog.readTextFile({
        properties: ['openFile'],
        filters: [{ name: 'Text / JSON', extensions: ['json', 'txt', 'jsonl', 'md', 'docs'] }]
      });
      if (!res) return;
      const ext = (res.path.split('.').pop() || '').toLowerCase();
      if (ext === 'json') {
        dataArea.value = res.content;
      } else if (a.kind === 'charLM') {
        dataArea.value = JSON.stringify({ text: res.content }, null, 2);
      } else {
        try {
          const lines = res.content.split('\n').filter(Boolean).map(l => JSON.parse(l));
          dataArea.value = JSON.stringify({ samples: lines }, null, 2);
        } catch (e) {
          dataArea.value = res.content;
        }
      }
      updateDataStats();
    });
  }

  $('#btn-save').addEventListener('click', saveEditor);
  $('#btn-dup').addEventListener('click', async () => {
    const dup = await window.nb.networks.duplicate(n.id);
    await refreshNetworks(); selectNetwork(dup.id);
  });
  $('#btn-reset-weights').addEventListener('click', () => {
    confirmModal(
      'Reset weights?',
      `This will permanently erase all trained weights, optimizer state, and metrics for "${n.name}". Architecture and settings are kept. The next training run will start from scratch.`,
      async () => {
        await window.nb.networks.update(n.id, { state: null, optimizerState: null, tokenizer: null, metrics: [] });
        // For plugin networks, also wipe the in-memory session so the Train tab
        // doesn't resume from a stale in-memory population on next mount.
        const pkind = n.architecture?.pluginKind;
        const pluginInfo = pkind && pluginRegistry.trainRenderers[pkind];
        if (pluginInfo) {
          try { await window.nb.plugins.invoke(pluginInfo.pluginId, `${pkind}:clearSession`, { instanceId: n.id }); } catch (_) {}
        }
        await loadCurrent(n.id); renderActiveTab(); toast('Weights reset.');
      }
    );
  });
  $('#btn-backup').addEventListener('click', () => openBackupModal(n.id));

  $('#btn-delete').addEventListener('click', () => {
    confirmModal('Delete network?', `This will permanently remove "${n.name}" and its weights.`, async () => {
      await window.nb.networks.delete(n.id);
      state.selectedId = null; state.current = null;
      await refreshNetworks(); state.activeTab = 'networks'; renderActiveTab();
    });
  });

  const encBtn = $('#btn-encrypt');
  if (encBtn) encBtn.addEventListener('click', () => {
    passphraseModal('Set passphrase', 'Create a passphrase to encrypt this network', async (pass) => {
      if (!pass) return;
      await window.nb.networks.update(n.id, { encryptionIntent: 'enable', passphrase: pass });
      await loadCurrent(n.id); renderActiveTab(); toast('Network encrypted.');
    });
  });
  const decBtn = $('#btn-decrypt');
  if (decBtn) decBtn.addEventListener('click', () => {
    passphraseModal('Decrypt network', 'Enter the passphrase to disable encryption', async (pass) => {
      if (!pass) return;
      try {
        await window.nb.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
        await loadCurrent(n.id); renderActiveTab(); toast('Network decrypted.');
      } catch (e) { toast('Decrypt failed: ' + e.message); }
    });
  });

  // layer add/remove
  bindArchEditor(a);
  // deferred so DOM is fully painted before we query fields
  setTimeout(refreshHints, 0);
}

function updateDataStats() {
  const tag = $('#data-stats');
  if (!tag) return;
  if (state.current?.architecture?.kind === 'gpt') {
    const ed = state.gptEditor;
    if (ed.modality === 'sft') {
      const complete = ed.pairs.filter(p => p.user && p.assistant).length;
      tag.textContent = `${ed.pairs.length} pair${ed.pairs.length !== 1 ? 's' : ''} · ${complete} complete`;
    } else {
      const chars = ed.docs.reduce((s, d) => s + (d.content?.length || 0), 0);
      tag.textContent = `${ed.docs.length} doc${ed.docs.length !== 1 ? 's' : ''} · ${chars.toLocaleString()} chars`;
    }
    return;
  }
  const area = $('#data-json');
  if (!area) return;
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

function getDataInfo() {
  const area = $('#data-json');
  if (!area) return { chars: 0, samples: 0, isChat: false };
  try {
    const parsed = JSON.parse(area.value);
    if (typeof parsed.text === 'string') return { chars: parsed.text.length, samples: 0, isChat: false };
    if (Array.isArray(parsed.samples)) {
      const chatCount = parsed.samples.filter(s => s && (typeof s.user === 'string' || Array.isArray(s.messages) || Array.isArray(s.conversation))).length;
      return { chars: area.value.length, samples: parsed.samples.length, isChat: chatCount > 0 };
    }
  } catch {}
  return { chars: 0, samples: 0, isChat: false };
}

function archRecs(a) {
  const di = getDataInfo();
  const vocab = a.vocabSize || 0;
  const recs = {};
  if (a.kind !== 'charLM') return recs;

  // embDim
  if (vocab > 0) {
    const suggested = vocab <= 60 ? 16 : vocab <= 150 ? 32 : vocab <= 300 ? 64 : 128;
    recs['a-embdim'] = `suggested: ${suggested} for vocab ${vocab}`;
  } else if (di.chars > 0) {
    recs['a-embdim'] = di.chars < 5000 ? 'suggested: 16–32 (small corpus)' : di.chars < 50000 ? 'suggested: 32–64' : 'suggested: 64–128';
  }

  // contextLen
  if (di.isChat && di.samples > 0) {
    const suggested = di.samples < 50 ? 64 : di.samples < 200 ? 128 : 256;
    recs['a-ctx'] = `suggested: ${suggested} for ${di.samples} chat pairs`;
  } else if (di.chars > 0) {
    const suggested = di.chars < 2000 ? 32 : di.chars < 20000 ? 64 : di.chars < 100000 ? 128 : 256;
    recs['a-ctx'] = `suggested: ${suggested} for ${di.chars.toLocaleString()} chars`;
  }

  // hidden
  if (vocab > 0 || di.chars > 0) {
    const scale = vocab > 200 || di.chars > 50000 ? 'large' : di.chars > 10000 ? 'medium' : 'small';
    recs['a-hidden'] = scale === 'large' ? 'suggested: 256,256 or 512,256' : scale === 'medium' ? 'suggested: 128,128' : 'suggested: 64,64';
  }

  // dropout
  if (di.samples > 0 || di.chars > 0) {
    const small = (di.isChat ? di.samples : di.chars) < (di.isChat ? 100 : 10000);
    recs['a-drop'] = small ? 'suggested: 0.1–0.2 (small data)' : 'suggested: 0 or 0.1';
  }

  return recs;
}

function trainingRecs() {
  const di = getDataInfo();
  const n = state.current;
  if (!n) return {};
  const a = n.architecture;
  const recs = {};
  const size = di.isChat ? di.samples : di.chars;
  const isSmall = size < (di.isChat ? 100 : 10000);
  const isMed = !isSmall && size < (di.isChat ? 500 : 100000);

  recs['t-lr'] = isSmall ? 'suggested: 0.001–0.003' : isMed ? 'suggested: 0.0005–0.001' : 'suggested: 0.0001–0.0005';
  recs['t-bs'] = isSmall ? 'suggested: 4–8' : isMed ? 'suggested: 16–32' : 'suggested: 32–64';
  recs['t-ep'] = isSmall ? 'suggested: 50–200' : isMed ? 'suggested: 20–50' : 'suggested: 5–20';

  return recs;
}

function applyHints(recs) {
  for (const [id, text] of Object.entries(recs)) {
    const el = $('#' + id);
    if (!el) continue;
    const label = el.closest('label.field');
    if (!label) continue;
    let hint = label.querySelector('.rec-hint');
    if (!hint) {
      hint = document.createElement('span');
      hint.className = 'rec-hint';
      hint.style.cssText = 'display:block;font-size:10px;color:#666;margin-bottom:2px;font-style:italic;';
      label.querySelector('span').after(hint);
    }
    hint.textContent = text;
  }
}

function refreshHints() {
  if (!state.current) return;
  applyHints(archRecs(state.current.architecture));
  applyHints(trainingRecs());
}

function formatBytes(n) {
  if (n < 1024) return n + ' B';
  if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
  return (n / 1024 / 1024).toFixed(1) + ' MB';
}

function gptEditorHtml() {
  const ed = state.gptEditor;
  const isCorpus = ed.modality !== 'sft';
  return `
    <div class="modality-grid" id="gpt-modality-grid">
      <div class="modality-card${isCorpus ? ' active' : ''}" data-mode="corpus">
        <div class="mc-header">
          <span class="mc-title">Corpus Training</span>
        </div>
        <div class="mc-desc">Upload documents and train directly on the raw text. Good for building general knowledge and writing style.</div>
      </div>
      <div class="modality-card${!isCorpus ? ' active' : ''}" data-mode="sft">
        <div class="mc-header">
          <span class="mc-title">Instruction Fine-Tuning</span>
          <span class="badge">SFT</span>
        </div>
        <div class="mc-desc">Train on structured User → AI pairs. Teaches proper conversation boundaries and reliable stop-token generation.</div>
      </div>
    </div>
    <div id="gpt-mode-content"></div>
    <div class="row" style="margin-top:8px;">
      <div class="spacer"></div>
      <span class="inline-tag" id="data-stats"></span>
    </div>
  `;
}

function renderGptDocList() {
  const list = $('#gpt-doc-list');
  if (!list) return;
  const ed = state.gptEditor;
  if (!ed.docs.length) {
    list.innerHTML = '<div class="hint" style="padding:4px 0;">No documents added yet.</div>';
    bindGptDocEvents();
    return;
  }
  list.innerHTML = ed.docs.map((d, i) => `
    <div style="display:flex;align-items:center;gap:6px;padding:5px 8px;background:#1a1a1a;border-radius:4px;margin-bottom:3px;">
      <span style="flex:1;font-size:13px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" title="${escapeHtml(d.name)}">${escapeHtml(d.name)}</span>
      <span class="inline-tag">${formatBytes(d.size)}${d.truncated ? ' · truncated' : ''}</span>
      <button class="btn sm gpt-doc-edit" data-idx="${i}">Edit</button>
      <button class="btn sm danger gpt-doc-remove" data-idx="${i}">✕</button>
    </div>
    ${ed.editingDocIdx === i ? `
      <div style="margin-bottom:6px;padding:0 2px;">
        <textarea class="code-editor gpt-doc-content" data-idx="${i}" style="min-height:120px;margin-bottom:4px;">${escapeHtml(d.content || '')}</textarea>
        <div style="display:flex;gap:6px;">
          <button class="btn sm gpt-doc-save-edit" data-idx="${i}">Save</button>
          <button class="btn sm gpt-doc-cancel-edit" data-idx="${i}">Cancel</button>
        </div>
      </div>
    ` : ''}
  `).join('');
  bindGptDocEvents();
}

function bindGptDocEvents() {
  const list = $('#gpt-doc-list');
  if (!list) return;
  list.querySelectorAll('.gpt-doc-remove').forEach(b => b.addEventListener('click', () => {
    state.gptEditor.docs.splice(parseInt(b.dataset.idx), 1);
    state.gptEditor.editingDocIdx = null;
    renderGptDocList(); updateDataStats();
  }));
  list.querySelectorAll('.gpt-doc-edit').forEach(b => b.addEventListener('click', () => {
    const idx = parseInt(b.dataset.idx);
    state.gptEditor.editingDocIdx = state.gptEditor.editingDocIdx === idx ? null : idx;
    renderGptDocList();
  }));
  list.querySelectorAll('.gpt-doc-save-edit').forEach(b => b.addEventListener('click', () => {
    const idx = parseInt(b.dataset.idx);
    const content = list.querySelector(`.gpt-doc-content[data-idx="${idx}"]`).value;
    state.gptEditor.docs[idx].content = content;
    state.gptEditor.docs[idx].size = content.length;
    state.gptEditor.docs[idx].truncated = false;
    state.gptEditor.editingDocIdx = null;
    renderGptDocList(); updateDataStats();
  }));
  list.querySelectorAll('.gpt-doc-cancel-edit').forEach(b => b.addEventListener('click', () => {
    state.gptEditor.editingDocIdx = null;
    renderGptDocList();
  }));
}

// Renders either the corpus doc list or the SFT pair editor into #gpt-mode-content.
function renderGptModeContent() {
  const root = $('#gpt-mode-content');
  if (!root) return;
  const ed = state.gptEditor;

  if (ed.modality !== 'sft') {
    root.innerHTML = `
      <p class="hint" style="margin:10px 0 8px;">Upload documents (md, txt, etc.) as the training corpus. The model learns from the raw text of each file.</p>
      <div id="gpt-doc-list" style="margin-bottom:8px;"></div>
      <button class="btn sm" id="btn-add-docs">+ Add documents</button>
    `;
    renderGptDocList();
    $('#btn-add-docs').addEventListener('click', async () => {
      const docs = await window.nb.gpt.pickDocuments();
      if (!docs.length) return;
      for (const d of docs) {
        const idx = ed.docs.findIndex(x => x.name === d.name);
        if (idx >= 0) ed.docs[idx] = d; else ed.docs.push(d);
      }
      renderGptDocList(); updateDataStats();
    });
  } else {
    root.innerHTML = `
      <p class="hint" style="margin:10px 0 8px;">Add User → AI pairs. Each pair is wrapped in role tags during training so the model learns to respond and generate proper stop tokens.</p>
      <div id="gpt-pair-list" style="margin-bottom:8px;"></div>
      <div class="row">
        <button class="btn sm" id="btn-add-pair">+ Add pair</button>
        <button class="btn sm ghost" id="btn-import-pairs">Import JSONL</button>
      </div>
    `;
    renderGptPairList();
    $('#btn-add-pair').addEventListener('click', () => {
      ed.pairs.push({ user: '', assistant: '' });
      renderGptPairList(); updateDataStats();
    });
    $('#btn-import-pairs').addEventListener('click', async () => {
      const res = await window.nb.dialog.readTextFile({
        filters: [{ name: 'JSONL / JSON', extensions: ['jsonl', 'json'] }]
      });
      if (!res) return;
      try {
        const lines = res.content.trim().split('\n').filter(Boolean).map(l => JSON.parse(l));
        const valid = lines.filter(l => typeof l.user === 'string' || typeof l.input === 'string');
        if (!valid.length) throw new Error('No valid pairs found. Expected { "user": "...", "assistant": "..." } per line.');
        for (const l of valid) {
          ed.pairs.push({ user: l.user || l.input || '', assistant: l.assistant || l.output || '' });
        }
        renderGptPairList(); updateDataStats();
        toast(`Imported ${valid.length} pair${valid.length !== 1 ? 's' : ''}.`);
      } catch (e) { toast('Import failed: ' + e.message); }
    });
  }
}

function renderGptPairList() {
  const list = $('#gpt-pair-list');
  if (!list) return;
  const ed = state.gptEditor;
  if (!ed.pairs.length) {
    list.innerHTML = '<div class="hint" style="padding:4px 0;">No pairs yet. Click "+ Add pair" or import a JSONL file.</div>';
    return;
  }
  list.innerHTML = ed.pairs.map((p, i) => `
    <div class="pair-row" data-idx="${i}">
      <div class="pair-col">
        <div class="pair-label">User</div>
        <textarea class="gpt-pair-user" data-idx="${i}" rows="3" placeholder="User message…">${escapeHtml(p.user || '')}</textarea>
      </div>
      <div class="pair-col">
        <div class="pair-label">Assistant</div>
        <textarea class="gpt-pair-assistant" data-idx="${i}" rows="3" placeholder="AI response…">${escapeHtml(p.assistant || '')}</textarea>
      </div>
      <button class="btn sm danger gpt-pair-remove" data-idx="${i}" style="align-self:flex-start;margin-top:18px;">✕</button>
    </div>
  `).join('');
  list.querySelectorAll('.gpt-pair-user').forEach(t => t.addEventListener('input', () => {
    ed.pairs[parseInt(t.dataset.idx)].user = t.value; updateDataStats();
  }));
  list.querySelectorAll('.gpt-pair-assistant').forEach(t => t.addEventListener('input', () => {
    ed.pairs[parseInt(t.dataset.idx)].assistant = t.value; updateDataStats();
  }));
  list.querySelectorAll('.gpt-pair-remove').forEach(b => b.addEventListener('click', () => {
    ed.pairs.splice(parseInt(b.dataset.idx), 1);
    renderGptPairList(); updateDataStats();
  }));
}

function bindGptEditor() {
  // Bind modality card clicks — switch between Corpus and SFT modes.
  $('#gpt-modality-grid')?.querySelectorAll('.modality-card').forEach(card => {
    card.addEventListener('click', () => {
      state.gptEditor.modality = card.dataset.mode;
      $('#gpt-modality-grid').querySelectorAll('.modality-card')
        .forEach(c => c.classList.toggle('active', c === card));
      renderGptModeContent();
      updateDataStats();
    });
  });
  renderGptModeContent();
  updateDataStats();
}

function dataFormatHint(arch) {
  if (arch.kind === 'classifier') return 'Format: <code>{"samples":[{"input":[…],"label":0}, …]}</code>. Input length must equal Input dim.';
  if (arch.kind === 'regressor')  return 'Format: <code>{"samples":[{"input":[…],"output":[…]}, …]}</code>.';
  if (arch.kind === 'charLM')     return 'Either <code>{"text":"…raw text…"}</code> for free text, or <code>{"samples":[{"user":"…","assistant":"…"}, …]}</code> for chat pairs. The app auto-wraps each turn with <code>&lt;|user|&gt;</code> / <code>&lt;|assistant|&gt;</code> tags so the model learns to respond.';
  if (arch.kind === 'gpt')        return '';
  return 'See Docs → Training data for the expected format.';
}

function renderPluginArchEditor(fields, a) {
  const rows = fields.map(f => {
    const val  = a[f.id] !== undefined ? a[f.id] : f.default;
    const hint = f.hint ? ` title="${escapeHtml(f.hint)}"` : '';
    if (f.type === 'number') {
      const min  = f.min  != null ? ` min="${f.min}"`   : '';
      const max  = f.max  != null ? ` max="${f.max}"`   : '';
      const step = f.step != null ? ` step="${f.step}"` : '';
      return `<label class="field"><span>${escapeHtml(f.label)}</span><input id="paf-${f.id}" type="number" value="${val}"${min}${max}${step}${hint}></label>`;
    }
    if (f.type === 'boolean') {
      return `<label class="field" style="flex-direction:row;align-items:center;gap:8px;"><input id="paf-${f.id}" type="checkbox"${val ? ' checked' : ''} style="width:16px;height:16px;flex-shrink:0;"${hint}><span>${escapeHtml(f.label)}</span></label>`;
    }
    if (f.type === 'activation') {
      const opts = ['relu', 'leakyRelu', 'tanh', 'sigmoid', 'gelu'].map(k => `<option${val === k ? ' selected' : ''} value="${k}">${k}</option>`).join('');
      return `<label class="field"><span>${escapeHtml(f.label)}</span><select id="paf-${f.id}"${hint}>${opts}</select></label>`;
    }
    if (f.type === 'layers') {
      return `<label class="field"><span>${escapeHtml(f.label)}</span><input id="paf-${f.id}" type="text" value="${Array.isArray(val) ? val.join(',') : val}"${hint}></label>`;
    }
    return '';
  }).join('');
  return `<div class="grid-3">${rows}</div>`;
}

function archEditor(a) {
  if (a.pluginKind && pluginRegistry.archFields[a.pluginKind]) {
    return renderPluginArchEditor(pluginRegistry.archFields[a.pluginKind].fields, a);
  }
  if (a.pluginKind) {
    return `
      <div class="grid-3">
        <label class="field"><span>Input dim</span><input readonly value="${a.inputDim}" title="Fixed by the plugin simulation"></label>
        <label class="field"><span>Output dim</span><input readonly value="${a.outputDim}" title="Fixed by the plugin simulation"></label>
        <label class="field"><span>Activation</span>
          <select id="a-act">
            ${['relu', 'leakyRelu', 'tanh', 'sigmoid', 'gelu'].map(k => `<option ${a.activation === k ? 'selected' : ''} value="${k}">${k}</option>`).join('')}
          </select>
        </label>
        <label class="field"><span>Hidden layers (comma-separated)</span><input id="a-hidden" type="text" value="${(a.hidden || []).join(',')}"></label>
      </div>
      <p style="font-size:11px;color:#555;margin:6px 0 0;">Input and output dimensions are fixed by the ${pluginKindLabel(a.pluginKind)} simulation and cannot be changed here.</p>
    `;
  }
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
  if (a.kind === 'charLM' || a.kind === 'gpt') {
    const tok = a.tokenizerKind || 'char';
    return `
      <div class="grid-3">
        <label class="field"><span>Tokenizer</span>
          <select id="a-tok">
            <option value="wordpart" ${tok === 'wordpart' ? 'selected' : ''}>Word-part (BPE subword)</option>
            <option value="char"     ${tok === 'char'     ? 'selected' : ''}>Character (char-level)</option>
            <option value="word"     ${tok === 'word'     ? 'selected' : ''}>Word (word-level)</option>
          </select>
        </label>
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
  if (a.pluginKind) {
    const spec = pluginRegistry.archFields[a.pluginKind];
    if (spec) {
      for (const f of spec.fields) {
        const el = document.getElementById(`paf-${f.id}`);
        if (!el) continue;
        if (f.type === 'number') {
          let v = parseFloat(el.value);
          if (isNaN(v)) v = f.default ?? 0;
          if (f.min != null) v = Math.max(f.min, v);
          if (f.max != null) v = Math.min(f.max, v);
          a[f.id] = v;
        } else if (f.type === 'boolean') {
          a[f.id] = el.checked;
        } else if (f.type === 'activation') {
          a[f.id] = el.value;
        } else if (f.type === 'layers') {
          const parsed = el.value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
          a[f.id] = parsed.length ? parsed : (Array.isArray(f.default) ? f.default : []);
        }
      }
      if (spec.computeDims) {
        const { inputDim, outputDim } = spec.computeDims(a);
        if (inputDim  != null) a.inputDim  = inputDim;
        if (outputDim != null) a.outputDim = outputDim;
      }
    }
  } else if (a.kind === 'classifier' || a.kind === 'mlp' || a.kind === 'regressor') {
    a.inputDim = parseInt($('#a-in').value);
    a.outputDim = parseInt($('#a-out').value);
    a.activation = $('#a-act').value;
    a.dropout = parseFloat($('#a-drop').value) || 0;
    a.hidden = $('#a-hidden').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
    if (a.kind === 'classifier') a.classes = $('#a-classes').value.split(',').map(s => s.trim()).filter(Boolean);
  } else if (a.kind === 'charLM' || a.kind === 'gpt') {
    a.tokenizerKind = $('#a-tok').value;
    a.embDim = parseInt($('#a-embdim').value);
    a.contextLen = parseInt($('#a-ctx').value);
    a.activation = $('#a-act').value;
    a.dropout = parseFloat($('#a-drop').value) || 0;
    a.hidden = $('#a-hidden').value.split(',').map(s => parseInt(s.trim())).filter(n => !isNaN(n) && n > 0);
  }
  patch.architecture = a;
  if (a.pluginKind && pluginRegistry.trainEditors[a.pluginKind]) {
    // Plugin manages its own trainingData — don't overwrite on save
  } else if (a.kind === 'gpt') {
    const ed = state.gptEditor;
    patch.trainingData = { modality: ed.modality, documents: ed.docs, samples: ed.pairs };
  } else {
    let data;
    try { data = JSON.parse($('#data-json').value); }
    catch (e) { toast('Training data JSON invalid: ' + e.message); return; }
    patch.trainingData = data;
  }
  // If arch shape changed, invalidate everything tied to the old shape:
  // weights, optimizer momentum/variance buffers, tokenizer, history.
  const shapeChanged = JSON.stringify(n.architecture) !== JSON.stringify(a);
  if (shapeChanged && n.state) {
    patch.state = null;
    patch.optimizerState = null;
    patch.tokenizer = null;
    patch.metrics = [];
  }
  // Always clear the stored tokenizer when the tokenizer kind changes so that
  // the next training run rebuilds the vocab from scratch. This runs regardless
  // of whether the network has been trained (n.state may be null).
  if ((a.kind === 'charLM' || a.kind === 'gpt') && a.tokenizerKind !== n.architecture.tokenizerKind) {
    patch.tokenizer = null;
  }
  try {
    await window.nb.networks.update(n.id, patch);
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
  // Plugin-managed training: hand the full Train tab to the plugin's registerTrainRenderer.
  if (n.architecture?.pluginKind && pluginRegistry.trainRenderers[n.architecture.pluginKind]) {
    const { fn, pluginId } = pluginRegistry.trainRenderers[n.architecture.pluginKind];
    const nb = { invoke: (ch, ...args) => window.nb.plugins.invoke(pluginId, ch, ...args) };
    root.innerHTML = '';
    fn(root, n, nb);
    return;
  }
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

      <div class="section">
        <h3>Engine</h3>
        <p class="hint" style="margin-bottom:10px;">Select the compute backend for this training run. JS is always available; WASM and WebGPU require native module support.</p>
        <div class="engine-grid" id="engine-grid">
          <div class="engine-card${state.enginePreset === 'js' ? ' active' : ''}" data-engine="js">
            <div class="ec-title">JS</div>
            <div class="ec-desc">Pure JavaScript. Runs everywhere, no setup required.</div>
          </div>
          <div class="engine-card${state.enginePreset === 'wasm' ? ' active' : ''}" data-engine="wasm">
            <div class="ec-title">WASM</div>
            <div class="ec-desc">WebAssembly / Rust native module. Faster matrix ops on supported platforms.</div>
          </div>
          <div class="engine-card${state.enginePreset === 'webgpu' ? ' active' : ''}" data-engine="webgpu">
            <div class="ec-title">WebGPU</div>
            <div class="ec-desc">GPU-accelerated compute. Requires a WebGPU-capable renderer process.</div>
          </div>
        </div>
      </div>
    </div>
  `;

  const launch = (fromScratch) => {
    if (n.stateLocked) {
      passphraseModal('Decrypt to train', 'Passphrase required to train an encrypted network', async (pass) => {
        if (!pass) return;
        await window.nb.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
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
    await window.nb.training.stop(n.id);
  });
  $('#engine-grid')?.querySelectorAll('.engine-card').forEach(card => {
    card.addEventListener('click', () => {
      state.enginePreset = card.dataset.engine;
      $('#engine-grid').querySelectorAll('.engine-card')
        .forEach(c => c.classList.toggle('active', c === card));
    });
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
    await window.nb.training.start(state.current.id, { fromScratch: !!opts.fromScratch, engine: state.enginePreset });
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
  const a = n.architecture;
  // Plugin inference renderers always get called — they handle the untrained
  // state themselves (e.g. showing the board but disabling AI buttons).
  const hookKind = a.pluginKind || null;
  if (hookKind && pluginRegistry.inferenceRenderers[hookKind]) {
    const { fn, pluginId } = pluginRegistry.inferenceRenderers[hookKind];
    const nb = { invoke: (ch, ...args) => window.nb.plugins.invoke(pluginId, ch, ...args) };
    root.innerHTML = '';
    fn(root, n, nb);
    return;
  }
  if (!n.state && !n.stateLocked) {
    root.innerHTML = `<div class="empty"><div class="big">Network is untrained</div><div>Train it on the Train tab first.</div></div>`;
    return;
  }
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
          <button class="btn" id="btn-stop-gen" style="display:none;">Stop generating</button>
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
  $('#btn-stop-gen').addEventListener('click', () => window.nb.inference.cancel());
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
    const result = await window.nb.inference.run(id, {
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
    const result = await window.nb.inference.run(id, {
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
      await window.nb.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
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
    const result = await window.nb.inference.run(id, payload);
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
  if (a.kind === 'charLM' || a.kind === 'gpt') {
    const isChat = !!a.isChat;
    return `
      ${isChat ? `<label class="field"><span>System (optional)</span><input id="inp-system" type="text" value=""></label>` : ''}
      <label class="field"><span>${isChat ? 'Your message' : 'Sentence stem / Prompt'}</span><textarea id="inp-prompt" rows="3">${isChat ? 'Hello, I need help' : 'The future of AI is'}</textarea></label>
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
      await window.nb.networks.update(n.id, { encryptionIntent: 'disable', passphrase: pass });
      await loadCurrent(n.id); runInference();
    });
    return;
  }
  const a = n.architecture;
  let payload;
  try {
    if (a.kind === 'charLM' || a.kind === 'gpt') {
      payload = {
        prompt: $('#inp-prompt').value,
        maxTokens: parseInt($('#inp-max').value) || 80,
        temperature: parseFloat($('#inp-temp').value) || 1.0,
        topK: parseInt($('#inp-topk').value) || 0
      };
      const sys = document.getElementById('inp-system');
      if (sys) payload.system = sys.value;

      // Streaming: show tokens as they arrive.
      const out = $('#infer-output');
      const runBtn = $('#btn-run');
      const stopBtn = $('#btn-stop-gen');
      out.textContent = '';
      runBtn.disabled = true;
      stopBtn.style.display = '';
      let streamed = '';
      try {
        const result = await window.nb.inference.streamStart(n.id, payload, (chunk) => {
          streamed += chunk;
          out.textContent = streamed;
        });
        // Replace with final trimmed text (strips any partial end-tags that
        // were emitted just before stop detection).
        if (result && result.kind === 'generation') out.textContent = result.text;
        else if (result) renderInferenceResult(result);
      } catch (e) {
        if (!streamed) out.textContent = 'ERROR: ' + e.message;
      } finally {
        runBtn.disabled = false;
        stopBtn.style.display = 'none';
      }
    } else {
      const vec = $('#inp-vec').value.split(',').map(v => parseFloat(v.trim()));
      if (vec.some(isNaN)) throw new Error('Vector has non-numeric value');
      payload = { input: vec };
      const result = await window.nb.inference.run(n.id, payload);
      renderInferenceResult(result);
    }
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
// ── Plugin API documentation blocks ──────────────────────────────────────────
function pluginApiDocs(arch, url) {
  const base = url || 'http://HOST:PORT';
  const pre  = (s) => `<pre style="background:#0d0d0d;border:1px solid #2a2a2a;border-radius:4px;padding:10px 12px;font-size:11px;line-height:1.6;overflow-x:auto;white-space:pre-wrap;word-break:break-all;">${escapeHtml(s)}</pre>`;
  const dim  = (s) => `<span style="color:#555;">${s}</span>`;

  if (arch.pluginKind === 'self-driving-car') {
    const ex = JSON.stringify({ inputs: [1,1,1,0.6,1,0.4,1,1,1, 0.35, 0.08] });
    return `
      <div class="section">
        <h3>Self-Driving Car — API Reference</h3>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">
          Send 11 raw sensor floats, receive steering and throttle commands from the best evolved brain.
          The model must be saved (pause training) before the server can start.
        </p>

        <h4 style="font-size:12px;color:#aaa;margin:14px 0 6px;">POST /predict</h4>
        <div class="kv-table" style="margin-bottom:8px;">
          <div class="k" style="color:#64b5f6;">inputs[0–8]</div><div>9 ray-cast distances (0.0 = wall ahead, 1.0 = clear), evenly spaced −80° → +80° relative to heading</div>
          <div class="k" style="color:#64b5f6;">inputs[9]</div><div>Normalised speed  ${dim('(current_speed / 340)')}</div>
          <div class="k" style="color:#64b5f6;">inputs[10]</div><div>Normalised heading angle  ${dim('(angle / 2π)')}</div>
        </div>
        <b style="font-size:11px;color:#666;">Request</b>
        ${pre(`curl -s -X POST ${base}/predict \\
  -H "Content-Type: application/json" \\
  -d '${ex}'`)}
        <b style="font-size:11px;color:#666;">Response</b>
        ${pre(`{ "steer": 0.14, "throttle": 0.82, "outputs": [0.14, 0.82] }`)}
        <div class="kv-table" style="margin-top:6px;">
          <div class="k" style="color:#a5d6a7;">steer</div><div>−1.0 (full left) → 1.0 (full right), tanh-clamped</div>
          <div class="k" style="color:#a5d6a7;">throttle</div><div>−1.0 (full brake) → 1.0 (full throttle), tanh-clamped</div>
        </div>
      </div>`;
  }

  if (arch.pluginKind === 'snake-neuro') {
    const grid = new Array(255).fill(0); grid[127] = 0.5; grid[142] = 0.25; grid[157] = -0.5; grid[60] = 1.0;
    const ex = JSON.stringify({ inputs: grid });
    return `
      <div class="section">
        <h3>Snake (Neuroevolution) — API Reference</h3>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">
          Send the 15×17 grid occupancy map, receive the next move from the best evolved snake brain.
          The model must be saved (pause training) before the server can start.
        </p>

        <h4 style="font-size:12px;color:#aaa;margin:14px 0 6px;">POST /predict</h4>
        <div class="kv-table" style="margin-bottom:8px;">
          <div class="k" style="color:#64b5f6;">inputs</div><div>255 floats — 15×17 grid, row-major (y × GRID_W + x)</div>
          <div class="k" style="color:#64b5f6;">0.0</div><div>Empty cell</div>
          <div class="k" style="color:#64b5f6;">1.0</div><div>Apple</div>
          <div class="k" style="color:#64b5f6;">0.5</div><div>Snake head</div>
          <div class="k" style="color:#64b5f6;">0.25</div><div>Snake body</div>
          <div class="k" style="color:#64b5f6;">−0.5</div><div>Snake tail</div>
        </div>
        <b style="font-size:11px;color:#666;">Request</b>
        ${pre(`curl -s -X POST ${base}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": [0,0,...,0.5,...,0.25,...,-0.5,...,1.0,...,0]}'
# inputs is a flat array of 255 values`)}
        <b style="font-size:11px;color:#666;">Response</b>
        ${pre(`{ "direction": 1, "directionName": "right", "outputs": [-0.3, 1.1, 0.2, -0.8] }`)}
        <div class="kv-table" style="margin-top:6px;">
          <div class="k" style="color:#a5d6a7;">direction</div><div>0 = up · 1 = right · 2 = down · 3 = left</div>
          <div class="k" style="color:#a5d6a7;">directionName</div><div>Human-readable direction string</div>
          <div class="k" style="color:#a5d6a7;">outputs</div><div>Raw logits [4] — argmax gives the chosen direction</div>
        </div>
      </div>`;
  }

  if (arch.pluginKind === 'warehouse-robot') {
    const nBoxes = Math.round((arch.inputDim - 5) / 6) || 1;
    const dim_   = 5 + nBoxes * 6;
    const ex = JSON.stringify({ inputs: new Array(dim_).fill(0).map(() => +(Math.random().toFixed(3))) });
    return `
      <div class="section">
        <h3>Warehouse Robot (DQN) — API Reference</h3>
        <p style="font-size:12px;color:#888;margin:0 0 12px;">
          Send the robot's encoded state, receive the next action from the trained DQN policy.
          This model was trained with <b style="color:#ccc;">${nBoxes} box${nBoxes > 1 ? 'es' : ''}</b>
          so inputs must be <b style="color:#ccc;">${dim_} floats</b>.
          The model must be saved (pause training) before the server can start.
        </p>

        <h4 style="font-size:12px;color:#aaa;margin:14px 0 6px;">POST /predict — Input layout (${dim_} floats)</h4>
        <div class="kv-table" style="margin-bottom:8px;">
          <div class="k" style="color:#64b5f6;">[0–1]</div><div>Robot position [row, col] normalised to 0–1 on an 8×8 grid</div>
          <div class="k" style="color:#64b5f6;">[2]</div><div>Carrying flag: 1.0 if holding a box, 0.0 otherwise</div>
          <div class="k" style="color:#64b5f6;">[3–${2+nBoxes*2}]</div><div>Box positions [row, col] × ${nBoxes} (normalised)</div>
          <div class="k" style="color:#64b5f6;">[${3+nBoxes*2}–${2+nBoxes*4}]</div><div>Target positions [row, col] × ${nBoxes} (normalised)</div>
          <div class="k" style="color:#64b5f6;">[${3+nBoxes*4}–${2+nBoxes*6}]</div><div>Box→target offset [Δrow, Δcol] × ${nBoxes}</div>
          <div class="k" style="color:#64b5f6;">[${3+nBoxes*6}–${4+nBoxes*6}]</div><div>Robot→nearest goal offset [Δrow, Δcol]</div>
        </div>
        <b style="font-size:11px;color:#666;">Request</b>
        ${pre(`curl -s -X POST ${base}/predict \\
  -H "Content-Type: application/json" \\
  -d '{"inputs": [0.125, 0.25, 0, 0.5, 0.5, 0.875, 0.875, 0.375, 0.375, 0.75, 0.75]}'
# Example for 1-box config (11 inputs total)`)}
        <b style="font-size:11px;color:#666;">Response</b>
        ${pre(`{ "action": 0, "actionName": "up", "outputs": [2.1, 0.4, -0.3, 0.8] }`)}
        <div class="kv-table" style="margin-top:6px;">
          <div class="k" style="color:#a5d6a7;">action</div><div>0 = up · 1 = down · 2 = left · 3 = right</div>
          <div class="k" style="color:#a5d6a7;">actionName</div><div>Human-readable action string</div>
          <div class="k" style="color:#a5d6a7;">outputs</div><div>Raw Q-values [4] — argmax gives the greedy action</div>
        </div>
      </div>`;
  }

  return '';
}

async function renderApiPanel() {
  const root = $('#content');
  const allActive = await window.nb.api.list();
  state.apiServers = new Map(allActive.map(s => [s.id, s]));
  if (!state.current) {
    root.innerHTML = `<div class="empty"><div class="big">No network selected</div></div>`;
    return;
  }
  const n       = state.current;
  const info    = await window.nb.system.info();
  const running = state.apiServers.get(n.id);
  const arch    = n.architecture || {};
  const isPlugin = !!arch.pluginKind;
  const serverUrl = running ? running.url : 'http://' + info.hostIp + ':PORT';

  const standardEndpoints = isPlugin ? `
    <div class="kv-table">
      <div class="k">GET</div><div>/health — liveness check: <code>{"ok":true}</code></div>
      <div class="k">GET</div><div>/info — model metadata, input spec, plugin kind</div>
      <div class="k">POST</div><div>/predict — body: <code>{"inputs":[…]}</code> → plugin-specific response</div>
      <div class="k">GET</div><div>/metrics — Prometheus-format request counters</div>
    </div>` : `
    <div class="kv-table">
      <div class="k">GET</div><div>/health — liveness check: <code>{"ok":true}</code></div>
      <div class="k">GET</div><div>/info — network metadata and input spec</div>
      <div class="k">POST</div><div>/predict — body matches the Inference tab shape</div>
      ${arch.isChat ? `<div class="k">POST</div><div>/chat — stateful chat (sessionId in body, or omit to start new)</div>
      <div class="k">POST</div><div>/chat/reset — clear a session: <code>{"sessionId":"…"}</code></div>` : ''}
      <div class="k">GET</div><div>/metrics — Prometheus-format request counters</div>
    </div>`;

  root.innerHTML = `
    <div class="panel">
      <h2>API — ${escapeHtml(n.name)}</h2>
      <p class="hint">Expose this model on your local network. Other devices can call it at the URL below.${isPlugin ? ' Save the model first (pause training) so the server has weights to serve.' : ''}</p>
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
        ${standardEndpoints}
      </div>
      ${isPlugin ? pluginApiDocs(arch, serverUrl) : ''}
      <div class="section">
        <h3>Log</h3>
        <div class="log" id="api-log"></div>
      </div>
    </div>
  `;

  const startBtn = $('#btn-start-api');
  if (startBtn) startBtn.addEventListener('click', async () => {
    if (n.stateLocked) { toast('Decrypt the network first.'); return; }
    if (isPlugin && !n.state) { toast('Pause training first to save the model, then start the server.'); return; }
    const port = parseInt($('#api-port').value) || 0;
    try {
      const r = await window.nb.api.start(n.id, port);
      toast('API running at ' + r.url);
      renderApiPanel();
    } catch (e) { toast('Failed: ' + e.message); }
  });
  const stopBtn = $('#btn-stop-api');
  if (stopBtn) stopBtn.addEventListener('click', async () => {
    await window.nb.api.stop(n.id);
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
      const r = await window.nb.script.run(state.current?.id || null, $('#script-area').value);
      out.textContent = (r.output || '') + (r.ok ? '' : '\n[error] ' + r.error);
    } catch (e) { out.textContent = 'ERROR: ' + e.message; }
  });
  $('#btn-save-script').addEventListener('click', async () => {
    if (!state.current) { toast('No network selected.'); return; }
    await window.nb.networks.update(state.current.id, { script: $('#script-area').value });
    await loadCurrent(state.current.id);
    toast('Saved.');
  });
}

// ---------- plugin restart tracking ----------

// Set to true whenever a plugin is installed or uninstalled this session.
// Cleared only by a restart.
let pluginNeedsRestart = false;

// Returns true if the named plugin is currently active in the running registry.
function isPluginActive(pluginId) {
  if (pluginRegistry.templates.some(t => (t.id === pluginId || t.pluginKind === pluginId))) return true;
  if (Object.values(pluginRegistry.inferenceRenderers).some(r => r.pluginId === pluginId)) return true;
  if (Object.values(pluginRegistry.trainEditors).some(r => r.pluginId === pluginId)) return true;
  return false;
}

// ---------- docs helpers ----------

// Convert a doc page's HTML body + title to clean Markdown for LLM consumption.
function docPageToMarkdown(title, html) {
  const div = document.createElement('div');
  div.innerHTML = html;

  function nodeToMd(node) {
    if (node.nodeType === 3) return node.nodeValue; // TEXT_NODE
    if (node.nodeType !== 1) return '';              // ELEMENT_NODE only

    const tag  = node.tagName.toLowerCase();
    const inner = () => Array.from(node.childNodes).map(nodeToMd).join('');

    switch (tag) {
      case 'h1': return '\n# '   + inner().trim() + '\n\n';
      case 'h2': return '\n## '  + inner().trim() + '\n\n';
      case 'h3': return '\n### ' + inner().trim() + '\n\n';
      case 'h4': return '\n#### '+ inner().trim() + '\n\n';
      case 'p':  return '\n' + inner().trim() + '\n\n';
      case 'strong': case 'b': return '**' + inner() + '**';
      case 'em':     case 'i': return '_'  + inner() + '_';
      case 'a': {
        const href = node.getAttribute('href') || '';
        const text = inner().trim();
        return href ? `[${text}](${href})` : text;
      }
      case 'code': {
        if (node.parentNode && node.parentNode.tagName.toLowerCase() === 'pre') return node.textContent;
        return '`' + node.textContent + '`';
      }
      case 'pre': {
        const codeEl = node.querySelector('code');
        return '\n```\n' + (codeEl || node).textContent + '\n```\n\n';
      }
      case 'ul': {
        const items = Array.from(node.querySelectorAll(':scope > li'))
          .map(li => '- ' + li.textContent.replace(/\s+/g, ' ').trim());
        return '\n' + items.join('\n') + '\n\n';
      }
      case 'ol': {
        const items = Array.from(node.querySelectorAll(':scope > li'))
          .map((li, i) => `${i + 1}. ` + li.textContent.replace(/\s+/g, ' ').trim());
        return '\n' + items.join('\n') + '\n\n';
      }
      case 'li': return '';
      case 'table': {
        const rows = Array.from(node.querySelectorAll('tr'));
        if (!rows.length) return '';
        let md = '\n'; let sepDone = false;
        for (const row of rows) {
          const cells = Array.from(row.querySelectorAll('th, td'));
          if (!cells.length) continue;
          md += '| ' + cells.map(c => c.textContent.replace(/\s+/g, ' ').trim()).join(' | ') + ' |\n';
          if (!sepDone && row.querySelector('th')) {
            md += '| ' + cells.map(() => '---').join(' | ') + ' |\n';
            sepDone = true;
          }
        }
        return md + '\n';
      }
      case 'tr': case 'th': case 'td': case 'thead': case 'tbody': return '';
      case 'br': return '\n';
      case 'hr': return '\n---\n\n';
      default:   return inner();
    }
  }

  let md = `# ${title}\n\n`;
  md += Array.from(div.childNodes).map(nodeToMd).join('');
  return md.replace(/\n{3,}/g, '\n\n').trim() + '\n';
}

// Pure-JS ZIP writer — creates a STORED (no compression) ZIP archive from
// an array of { name: string, content: string } objects.
// Returns a Uint8Array.
function buildZip(files) {
  const te = new TextEncoder();

  // CRC-32 lookup table
  const crcT = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = (c & 1) ? (0xedb88320 ^ (c >>> 1)) : (c >>> 1);
    crcT[i] = c >>> 0;
  }
  function crc32(buf) {
    let c = 0xffffffff;
    for (let i = 0; i < buf.length; i++) c = crcT[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
    return (c ^ 0xffffffff) >>> 0;
  }
  function u16(n) { return [n & 0xff, (n >> 8) & 0xff]; }
  function u32(n) { return [n & 0xff, (n >> 8) & 0xff, (n >> 16) & 0xff, (n >> 24) & 0xff]; }

  const locals  = [];
  const central = [];
  let   pos     = 0;

  for (const file of files) {
    const nb   = te.encode(file.name);
    const data = te.encode(file.content);
    const crc  = crc32(data);
    const sz   = data.length;

    // Local file header (30 bytes + filename)
    const local = new Uint8Array([
      0x50, 0x4b, 0x03, 0x04,      // local file header signature
      0x14, 0x00,                   // version needed: 2.0
      0x00, 0x00,                   // general purpose bit flag
      0x00, 0x00,                   // compression: STORED
      0x00, 0x00, 0x00, 0x00,       // last mod time + date
      ...u32(crc),
      ...u32(sz),                   // compressed size
      ...u32(sz),                   // uncompressed size
      ...u16(nb.length),            // filename length
      0x00, 0x00,                   // extra field length
      ...nb
    ]);
    const hdrOffset = pos;
    locals.push(local, data);
    pos += local.length + data.length;

    // Central directory entry (46 bytes + filename)
    central.push(new Uint8Array([
      0x50, 0x4b, 0x01, 0x02,      // central directory signature
      0x14, 0x00,                   // version made by
      0x14, 0x00,                   // version needed
      0x00, 0x00,                   // flags
      0x00, 0x00,                   // compression: STORED
      0x00, 0x00, 0x00, 0x00,       // last mod time + date
      ...u32(crc),
      ...u32(sz),                   // compressed size
      ...u32(sz),                   // uncompressed size
      ...u16(nb.length),            // filename length
      0x00, 0x00,                   // extra field length
      0x00, 0x00,                   // file comment length
      0x00, 0x00,                   // disk number start
      0x00, 0x00,                   // internal attributes
      0x00, 0x00, 0x00, 0x00,       // external attributes
      ...u32(hdrOffset),            // relative offset of local file header
      ...nb
    ]));
  }

  const cdOffset = pos;
  const cdSize   = central.reduce((s, p) => s + p.length, 0);
  const eocd = new Uint8Array([
    0x50, 0x4b, 0x05, 0x06,        // end of central directory signature
    0x00, 0x00, 0x00, 0x00,        // disk number / disk with start of CD
    ...u16(files.length),           // entries on this disk
    ...u16(files.length),           // total entries
    ...u32(cdSize),                 // size of central directory
    ...u32(cdOffset),               // offset of central directory
    0x00, 0x00                      // comment length
  ]);

  const all   = [...locals, ...central, eocd];
  const total = all.reduce((s, p) => s + p.length, 0);
  const out   = new Uint8Array(total);
  let   off   = 0;
  for (const p of all) { out.set(p, off); off += p.length; }
  return out;
}

// Trigger a browser-style file download from a Blob or Uint8Array.
function triggerDownload(data, filename, mimeType) {
  const blob = data instanceof Blob ? data : new Blob([data], { type: mimeType || 'application/octet-stream' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}

// Docs tab
function renderDocsTab(root) {
  const doc = window.NB_DOCS.find(d => d.id === state.currentDocId) || window.NB_DOCS[0];
  root.innerHTML = `
    <div class="panel">
      <div class="docs-layout">
        <div class="docs-nav">
          ${window.NB_DOCS.map(d => `<button data-id="${d.id}" class="${d.id === doc.id ? 'active' : ''}">${d.title}</button>`).join('')}
          <div style="border-top:1px solid #2a2a2a;margin-top:8px;padding-top:8px;">
            <button class="btn sm" id="btn-docs-zip" style="width:100%;font-size:11px;text-align:left;padding:5px 8px;">⬇ Download all for LLM (.zip)</button>
          </div>
        </div>
        <div style="display:flex;flex-direction:column;min-height:0;flex:1;">
          <div style="display:flex;align-items:center;justify-content:flex-end;padding:0 0 8px 0;gap:8px;">
            <button class="btn sm" id="btn-docs-llm" style="font-size:11px;">⬇ Download page for LLM (.md)</button>
          </div>
          <div class="docs-body" style="user-select:text;flex:1;">${doc.body}</div>
        </div>
      </div>
    </div>
  `;

  // Navigate between doc pages
  root.querySelectorAll('.docs-nav button[data-id]').forEach(b => b.addEventListener('click', () => {
    state.currentDocId = b.dataset.id;
    renderDocsTab(root);
  }));

  // Download current page as Markdown for LLM ingestion
  root.querySelector('#btn-docs-llm').addEventListener('click', () => {
    const md       = docPageToMarkdown(doc.title, doc.body);
    const safeName = doc.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
    triggerDownload(md, `neuralcabin-doc-${safeName}.md`, 'text/markdown');
  });

  // Download all pages as a ZIP of Markdown files for LLM ingestion
  root.querySelector('#btn-docs-zip').addEventListener('click', () => {
    const files = window.NB_DOCS.map(d => {
      const safeName = d.title.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '');
      return {
        name:    `neuralcabin-docs/${safeName}.md`,
        content: docPageToMarkdown(d.title, d.body)
      };
    });
    const zipBytes = buildZip(files);
    triggerDownload(zipBytes, 'neuralcabin-docs.zip', 'application/zip');
  });
}

// ---------- plugins tab ----------

async function renderPluginsTab(root) {
  const installed = await window.nb.plugins.list();

  // Restart-required banner (shown after any install/uninstall this session)
  const restartBanner = pluginNeedsRestart ? `
    <div id="plugin-restart-banner" style="background:#1a1200;border:1px solid #665500;border-radius:6px;padding:12px 16px;margin-bottom:14px;display:flex;align-items:center;gap:12px;">
      <div style="flex:1;">
        <div style="font-size:13px;font-weight:600;color:#f0b429;">⚠ Restart required</div>
        <div style="font-size:12px;color:#aaa;margin-top:2px;">Plugin changes take effect after a full application restart.</div>
      </div>
      <button class="btn primary sm" id="btn-restart-now" style="white-space:nowrap;">Restart now</button>
    </div>
  ` : '';

  root.innerHTML = `
    <div class="panel">
      <h2>Plugins</h2>
      <p class="hint">Plugins add new model types, training data editors, and inference UIs. Each plugin is a <code>.nbpl</code> file — see Docs → Plugin system for the format reference.</p>
      <div class="section">
        ${restartBanner}
        <div class="row" style="margin-bottom:14px;gap:8px;">
          <button class="btn primary" id="btn-install-plugin">Install Plugin (.nbpl)</button>
        </div>

        <div id="plugin-list">
          ${installed.length === 0
            ? `<div class="empty" style="padding:24px 0;"><div>No plugins installed.</div><div style="font-size:12px;color:#666;margin-top:6px;">Click "Install Plugin" to add a .nbpl file.</div></div>`
            : installed.map(p => {
                const active = isPluginActive(p.id);
                const badge  = active
                  ? `<span style="font-size:10px;font-weight:700;background:#1a3320;color:#4caf50;border:1px solid #2d5c3a;border-radius:10px;padding:2px 8px;letter-spacing:0.04em;">● ACTIVE</span>`
                  : `<span style="font-size:10px;font-weight:700;background:#1f1900;color:#f0b429;border:1px solid #4d3d00;border-radius:10px;padding:2px 8px;letter-spacing:0.04em;">↺ NEEDS RESTART</span>`;
                return `
                  <div class="section" style="background:#141414;border:1px solid #2a2a2a;border-radius:6px;padding:12px 16px;margin-bottom:8px;">
                    <div class="row" style="align-items:flex-start;gap:10px;">
                      <div style="flex:1;min-width:0;">
                        <div class="row" style="gap:8px;align-items:center;margin-bottom:2px;">
                          <span style="font-weight:600;font-size:14px;color:#e0e0e0;">${escapeHtml(p.manifest.name || p.id)}</span>
                          ${badge}
                        </div>
                        <div style="font-size:12px;color:#666;">v${escapeHtml(p.manifest.version || '0.0.0')} · by ${escapeHtml(p.manifest.author || 'unknown')}</div>
                        ${p.manifest.description ? `<div style="font-size:13px;color:#aaa;margin-top:5px;">${escapeHtml(p.manifest.description)}</div>` : ''}
                        <div style="font-size:11px;color:#555;margin-top:4px;font-family:monospace;">id: ${escapeHtml(p.id)}</div>
                      </div>
                      <button class="btn danger sm" data-uninstall="${escapeHtml(p.id)}" style="flex-shrink:0;">Uninstall</button>
                    </div>
                  </div>`;
              }).join('')}
        </div>

        <div class="hint" style="margin-top:10px;font-size:11px;">
          Plugin changes take effect after a restart. Active plugins were loaded at startup.
          See <strong>Docs → Plugin system</strong> for file format details.
        </div>
      </div>
    </div>
  `;

  // Restart now button
  const restartBtn = root.querySelector('#btn-restart-now');
  if (restartBtn) {
    restartBtn.addEventListener('click', async () => {
      restartBtn.disabled = true;
      restartBtn.textContent = 'Restarting…';
      try { await window.nb.app.restart(); } catch (e) { toast('Restart failed: ' + e.message); restartBtn.disabled = false; restartBtn.textContent = 'Restart now'; }
    });
  }

  // Install plugin
  root.querySelector('#btn-install-plugin').addEventListener('click', async () => {
    let result;
    try { result = await window.nb.plugins.install(); }
    catch (e) { toast('Install failed: ' + e.message); return; }

    if (!result) return; // user cancelled dialog

    pluginNeedsRestart = true;
    toast(`✓ "${result.name}" installed. Restart to activate.`);
    renderPluginsTab(root);
  });

  // Uninstall plugin
  root.querySelectorAll('[data-uninstall]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id   = btn.dataset.uninstall;
      const name = btn.closest('.section')?.querySelector('[style*="font-weight:600"]')?.textContent?.trim() || id;
      confirmModal('Uninstall plugin?', `Remove "${escapeHtml(name)}"? Changes take effect after restart.`, async () => {
        try {
          await window.nb.plugins.uninstall(id);
          pluginNeedsRestart = true;
          toast(`✓ "${name}" removed. Restart to apply.`);
          renderPluginsTab(root);
        } catch (e) { toast('Uninstall failed: ' + e.message); }
      });
    });
  });
}

// ---------- backup modal ----------

async function openBackupModal(netId) {
  const modal = $('#modal-backup');
  $('#backup-net-name').textContent = state.current?.name || '';
  modal.hidden = false;
  await renderBackupList(netId);
}

async function renderBackupList(netId) {
  const body = $('#backup-list-body');
  if (!body) return;
  body.innerHTML = '<div style="padding:16px;color:#8a8a8a;">Loading…</div>';
  let backups;
  try { backups = await window.nb.backups.list(netId); }
  catch (e) { body.innerHTML = `<div style="padding:16px;color:#ff7b7b;">Failed to load backups: ${escapeHtml(e.message)}</div>`; return; }

  if (backups.length === 0) {
    body.innerHTML = `
      <div class="backup-empty">
        <div class="big" style="font-size:15px;margin-bottom:8px;">No backups yet</div>
        <div style="color:#8a8a8a;margin-bottom:16px;">Save a snapshot of this network's weights and settings.</div>
        <button class="btn primary" id="btn-backup-first">Create your first backup</button>
      </div>`;
    $('#btn-backup-first').addEventListener('click', async () => {
      try { await window.nb.backups.create(netId, ''); await renderBackupList(netId); toast('Backup created.'); }
      catch (e) { toast('Backup failed: ' + e.message); }
    });
    return;
  }

  body.innerHTML = `
    <div style="margin-bottom:14px;">
      <button class="btn primary sm" id="btn-backup-new">+ Create backup</button>
    </div>
    <div id="backup-items">
      ${backups.map(b => `
        <div class="backup-item" data-id="${b.id}">
          <div class="backup-info">
            <div class="backup-label">${escapeHtml(b.label)}</div>
            <div class="backup-meta">${new Date(b.createdAt).toLocaleString()} · ${b.hasWeights ? 'trained weights' : 'no weights'}</div>
          </div>
          <div class="backup-btns">
            <button class="btn sm" data-act="restore" data-id="${b.id}">Restore</button>
            <button class="btn sm" data-act="download" data-id="${b.id}">Download</button>
            <button class="btn sm danger" data-act="delete" data-id="${b.id}">Delete</button>
          </div>
        </div>`).join('')}
    </div>`;

  $('#btn-backup-new').addEventListener('click', async () => {
    try { await window.nb.backups.create(netId, ''); await renderBackupList(netId); toast('Backup created.'); }
    catch (e) { toast('Backup failed: ' + e.message); }
  });

  body.querySelectorAll('[data-act]').forEach(btn => {
    btn.addEventListener('click', async () => {
      const act = btn.dataset.act, backupId = btn.dataset.id;
      if (act === 'restore') {
        confirmModal('Restore backup?',
          'This overwrites the current network state. The current state will be lost unless you have another backup.',
          async () => {
            await window.nb.backups.restore(netId, backupId);
            await refreshNetworks(); await loadCurrent(netId);
            $('#modal-backup').hidden = true;
            renderActiveTab(); toast('Network restored from backup.');
          });
      } else if (act === 'download') {
        try {
          const p = await window.nb.backups.download(netId, backupId);
          if (p) toast('Saved to ' + p);
        } catch (e) { toast('Download failed: ' + e.message); }
      } else if (act === 'delete') {
        confirmModal('Delete backup?', 'This permanently removes this backup snapshot.', async () => {
          await window.nb.backups.delete(netId, backupId);
          await renderBackupList(netId); toast('Backup deleted.');
        });
      }
    });
  });
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
  const matches = kind === 'plugin'
    ? pluginRegistry.templates
    : window.NB_TEMPLATES.filter(t => {
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
    const t = window.NB_TEMPLATES.find(x => x.id === selectedTemplate);
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
        : ((kind === 'charLM' || kind === 'gpt') ? { text: '' } : { samples: [] })
    };
  }
  const net = await window.nb.networks.create(payload);
  $('#modal-new').hidden = true;
  await refreshNetworks();
  selectNetwork(net.id);
}

function defaultArch(kind) {
  if (kind === 'chat') return { kind: 'charLM', vocabSize: 0, embDim: 32, contextLen: 64, hidden: [96, 96], activation: 'gelu', dropout: 0.1, isChat: true, tokenizerKind: 'wordpart' };
  if (kind === 'gpt') return { kind: 'gpt', vocabSize: 0, embDim: 32, contextLen: 96, hidden: [96, 96], activation: 'gelu', dropout: 0.1, tokenizerKind: 'wordpart' };
  if (kind === 'charLM') return { kind: 'charLM', vocabSize: 0, embDim: 16, contextLen: 32, hidden: [64], activation: 'gelu', dropout: 0, tokenizerKind: 'wordpart' };
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
    const info = await window.nb.system.info();
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
