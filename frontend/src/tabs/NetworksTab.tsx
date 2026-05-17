import { useEffect, useMemo, useState } from 'react'
import { networks, exporter, Activation, ExportFormat, Layer, Network, NetworkKind } from '../api'
import type { TabProps } from '../App'
import PluginErrorBoundary from '../components/PluginErrorBoundary'

const ACTIVATIONS: Activation[] = ['identity', 'relu', 'sigmoid', 'tanh', 'softmax']

interface FormState {
  name: string
  kind: NetworkKind
  seed: number
  // Feedforward fields
  inputDim: number
  hiddenSpec: string
  outputDim: number
  outputActivation: Activation
  // Next-token fields
  vocabBudget: number
  contextSize: number
  embedHidden: string
  // Transformer fields
  tNCtx:    number
  tNEmbd:   number
  tNLayers: number
  tNHeads:  number
  tNFf:     number
}

function defaultForm(kind: NetworkKind): FormState {
  const base = {
    seed: 42,
    inputDim: 2, hiddenSpec: '8,tanh', outputDim: 1, outputActivation: 'sigmoid' as Activation,
    vocabBudget: 64, contextSize: 16, embedHidden: '64,relu',
    tNCtx: 64, tNEmbd: 64, tNLayers: 2, tNHeads: 4, tNFf: 256,
  }
  if (kind === 'feedforward') return { ...base, name: 'xor-mlp', kind }
  if (kind === 'transformer') return { ...base, name: 'nano-llama', kind }
  return { ...base, name: 'tiny-lm', kind }
}

/**
 * Parse a hidden-layer spec like "8,relu,4,relu" into a sequence of
 * (linearOutDim | activation) tokens. Returns null + error on parse failure.
 */
function parseSpec(spec: string): { tokens: Array<number | Activation>; error?: string } {
  const parts = spec.split(',').map(s => s.trim()).filter(s => s.length > 0)
  const tokens: Array<number | Activation> = []
  for (const p of parts) {
    if (ACTIVATIONS.includes(p as Activation)) {
      tokens.push(p as Activation)
    } else if (/^\d+$/.test(p)) {
      const n = parseInt(p, 10)
      if (n <= 0) return { tokens: [], error: `dim must be > 0 (got ${p})` }
      tokens.push(n)
    } else {
      return { tokens: [], error: `unknown spec token '${p}' (expected a positive integer or one of ${ACTIVATIONS.join(',')})` }
    }
  }
  return { tokens }
}

function buildLayers(form: FormState): { layers: Layer[]; inputDim: number; outputDim: number; error?: string } {
  if (form.kind === 'feedforward') {
    const parsed = parseSpec(form.hiddenSpec)
    if (parsed.error) return { layers: [], inputDim: 0, outputDim: 0, error: parsed.error }
    if (form.inputDim <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'input dim must be > 0' }
    if (form.outputDim <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'output dim must be > 0' }

    const layers: Layer[] = []
    let cur = form.inputDim
    for (const tok of parsed.tokens) {
      if (typeof tok === 'number') {
        layers.push({ type: 'linear', in_dim: cur, out_dim: tok })
        cur = tok
      } else {
        layers.push({ type: 'activation', activation: tok })
      }
    }
    layers.push({ type: 'linear', in_dim: cur, out_dim: form.outputDim })
    if (form.outputActivation !== 'identity') {
      layers.push({ type: 'activation', activation: form.outputActivation })
    }
    return { layers, inputDim: form.inputDim, outputDim: form.outputDim }
  }

  if (form.kind === 'transformer') {
    // Transformer's architecture is parameterised by hparams, not a layer list.
    // We report inputDim/outputDim as 0 here (they're driven by the vocab built
    // from the attached corpus). The ArchitecturePreview shows the hparams.
    return { layers: [], inputDim: 0, outputDim: 0 }
  }

  // next_token
  // Reserved tokens (4) + user vocab budget — final vocab built from corpus,
  // but we need to size the input/output layers now. We approximate with the
  // budget; if real vocab differs we'll surface a clear error at training time.
  const vocab = Math.max(form.vocabBudget, 8)
  const inputDim = vocab * form.contextSize
  const parsed = parseSpec(form.embedHidden)
  if (parsed.error) return { layers: [], inputDim: 0, outputDim: 0, error: parsed.error }
  if (form.contextSize <= 0) return { layers: [], inputDim: 0, outputDim: 0, error: 'context size must be > 0' }

  const layers: Layer[] = []
  let cur = inputDim
  for (const tok of parsed.tokens) {
    if (typeof tok === 'number') {
      layers.push({ type: 'linear', in_dim: cur, out_dim: tok })
      cur = tok
    } else {
      layers.push({ type: 'activation', activation: tok })
    }
  }
  // Output layer projects to vocab. Loss is softmax+cross-entropy, so leave logits raw.
  layers.push({ type: 'linear', in_dim: cur, out_dim: vocab })
  return { layers, inputDim, outputDim: vocab }
}

export default function NetworksTab({ refreshNetworks, onSelect, pluginRegistry }: TabProps & { onSelect: (id: string) => void }) {
  const [list, setList] = useState<Network[]>([])
  const [error, setError] = useState<string | null>(null)
  const [showForm, setShowForm] = useState(false)
  const [form, setForm] = useState<FormState>(defaultForm('feedforward'))
  const [busy, setBusy] = useState(false)
  /** Encoded "<pluginId>::<typeId>" when the Type dropdown points at a plugin
   * type, otherwise null and the built-in form is used. */
  const [selectedPluginType, setSelectedPluginType] = useState<string | null>(null)
  const pluginTypes = pluginRegistry?.networkTypes ?? []
  const activePluginType = selectedPluginType
    ? pluginTypes.find(p => `${p.plugin.id}::${p.type.id}` === selectedPluginType) ?? null
    : null

  useEffect(() => { void load() }, [])

  const load = async () => {
    try { setList(await networks.list()); setError(null) }
    catch (e) { setError(String(e)) }
  }

  const preview = useMemo(() => buildLayers(form), [form])

  const onSubmit = async () => {
    if (preview.error) { setError(preview.error); return }
    setBusy(true); setError(null)
    try {
      const created = await networks.create({
        name: form.name.trim() || 'untitled',
        kind: form.kind,
        seed: form.seed,
        layers: preview.layers,
        input_dim: preview.inputDim,
        context_size: form.kind === 'next_token' ? form.contextSize
                     : form.kind === 'transformer' ? form.tNCtx
                     : null,
        transformer: form.kind === 'transformer' ? {
          n_ctx:    form.tNCtx,
          n_embd:   form.tNEmbd,
          n_layers: form.tNLayers,
          n_heads:  form.tNHeads,
          n_ff:     form.tNFf,
          rope_theta: 10000,
          rms_eps:    1e-5,
        } : null,
      })
      setShowForm(false)
      setForm(defaultForm(form.kind))
      await load()
      await refreshNetworks()
      onSelect(created.id)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  const onDelete = async (id: string) => {
    try { await networks.delete(id); await load(); await refreshNetworks() }
    catch (e) { setError(String(e)) }
  }

  return (
    <div className="tab-content">
      <h2>Networks</h2>
      <p className="muted">Create and manage neural-network architectures.</p>

      {error && <div className="status error">{error}</div>}

      <div className="card">
        <div className="card-row">
          <h3 style={{ margin: 0 }}>{showForm ? 'Create network' : `${list.length} network${list.length === 1 ? '' : 's'}`}</h3>
          <button onClick={() => setShowForm(s => !s)} className={showForm ? 'secondary' : ''}>
            {showForm ? 'Cancel' : 'New network'}
          </button>
        </div>

        {showForm && (
          <div className="mt-2 flex-col">
            <div className="grid-2">
              <div>
                <label>Name</label>
                <input value={form.name} onChange={e => setForm({ ...form, name: e.target.value })} />
              </div>
              <div>
                <label>Type</label>
                <select
                  value={selectedPluginType ?? form.kind}
                  onChange={e => {
                    const v = e.target.value
                    if (v.startsWith('plugin::')) {
                      setSelectedPluginType(v.slice('plugin::'.length))
                    } else {
                      setSelectedPluginType(null)
                      setForm(defaultForm(v as NetworkKind))
                    }
                  }}
                >
                  <option value="feedforward">Feed-forward (regression / classification)</option>
                  <option value="next_token">Next-token MLP (text on one-hot window)</option>
                  <option value="transformer">Transformer LM (llama-style — exports to GGUF for llama.cpp / LM Studio)</option>
                  {pluginTypes.map(({ plugin, type }) => (
                    <option key={`${plugin.id}::${type.id}`} value={`plugin::${plugin.id}::${type.id}`}>
                      {type.label} (plugin: {plugin.name})
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {activePluginType && (
              <PluginErrorBoundary fallbackTitle={`${activePluginType.plugin.name} create form crashed`}>
                <div>
                  <p className="muted">{activePluginType.type.description}</p>
                  <activePluginType.type.CreateForm
                    context={pluginRegistry!.context}
                    onCreated={async (id) => {
                      setShowForm(false)
                      setSelectedPluginType(null)
                      await load()
                      await refreshNetworks()
                      onSelect(id)
                    }}
                  />
                </div>
              </PluginErrorBoundary>
            )}

            {!activePluginType && <>
            {form.kind === 'transformer' ? (
              <>
                <div className="grid-3">
                  <div>
                    <label>Context length (n_ctx)</label>
                    <input type="number" min={2} max={4096} value={form.tNCtx}
                      onChange={e => setForm({ ...form, tNCtx: parseInt(e.target.value) || 2 })} />
                    <small>Maximum tokens per sequence.</small>
                  </div>
                  <div>
                    <label>Embedding dim (n_embd)</label>
                    <input type="number" min={8} step={8} value={form.tNEmbd}
                      onChange={e => setForm({ ...form, tNEmbd: parseInt(e.target.value) || 8 })} />
                    <small>Must be divisible by n_heads; head_dim must be even (for RoPE).</small>
                  </div>
                  <div>
                    <label>Heads (n_heads)</label>
                    <input type="number" min={1} value={form.tNHeads}
                      onChange={e => setForm({ ...form, tNHeads: parseInt(e.target.value) || 1 })} />
                  </div>
                </div>
                <div className="grid-3">
                  <div>
                    <label>Layers (n_layers)</label>
                    <input type="number" min={1} max={64} value={form.tNLayers}
                      onChange={e => setForm({ ...form, tNLayers: parseInt(e.target.value) || 1 })} />
                  </div>
                  <div>
                    <label>Feed-forward dim (n_ff)</label>
                    <input type="number" min={8} value={form.tNFf}
                      onChange={e => setForm({ ...form, tNFf: parseInt(e.target.value) || 8 })} />
                    <small>Typical: 4 × n_embd.</small>
                  </div>
                  <div>
                    <label>Seed</label>
                    <input type="number" value={form.seed}
                      onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
                  </div>
                </div>
                <div className="card" style={{ background: 'var(--bg-input)', margin: 0 }}>
                  <small className="muted">
                    Pre-norm decoder-only transformer with multi-head causal
                    self-attention, RoPE, and SwiGLU feed-forward. Architecture
                    matches llama 2 so exported GGUF files load in llama.cpp /
                    LM Studio. Vocabulary is built from the attached corpus on
                    the Corpus tab — the embedding layer is resized to match.
                  </small>
                </div>
              </>
            ) : form.kind === 'feedforward' ? (
              <>
                <div className="grid-3">
                  <div>
                    <label>Input dim</label>
                    <input type="number" min={1} value={form.inputDim}
                      onChange={e => setForm({ ...form, inputDim: parseInt(e.target.value) || 1 })} />
                  </div>
                  <div>
                    <label>Output dim</label>
                    <input type="number" min={1} value={form.outputDim}
                      onChange={e => setForm({ ...form, outputDim: parseInt(e.target.value) || 1 })} />
                  </div>
                  <div>
                    <label>Output activation</label>
                    <select value={form.outputActivation}
                      onChange={e => setForm({ ...form, outputActivation: e.target.value as Activation })}>
                      {ACTIVATIONS.map(a => <option key={a} value={a}>{a}</option>)}
                    </select>
                  </div>
                </div>
                <div>
                  <label>Hidden layers (comma-separated dims and activations)</label>
                  <input value={form.hiddenSpec}
                    onChange={e => setForm({ ...form, hiddenSpec: e.target.value })}
                    placeholder="8,tanh,4,tanh" />
                  <small>e.g. <code>8,tanh,4,tanh</code> → Linear 2→8 · tanh · Linear 8→4 · tanh · Linear 4→{form.outputDim}</small>
                </div>
              </>
            ) : (
              <>
                <div className="grid-3">
                  <div>
                    <label>Vocab budget</label>
                    <input type="number" min={8} value={form.vocabBudget}
                      onChange={e => setForm({ ...form, vocabBudget: parseInt(e.target.value) || 8 })} />
                    <small>Estimated vocabulary size. Sizes input/output layers. Must match real corpus vocab at train time.</small>
                  </div>
                  <div>
                    <label>Context size</label>
                    <input type="number" min={1} max={512} value={form.contextSize}
                      onChange={e => setForm({ ...form, contextSize: parseInt(e.target.value) || 1 })} />
                    <small>Tokens fed per prediction.</small>
                  </div>
                  <div>
                    <label>Seed</label>
                    <input type="number" value={form.seed}
                      onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
                  </div>
                </div>
                <div>
                  <label>Hidden layers (between input and vocab projection)</label>
                  <input value={form.embedHidden}
                    onChange={e => setForm({ ...form, embedHidden: e.target.value })}
                    placeholder="64,relu,32,relu" />
                  <small>Output layer projects to <code>vocab_size</code> logits (softmax+cross-entropy is applied at training time).</small>
                </div>
              </>
            )}

            {form.kind === 'feedforward' && (
              <div>
                <label>Seed</label>
                <input type="number" value={form.seed}
                  onChange={e => setForm({ ...form, seed: parseInt(e.target.value) || 0 })} />
              </div>
            )}

            {form.kind === 'transformer'
              ? <TransformerPreview form={form} />
              : <ArchitecturePreview preview={preview} />}

            <div className="flex">
              <button onClick={onSubmit} disabled={busy || !!preview.error}>
                {busy ? 'Creating…' : 'Create network'}
              </button>
              <button className="secondary" onClick={() => setShowForm(false)}>Cancel</button>
            </div>
            </>}
          </div>
        )}
      </div>

      {list.length === 0 ? (
        <div className="card"><p className="muted">No networks yet.</p></div>
      ) : list.map(n => (
        <div key={n.id} className="list-item">
          <div className="flex-1">
            <strong>{n.name}</strong>
            <p>
              <span className="chip">{n.kind === 'next_token' ? 'next-token' : n.kind === 'transformer' ? 'transformer' : 'feed-forward'}</span>
              {' '}<span className="chip">{n.input_dim} → {n.output_dim}</span>
              {' '}<span className="chip">{n.layers.length} layers</span>
              {' '}<span className="chip">{n.parameter_count.toLocaleString()} params</span>
              {n.context_size ? <> {' '}<span className="chip">ctx {n.context_size}</span></> : null}
              {n.trained ? <> {' '}<span className="chip accent">trained</span></> : null}
            </p>
          </div>
          <div className="flex">
            <ExportButton network={n} onError={setError} />
            <button className="danger" onClick={() => onDelete(n.id)}>Delete</button>
          </div>
        </div>
      ))}
    </div>
  )
}

function ExportButton({ network, onError }: { network: Network; onError: (e: string) => void }) {
  const [open, setOpen] = useState(false)
  const [busy, setBusy] = useState<ExportFormat | null>(null)

  const formats: { id: ExportFormat; label: string; enabled: boolean; reason?: string }[] = [
    { id: 'pytorch', label: 'PyTorch (.pt)', enabled: true },
    {
      id: 'onnx', label: 'ONNX (.onnx)',
      enabled: network.kind !== 'transformer',
      reason: network.kind === 'transformer' ? 'ONNX export for transformers is not yet implemented' : undefined,
    },
    {
      id: 'gguf', label: 'GGUF (.gguf)',
      enabled: network.kind !== 'feedforward',
      reason: network.kind === 'feedforward'
        ? 'GGUF is only useful for text-generation networks'
        : (network.kind === 'next_token'
            ? 'Uses a custom architecture — not loadable in llama.cpp / LM Studio. Use the transformer kind for that.'
            : undefined),
    },
  ]

  const doExport = async (fmt: ExportFormat) => {
    setBusy(fmt)
    try {
      const payload = await exporter.run(network.id, fmt)
      // Decode base64 → Blob → trigger download.
      const bin = atob(payload.data_b64)
      const bytes = new Uint8Array(bin.length)
      for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i)
      const blob = new Blob([bytes], { type: 'application/octet-stream' })
      const url = URL.createObjectURL(blob)
      const a = document.createElement('a')
      a.href = url; a.download = payload.filename
      document.body.appendChild(a); a.click()
      document.body.removeChild(a); URL.revokeObjectURL(url)
      setOpen(false)
    } catch (e) {
      onError(String(e))
    } finally { setBusy(null) }
  }

  if (!open) {
    return <button className="secondary" onClick={() => setOpen(true)} disabled={!network.trained && network.kind === 'next_token'}>
      Export
    </button>
  }

  return (
    <div className="flex" style={{ gap: 4 }}>
      {formats.map(f => (
        <button
          key={f.id}
          className="secondary"
          disabled={!f.enabled || busy !== null}
          title={f.reason}
          onClick={() => doExport(f.id)}
        >
          {busy === f.id ? 'Exporting…' : f.label}
        </button>
      ))}
      <button className="secondary" onClick={() => setOpen(false)}>Cancel</button>
    </div>
  )
}

function TransformerPreview({ form }: { form: FormState }) {
  const headDim = form.tNHeads ? form.tNEmbd / form.tNHeads : 0
  const errors: string[] = []
  if (form.tNEmbd % form.tNHeads !== 0) errors.push(`n_embd (${form.tNEmbd}) must be divisible by n_heads (${form.tNHeads})`)
  if (headDim % 2 !== 0)                errors.push(`head_dim (n_embd/n_heads = ${headDim}) must be even for RoPE`)

  // Rough parameter-count estimate (excluding embedding, which depends on vocab):
  // per layer: 4 * n_embd^2 (attn QKVO) + 3 * n_embd * n_ff (gate/up/down) + 2*n_embd (norms)
  const perLayer = 4 * form.tNEmbd * form.tNEmbd + 3 * form.tNEmbd * form.tNFf + 2 * form.tNEmbd
  const coreParams = perLayer * form.tNLayers + form.tNEmbd

  return (
    <div className="card" style={{ background: 'var(--bg-input)', margin: 0 }}>
      <h4>Architecture preview</h4>
      <div className="flex" style={{ flexWrap: 'wrap' }}>
        <span className="chip">n_ctx {form.tNCtx}</span>
        <span className="chip">n_embd {form.tNEmbd}</span>
        <span className="chip">n_layers {form.tNLayers}</span>
        <span className="chip">n_heads {form.tNHeads}</span>
        <span className="chip">head_dim {headDim}</span>
        <span className="chip">n_ff {form.tNFf}</span>
        <span className="chip accent">~{coreParams.toLocaleString()} core params (+ vocab × n_embd × 2 for embedding & LM head)</span>
      </div>
      {errors.length > 0 && <div className="status error mt-1">{errors.join('; ')}</div>}
    </div>
  )
}

function ArchitecturePreview({ preview }: {
  preview: { layers: Layer[]; inputDim: number; outputDim: number; error?: string }
}) {
  if (preview.error) {
    return <div className="status error">{preview.error}</div>
  }
  return (
    <div className="card" style={{ background: 'var(--bg-input)', margin: 0 }}>
      <h4>Architecture preview</h4>
      <div className="flex" style={{ flexWrap: 'wrap' }}>
        <span className="chip">in {preview.inputDim}</span>
        {preview.layers.map((l, i) => (
          <span key={i} className="chip">
            {l.type === 'linear' ? `linear → ${l.out_dim}` : l.activation}
          </span>
        ))}
        <span className="chip accent">out {preview.outputDim}</span>
      </div>
    </div>
  )
}

// PluginErrorBoundary used here is imported from the components folder.

