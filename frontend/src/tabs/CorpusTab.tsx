import { useEffect, useMemo, useState } from 'react'
import {
  corpus, FineTunePair, FeedforwardCorpus,
  CorpusStats, Stage, Network,
} from '../api'
import type { TabProps } from '../App'

export default function CorpusTab({ network, pluginRegistry }: TabProps) {
  const [stats, setStats] = useState<CorpusStats | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [info, setInfo] = useState<string | null>(null)

  useEffect(() => {
    if (network) void loadStats(network.id)
    else setStats(null)
  }, [network])

  // Plugin-managed networks own their own corpus UI.
  const pluginType = network && pluginRegistry?.typeForNetwork(network.id)
  if (network && pluginType?.type.CorpusUI) {
    const PluginCorpus = pluginType.type.CorpusUI
    return (
      <div className="tab-content">
        <h2>Corpus</h2>
        <p className="muted">
          Managed by plugin <strong>{pluginType.plugin.name}</strong> · type <strong>{pluginType.type.label}</strong>.
        </p>
        <PluginCorpus network={network} context={pluginRegistry!.context} />
      </div>
    )
  }

  const loadStats = async (id: string) => {
    try { setStats(await corpus.stats(id)) }
    catch (e) { setStats(null); setError(String(e)) }
  }

  return (
    <div className="tab-content">
      <h2>Corpus</h2>
      <p className="muted">
        Attach training data to a network. The shape and uploader change with
        the network type.
      </p>

      {error && <div className="status error">{error}</div>}
      {info  && <div className="status success">{info}</div>}

      {!network ? (
        <div className="card">
          <p className="muted">Create or select a network to manage its corpus.</p>
        </div>
      ) : network.kind === 'feedforward' ? (
        <FeedforwardCorpusEditor
          network={network}
          stats={stats}
          onSaved={async () => { await loadStats(network.id); setInfo('Corpus saved.'); setError(null) }}
          onError={(e) => { setError(e); setInfo(null) }}
        />
      ) : (
        <NextTokenCorpusEditor
          network={network}
          stats={stats}
          onSaved={async () => { await loadStats(network.id); setInfo('Corpus saved.'); setError(null) }}
          onError={(e) => { setError(e); setInfo(null) }}
        />
      )}

      {stats && <StatsCard stats={stats} />}
    </div>
  )
}

// ─── Stats card ─────────────────────────────────────────────────────────────

function StatsCard({ stats }: { stats: CorpusStats }) {
  const rows: [string, string | number | undefined][] = []
  if (stats.kind === 'feedforward') {
    rows.push(['Rows', stats.rows], ['Input dim', stats.in_dim], ['Output dim', stats.out_dim])
  } else {
    rows.push(
      ['Stage', stats.stage ?? '—'],
      ['Vocab mode', stats.vocab_mode ?? '—'],
      ['Vocab size', stats.vocab_size],
      ['Text characters', stats.text_chars],
      ['Text tokens', stats.text_tokens],
      ['Pair count', stats.pair_count],
    )
  }
  rows.push(['Training examples', stats.training_examples])

  return (
    <div className="card">
      <h3>Corpus statistics</h3>
      <table>
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}><th>{k}</th><td>{v ?? '—'}</td></tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// ─── Feedforward editor ─────────────────────────────────────────────────────

function FeedforwardCorpusEditor({ network, onSaved, onError }: {
  network: Network; stats: CorpusStats | null;
  onSaved: () => Promise<void> | void; onError: (e: string) => void;
}) {
  // CSV-style text editor: each row is `feat1,feat2,...,target1,target2,...`
  // Length per row must equal (input_dim + output_dim).
  const [csv, setCsv] = useState<string>('')
  const [busy, setBusy] = useState(false)

  // Hydrate from saved corpus when the network changes.
  useEffect(() => {
    let cancelled = false
    void corpus.get(network.id).then(c => {
      if (cancelled) return
      if (c?.feedforward) {
        setCsv(serializeFeedforward(c.feedforward))
      } else {
        setCsv('')
      }
    }).catch(() => { if (!cancelled) setCsv('') })
    return () => { cancelled = true }
  }, [network.id])

  const parsed = useMemo(() => parseCsvFeedforward(csv, network.input_dim, network.output_dim), [csv, network])

  const onUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    const reader = new FileReader()
    reader.onload = () => setCsv(String(reader.result ?? ''))
    reader.readAsText(f)
    e.target.value = ''
  }

  const onSave = async () => {
    if (parsed.error || !parsed.corpus) { onError(parsed.error ?? 'no rows'); return }
    setBusy(true)
    try {
      await corpus.set({ network_id: network.id, feedforward: parsed.corpus })
      await onSaved()
    } catch (e) {
      onError(String(e))
    } finally { setBusy(false) }
  }

  const onExport = () => {
    if (!parsed.corpus) return
    const blob = new Blob([csv], { type: 'text/csv' })
    download(blob, `${network.name}-corpus.csv`)
  }

  return (
    <>
      <div className="card">
        <div className="card-row">
          <h3 style={{ margin: 0 }}>Feed-forward corpus</h3>
          <div className="flex">
            <label className="btn secondary" style={{ marginBottom: 0, cursor: 'pointer', textTransform: 'none', letterSpacing: 0 }}>
              Upload CSV
              <input type="file" accept=".csv,text/csv,text/plain" onChange={onUpload}
                     style={{ display: 'none' }} />
            </label>
            <button className="secondary" onClick={onExport} disabled={!parsed.corpus}>Export CSV</button>
          </div>
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          One row per example. Each row must have <code>{network.input_dim} + {network.output_dim}</code>{' '}
          comma-separated numbers — the first {network.input_dim} are inputs, the last {network.output_dim} are targets.
        </p>
        <textarea
          rows={10}
          value={csv}
          onChange={e => setCsv(e.target.value)}
          placeholder={
            network.input_dim === 2 && network.output_dim === 1
              ? '0,0,0\n0,1,1\n1,0,1\n1,1,0'
              : ''
          }
        />
        {parsed.error && <div className="status error mt-1">{parsed.error}</div>}
        {parsed.corpus && !parsed.error && (
          <div className="status success mt-1">
            Parsed {parsed.corpus.rows} row{parsed.corpus.rows === 1 ? '' : 's'}.
          </div>
        )}
        <div className="flex mt-2">
          <button onClick={onSave} disabled={busy || !!parsed.error || !parsed.corpus}>
            {busy ? 'Saving…' : 'Save corpus'}
          </button>
        </div>
      </div>
    </>
  )
}

function serializeFeedforward(ff: FeedforwardCorpus): string {
  const lines: string[] = []
  for (let r = 0; r < ff.rows; r++) {
    const row: number[] = []
    for (let i = 0; i < ff.in_dim; i++) row.push(ff.features[r * ff.in_dim + i])
    for (let o = 0; o < ff.out_dim; o++) row.push(ff.targets[r * ff.out_dim + o])
    lines.push(row.join(','))
  }
  return lines.join('\n')
}

function parseCsvFeedforward(csv: string, inDim: number, outDim: number): {
  corpus: FeedforwardCorpus | null; error?: string
} {
  const need = inDim + outDim
  const lines = csv.split(/\r?\n/).map(l => l.trim()).filter(l => l.length > 0)
  if (lines.length === 0) return { corpus: null }

  const features: number[] = []
  const targets: number[] = []
  for (let i = 0; i < lines.length; i++) {
    const cells = lines[i].split(',').map(s => s.trim())
    if (cells.length !== need) {
      return { corpus: null, error: `row ${i + 1}: expected ${need} values, got ${cells.length}` }
    }
    for (let j = 0; j < need; j++) {
      const n = Number(cells[j])
      if (!Number.isFinite(n)) return { corpus: null, error: `row ${i + 1} col ${j + 1}: '${cells[j]}' is not a number` }
      if (j < inDim) features.push(n)
      else targets.push(n)
    }
  }
  return {
    corpus: { features, targets, rows: lines.length, in_dim: inDim, out_dim: outDim },
  }
}

// ─── Next-token editor ──────────────────────────────────────────────────────

function NextTokenCorpusEditor({ network, stats, onSaved, onError }: {
  network: Network; stats: CorpusStats | null;
  onSaved: () => Promise<void> | void; onError: (e: string) => void;
}) {
  const [stage, setStage] = useState<Stage>(stats?.stage ?? 'pretrain')
  const [vocabMode, setVocabMode] = useState<'char' | 'word'>(
    (stats?.vocab_mode === 'word' ? 'word' : 'char')
  )
  const [text, setText] = useState<string>('')
  const [pairs, setPairs] = useState<FineTunePair[]>([{ input: '', output: '' }])
  const [busy, setBusy] = useState(false)

  // Hydrate from saved corpus when the network changes. Without this, switching
  // tabs and coming back clears the editor even though the backend still has
  // the data — looks like "save lost everything".
  useEffect(() => {
    let cancelled = false
    void corpus.get(network.id).then(c => {
      if (cancelled) return
      setText(c?.text ?? '')
      setPairs(c?.pairs && c.pairs.length > 0 ? c.pairs : [{ input: '', output: '' }])
      if (c?.stage) setStage(c.stage)
    }).catch(() => {/* leave defaults */})
    return () => { cancelled = true }
  }, [network.id])

  // Pick up vocab_mode from stats when it becomes known.
  useEffect(() => {
    if (stats?.vocab_mode === 'word' || stats?.vocab_mode === 'char') {
      setVocabMode(stats.vocab_mode)
    }
  }, [stats?.vocab_mode])

  // If finetune was selected but pre-training hasn't completed, snap back
  // to pretrain — the backend would reject the save anyway.
  useEffect(() => {
    if (!network.pretrained && stage === 'finetune') setStage('pretrain')
  }, [network.pretrained, stage])

  // ─── Pretrain: bulk file upload ───
  const onUploadText = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = Array.from(e.target.files ?? [])
    if (files.length === 0) return
    Promise.all(files.map(f => f.text())).then(parts => {
      setText(prev => (prev ? prev + '\n\n' : '') + parts.join('\n\n'))
    }).catch(err => onError(String(err)))
    e.target.value = ''
  }

  // ─── Fine-tune: JSON import/export ───
  const onImportPairs = (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0]
    if (!f) return
    f.text().then(raw => {
      try {
        const data = JSON.parse(raw)
        if (!Array.isArray(data)) throw new Error('top-level JSON must be an array')
        const cleaned: FineTunePair[] = data.map((row: unknown, i: number) => {
          if (!row || typeof row !== 'object') throw new Error(`item ${i} is not an object`)
          const r = row as Record<string, unknown>
          const input = typeof r.input === 'string' ? r.input : String(r.input ?? '')
          const output = typeof r.output === 'string' ? r.output : String(r.output ?? '')
          return { input, output }
        })
        if (cleaned.length === 0) throw new Error('no pairs in file')
        setPairs(cleaned)
      } catch (err) { onError(`JSON parse failed: ${err}`) }
    }).catch(err => onError(String(err)))
    e.target.value = ''
  }

  const onExportPairs = () => {
    const blob = new Blob([JSON.stringify(pairs, null, 2)], { type: 'application/json' })
    download(blob, `${network.name}-finetune.json`)
  }

  const updatePair = (i: number, field: 'input' | 'output', value: string) =>
    setPairs(prev => prev.map((p, idx) => idx === i ? { ...p, [field]: value } : p))
  const addPair = () => setPairs(prev => [...prev, { input: '', output: '' }])
  const removePair = (i: number) => setPairs(prev => prev.filter((_, idx) => idx !== i))

  const onSave = async () => {
    setBusy(true)
    try {
      const payload =
        stage === 'pretrain'
          ? {
              network_id: network.id,
              stage: 'pretrain' as const,
              vocab_mode: vocabMode,
              text: text.trim().length > 0 ? text : undefined,
            }
          : {
              network_id: network.id,
              stage: 'finetune' as const,
              vocab_mode: vocabMode,
              pairs: pairs.filter(p => p.input.length > 0 || p.output.length > 0),
            }
      if (stage === 'pretrain' && !payload.text) { onError('Pretraining requires non-empty text.'); return }
      if (stage === 'finetune' && (!('pairs' in payload) || payload.pairs!.length === 0)) {
        onError('Fine-tuning requires at least one input/output pair.'); return
      }
      await corpus.set(payload)
      await onSaved()
    } catch (e) { onError(String(e)) }
    finally { setBusy(false) }
  }

  return (
    <>
      <div className="card">
        <div className="card-row">
          <h3 style={{ margin: 0 }}>Stage</h3>
          <div className="flex">
            <button
              className={stage === 'pretrain' ? '' : 'secondary'}
              onClick={() => setStage('pretrain')}
            >Pretraining</button>
            <button
              className={stage === 'finetune' ? '' : 'secondary'}
              onClick={() => network.pretrained && setStage('finetune')}
              disabled={!network.pretrained}
              title={network.pretrained
                ? 'Fine-tune on input/output pairs'
                : 'Complete a pre-training run before fine-tuning'}
            >Fine-tuning {!network.pretrained && '🔒'}</button>
          </div>
        </div>
        <p className="muted" style={{ marginTop: 8 }}>
          {stage === 'pretrain'
            ? 'Upload one or more plain-text files. The vocabulary will be built from this text and the model will learn to predict the next token from a sliding window. Pre-training must complete before fine-tuning is unlocked.'
            : 'Provide input → output pairs. The model is fine-tuned on top of the pre-trained weights — vocabulary and base representations are preserved. Use layer locking on the Training tab to freeze parts of the network.'}
        </p>
        {!network.pretrained && (
          <p className="muted" style={{ marginTop: 6, fontSize: 12 }}>
            🔒 Fine-tuning is locked until a pre-training run completes. Your pre-training and fine-tuning data are saved independently — editing one doesn't erase the other.
          </p>
        )}
      </div>

      <div className="card">
        <div className="grid-2">
          <div>
            <label>Vocab mode</label>
            <select value={vocabMode} onChange={e => setVocabMode(e.target.value as 'char' | 'word')}>
              <option value="char">Character-level</option>
              <option value="word">Word-level (whitespace + punctuation)</option>
            </select>
          </div>
          <div>
            <label>Context size (network)</label>
            <input value={network.context_size ?? '—'} disabled />
            <small>Set when the network was created.</small>
          </div>
        </div>
      </div>

      {stage === 'pretrain' ? (
        <div className="card">
          <div className="card-row">
            <h3 style={{ margin: 0 }}>Text corpus</h3>
            <label className="btn secondary" style={{ marginBottom: 0, cursor: 'pointer', textTransform: 'none', letterSpacing: 0 }}>
              Upload .txt files
              <input type="file" accept=".txt,text/plain" multiple onChange={onUploadText}
                     style={{ display: 'none' }} />
            </label>
          </div>
          <textarea
            rows={14}
            value={text}
            onChange={e => setText(e.target.value)}
            placeholder="Paste or upload training text here…"
          />
          <p className="muted mt-1">
            {text.length.toLocaleString()} characters
          </p>
        </div>
      ) : (
        <div className="card">
          <div className="card-row">
            <h3 style={{ margin: 0 }}>Input/Output pairs</h3>
            <div className="flex">
              <label className="btn secondary" style={{ marginBottom: 0, cursor: 'pointer', textTransform: 'none', letterSpacing: 0 }}>
                Import JSON
                <input type="file" accept=".json,application/json" onChange={onImportPairs}
                       style={{ display: 'none' }} />
              </label>
              <button className="secondary" onClick={onExportPairs}>Export JSON</button>
              <button className="secondary" onClick={addPair}>+ Pair</button>
            </div>
          </div>
          <p className="muted" style={{ marginTop: 8 }}>
            JSON format: <code>[{`{"input": "...", "output": "..."}`}]</code>
          </p>
          <div className="flex-col mt-2">
            {pairs.map((p, i) => (
              <div key={i} className="card" style={{ background: 'var(--bg-input)', margin: 0 }}>
                <div className="card-row">
                  <h4>Pair {i + 1}</h4>
                  <button className="ghost" onClick={() => removePair(i)} disabled={pairs.length <= 1}>Remove</button>
                </div>
                <div className="grid-2 mt-1">
                  <div>
                    <label>User input</label>
                    <textarea rows={3} value={p.input}
                      onChange={e => updatePair(i, 'input', e.target.value)} />
                  </div>
                  <div>
                    <label>Assistant output</label>
                    <textarea rows={3} value={p.output}
                      onChange={e => updatePair(i, 'output', e.target.value)} />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      <div className="card">
        <button onClick={onSave} disabled={busy}>
          {busy ? 'Saving…' : `Save corpus (${stage})`}
        </button>
      </div>
    </>
  )
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function download(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob)
  const a = document.createElement('a')
  a.href = url; a.download = filename
  document.body.appendChild(a); a.click(); document.body.removeChild(a)
  URL.revokeObjectURL(url)
}
