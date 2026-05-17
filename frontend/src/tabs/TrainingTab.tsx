import { useEffect, useRef, useState } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import {
  training, corpus, OptimizerConfig, CorpusStats, TrainingRun,
} from '../api'
import type { TabProps } from '../App'

interface RunState {
  running: boolean
  epoch: number
  totalEpochs: number
  lastLoss: number
  lossHistory: number[]
  elapsedSecs: number
  finalStatus?: 'completed' | 'cancelled' | 'aborted'
}

const EMPTY_RUN: RunState = { running: false, epoch: 0, totalEpochs: 0, lastLoss: 0, lossHistory: [], elapsedSecs: 0 }

export default function TrainingTab({ network, refreshNetworks }: TabProps) {
  const [stats, setStats] = useState<CorpusStats | null>(null)

  const [epochs, setEpochs] = useState(500)
  const [batchSize, setBatchSize] = useState(32)
  const [seed, setSeed] = useState(42)
  const [optKind, setOptKind] = useState<'adam' | 'adamw' | 'lamb' | 'sgd'>('adam')
  const [lr, setLr] = useState(0.01)
  const [momentum, setMomentum] = useState(0.9)
  const [maskUserTokens, setMaskUserTokens] = useState(true)
  const [frozenLayers, setFrozenLayers] = useState<Set<string>>(new Set())

  const [trainingId, setTrainingId] = useState<string | null>(null)
  const [run, setRun] = useState<RunState>(EMPTY_RUN)
  const [error, setError] = useState<string | null>(null)
  const [historyVersion, setHistoryVersion] = useState(0)
  const finishedListenersRef = useRef<UnlistenFn[]>([])

  useEffect(() => {
    if (network) void loadStats(network.id)
    else setStats(null)
  }, [network])

  const isNextToken = network?.kind === 'next_token'
  const isTransformer = network?.kind === 'transformer'
  const isFinetune = stats?.stage === 'finetune'
  const finetuneBlocked = isFinetune && network && !network.pretrained
  const lossLabel = isNextToken ? 'crossentropy (forced for next-token)' : 'mse'
  const toggleFrozen = (key: string) =>
    setFrozenLayers(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key); else next.add(key)
      return next
    })

  // Subscribe to training events whenever a job is active
  useEffect(() => {
    if (!trainingId) return
    let cancelled = false
    const cleanups: UnlistenFn[] = []

    const setup = async () => {
      const u = await training.onUpdate(u => {
        if (cancelled || u.training_id !== trainingId) return
        setRun({
          running: true, epoch: u.epoch, totalEpochs: u.total_epochs,
          lastLoss: u.loss, lossHistory: u.loss_history, elapsedSecs: u.elapsed_secs,
        })
      })
      const f = await training.onFinished(r => {
        if (cancelled || r.training_id !== trainingId) return
        setRun(prev => ({
          ...prev,
          running: false,
          lastLoss: r.final_loss,
          elapsedSecs: r.elapsed_secs,
          finalStatus: r.status,
        }))
        // Reload network list so "trained" badge updates, and bump history.
        void refreshNetworks()
        setHistoryVersion(v => v + 1)
      })
      const e = await training.onError(err => {
        if (cancelled || err.training_id !== trainingId) return
        setError(err.message)
        setRun(prev => ({ ...prev, running: false }))
        setHistoryVersion(v => v + 1)
      })
      cleanups.push(u, f, e)
    }
    void setup()
    finishedListenersRef.current = cleanups
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [trainingId])

  const loadStats = async (id: string) => {
    try { setStats(await corpus.stats(id)) }
    catch (e) { setStats(null) /* not having a corpus is OK */ }
  }

  const start = async () => {
    if (!network) return
    setError(null); setRun({ ...EMPTY_RUN, running: true, totalEpochs: epochs })
    const optimizer: OptimizerConfig = { kind: optKind, lr }
    if (optKind === 'sgd') optimizer.momentum = momentum
    try {
      const r = await training.start({
        network_id: network.id,
        config: {
          epochs, batch_size: batchSize, optimizer,
          loss: isNextToken ? 'crossentropy' : 'mse',
          seed,
          mask_user_tokens: isNextToken ? maskUserTokens : undefined,
          frozen_layers: frozenLayers.size > 0 ? Array.from(frozenLayers) : undefined,
        },
      })
      setTrainingId(r.training_id)
    } catch (e) {
      setError(String(e))
      setRun(EMPTY_RUN)
    }
  }

  const reset = () => { setTrainingId(null); setRun(EMPTY_RUN); setError(null) }

  const examples = stats?.training_examples ?? 0
  const corpusReady = examples > 0

  return (
    <div className="tab-content">
      <h2>Training</h2>
      <p className="muted">Train the selected network on its attached corpus.</p>

      {error && <div className="status error">{error}</div>}

      {network && (
        <div className="card">
          <p className="muted">
            <span className="chip">{network.name}</span>{' '}
            <span className="chip">{network.input_dim} → {network.output_dim}</span>{' '}
            <span className="chip">{network.parameter_count.toLocaleString()} params</span>{' '}
            <span className="chip">{examples.toLocaleString()} training examples</span>
            {!corpusReady && <span className="status error" style={{ display: 'inline-block', marginLeft: 8 }}>
              No corpus attached. Add data on the Corpus tab.
            </span>}
          </p>
        </div>
      )}

      {network && !trainingId && (
        <HistoryPanel networkId={network.id} refreshKey={historyVersion} />
      )}

      {!trainingId ? (
        <div className="card">
          <h3>Configuration</h3>
          <div className="grid-3">
            <div>
              <label>Epochs</label>
              <input type="number" min={1} value={epochs}
                onChange={e => setEpochs(Math.max(1, parseInt(e.target.value) || 1))} />
            </div>
            <div>
              <label>Batch size</label>
              <input type="number" min={1} value={batchSize}
                onChange={e => setBatchSize(Math.max(1, parseInt(e.target.value) || 1))} />
            </div>
            <div>
              <label>Seed</label>
              <input type="number" value={seed}
                onChange={e => setSeed(parseInt(e.target.value) || 0)} />
            </div>
            <div>
              <label>Optimizer</label>
              <select value={optKind} onChange={e => setOptKind(e.target.value as 'adam' | 'adamw' | 'lamb' | 'sgd')}>
                <option value="adam">Adam</option>
                <option value="adamw">AdamW</option>
                <option value="lamb">LAMB</option>
                <option value="sgd">SGD</option>
              </select>
            </div>
            <div>
              <label>Learning rate</label>
              <input type="number" step="0.001" min={0} value={lr}
                onChange={e => setLr(parseFloat(e.target.value) || 0)} />
            </div>
            {optKind === 'sgd' && (
              <div>
                <label>Momentum</label>
                <input type="number" step="0.01" min={0} max={1} value={momentum}
                  onChange={e => setMomentum(parseFloat(e.target.value) || 0)} />
              </div>
            )}
            <div>
              <label>Loss</label>
              <input value={lossLabel} disabled />
            </div>
            {isNextToken && stats?.stage === 'finetune' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <label>Mask user tokens</label>
                <label style={{ display: 'flex', alignItems: 'center', gap: 8, textTransform: 'none', letterSpacing: 0, color: 'var(--text)' }}>
                  <input type="checkbox" style={{ width: 'auto' }} checked={maskUserTokens}
                    onChange={e => setMaskUserTokens(e.target.checked)} />
                  Score loss only on assistant output
                </label>
              </div>
            )}
          </div>
          {isTransformer && isFinetune && network?.pretrained && (
            <div style={{ marginTop: 16, padding: 12, border: '1px solid var(--border-soft)',
                          borderRadius: 'var(--radius)', background: 'var(--bg-input)' }}>
              <h4 style={{ margin: 0 }}>Layer locking</h4>
              <p className="muted small" style={{ marginTop: 6 }}>
                Freeze parts of the pre-trained network so fine-tuning can't drift them. GPT-style fine-tuning typically freezes the token embedding and early blocks, leaving only the later blocks and LM head trainable.
              </p>
              <div className="flex" style={{ flexWrap: 'wrap', gap: 6, marginTop: 8 }}>
                <FreezeChip label="Token embedding" k="embedding" frozen={frozenLayers} onToggle={toggleFrozen} />
                {Array.from({ length: network.transformer?.n_layers ?? 0 }, (_, i) => (
                  <FreezeChip key={i} label={`Block ${i}`} k={`block:${i}`} frozen={frozenLayers} onToggle={toggleFrozen} />
                ))}
                <FreezeChip label="Final norm" k="output_norm" frozen={frozenLayers} onToggle={toggleFrozen} />
                <FreezeChip label="LM head" k="output" frozen={frozenLayers} onToggle={toggleFrozen} />
              </div>
              <p className="muted small" style={{ marginTop: 8 }}>
                {frozenLayers.size === 0
                  ? 'All parameters will be updated.'
                  : `${frozenLayers.size} component${frozenLayers.size === 1 ? '' : 's'} frozen.`}
              </p>
            </div>
          )}
          {finetuneBlocked && (
            <div className="status error mt-2">
              Fine-tuning is locked — complete a pre-training run on this network first.
            </div>
          )}
          <div className="flex mt-2">
            <button onClick={start} disabled={!network || !corpusReady || !!finetuneBlocked}>
              {isFinetune ? 'Start fine-tuning' : 'Start training'}
            </button>
          </div>
        </div>
      ) : (
        <RunView run={run} trainingId={trainingId} onReset={reset} onError={setError} />
      )}
    </div>
  )
}

function formatDuration(secs: number): string {
  if (secs < 60) return `${secs.toFixed(0)}s`
  const m = Math.floor(secs / 60)
  const s = Math.floor(secs % 60)
  if (m < 60) return `${m}m ${s}s`
  const h = Math.floor(m / 60)
  return `${h}h ${m % 60}m`
}

function RunView({ run, trainingId, onReset, onError }: {
  run: RunState
  trainingId: string | null
  onReset: () => void
  onError: (e: string | null) => void
}) {
  const [stopping, setStopping] = useState<'stop' | 'abort' | null>(null)

  const onStopHere = async () => {
    if (!trainingId) return
    setStopping('stop'); onError(null)
    try { await training.stop(trainingId) }
    catch (e) { onError(String(e)) }
  }
  const onAbort = async () => {
    if (!trainingId) return
    setStopping('abort'); onError(null)
    try { await training.abort(trainingId) }
    catch (e) { onError(String(e)) }
  }

  const epochsPerSec = run.epoch > 0 && run.elapsedSecs > 0
    ? run.epoch / run.elapsedSecs
    : null
  const etaSecs = epochsPerSec && run.running && run.epoch < run.totalEpochs
    ? (run.totalEpochs - run.epoch) / epochsPerSec
    : null

  return (
    <>
      <div className={`status ${run.running ? '' : (run.finalStatus === 'aborted' ? 'error' : 'success')}`}>
        {run.running
          ? `Training… epoch ${run.epoch} / ${run.totalEpochs}`
          : run.finalStatus === 'aborted'
            ? `Aborted — model rolled back to pre-training weights (${formatDuration(run.elapsedSecs)})`
            : run.finalStatus === 'cancelled'
              ? `Stopped at epoch ${run.epoch} — kept current weights (${formatDuration(run.elapsedSecs)})`
              : `Done — ${run.totalEpochs} epochs in ${formatDuration(run.elapsedSecs)}`}
      </div>

      <div className="card">
        <div className="grid-3" style={{ gridTemplateColumns: 'repeat(4, 1fr)' }}>
          <Metric label="Epoch" value={`${run.epoch} / ${run.totalEpochs}`} />
          <Metric label="Loss"  value={run.lastLoss.toFixed(6)} />
          <Metric label="Elapsed"  value={formatDuration(run.elapsedSecs)} />
          <Metric
            label={run.running ? 'ETA' : 'Speed'}
            value={run.running
              ? (etaSecs !== null ? formatDuration(etaSecs) : '—')
              : (epochsPerSec !== null ? `${epochsPerSec.toFixed(1)} ep/s` : '—')}
          />
        </div>
        <ProgressBar epoch={run.epoch} total={run.totalEpochs} />

        {run.running && (
          <div className="flex mt-2">
            <button
              className="secondary"
              onClick={onStopHere}
              disabled={stopping !== null}
              title="Halt training and keep whatever the model has learned so far."
            >
              {stopping === 'stop' ? 'Stopping…' : 'Stop here'}
            </button>
            <button
              className="danger"
              onClick={onAbort}
              disabled={stopping !== null}
              title="Halt training AND revert the model to its pre-training weights."
            >
              {stopping === 'abort' ? 'Aborting…' : 'Abort'}
            </button>
          </div>
        )}
      </div>

      <div className="plot">
        <h3>Loss over epochs</h3>
        <LossPlot history={run.lossHistory} />
      </div>

      {!run.running && (
        <button onClick={onReset}>Start a new run</button>
      )}
    </>
  )
}

function FreezeChip({ label, k, frozen, onToggle }: {
  label: string; k: string; frozen: Set<string>; onToggle: (k: string) => void
}) {
  const isFrozen = frozen.has(k)
  return (
    <button
      type="button"
      className={isFrozen ? '' : 'secondary'}
      onClick={() => onToggle(k)}
      style={{ fontSize: 11, padding: '4px 10px', textTransform: 'none', letterSpacing: 0 }}
      title={isFrozen ? `${label} is frozen (weights won't update)` : `${label} is trainable`}
    >
      {isFrozen ? '🔒 ' : ''}{label}
    </button>
  )
}

function Metric({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <p className="muted small">{label}</p>
      <p style={{ fontSize: 22, fontWeight: 600, fontVariantNumeric: 'tabular-nums' }}>{value}</p>
    </div>
  )
}

function ProgressBar({ epoch, total }: { epoch: number; total: number }) {
  const pct = total > 0 ? (epoch / total) * 100 : 0
  return (
    <div style={{ marginTop: 14 }}>
      <div style={{ height: 6, background: 'var(--bg-input)', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{ height: '100%', width: `${pct}%`, background: 'var(--accent)', transition: 'width 0.2s ease' }} />
      </div>
      <p className="muted small mt-1" style={{ textAlign: 'right' }}>{pct.toFixed(1)}%</p>
    </div>
  )
}

function LossPlot({ history, color = 'var(--accent)' }: { history: number[]; color?: string }) {
  const [logScale, setLogScale] = useState(false)

  if (history.length === 0) {
    return <p className="muted">Waiting for first epoch…</p>
  }

  const allPositive = history.every(v => v > 0)
  const useLog = logScale && allPositive

  const transformed = useLog ? history.map(v => Math.log10(v)) : history
  const max = Math.max(...transformed)
  const min = Math.min(...transformed)
  const range = max - min || 1

  const toY = (v: number) => 100 - ((v - min) / range) * 90 - 5

  const points = transformed.map((v, i) => {
    const x = (i / Math.max(history.length - 1, 1)) * 100
    const y = toY(v)
    return `${x},${y}`
  }).join(' ')

  // Best (minimum loss) epoch
  const rawMin = Math.min(...history)
  const bestIdx = history.indexOf(rawMin)
  const bestX = (bestIdx / Math.max(history.length - 1, 1)) * 100
  const bestY = toY(transformed[bestIdx])

  // Grid lines: 3 evenly spaced between min and max
  const gridLines = [25, 50, 75].map(pct => {
    const v = min + (range * pct / 100)
    return { y: toY(v), label: useLog ? `10^${v.toFixed(1)}` : v.toFixed(4) }
  })

  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 6 }}>
        <button
          className={logScale ? '' : 'secondary'}
          onClick={() => setLogScale(s => !s)}
          style={{ fontSize: 11, padding: '3px 10px', textTransform: 'none', letterSpacing: 0 }}
          title={allPositive ? 'Toggle log scale' : 'Log scale unavailable (loss values ≤ 0)'}
          disabled={!allPositive}
        >
          log scale
        </button>
      </div>
      <svg viewBox="0 0 100 100" preserveAspectRatio="none"
           style={{ width: '100%', height: 220, background: 'var(--bg-input)', borderRadius: 4 }}>
        {gridLines.map(({ y }, i) => (
          <line key={i} x1="0" y1={y} x2="100" y2={y}
                stroke="var(--border-soft)" strokeWidth="0.2" />
        ))}
        <polyline points={points} fill="none" stroke={color} strokeWidth="1"
                  vectorEffect="non-scaling-stroke" />
        {history.length > 1 && (
          <>
            <circle cx={bestX} cy={bestY} r="1.8" fill="var(--success)"
                    vectorEffect="non-scaling-stroke" />
            <line x1={bestX} y1={bestY} x2={bestX} y2="100"
                  stroke="var(--success)" strokeWidth="0.3" strokeDasharray="1,1"
                  vectorEffect="non-scaling-stroke" />
          </>
        )}
      </svg>
      <p className="muted small mt-1">
        min {rawMin.toFixed(6)} @ epoch {bestIdx + 1}
        {' · '}last {history[history.length - 1].toFixed(6)}
        {useLog ? ' (log₁₀ scale)' : ''}
      </p>
    </div>
  )
}

// ─── Training history panel ──────────────────────────────────────────────────

function HistoryPanel({ networkId, refreshKey }: { networkId: string; refreshKey: number }) {
  const [runs, setRuns] = useState<TrainingRun[]>([])
  const [expanded, setExpanded] = useState<string | null>(null)
  const [clearing, setClearing] = useState(false)

  useEffect(() => {
    training.history(networkId).then(setRuns).catch(() => setRuns([]))
  }, [networkId, refreshKey])

  if (runs.length === 0) return null

  const clearAll = async () => {
    setClearing(true)
    try {
      await training.clearHistory(networkId)
      setRuns([])
      setExpanded(null)
    } finally {
      setClearing(false)
    }
  }

  return (
    <div className="card">
      <div className="card-row" style={{ marginBottom: 12 }}>
        <h3 style={{ marginBottom: 0 }}>Training History</h3>
        <button className="ghost" style={{ fontSize: 12, padding: '4px 10px' }}
          onClick={clearAll} disabled={clearing}>
          {clearing ? 'Clearing…' : 'Clear all'}
        </button>
      </div>
      {runs.map(run => (
        <HistoryRunItem
          key={run.id}
          run={run}
          expanded={expanded === run.id}
          onToggle={() => setExpanded(prev => prev === run.id ? null : run.id)}
        />
      ))}
    </div>
  )
}

const STATUS_COLORS: Record<string, string> = {
  completed: 'var(--success)',
  cancelled:  'var(--text-muted)',
  aborted:    'var(--error)',
  error:      'var(--error)',
}

function HistoryRunItem({ run, expanded, onToggle }: {
  run: TrainingRun
  expanded: boolean
  onToggle: () => void
}) {
  const date = new Date(run.started_at)
  const dateStr = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })
  const statusColor = STATUS_COLORS[run.status] ?? 'var(--text-muted)'
  const plotColor = run.status === 'completed' ? 'var(--accent)' : 'var(--text-faint)'

  return (
    <div style={{ marginBottom: 6 }}>
      <button
        onClick={onToggle}
        style={{
          display: 'flex', alignItems: 'center', gap: 10,
          width: '100%', textAlign: 'left',
          background: expanded ? 'var(--bg-elev-2)' : 'transparent',
          border: '1px solid var(--border-soft)',
          borderRadius: 'var(--radius)',
          padding: '8px 12px',
          cursor: 'pointer',
          color: 'var(--text)',
          fontFamily: 'var(--font-sans)',
          fontSize: 13,
          transition: 'background 0.12s ease',
        }}
      >
        <span style={{ color: statusColor, fontSize: 11, fontWeight: 600,
                        textTransform: 'uppercase', letterSpacing: '0.06em', minWidth: 64 }}>
          {run.status}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 12, minWidth: 90 }}>
          {dateStr} {timeStr}
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          {run.config_summary.optimizer} lr={run.config_summary.lr}
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          {run.epochs_run}/{run.total_epochs} epochs
        </span>
        <span className="chip" style={{ fontSize: 11 }}>
          loss {run.final_loss.toFixed(6)}
        </span>
        <span style={{ color: 'var(--text-faint)', fontSize: 12, marginLeft: 'auto' }}>
          {formatDuration(run.elapsed_secs)}
          {run.elapsed_secs > 0 && run.epochs_run > 0
            ? ` · ${(run.epochs_run / run.elapsed_secs).toFixed(1)} ep/s`
            : ''}
        </span>
        <span style={{ color: 'var(--text-muted)', fontSize: 11 }}>
          {expanded ? '▲' : '▼'}
        </span>
      </button>
      {expanded && run.loss_history.length > 0 && (
        <div style={{ padding: '12px 12px 4px', background: 'var(--bg-elev-2)',
                      border: '1px solid var(--border-soft)', borderTop: 'none',
                      borderRadius: '0 0 var(--radius) var(--radius)' }}>
          <LossPlot history={run.loss_history} color={plotColor} />
        </div>
      )}
    </div>
  )
}
