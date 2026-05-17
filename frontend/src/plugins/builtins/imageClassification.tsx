import { useEffect, useRef, useState, useCallback } from 'react'
import { networks, corpus, inference, type Activation } from '../../api'
import NetworkViz from '../../components/NetworkViz'
import { idbKV } from '../storage'
import type {
  NeuralCabinPlugin,
  NetworkTypeDescriptor,
  CreateFormProps,
  NetworkTypeRenderProps,
} from '../types'

// ─── Per-network metadata (stored in plugin meta) ─────────────────────────────

/** Small bookkeeping persisted via the host's localStorage-backed meta API. */
interface ImageClassMeta {
  sizeX: number
  sizeY: number
  colored: boolean
  classes: string[]
  /** Create-time hyperparams, preserved so we can rebuild the underlying
   * feed-forward network if the class count later changes. */
  name: string
  hiddenSpec: string
  outputAct: Activation
  seed: number
}

/** A single image sample. Pixel data is too big for localStorage (a 64×64 RGB
 * sample is ~12k floats), so samples are persisted in IndexedDB keyed by
 * network id — see `samplesKey` below. */
interface ImageSample { pixels: number[]; classIdx: number }

const samplesKey = (networkId: string) => `image-classification.samples.${networkId}`
const loadSamples = (id: string) => idbKV.get<ImageSample[]>(samplesKey(id)).then(v => v ?? [])
const saveSamples = (id: string, s: ImageSample[]) => idbKV.set(samplesKey(id), s)
const deleteSamples = (id: string) => idbKV.delete(samplesKey(id))

const DEFAULT_META = (
  sizeX: number, sizeY: number, colored: boolean,
  name = 'image-classifier', hiddenSpec = '64,relu,32,relu',
  outputAct: Activation = 'softmax', seed = 42,
): ImageClassMeta => ({
  sizeX, sizeY, colored, classes: [],
  name, hiddenSpec, outputAct, seed,
})

/** Parse `"64,relu,32,relu"` into a layer list given an input dim and a final
 * output dim. Shared between the create form and the rebuild path so both
 * routes produce byte-identical architectures. */
function buildLayers(
  inputDim: number, outputDim: number, hiddenSpec: string, outputAct: Activation,
): any[] {
  const layers: any[] = []
  let cur = inputDim
  const parts = hiddenSpec.split(',').map(s => s.trim()).filter(Boolean)
  for (const p of parts) {
    if (['relu', 'sigmoid', 'tanh', 'softmax', 'identity'].includes(p)) {
      layers.push({ type: 'activation', activation: p })
    } else if (/^\d+$/.test(p)) {
      const n = parseInt(p, 10)
      layers.push({ type: 'linear', in_dim: cur, out_dim: n })
      cur = n
    } else {
      throw new Error(`unknown hidden-layer token '${p}'`)
    }
  }
  layers.push({ type: 'linear', in_dim: cur, out_dim: Math.max(1, outputDim) })
  if (outputAct !== 'identity') layers.push({ type: 'activation', activation: outputAct })
  return layers
}

// ─── Helpers: image <-> feature-vector ───────────────────────────────────────

function pixelsFromImage(img: HTMLImageElement, w: number, h: number, colored: boolean): number[] {
  const canvas = document.createElement('canvas')
  canvas.width = w; canvas.height = h
  const ctx = canvas.getContext('2d')!
  ctx.drawImage(img, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h).data
  const out: number[] = []
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4] / 255
    const g = data[i * 4 + 1] / 255
    const b = data[i * 4 + 2] / 255
    if (colored) { out.push(r, g, b) }
    else { out.push((r + g + b) / 3) }
  }
  return out
}

function pixelsFromCanvas(canvas: HTMLCanvasElement, w: number, h: number, colored: boolean): number[] {
  const tmp = document.createElement('canvas')
  tmp.width = w; tmp.height = h
  const ctx = tmp.getContext('2d')!
  ctx.fillStyle = '#000'
  ctx.fillRect(0, 0, w, h)
  ctx.drawImage(canvas, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h).data
  const out: number[] = []
  for (let i = 0; i < w * h; i++) {
    const r = data[i * 4] / 255
    const g = data[i * 4 + 1] / 255
    const b = data[i * 4 + 2] / 255
    if (colored) { out.push(r, g, b) }
    else { out.push((r + g + b) / 3) }
  }
  return out
}

function featureDim(m: Pick<ImageClassMeta, 'sizeX' | 'sizeY' | 'colored'>): number {
  return m.sizeX * m.sizeY * (m.colored ? 3 : 1)
}

// ─── Drawing panel ───────────────────────────────────────────────────────────

interface DrawPanelHandle {
  clear: () => void
  getPixels: () => number[]
}

function DrawPanel({
  width, height, colored,
  pixelScale = 12, onChange, panelRef,
}: {
  width: number; height: number; colored: boolean
  pixelScale?: number
  onChange?: () => void
  panelRef?: React.MutableRefObject<DrawPanelHandle | null>
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)
  const [color, setColor] = useState<string>(colored ? '#ffffff' : '#ffffff')
  const drawingRef = useRef(false)
  const lastRef = useRef<{ x: number; y: number } | null>(null)

  // Initialize blank canvas on first mount / when dims change.
  useEffect(() => {
    const c = canvasRef.current
    if (!c) return
    const ctx = c.getContext('2d')!
    ctx.fillStyle = '#000'
    ctx.fillRect(0, 0, c.width, c.height)
  }, [width, height])

  useEffect(() => {
    if (!panelRef) return
    panelRef.current = {
      clear: () => {
        const c = canvasRef.current; if (!c) return
        const ctx = c.getContext('2d')!
        ctx.fillStyle = '#000'
        ctx.fillRect(0, 0, c.width, c.height)
        onChange?.()
      },
      getPixels: () => {
        const c = canvasRef.current
        if (!c) return new Array(width * height * (colored ? 3 : 1)).fill(0)
        return pixelsFromCanvas(c, width, height, colored)
      },
    }
  })

  const point = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const c = canvasRef.current; if (!c) return null
    const rect = c.getBoundingClientRect()
    return {
      x: ((e.clientX - rect.left) / rect.width) * c.width,
      y: ((e.clientY - rect.top) / rect.height) * c.height,
    }
  }

  const stroke = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const p = point(e); if (!p) return
    const c = canvasRef.current!
    const ctx = c.getContext('2d')!
    ctx.strokeStyle = colored ? color : '#ffffff'
    ctx.fillStyle = ctx.strokeStyle
    ctx.lineCap = 'round'
    ctx.lineWidth = Math.max(2, pixelScale * 0.9)
    if (lastRef.current) {
      ctx.beginPath()
      ctx.moveTo(lastRef.current.x, lastRef.current.y)
      ctx.lineTo(p.x, p.y)
      ctx.stroke()
    } else {
      ctx.beginPath()
      ctx.arc(p.x, p.y, ctx.lineWidth / 2, 0, Math.PI * 2)
      ctx.fill()
    }
    lastRef.current = p
    onChange?.()
  }

  return (
    <div>
      <canvas
        ref={canvasRef}
        width={width * pixelScale}
        height={height * pixelScale}
        style={{
          background: '#000',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius)',
          imageRendering: 'pixelated',
          touchAction: 'none',
          cursor: 'crosshair',
          maxWidth: '100%',
        }}
        onPointerDown={e => {
          drawingRef.current = true
          ;(e.target as Element).setPointerCapture(e.pointerId)
          lastRef.current = null
          stroke(e)
        }}
        onPointerMove={e => { if (drawingRef.current) stroke(e) }}
        onPointerUp={() => { drawingRef.current = false; lastRef.current = null }}
        onPointerLeave={() => { drawingRef.current = false; lastRef.current = null }}
      />
      <div className="flex mt-1" style={{ gap: 8, alignItems: 'center' }}>
        {colored && (
          <>
            <label style={{ marginBottom: 0 }}>Brush color</label>
            <input type="color" value={color} onChange={e => setColor(e.target.value)} />
          </>
        )}
        <button
          className="secondary"
          onClick={() => { panelRef?.current?.clear() }}
        >
          Clear
        </button>
      </div>
    </div>
  )
}

// ─── Create form ─────────────────────────────────────────────────────────────

function CreateForm({ context, onCreated }: CreateFormProps) {
  const [name, setName] = useState('image-classifier')
  const [sizeX, setSizeX] = useState(16)
  const [sizeY, setSizeY] = useState(16)
  const [hidden, setHidden] = useState('64,relu,32,relu')
  const [outputAct, setOutputAct] = useState<Activation>('softmax')
  const [colored, setColored] = useState(false)
  const [seed, setSeed] = useState(42)
  const [numClasses, setNumClasses] = useState(3)
  const [error, setError] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)

  const inputDim = sizeX * sizeY * (colored ? 3 : 1)

  const create = async () => {
    setError(null); setBusy(true)
    try {
      const layers = buildLayers(inputDim, Math.max(1, numClasses), hidden, outputAct)
      const net = await networks.create({
        name, kind: 'feedforward', seed, layers, input_dim: inputDim,
      })
      context.tagNetwork(net.id, 'image-classification')
      context.setMeta<ImageClassMeta>(
        net.id,
        DEFAULT_META(sizeX, sizeY, colored, name, hidden, outputAct, seed),
      )
      await context.refreshNetworks()
      onCreated(net.id)
    } catch (e) {
      setError(String(e))
    } finally { setBusy(false) }
  }

  return (
    <div>
      <div className="grid-3">
        <div>
          <label>Name</label>
          <input value={name} onChange={e => setName(e.target.value)} />
        </div>
        <div>
          <label>Image size X</label>
          <input type="number" min={2} max={128} value={sizeX}
            onChange={e => setSizeX(Math.max(2, Math.min(128, parseInt(e.target.value) || 16)))} />
        </div>
        <div>
          <label>Image size Y</label>
          <input type="number" min={2} max={128} value={sizeY}
            onChange={e => setSizeY(Math.max(2, Math.min(128, parseInt(e.target.value) || 16)))} />
        </div>
        <div>
          <label>Hidden layers</label>
          <input value={hidden} onChange={e => setHidden(e.target.value)} placeholder="64,relu,32,relu" />
          <small>e.g. <code>64,relu,32,relu</code></small>
        </div>
        <div>
          <label>Output activation</label>
          <select value={outputAct} onChange={e => setOutputAct(e.target.value as Activation)}>
            <option value="softmax">softmax</option>
            <option value="sigmoid">sigmoid</option>
            <option value="identity">identity</option>
          </select>
        </div>
        <div>
          <label>Initial class count</label>
          <input type="number" min={1} max={1024} value={numClasses}
            onChange={e => setNumClasses(Math.max(1, parseInt(e.target.value) || 1))} />
          <small>You can add labels later in the Corpus tab.</small>
        </div>
        <div>
          <label>Colored (RGB)</label>
          <select value={colored ? 'rgb' : 'bw'} onChange={e => setColored(e.target.value === 'rgb')}>
            <option value="bw">White / Black</option>
            <option value="rgb">RGB</option>
          </select>
        </div>
        <div>
          <label>Seed</label>
          <input type="number" value={seed}
            onChange={e => setSeed(parseInt(e.target.value) || 0)} />
        </div>
      </div>
      <p className="muted mt-1">Input dim = {inputDim} ({sizeX}×{sizeY}{colored ? '×3' : ' grayscale'}).</p>
      {error && <div className="status error">{error}</div>}
      <div className="flex mt-1">
        <button onClick={create} disabled={busy}>{busy ? 'Creating…' : 'Create network'}</button>
      </div>
    </div>
  )
}

// ─── Corpus UI ───────────────────────────────────────────────────────────────

function loadMeta(context: NetworkTypeRenderProps['context'], networkId: string): ImageClassMeta {
  // The host's meta store may carry a legacy `samples` field from earlier
  // versions of this plugin (when samples lived in localStorage). We don't
  // restore it here — `useSamples` migrates it into IDB asynchronously and
  // strips it out so we never re-hit the quota.
  const m = context.getMeta<Partial<ImageClassMeta>>(networkId)
  return {
    sizeX: m?.sizeX ?? 16,
    sizeY: m?.sizeY ?? 16,
    colored: m?.colored ?? false,
    classes: Array.isArray(m?.classes) ? m!.classes! : [],
    name: m?.name ?? 'image-classifier',
    hiddenSpec: m?.hiddenSpec ?? '64,relu,32,relu',
    outputAct: (m?.outputAct as Activation) ?? 'softmax',
    seed: m?.seed ?? 42,
  }
}

/** Manage `samples` for a network: load from IDB on mount, migrate any
 * legacy localStorage-backed samples on the first run, persist updates back
 * to IDB. Returns the live list, a setter, and a "ready" flag so the UI can
 * avoid showing a misleadingly-empty state during the initial load. */
function useSamples(
  context: NetworkTypeRenderProps['context'],
  networkId: string,
): [ImageSample[], (next: ImageSample[]) => void, boolean] {
  const [samples, setSamplesState] = useState<ImageSample[]>([])
  const [ready, setReady] = useState(false)

  useEffect(() => {
    let cancelled = false
    setReady(false)
    ;(async () => {
      let loaded = await loadSamples(networkId).catch(() => [])
      // One-time migration: legacy meta had `samples` inline. Pull them out
      // of localStorage, move them into IDB, and rewrite meta without them.
      const legacy = context.getMeta<{ samples?: ImageSample[] }>(networkId)
      if (legacy?.samples && legacy.samples.length > 0 && loaded.length === 0) {
        loaded = legacy.samples
        try { await saveSamples(networkId, loaded) } catch { /* leave inline as fallback */ }
        const { samples: _omit, ...rest } = legacy as any
        context.setMeta(networkId, rest)
      }
      if (!cancelled) {
        setSamplesState(loaded)
        setReady(true)
      }
    })()
    return () => { cancelled = true }
  }, [networkId])

  const setSamples = useCallback((next: ImageSample[]) => {
    setSamplesState(next)
    // Fire-and-forget; errors here (e.g. IDB full) should be surfaced by the
    // caller via the corpus-save path rather than crash the UI.
    void saveSamples(networkId, next).catch(err => {
      console.error('[image-classification] failed to persist samples', err)
    })
  }, [networkId])

  return [samples, setSamples, ready]
}

function CorpusUI({ network, context }: NetworkTypeRenderProps) {
  const [meta, setMeta] = useState<ImageClassMeta>(() => loadMeta(context, network.id))
  const [samples, setSamples, samplesReady] = useSamples(context, network.id)
  const [newClass, setNewClass] = useState('')
  const [selectedClass, setSelectedClass] = useState<number>(0)
  const [status, setStatus] = useState<string | null>(null)
  const [error, setError] = useState<string | null>(null)
  const drawRef = useRef<DrawPanelHandle | null>(null)

  // Re-hydrate when the selected network changes; useState's initial value
  // only fires once so without this we'd display stale meta after a switch.
  useEffect(() => {
    setMeta(loadMeta(context, network.id))
    setSelectedClass(0)
    setStatus(null)
    setError(null)
  }, [network.id])

  useEffect(() => { context.setMeta(network.id, meta) }, [meta, network.id])

  const addClass = () => {
    const name = newClass.trim(); if (!name) return
    if (meta.classes.includes(name)) { setError(`class '${name}' already exists`); return }
    setMeta({ ...meta, classes: [...meta.classes, name] })
    setNewClass('')
  }

  const removeClass = (idx: number) => {
    const nextSamples = samples
      .filter(s => s.classIdx !== idx)
      .map(s => ({ ...s, classIdx: s.classIdx > idx ? s.classIdx - 1 : s.classIdx }))
    setMeta({ ...meta, classes: meta.classes.filter((_, i) => i !== idx) })
    setSamples(nextSamples)
    if (selectedClass >= meta.classes.length - 1) setSelectedClass(0)
  }

  const addSample = (pixels: number[]) => {
    if (meta.classes.length === 0) { setError('add at least one class first'); return }
    const next = [...samples, { pixels, classIdx: selectedClass }]
    setSamples(next)
    setStatus(`Added sample for "${meta.classes[selectedClass]}" (${next.length} total)`)
  }

  const onUpload = async (files: FileList | null) => {
    if (!files || files.length === 0) return
    setError(null)
    for (const file of Array.from(files)) {
      try {
        const url = URL.createObjectURL(file)
        const img = new Image()
        await new Promise<void>((res, rej) => {
          img.onload = () => res(); img.onerror = () => rej(new Error('image load failed')); img.src = url
        })
        const pixels = pixelsFromImage(img, meta.sizeX, meta.sizeY, meta.colored)
        URL.revokeObjectURL(url)
        addSample(pixels)
      } catch (e) {
        setError(String(e))
      }
    }
  }

  const addDrawnSample = () => {
    const pixels = drawRef.current?.getPixels()
    if (!pixels) return
    addSample(pixels)
    drawRef.current?.clear()
  }

  const saveCorpusToBackend = async () => {
    setError(null); setStatus(null)
    try {
      if (samples.length === 0) throw new Error('add some samples first')
      if (meta.classes.length === 0) throw new Error('add at least one class first')
      const inDim = featureDim(meta)
      const outDim = meta.classes.length

      // The underlying feed-forward network's output_dim is fixed at create
      // time, but the user can add classes after the fact. If the class count
      // has drifted, rebuild the network so its output layer matches before
      // we save the corpus — otherwise the backend rejects with a shape
      // mismatch ("feedforward out_dim N doesn't match network output_dim M").
      let targetNetworkId = network.id
      if (network.output_dim !== outDim) {
        if (network.trained) {
          throw new Error(
            `Class count (${outDim}) doesn't match the trained network's output dim (${network.output_dim}). ` +
            `Rebuilding would discard trained weights — delete the network manually and create a new one if that's what you want.`,
          )
        }
        const layers = buildLayers(inDim, outDim, meta.hiddenSpec, meta.outputAct)
        const fresh = await networks.create({
          name: meta.name, kind: 'feedforward', seed: meta.seed,
          layers, input_dim: inDim,
        })
        // Re-tag, copy meta + move samples to the new id, then delete the old.
        context.tagNetwork(fresh.id, 'image-classification')
        context.setMeta<ImageClassMeta>(fresh.id, meta)
        try { await saveSamples(fresh.id, samples) } catch { /* best-effort */ }
        try { await deleteSamples(network.id) } catch { /* best-effort */ }
        try { await networks.delete(network.id) } catch { /* best-effort */ }
        targetNetworkId = fresh.id
      }

      const features: number[] = []
      const targets: number[] = []
      for (const s of samples) {
        features.push(...s.pixels)
        for (let i = 0; i < outDim; i++) targets.push(i === s.classIdx ? 1 : 0)
      }
      await corpus.set({
        network_id: targetNetworkId,
        feedforward: {
          features, targets,
          rows: samples.length,
          in_dim: inDim, out_dim: outDim,
        },
      })

      if (targetNetworkId !== network.id) {
        await context.refreshNetworks()
        context.selectNetwork(targetNetworkId)
        setStatus(
          `Output dim grew from ${network.output_dim} to ${outDim}; rebuilt the network and ` +
          `saved ${samples.length} samples to it.`,
        )
      } else {
        setStatus(`Saved ${samples.length} samples × ${outDim} classes to the backend.`)
      }
    } catch (e) { setError(String(e)) }
  }

  const samplesByClass = meta.classes.map((_, i) => samples.filter(s => s.classIdx === i).length)

  return (
    <>
      <div className="card">
        <h3>Classes</h3>
        {meta.classes.length !== network.output_dim && meta.classes.length > 0 && (
          <div className="status mt-1">
            Network currently has <strong>{network.output_dim}</strong> output neuron(s) but you have <strong>{meta.classes.length}</strong> class(es).
            {network.trained
              ? ' The network is trained — saving will fail until classes match. Create a new network if you need more outputs.'
              : ' Saving the corpus will automatically rebuild the network with the new output dim.'}
          </div>
        )}
        <div className="flex" style={{ gap: 8, marginBottom: 12 }}>
          <input
            value={newClass} onChange={e => setNewClass(e.target.value)}
            onKeyDown={e => { if (e.key === 'Enter') addClass() }}
            placeholder="e.g. Cat" style={{ flex: 1 }}
          />
          <button onClick={addClass}>Add class</button>
        </div>
        {meta.classes.length === 0 ? (
          <p className="muted">No classes yet. Each class becomes an output neuron.</p>
        ) : (
          <table>
            <thead><tr><th>#</th><th>Label</th><th>Samples</th><th></th></tr></thead>
            <tbody>
              {meta.classes.map((c, i) => (
                <tr key={i}>
                  <td><code>{i}</code></td>
                  <td>{c}</td>
                  <td>{samplesByClass[i]}</td>
                  <td><button className="secondary" onClick={() => removeClass(i)}>Remove</button></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>Add samples</h3>
        <p className="muted">
          Image size: {meta.sizeX}×{meta.sizeY} · {meta.colored ? 'RGB' : 'grayscale'}.
          Uploaded images are resized to these dimensions automatically.
        </p>
        <div className="grid-2">
          <div>
            <label>Assign new samples to</label>
            <select value={selectedClass}
              onChange={e => setSelectedClass(parseInt(e.target.value))}
              disabled={meta.classes.length === 0}>
              {meta.classes.map((c, i) => (
                <option key={i} value={i}>{c}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Upload image(s)</label>
            <input type="file" accept="image/*" multiple onChange={e => onUpload(e.target.files)} />
          </div>
        </div>
        <div className="mt-2">
          <label>…or draw one</label>
          <DrawPanel
            width={meta.sizeX} height={meta.sizeY} colored={meta.colored}
            panelRef={drawRef}
          />
          <div className="flex mt-1">
            <button onClick={addDrawnSample} disabled={meta.classes.length === 0}>
              Add drawn sample
            </button>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Save to backend</h3>
        <p className="muted">
          Encodes samples as a feedforward corpus ({featureDim(meta)} input dims, one-hot targets) and
          writes it to the engine so training can pick it up.
          {' '}
          {samplesReady
            ? `${samples.length} sample(s) ready.`
            : 'Loading samples…'}
        </p>
        <button onClick={saveCorpusToBackend} disabled={!samplesReady}>Save corpus</button>
        {status && <div className="status mt-1">{status}</div>}
        {error && <div className="status error mt-1">{error}</div>}
      </div>
    </>
  )
}

// ─── Inference UI ────────────────────────────────────────────────────────────

function InferenceUI({ network, context }: NetworkTypeRenderProps) {
  const [meta, setMetaState] = useState<ImageClassMeta>(() => loadMeta(context, network.id))
  const [mode, setMode] = useState<'draw' | 'upload'>('draw')
  const [realtime, setRealtime] = useState(false)
  const [showViz, setShowViz] = useState(true)
  const [output, setOutput] = useState<number[] | null>(null)
  const [activations, setActivations] = useState<{ names: string[]; data: number[][] } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const drawRef = useRef<DrawPanelHandle | null>(null)
  const pendingRef = useRef(false)
  const queuedRef = useRef<number[] | null>(null)

  useEffect(() => {
    setMetaState(loadMeta(context, network.id))
    setOutput(null); setActivations(null); setError(null)
  }, [network.id])

  const runOnPixels = useCallback(async (pixels: number[]) => {
    setError(null)
    if (pixels.length !== featureDim(meta)) {
      setError(`expected ${featureDim(meta)} pixels, got ${pixels.length}`); return
    }
    if (pendingRef.current) { queuedRef.current = pixels; return }
    pendingRef.current = true
    try {
      const res = await inference.runWithActivations(network.id, pixels)
      setActivations({ names: res.layer_names, data: res.activations })
      setOutput(res.activations[res.activations.length - 1])
    } catch (e) {
      setError(String(e))
    } finally {
      pendingRef.current = false
      if (queuedRef.current) {
        const next = queuedRef.current
        queuedRef.current = null
        void runOnPixels(next)
      }
    }
  }, [network.id, meta])

  const handleDrawChange = () => {
    if (!realtime) return
    const pixels = drawRef.current?.getPixels()
    if (pixels) void runOnPixels(pixels)
  }

  const inferDrawn = () => {
    const pixels = drawRef.current?.getPixels()
    if (pixels) void runOnPixels(pixels)
  }

  const onUpload = async (file: File | null) => {
    if (!file) return
    setError(null)
    try {
      const url = URL.createObjectURL(file)
      const img = new Image()
      await new Promise<void>((res, rej) => {
        img.onload = () => res(); img.onerror = () => rej(new Error('load')); img.src = url
      })
      const pixels = pixelsFromImage(img, meta.sizeX, meta.sizeY, meta.colored)
      URL.revokeObjectURL(url)
      await runOnPixels(pixels)
    } catch (e) { setError(String(e)) }
  }

  const best = output ? output.indexOf(Math.max(...output)) : -1

  return (
    <>
      <div className="card">
        <h3>Input</h3>
        <div className="flex" style={{ gap: 8, marginBottom: 12 }}>
          <button className={mode === 'draw' ? '' : 'secondary'} onClick={() => setMode('draw')}>Draw</button>
          <button className={mode === 'upload' ? '' : 'secondary'} onClick={() => setMode('upload')}>Upload</button>
          <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={realtime} onChange={e => setRealtime(e.target.checked)} />
            Real-time inference
          </label>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={showViz} onChange={e => setShowViz(e.target.checked)} />
            Show network
          </label>
        </div>
        {mode === 'draw' ? (
          <>
            <DrawPanel
              width={meta.sizeX} height={meta.sizeY} colored={meta.colored}
              panelRef={drawRef}
              onChange={handleDrawChange}
            />
            <div className="flex mt-1">
              <button onClick={inferDrawn}>Predict</button>
            </div>
          </>
        ) : (
          <input type="file" accept="image/*" onChange={e => onUpload(e.target.files?.[0] ?? null)} />
        )}
        {error && <div className="status error mt-1">{error}</div>}
      </div>

      {output && (
        <div className="card">
          <h3>Prediction</h3>
          {best >= 0 && meta.classes[best] !== undefined && (
            <p>Top class: <strong>{meta.classes[best]}</strong> ({(output[best] * 100).toFixed(2)}%)</p>
          )}
          <table>
            <thead><tr><th>Class</th><th>Confidence</th></tr></thead>
            <tbody>
              {output.map((v, i) => (
                <tr key={i}>
                  <td>{meta.classes[i] ?? <em>class {i}</em>}</td>
                  <td style={{ fontVariantNumeric: 'tabular-nums' }}>{(v * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showViz && activations && (
        <div className="card">
          <h3>Network</h3>
          <p className="muted">Activations after the most recent forward pass.</p>
          <NetworkViz
            layerNames={activations.names}
            activations={activations.data}
            outputLabels={meta.classes}
          />
        </div>
      )}
    </>
  )
}

// ─── Plugin descriptor ───────────────────────────────────────────────────────

const networkType: NetworkTypeDescriptor = {
  id: 'image-classification',
  label: 'Image Classification',
  description: 'Classify images from upload or a drawing panel. Backed by a feed-forward MLP.',
  CreateForm,
  CorpusUI,
  InferenceUI,
}

const plugin: NeuralCabinPlugin = {
  id: 'neuralcabin.image-classification',
  name: 'Image Classification',
  version: '0.1.0',
  author: 'NeuralCabin',
  description: 'Out-of-the-box plugin: train an MLP to classify images you upload or draw.',
  networkTypes: [networkType],
}

export default plugin
