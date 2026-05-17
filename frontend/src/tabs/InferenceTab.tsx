import { useEffect, useRef, useState } from 'react'
import { type UnlistenFn } from '@tauri-apps/api/event'
import { inference, vocabulary, corpus, Network, InferenceToken, CorpusStats } from '../api'
import type { TabProps } from '../App'
import NetworkViz from '../components/NetworkViz'
import PluginErrorBoundary from '../components/PluginErrorBoundary'

export default function InferenceTab({ network, pluginRegistry }: TabProps) {
  const [error, setError] = useState<string | null>(null)
  const [corpusStats, setCorpusStats] = useState<CorpusStats | null>(null)

  // Plugin-managed networks own their own inference UI.
  const pluginType = network && pluginRegistry?.typeForNetwork(network.id)
  if (network && pluginType?.type.InferenceUI) {
    const PluginInference = pluginType.type.InferenceUI
    return (
      <div className="tab-content">
        <h2>Inference</h2>
        <p className="muted">
          Managed by plugin <strong>{pluginType.plugin.name}</strong> · type <strong>{pluginType.type.label}</strong>.
        </p>
        <PluginErrorBoundary
          key={`${pluginType.plugin.id}:${network.id}`}
          fallbackTitle={`${pluginType.plugin.name} inference UI crashed`}>
          <PluginInference network={network} context={pluginRegistry!.context} />
        </PluginErrorBoundary>
      </div>
    )
  }

  useEffect(() => {
    if (!network) {
      setCorpusStats(null)
      return
    }
    const fetch = async () => {
      try {
        const stats = await corpus.stats(network.id)
        setCorpusStats(stats)
      } catch (e) {
        setCorpusStats(null)
      }
    }
    void fetch()
  }, [network?.id])

  const isFinetuned = network?.kind === 'next_token' && corpusStats?.stage === 'finetune'

  return (
    <div className="tab-content">
      <h2>Inference</h2>
      <p className="muted">
        {isFinetuned ? 'Chat with a fine-tuned model.' : 'Run a trained network on new inputs.'}
      </p>

      {error && <div className="status error">{error}</div>}

      {network && !network.trained && (
        <div className="status mt-1">
          This network hasn't been trained yet — predictions will be random.
        </div>
      )}

      {!network ? (
        <div className="card"><p className="muted">Select a network to run inference.</p></div>
      ) : network.kind === 'feedforward' ? (
        <FeedforwardInference network={network} onError={setError} />
      ) : isFinetuned ? (
        <ChatInference network={network} onError={setError} />
      ) : (
        <NextTokenInference network={network} onError={setError} />
      )}
    </div>
  )
}

// ─── Chat inference (fine-tuned models) ───────────────────────────────────────

type ChatMode = 'simple' | 'multiturn'
interface Message { role: 'user' | 'assistant'; text: string }

function ChatInference({ network, onError }: {
  network: Network; onError: (e: string | null) => void
}) {
  const [mode, setMode] = useState<ChatMode>('simple')
  const [maxNewTokens, setMaxNewTokens] = useState(128)
  const [temperature, setTemperature] = useState(0.7)

  return (
    <>
      <div className="card">
        <h3>Mode</h3>
        <div className="flex" style={{ gap: 8 }}>
          <button
            className={mode === 'simple' ? '' : 'secondary'}
            onClick={() => setMode('simple')}
          >
            Simple Q&A
          </button>
          <button
            className={mode === 'multiturn' ? '' : 'secondary'}
            onClick={() => setMode('multiturn')}
          >
            Multi-turn Chat
          </button>
        </div>
      </div>

      {mode === 'simple' ? (
        <SimpleChat network={network} maxTokens={maxNewTokens} temp={temperature} onError={onError} />
      ) : (
        <MultiTurnChat network={network} maxTokens={maxNewTokens} temp={temperature} onError={onError} />
      )}

      <div className="card">
        <h3>Generation settings</h3>
        <div className="grid-2">
          <div>
            <label>Max new tokens</label>
            <input type="number" min={1} max={2048} value={maxNewTokens}
              onChange={e => setMaxNewTokens(Math.max(1, Math.min(2048, parseInt(e.target.value) || 128)))} />
          </div>
          <div>
            <label>Temperature</label>
            <input type="number" step="0.1" min={0} max={2} value={temperature}
              onChange={e => setTemperature(Math.max(0, parseFloat(e.target.value) || 0.7))} />
            <small>0 = greedy. 0.5-1.5 = creative.</small>
          </div>
        </div>
      </div>
    </>
  )
}

const UNK_ID = 1

function SimpleChat({ network, maxTokens, temp, onError }: {
  network: Network; maxTokens: number; temp: number; onError: (e: string | null) => void
}) {
  const [userMessage, setUserMessage] = useState('')
  const [assistantMessage, setAssistantMessage] = useState<string | null>(null)
  const [busy, setBusy] = useState(false)
  const [inferenceId, setInferenceId] = useState<string | null>(null)
  const activeIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)

  useEffect(() => {
    setUserMessage(''); setAssistantMessage(null); setInferenceId(null)
    activeIdRef.current = null
    busyRef.current = false
    setBusy(false)
  }, [network.id])

  useEffect(() => {
    let cancelled = false
    const cleanups: UnlistenFn[] = []
    const setup = async () => {
      const u1 = await inference.onToken(t => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && t.inference_id !== activeIdRef.current) return
        setAssistantMessage(prev => (prev || '') + t.token)
      })
      const u2 = await inference.onFinished(r => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && r.inference_id !== activeIdRef.current) return
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      const u3 = await inference.onError(err => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && err.inference_id !== activeIdRef.current) return
        onError(err.message)
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      if (cancelled) { u1(); u2(); u3(); return }
      cleanups.push(u1, u2, u3)
    }
    void setup()
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [network.id])

  const generate = async () => {
    if (!userMessage.trim()) return
    onError(null)
    setAssistantMessage('')
    busyRef.current = true
    activeIdRef.current = null
    setBusy(true)
    try {
      const toks = await vocabulary.tokenize(network.id, userMessage)
      const unknown = toks.filter(([id]) => id === UNK_ID)
      if (unknown.length > 0) {
        const sample = unknown.slice(0, 3).map(([, s]) => JSON.stringify(s)).join(', ')
        throw new Error(`Input contains unknown tokens (e.g. ${sample}).`)
      }
      const r = await inference.run({
        network_id: network.id,
        prompt: userMessage,
        max_new_tokens: maxTokens,
        temperature: temp,
      })
      const id = r.inference_id ?? null
      activeIdRef.current = id
      setInferenceId(id)
      if (!id) {
        busyRef.current = false
        setBusy(false)
      }
    } catch (e) {
      busyRef.current = false
      activeIdRef.current = null
      setBusy(false)
      onError(String(e))
    }
  }

  const stop = async () => {
    const id = activeIdRef.current ?? inferenceId
    if (!id) {
      busyRef.current = false
      setBusy(false)
      setInferenceId(null)
      return
    }
    try {
      await inference.stop(id)
    } catch (e) {
      onError(String(e))
    }
    setTimeout(() => {
      if (busyRef.current && activeIdRef.current === id) {
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      }
    }, 2000)
  }

  const reset = () => {
    setUserMessage('')
    setAssistantMessage(null)
    busyRef.current = false
    setBusy(false)
    setInferenceId(null)
    activeIdRef.current = null
    onError(null)
  }

  const regenerate = async () => {
    if (!userMessage.trim()) return
    onError(null)
    setAssistantMessage('')
    busyRef.current = true
    activeIdRef.current = null
    setBusy(true)
    try {
      const r = await inference.run({
        network_id: network.id,
        prompt: userMessage,
        max_new_tokens: maxTokens,
        temperature: temp,
      })
      const id = r.inference_id ?? null
      activeIdRef.current = id
      setInferenceId(id)
      if (!id) {
        busyRef.current = false
        setBusy(false)
      }
    } catch (e) {
      busyRef.current = false
      activeIdRef.current = null
      setBusy(false)
      onError(String(e))
    }
  }

  return (
    <div className="card">
      <h3>Message</h3>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
        <div>
          <label style={{ display: 'block', marginBottom: 6 }}>Your question</label>
          <textarea
            rows={3}
            value={userMessage}
            onChange={e => setUserMessage(e.target.value)}
            disabled={busy}
            placeholder="Ask the model something..."
          />
        </div>
        <div className="flex" style={{ gap: 8 }}>
          <button onClick={generate} disabled={busy || !userMessage.trim()}>
            {busy ? 'Generating...' : 'Generate'}
          </button>
          {busy && <button className="secondary" onClick={stop}>Stop</button>}
          {assistantMessage && <button className="secondary" onClick={regenerate} disabled={busy}>Regenerate</button>}
          {assistantMessage && <button className="secondary" onClick={reset}>Reset</button>}
        </div>
      </div>

      {assistantMessage && (
        <div style={{ marginTop: 20 }}>
          <h4 style={{ marginBottom: 8 }}>Response</h4>
          <div style={{
            background: 'var(--bg-input)', padding: 12, borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', whiteSpace: 'pre-wrap',
            fontFamily: 'var(--font-mono)', fontSize: 13,
          }}>
            {assistantMessage}
          </div>
        </div>
      )}
    </div>
  )
}

function MultiTurnChat({ network, maxTokens, temp, onError }: {
  network: Network; maxTokens: number; temp: number; onError: (e: string | null) => void
}) {
  const [messages, setMessages] = useState<Message[]>([])
  const [userInput, setUserInput] = useState('')
  const [busy, setBusy] = useState(false)
  const [inferenceId, setInferenceId] = useState<string | null>(null)
  const activeIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)

  useEffect(() => {
    setMessages([]); setUserInput(''); setInferenceId(null)
    activeIdRef.current = null
    busyRef.current = false
    setBusy(false)
  }, [network.id])

  useEffect(() => {
    let cancelled = false
    const cleanups: UnlistenFn[] = []
    const setup = async () => {
      const u1 = await inference.onToken(t => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && t.inference_id !== activeIdRef.current) return
        setMessages(prev => {
          const last = prev[prev.length - 1]
          if (last && last.role === 'assistant') {
            return [...prev.slice(0, -1), { ...last, text: last.text + t.token }]
          }
          return prev
        })
      })
      const u2 = await inference.onFinished(r => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && r.inference_id !== activeIdRef.current) return
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      const u3 = await inference.onError(err => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && err.inference_id !== activeIdRef.current) return
        onError(err.message)
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      if (cancelled) { u1(); u2(); u3(); return }
      cleanups.push(u1, u2, u3)
    }
    void setup()
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [network.id])

  const send = async () => {
    if (!userInput.trim()) return
    onError(null)
    const newMessages = [...messages, { role: 'user' as const, text: userInput }]
    setMessages(newMessages)
    setUserInput('')
    busyRef.current = true
    activeIdRef.current = null
    setBusy(true)

    try {
      const toks = await vocabulary.tokenize(network.id, userInput)
      const unknown = toks.filter(([id]) => id === UNK_ID)
      if (unknown.length > 0) {
        const sample = unknown.slice(0, 3).map(([, s]) => JSON.stringify(s)).join(', ')
        throw new Error(`Input contains unknown tokens (e.g. ${sample}).`)
      }

      // Build the full conversation history so the model gets actual
      // multi-turn context. `newMessages` already includes the just-typed
      // user message; the empty assistant placeholder is for streaming UI
      // only and must NOT be sent to the backend.
      const history = newMessages.map(m => ({ role: m.role, text: m.text }))
      newMessages.push({ role: 'assistant', text: '' })
      setMessages([...newMessages])

      const r = await inference.run({
        network_id: network.id,
        messages: history,
        max_new_tokens: maxTokens,
        temperature: temp,
      })
      const id = r.inference_id ?? null
      activeIdRef.current = id
      setInferenceId(id)
      if (!id) {
        busyRef.current = false
        setBusy(false)
      }
    } catch (e) {
      busyRef.current = false
      activeIdRef.current = null
      setBusy(false)
      onError(String(e))
    }
  }

  const stop = async () => {
    const id = activeIdRef.current ?? inferenceId
    if (!id) {
      busyRef.current = false
      setBusy(false)
      setInferenceId(null)
      return
    }
    try {
      await inference.stop(id)
    } catch (e) {
      onError(String(e))
    }
    setTimeout(() => {
      if (busyRef.current && activeIdRef.current === id) {
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      }
    }, 2000)
  }

  const deleteMessage = (index: number) => {
    setMessages(messages.filter((_, i) => i !== index))
  }

  const reset = () => {
    setMessages([])
    setUserInput('')
    busyRef.current = false
    setBusy(false)
    setInferenceId(null)
    activeIdRef.current = null
    onError(null)
  }

  return (
    <div className="card">
      <h3>Conversation</h3>
      <div style={{
        background: 'var(--bg-elev-1)', padding: 12, borderRadius: 'var(--radius)',
        border: '1px solid var(--border)', minHeight: 300, maxHeight: 400, overflow: 'auto',
        marginBottom: 12, display: 'flex', flexDirection: 'column', gap: 8,
      }}>
        {messages.length === 0 ? (
          <div className="muted" style={{ margin: 'auto' }}>No messages yet. Start a conversation!</div>
        ) : (
          messages.map((msg, i) => (
            <div key={i} style={{
              display: 'flex', justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
            }}>
              <div style={{
                maxWidth: '80%', padding: '8px 12px', borderRadius: 'var(--radius)',
                background: msg.role === 'user' ? 'var(--accent)' : 'var(--bg-elev-2)',
                color: msg.role === 'user' ? '#fff' : 'var(--text)',
                wordBreak: 'break-word',
              }}>
                <div style={{ whiteSpace: 'pre-wrap', fontSize: 13, lineHeight: 1.5 }}>
                  {msg.text}
                </div>
                {msg.role === 'assistant' && (
                  <button
                    className="secondary"
                    onClick={() => deleteMessage(i)}
                    style={{ marginTop: 4, padding: '2px 6px', fontSize: 11 }}
                  >
                    Delete
                  </button>
                )}
              </div>
            </div>
          ))
        )}
        {busy && (
          <div style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
            <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>Generating...</span>
            <span style={{ animation: 'pulse 1.5s infinite', color: 'var(--accent)' }}>●</span>
          </div>
        )}
      </div>

      <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
        <input
          type="text"
          value={userInput}
          onChange={e => setUserInput(e.target.value)}
          onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); void send() } }}
          disabled={busy}
          placeholder="Type a message (Enter to send)..."
          style={{ flex: 1 }}
        />
        <button onClick={send} disabled={busy || !userInput.trim()}>
          {busy ? 'Sending...' : 'Send'}
        </button>
        {busy && <button className="secondary" onClick={stop}>Stop</button>}
      </div>

      {messages.length > 0 && (
        <button className="secondary" onClick={reset} style={{ width: '100%' }}>
          Reset conversation
        </button>
      )}
    </div>
  )
}

// ─── Feed-forward inference ─────────────────────────────────────────────────

function FeedforwardInference({ network, onError }: {
  network: Network; onError: (e: string | null) => void
}) {
  const [inputs, setInputs] = useState<string[]>(() => Array(network.input_dim).fill('0'))
  const [output, setOutput] = useState<number[] | null>(null)
  const [activations, setActivations] = useState<{ names: string[]; data: number[][] } | null>(null)
  const [showViz, setShowViz] = useState(true)
  const [busy, setBusy] = useState(false)

  useEffect(() => {
    setInputs(Array(network.input_dim).fill('0')); setOutput(null); setActivations(null)
  }, [network.id, network.input_dim])

  const setVal = (i: number, v: string) =>
    setInputs(prev => prev.map((x, idx) => idx === i ? v : x))

  const run = async () => {
    onError(null)
    const features = inputs.map(s => Number(s))
    if (features.some(n => !Number.isFinite(n))) {
      onError('All inputs must be valid numbers.')
      return
    }
    setBusy(true)
    try {
      const r = await inference.runWithActivations(network.id, features)
      setActivations({ names: r.layer_names, data: r.activations })
      setOutput(r.activations[r.activations.length - 1] ?? null)
    } catch (e) { onError(String(e)) }
    finally { setBusy(false) }
  }

  return (
    <>
      <div className="card">
        <h3>Inputs</h3>
        <div className="grid-3">
          {inputs.map((v, i) => (
            <div key={i}>
              <label>x{i}</label>
              <input value={v} onChange={e => setVal(i, e.target.value)} />
            </div>
          ))}
        </div>
        <div className="flex mt-2" style={{ gap: 12, alignItems: 'center' }}>
          <button onClick={run} disabled={busy}>
            {busy ? 'Running...' : 'Predict'}
          </button>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input type="checkbox" checked={showViz} onChange={e => setShowViz(e.target.checked)} />
            Show network
          </label>
        </div>
      </div>

      {output && (
        <div className="card">
          <h3>Output</h3>
          <table>
            <thead><tr><th>Index</th><th>Value</th></tr></thead>
            <tbody>
              {output.map((v, i) => (
                <tr key={i}><td><code>y{i}</code></td><td style={{ fontVariantNumeric: 'tabular-nums' }}>{v.toFixed(6)}</td></tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {showViz && activations && (
        <div className="card">
          <h3>Network</h3>
          <p className="muted">Neuron activations after the most recent forward pass.</p>
          <NetworkViz
            layerNames={activations.names}
            activations={activations.data}
            inputLabels={inputs.map((_, i) => `x${i}`)}
            outputLabels={output ? output.map((_, i) => `y${i}`) : undefined}
          />
        </div>
      )}
    </>
  )
}

// ─── Next-token inference (streaming) ───────────────────────────────────────

function NextTokenInference({ network, onError }: {
  network: Network; onError: (e: string | null) => void
}) {
  const [prompt, setPrompt] = useState<string>('')
  const [maxNewTokens, setMaxNewTokens] = useState(64)
  const [temperature, setTemperature] = useState(0.0)
  const [generated, setGenerated] = useState<string>('')
  const [tokens, setTokens] = useState<InferenceToken[]>([])
  const [busy, setBusy] = useState(false)
  const [inferenceId, setInferenceId] = useState<string | null>(null)
  const activeIdRef = useRef<string | null>(null)
  const busyRef = useRef(false)

  useEffect(() => {
    setGenerated(''); setTokens([]); setInferenceId(null)
    activeIdRef.current = null
    busyRef.current = false
    setBusy(false)
  }, [network.id])

  useEffect(() => {
    let cancelled = false
    const cleanups: UnlistenFn[] = []
    const setup = async () => {
      const u1 = await inference.onToken(t => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && t.inference_id !== activeIdRef.current) return
        setTokens(prev => [...prev, t])
        setGenerated(prev => prev + t.token)
      })
      const u2 = await inference.onFinished(r => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && r.inference_id !== activeIdRef.current) return
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      const u3 = await inference.onError(err => {
        if (cancelled || !busyRef.current) return
        if (activeIdRef.current && err.inference_id !== activeIdRef.current) return
        onError(err.message)
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      })
      if (cancelled) { u1(); u2(); u3(); return }
      cleanups.push(u1, u2, u3)
    }
    void setup()
    return () => { cancelled = true; cleanups.forEach(c => c()) }
  }, [network.id])

  const run = async () => {
    onError(null)
    setGenerated(''); setTokens([])
    busyRef.current = true
    activeIdRef.current = null
    setBusy(true)
    try {
      if (prompt) {
        const toks = await vocabulary.tokenize(network.id, prompt)
        const unknown = toks.filter(([id]) => id === UNK_ID)
        if (unknown.length > 0) {
          const sample = unknown.slice(0, 5).map(([, s]) => JSON.stringify(s)).join(', ')
          throw new Error(
            `Prompt contains ${unknown.length} token(s) not in the trained vocabulary` +
            ` (e.g. ${sample}). Train on text that includes these characters first,` +
            ` or remove them from the prompt.`
          )
        }
      }
      const r = await inference.run({
        network_id: network.id,
        prompt,
        max_new_tokens: maxNewTokens,
        temperature,
      })
      const id = r.inference_id ?? null
      activeIdRef.current = id
      setInferenceId(id)
      if (!id) {
        busyRef.current = false
        setBusy(false)
      }
    } catch (e) {
      busyRef.current = false
      activeIdRef.current = null
      setBusy(false)
      onError(String(e))
    }
  }

  const stop = async () => {
    const id = activeIdRef.current ?? inferenceId
    if (!id) {
      busyRef.current = false
      setBusy(false)
      setInferenceId(null)
      return
    }
    try {
      await inference.stop(id)
    } catch (e) {
      onError(String(e))
    }
    setTimeout(() => {
      if (busyRef.current && activeIdRef.current === id) {
        busyRef.current = false
        activeIdRef.current = null
        setBusy(false)
        setInferenceId(null)
      }
    }, 2000)
  }

  return (
    <>
      <div className="card">
        <h3>Prompt</h3>
        <textarea
          rows={4}
          value={prompt}
          onChange={e => setPrompt(e.target.value)}
          placeholder="Type a prompt to continue..."
        />
        <div className="grid-3 mt-2">
          <div>
            <label>Max new tokens</label>
            <input type="number" min={1} max={2048} value={maxNewTokens}
              onChange={e => setMaxNewTokens(Math.max(1, Math.min(2048, parseInt(e.target.value) || 1)))} />
          </div>
          <div>
            <label>Temperature</label>
            <input type="number" step="0.1" min={0} value={temperature}
              onChange={e => setTemperature(Math.max(0, parseFloat(e.target.value) || 0))} />
            <small>0 = greedy / argmax. Higher = more random.</small>
          </div>
        </div>
        <div className="flex mt-2">
          <button onClick={run} disabled={busy}>{busy ? 'Generating...' : 'Generate'}</button>
          {busy && (
            <button className="secondary" onClick={stop}>
              Stop
            </button>
          )}
        </div>
      </div>

      {generated && (
        <div className="card">
          <h3>Output</h3>
          <div style={{
            background: 'var(--bg-input)', padding: 14, borderRadius: 'var(--radius)',
            border: '1px solid var(--border)', whiteSpace: 'pre-wrap',
            fontFamily: 'var(--font-mono)', fontSize: 13,
          }}>
            <span className="muted">{prompt}</span>
            <span style={{ color: 'var(--accent)' }}>{generated}</span>
          </div>
        </div>
      )}

      {tokens.length > 0 && (
        <div className="card">
          <h3>Per-token probabilities</h3>
          <div style={{ maxHeight: 260, overflow: 'auto' }}>
            <table>
              <thead><tr><th style={{ width: 40 }}>#</th><th>Token</th><th style={{ width: 120 }}>Probability</th></tr></thead>
              <tbody>
                {tokens.map((s, i) => (
                  <tr key={i}>
                    <td><code>{i + 1}</code></td>
                    <td><code>{visualize(s.token)}</code></td>
                    <td style={{ fontVariantNumeric: 'tabular-nums' }}>{(s.probability * 100).toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </>
  )
}

function visualize(t: string): string {
  if (t === '\n') return '\\n'
  if (t === '\t') return '\\t'
  if (t === ' ')  return '·'
  return t
}
