// ─── Host abstraction ─────────────────────────────────────────────────────────
//
// NeuralCabin's UI talks to its backend through exactly two primitives:
//   • invoke(cmd, args)  — call a backend command, await a typed result
//   • listen(event, fn)  — subscribe to a backend-pushed event stream
//
// Historically these were Tauri's IPC functions. As part of the Tauri → .NET
// migration this module makes the UI host-agnostic: it transparently routes to
// Tauri when running inside the Tauri shell, or to the native .NET (Photino)
// shell's message bridge otherwise. The rest of the frontend imports `invoke`
// and `listen` from here and is unaware of which backend answered.
//
// Wire protocol with the .NET host (one JSON string per message, matching
// NeuralCabin.Host/Ipc/IpcMessages.cs):
//   UI  → host:  { kind:'invoke',   id, cmd, args }
//   host → UI:   { kind:'response', id, ok, result | error }
//   host → UI:   { kind:'event',    event, payload }

import { invoke as tauriInvoke } from '@tauri-apps/api/core'
import { listen as tauriListen, type UnlistenFn } from '@tauri-apps/api/event'

export type { UnlistenFn }

/** Event shape delivered to `listen` handlers (subset of Tauri's `Event`). */
export interface HostEvent<T> {
  payload: T
}

// ─── Host detection ─────────────────────────────────────────────────────────

type HostKind = 'tauri' | 'dotnet'

interface DotnetExternal {
  sendMessage(message: string): void
  receiveMessage(callback: (message: string) => void): void
}

interface HostWindow {
  __NEURALCABIN_HOST__?: HostKind
  __TAURI_INTERNALS__?: unknown
  external?: Partial<DotnetExternal>
}

function hostWindow(): HostWindow {
  return window as unknown as HostWindow
}

let cachedHost: HostKind | null = null

function detectHost(): HostKind {
  if (cachedHost) return cachedHost

  const w = hostWindow()

  // An explicit marker (set by a host that wants to force a mode) wins.
  if (w.__NEURALCABIN_HOST__ === 'dotnet' || w.__NEURALCABIN_HOST__ === 'tauri') {
    cachedHost = w.__NEURALCABIN_HOST__
    return cachedHost
  }

  // Photino injects window.external.{sendMessage,receiveMessage}; Tauri injects
  // window.__TAURI_INTERNALS__. Prefer Tauri if present, else detect the .NET
  // bridge, else fall back to Tauri (preserving pre-migration behavior).
  const ext = w.external
  const looksDotnet =
    !w.__TAURI_INTERNALS__ &&
    !!ext &&
    typeof ext.sendMessage === 'function' &&
    typeof ext.receiveMessage === 'function'

  cachedHost = looksDotnet ? 'dotnet' : 'tauri'
  return cachedHost
}

// ─── .NET (Photino) bridge ────────────────────────────────────────────────────

interface ResponseMessage {
  kind: 'response'
  id: string
  ok: boolean
  result?: unknown
  error?: string
}

interface EventMessage {
  kind: 'event'
  event: string
  payload?: unknown
}

interface Pending {
  resolve: (value: unknown) => void
  reject: (error: Error) => void
}

const pending = new Map<string, Pending>()
const listeners = new Map<string, Set<(payload: unknown) => void>>()
let bridgeInitialized = false
let invokeCounter = 0

function bridge(): DotnetExternal {
  const ext = hostWindow().external
  if (!ext || typeof ext.sendMessage !== 'function' || typeof ext.receiveMessage !== 'function') {
    throw new Error('NeuralCabin .NET host bridge is unavailable')
  }
  const ready = ext as DotnetExternal
  if (!bridgeInitialized) {
    bridgeInitialized = true
    ready.receiveMessage(dispatch)
  }
  return ready
}

function dispatch(raw: string): void {
  let message: ResponseMessage | EventMessage
  try {
    message = JSON.parse(raw) as ResponseMessage | EventMessage
  } catch {
    return
  }

  if (message.kind === 'response') {
    const waiter = pending.get(message.id)
    if (!waiter) return
    pending.delete(message.id)
    if (message.ok) waiter.resolve(message.result)
    else waiter.reject(new Error(message.error ?? 'invoke failed'))
    return
  }

  if (message.kind === 'event') {
    const set = listeners.get(message.event)
    if (!set) return
    for (const handler of set) handler(message.payload)
  }
}

function nextInvokeId(): string {
  invokeCounter += 1
  return `${Date.now().toString(36)}-${invokeCounter.toString(36)}-${Math.random().toString(36).slice(2, 8)}`
}

function dotnetInvoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  const ext = bridge()
  const id = nextInvokeId()
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: (value) => resolve(value as T), reject })
    ext.sendMessage(JSON.stringify({ kind: 'invoke', id, cmd, args: args ?? {} }))
  })
}

function dotnetListen<T>(event: string, handler: (e: HostEvent<T>) => void): Promise<UnlistenFn> {
  bridge()
  let set = listeners.get(event)
  if (!set) {
    set = new Set()
    listeners.set(event, set)
  }
  const wrapped = (payload: unknown) => handler({ payload: payload as T })
  set.add(wrapped)
  const unlisten: UnlistenFn = () => {
    set?.delete(wrapped)
  }
  return Promise.resolve(unlisten)
}

// ─── Public, host-agnostic API ────────────────────────────────────────────────

/** Call a backend command and await its typed result. */
export function invoke<T>(cmd: string, args?: Record<string, unknown>): Promise<T> {
  return detectHost() === 'dotnet' ? dotnetInvoke<T>(cmd, args) : tauriInvoke<T>(cmd, args)
}

/** Subscribe to a backend event stream; resolves to an unsubscribe function. */
export function listen<T>(event: string, handler: (e: HostEvent<T>) => void): Promise<UnlistenFn> {
  if (detectHost() === 'dotnet') return dotnetListen<T>(event, handler)
  return tauriListen<T>(event, (e) => handler({ payload: e.payload }))
}
