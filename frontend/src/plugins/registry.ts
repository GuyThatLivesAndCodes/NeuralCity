import { useEffect, useState, useCallback } from 'react'
import type { InstalledPlugin, NeuralCabinPlugin, NetworkTypeDescriptor, PluginContext } from './types'
import imageClassificationPlugin from './builtins/imageClassification'

// ─── Built-in plugins ────────────────────────────────────────────────────────

const BUILTIN_PLUGINS: NeuralCabinPlugin[] = [
  imageClassificationPlugin,
]

// ─── Persistent state ────────────────────────────────────────────────────────

const LS_INSTALLED = 'neuralcabin.plugins.installed'
const LS_NETWORK_TAGS = 'neuralcabin.plugins.networkTags'
const LS_META_PREFIX = 'neuralcabin.plugins.meta.'

function loadInstalled(): InstalledPlugin[] {
  try {
    const raw = localStorage.getItem(LS_INSTALLED)
    const parsed = raw ? (JSON.parse(raw) as InstalledPlugin[]) : []
    // Always reconcile built-ins so first-boot users see them.
    const byId = new Map(parsed.map(p => [p.id, p]))
    for (const b of BUILTIN_PLUGINS) {
      if (!byId.has(b.id)) {
        byId.set(b.id, {
          id: b.id, name: b.name, version: b.version,
          source: 'builtin', enabled: true,
        })
      }
    }
    return Array.from(byId.values())
  } catch {
    return BUILTIN_PLUGINS.map(b => ({
      id: b.id, name: b.name, version: b.version,
      source: 'builtin' as const, enabled: true,
    }))
  }
}

function saveInstalled(items: InstalledPlugin[]) {
  localStorage.setItem(LS_INSTALLED, JSON.stringify(items))
}

function loadNetworkTags(): Record<string, string> {
  try { return JSON.parse(localStorage.getItem(LS_NETWORK_TAGS) || '{}') } catch { return {} }
}
function saveNetworkTags(tags: Record<string, string>) {
  localStorage.setItem(LS_NETWORK_TAGS, JSON.stringify(tags))
}

// ─── Dynamic plugin loading ──────────────────────────────────────────────────
//
// Uploaded plugins are JS modules. We turn the source code into a Blob URL
// and `import()` it; the module's default export must be a NeuralCabinPlugin.
// Plugins run in the same JS realm as the host (no sandbox) — installing a
// plugin is therefore a trust decision the user makes explicitly. This is
// called out in the Plugins tab.

const loadedModules = new Map<string, NeuralCabinPlugin>()
for (const b of BUILTIN_PLUGINS) loadedModules.set(b.id, b)

export async function loadUploadedPlugin(source: string): Promise<NeuralCabinPlugin> {
  const blob = new Blob([source], { type: 'text/javascript' })
  const url = URL.createObjectURL(blob)
  try {
    const mod = await import(/* @vite-ignore */ url)
    const plugin: NeuralCabinPlugin = mod.default ?? mod.plugin
    if (!plugin?.id || !Array.isArray(plugin.networkTypes)) {
      throw new Error('Plugin module did not export a valid default export.')
    }
    loadedModules.set(plugin.id, plugin)
    return plugin
  } finally {
    // Revoke after a tick so the import() resolution can complete.
    setTimeout(() => URL.revokeObjectURL(url), 0)
  }
}

// ─── React hook ──────────────────────────────────────────────────────────────

export interface PluginRegistry {
  installed: InstalledPlugin[]
  /** Map of plugin id -> loaded plugin module. */
  modules: Map<string, NeuralCabinPlugin>
  /** All enabled network types, flattened across enabled plugins. */
  networkTypes: Array<{ plugin: NeuralCabinPlugin; type: NetworkTypeDescriptor }>
  /** Look up which plugin type a given network is tagged with, if any. */
  typeForNetwork: (networkId: string) => { plugin: NeuralCabinPlugin; type: NetworkTypeDescriptor } | null
  enable: (id: string, enabled: boolean) => void
  uninstall: (id: string) => void
  installFromSource: (source: string) => Promise<void>
  /** Used by tabs to mark/read plugin metadata for a network. */
  context: PluginContext
}

export function usePluginRegistry(refreshNetworks: () => Promise<void>): PluginRegistry {
  const [installed, setInstalled] = useState<InstalledPlugin[]>(() => loadInstalled())
  const [tags, setTags] = useState<Record<string, string>>(() => loadNetworkTags())
  const [, setBump] = useState(0)
  const bump = () => setBump(x => x + 1)

  // Re-hydrate uploaded plugins on mount.
  useEffect(() => {
    let cancelled = false
    ;(async () => {
      for (const p of installed) {
        if (p.source === 'uploaded' && p.source_code && !loadedModules.has(p.id)) {
          try { await loadUploadedPlugin(p.source_code) } catch (e) {
            console.warn('Failed to re-load uploaded plugin', p.id, e)
          }
        }
      }
      if (!cancelled) bump()
    })()
    return () => { cancelled = true }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  const update = (next: InstalledPlugin[]) => {
    setInstalled(next)
    saveInstalled(next)
  }

  const enable = (id: string, enabled: boolean) => {
    update(installed.map(p => p.id === id ? { ...p, enabled } : p))
  }

  const uninstall = (id: string) => {
    const plugin = installed.find(p => p.id === id)
    if (!plugin) return
    if (plugin.source === 'builtin') {
      // Built-ins can be disabled but not uninstalled.
      enable(id, false)
      return
    }
    update(installed.filter(p => p.id !== id))
    loadedModules.delete(id)
  }

  const installFromSource = async (source: string) => {
    const plugin = await loadUploadedPlugin(source)
    const entry: InstalledPlugin = {
      id: plugin.id,
      name: plugin.name,
      version: plugin.version,
      source: 'uploaded',
      enabled: true,
      source_code: source,
    }
    update([...installed.filter(p => p.id !== plugin.id), entry])
  }

  const context: PluginContext = {
    refreshNetworks,
    tagNetwork: (networkId, typeId) => {
      const next = { ...tags, [networkId]: typeId }
      setTags(next); saveNetworkTags(next)
    },
    getMeta: <T,>(networkId: string) => {
      try {
        const raw = localStorage.getItem(LS_META_PREFIX + networkId)
        return raw ? (JSON.parse(raw) as T) : null
      } catch { return null }
    },
    setMeta: (networkId, meta) => {
      localStorage.setItem(LS_META_PREFIX + networkId, JSON.stringify(meta))
    },
  }

  const typeForNetwork = useCallback((networkId: string) => {
    const typeId = tags[networkId]
    if (!typeId) return null
    for (const inst of installed) {
      if (!inst.enabled) continue
      const mod = loadedModules.get(inst.id)
      if (!mod) continue
      const type = mod.networkTypes.find(t => t.id === typeId)
      if (type) return { plugin: mod, type }
    }
    return null
  }, [tags, installed])

  const networkTypes: PluginRegistry['networkTypes'] = []
  for (const inst of installed) {
    if (!inst.enabled) continue
    const mod = loadedModules.get(inst.id)
    if (!mod) continue
    for (const t of mod.networkTypes) networkTypes.push({ plugin: mod, type: t })
  }

  return {
    installed,
    modules: loadedModules,
    networkTypes,
    typeForNetwork,
    enable,
    uninstall,
    installFromSource,
    context,
  }
}
