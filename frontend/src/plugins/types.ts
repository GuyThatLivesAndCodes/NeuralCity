import type { ComponentType } from 'react'
import type { Network } from '../api'

/**
 * NeuralCabin plugin API.
 *
 * Plugins are JS/TS modules whose default export is a `NeuralCabinPlugin`.
 * Each plugin contributes one or more "network types". A network type is a
 * higher-level abstraction over the engine's built-in network kinds: the
 * plugin owns the create form, the corpus UI, the training UI and the
 * inference UI for networks of that type. Under the hood the plugin usually
 * creates a normal feedforward or next-token network — the engine doesn't
 * need to know plugins exist.
 *
 * Plugin-managed networks are tagged in localStorage so the host knows which
 * plugin to route a network to once it's been created.
 */

export interface PluginContext {
  /** Refresh the network list in the header dropdown. */
  refreshNetworks: () => Promise<void>
  /** Mark a network as managed by this plugin's type. */
  tagNetwork: (networkId: string, typeId: string) => void
  /** Get/set arbitrary per-network plugin metadata (JSON-serializable). */
  getMeta: <T = any>(networkId: string) => T | null
  setMeta: <T = any>(networkId: string, meta: T) => void
}

export interface NetworkTypeRenderProps {
  network: Network
  context: PluginContext
}

export interface CreateFormProps {
  context: PluginContext
  /** Called by the form once the network has been successfully created. */
  onCreated: (networkId: string) => void
}

export interface NetworkTypeDescriptor {
  /** Stable id, scoped by plugin (e.g. "image-classification"). */
  id: string
  /** Display name shown in the Networks tab type selector. */
  label: string
  /** One-line summary shown next to the label. */
  description: string
  /** Custom create form rendered in the Networks tab when this type is selected. */
  CreateForm: ComponentType<CreateFormProps>
  /** Optional custom Corpus tab for this network type. */
  CorpusUI?: ComponentType<NetworkTypeRenderProps>
  /** Optional custom Training tab. If omitted, the host's standard training UI is used. */
  TrainingUI?: ComponentType<NetworkTypeRenderProps>
  /** Optional custom Inference tab. If omitted, the host's standard inference UI is used. */
  InferenceUI?: ComponentType<NetworkTypeRenderProps>
}

export interface NeuralCabinPlugin {
  /** Globally unique id (reverse-DNS recommended for third-party plugins). */
  id: string
  name: string
  version: string
  author?: string
  description: string
  /** One or more network types this plugin contributes. */
  networkTypes: NetworkTypeDescriptor[]
}

/** Stored representation of an installed plugin entry. */
export interface InstalledPlugin {
  id: string
  name: string
  version: string
  source: 'builtin' | 'uploaded' | 'marketplace'
  enabled: boolean
  /** For uploaded plugins, the raw module source (so we can re-load on boot). */
  source_code?: string
}
