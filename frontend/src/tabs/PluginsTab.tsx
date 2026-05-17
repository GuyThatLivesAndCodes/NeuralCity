import { useState } from 'react'
import type { PluginRegistry } from '../plugins/registry'

export default function PluginsTab({ registry }: { registry: PluginRegistry }) {
  const [error, setError] = useState<string | null>(null)
  const [installing, setInstalling] = useState(false)

  const onUpload = async (file: File | null) => {
    if (!file) return
    setError(null); setInstalling(true)
    try {
      const text = await file.text()
      await registry.installFromSource(text)
    } catch (e) {
      setError(`Failed to install plugin: ${String(e)}`)
    } finally { setInstalling(false) }
  }

  return (
    <div className="tab-content">
      <h2>Plugins</h2>
      <p className="muted">
        Plugins extend NeuralCabin with new network types. Each plugin owns its
        own create form, corpus UI, and inference UI for the types it
        contributes.
      </p>

      {error && <div className="status error">{error}</div>}

      <div className="card">
        <h3>Installed plugins</h3>
        {registry.installed.length === 0 ? (
          <p className="muted">No plugins installed.</p>
        ) : (
          <table>
            <thead>
              <tr><th>Name</th><th>Version</th><th>Source</th><th>Status</th><th></th></tr>
            </thead>
            <tbody>
              {registry.installed.map(p => (
                <tr key={p.id}>
                  <td>
                    <strong>{p.name}</strong>
                    <div className="muted" style={{ fontSize: 12 }}>{p.id}</div>
                  </td>
                  <td>{p.version}</td>
                  <td>{p.source}</td>
                  <td>{p.enabled ? <span style={{ color: 'var(--accent)' }}>enabled</span> : <span className="muted">disabled</span>}</td>
                  <td>
                    <div className="flex" style={{ gap: 6 }}>
                      <button
                        className="secondary"
                        onClick={() => registry.enable(p.id, !p.enabled)}
                      >
                        {p.enabled ? 'Disable' : 'Enable'}
                      </button>
                      {p.source !== 'builtin' && (
                        <button className="secondary" onClick={() => registry.uninstall(p.id)}>
                          Uninstall
                        </button>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      <div className="card">
        <h3>Install from file</h3>
        <p className="muted">
          Upload a JS plugin module. The module's default export must be a
          NeuralCabin plugin object.{' '}
          <strong>Warning:</strong> plugins run with full access to the app — only
          install plugins you trust.
        </p>
        <input type="file" accept=".js,.mjs,text/javascript"
          onChange={e => onUpload(e.target.files?.[0] ?? null)}
          disabled={installing} />
      </div>

      <div className="card">
        <h3>NeuralCabin Marketplace</h3>
        <p className="muted">
          A curated, signed plugin registry hosted on Cloudflare. Coming soon —
          once the backing infrastructure is wired up, you'll be able to browse
          and one-click install plugins from here.
        </p>
        <button className="secondary" disabled>Browse marketplace (coming soon)</button>
      </div>
    </div>
  )
}
