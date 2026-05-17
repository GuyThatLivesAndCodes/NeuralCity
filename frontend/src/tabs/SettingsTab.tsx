import { useState } from 'react'
import {
  AppSettings, applySettings, deriveHover, loadSettings, resetSettings,
  saveSettings,
} from '../settings'

interface Props {
  /** Called by App.tsx so the header / nav re-render when the theme changes. */
  onChange: (s: AppSettings) => void
}

const BG_PRESETS: { name: string; hex: string }[] = [
  { name: 'Cabin dark',    hex: '#1a1815' },
  { name: 'Midnight',      hex: '#0f1216' },
  { name: 'Slate',         hex: '#1b1f24' },
  { name: 'Forest',        hex: '#141a16' },
  { name: 'Charcoal',      hex: '#1d1d1d' },
  { name: 'Cocoa',         hex: '#1f1814' },
]

const PRESETS: { name: string; hex: string }[] = [
  { name: 'Cabin orange',  hex: '#d97757' },
  { name: 'Cobalt',        hex: '#3b82f6' },
  { name: 'Emerald',       hex: '#10b981' },
  { name: 'Rose',          hex: '#e11d48' },
  { name: 'Amber',         hex: '#f59e0b' },
  { name: 'Slate',         hex: '#64748b' },
  { name: 'Violet',        hex: '#8b5cf6' },
  { name: 'Teal',          hex: '#14b8a6' },
]

export default function SettingsTab({ onChange }: Props) {
  const [settings, setSettings] = useState<AppSettings>(() => loadSettings())
  const [saved, setSaved] = useState(false)

  const update = (patch: Partial<AppSettings>) => {
    const next = { ...settings, ...patch }
    setSettings(next)
    setSaved(false)
  }

  const onPickColor = (hex: string) => {
    update({ primaryColor: hex, primaryHover: deriveHover(hex) })
  }

  const onSave = () => {
    saveSettings(settings)
    applySettings(settings)
    onChange(settings)
    setSaved(true)
  }

  const onReset = () => {
    const fresh = resetSettings()
    setSettings(fresh)
    applySettings(fresh)
    onChange(fresh)
    setSaved(true)
  }

  return (
    <div className="tab-content">
      <h2>Settings</h2>
      <p className="muted">
        Tune the look of the app and how training uses your hardware. Changes
        are stored locally on this machine and survive restarts.
      </p>

      {saved && <div className="status success">Settings saved.</div>}

      <div className="card">
        <h3>Primary color</h3>
        <p className="muted" style={{ marginTop: 4 }}>
          Used for buttons, highlights, links, and the loss plot. Pick a preset
          or set an exact hex value below.
        </p>

        <div className="flex" style={{ flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
          {PRESETS.map(p => (
            <button
              key={p.hex}
              onClick={() => onPickColor(p.hex)}
              className={settings.primaryColor.toLowerCase() === p.hex.toLowerCase()
                ? '' : 'secondary'}
              title={p.hex}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                textTransform: 'none', letterSpacing: 0,
              }}
            >
              <span style={{
                width: 14, height: 14, borderRadius: 4,
                background: p.hex, border: '1px solid var(--border)',
                display: 'inline-block',
              }} />
              {p.name}
            </button>
          ))}
        </div>

        <div className="grid-2 mt-2">
          <div>
            <label>Hex value</label>
            <div className="flex" style={{ alignItems: 'center', gap: 8 }}>
              <input
                value={settings.primaryColor}
                onChange={e => onPickColor(e.target.value)}
                placeholder="#d97757"
                style={{ fontFamily: 'var(--font-mono)' }}
              />
              <input
                type="color"
                value={normalizeForColorInput(settings.primaryColor)}
                onChange={e => onPickColor(e.target.value)}
                style={{ width: 44, height: 38, padding: 0, cursor: 'pointer' }}
                aria-label="Pick a color"
              />
            </div>
          </div>
          <div>
            <label>Hover color</label>
            <input
              value={settings.primaryHover}
              onChange={e => update({ primaryHover: e.target.value })}
              placeholder="#c8633e"
              style={{ fontFamily: 'var(--font-mono)' }}
            />
            <small>Auto-derived from the primary, but you can override.</small>
          </div>
        </div>

        <div className="mt-2">
          <p className="muted small">Preview</p>
          <div className="flex" style={{ gap: 8, marginTop: 8 }}>
            <button>Primary action</button>
            <button className="secondary">Secondary</button>
            <span className="chip">Chip</span>
            <span className="status success" style={{ display: 'inline-block' }}>Success</span>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Background color</h3>
        <p className="muted" style={{ marginTop: 4 }}>
          The base canvas colour. Elevated cards, inputs, and the nav are
          auto-derived from this single value so the whole palette shifts
          together.
        </p>

        <div className="flex" style={{ flexWrap: 'wrap', gap: 8, marginTop: 12 }}>
          {BG_PRESETS.map(p => (
            <button
              key={p.hex}
              onClick={() => update({ backgroundColor: p.hex })}
              className={settings.backgroundColor.toLowerCase() === p.hex.toLowerCase()
                ? '' : 'secondary'}
              title={p.hex}
              style={{
                display: 'flex', alignItems: 'center', gap: 8,
                textTransform: 'none', letterSpacing: 0,
              }}
            >
              <span style={{
                width: 14, height: 14, borderRadius: 4,
                background: p.hex, border: '1px solid var(--border)',
                display: 'inline-block',
              }} />
              {p.name}
            </button>
          ))}
        </div>

        <div className="grid-2 mt-2">
          <div>
            <label>Hex value</label>
            <div className="flex" style={{ alignItems: 'center', gap: 8 }}>
              <input
                value={settings.backgroundColor}
                onChange={e => update({ backgroundColor: e.target.value })}
                placeholder="#1a1815"
                style={{ fontFamily: 'var(--font-mono)' }}
              />
              <input
                type="color"
                value={normalizeForColorInput(settings.backgroundColor)}
                onChange={e => update({ backgroundColor: e.target.value })}
                style={{ width: 44, height: 38, padding: 0, cursor: 'pointer' }}
                aria-label="Pick a background color"
              />
            </div>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Compute backend</h3>
        <p className="muted" style={{ marginTop: 4 }}>
          The engine now runs on the <strong>Burn</strong> framework with the
          <strong> WGPU </strong> backend. On laptops without a discrete GPU,
          WGPU automatically falls back to an integrated GPU or CPU compute
          pipeline — there's no driver to install. Pick <em>CPU only</em> to
          force a CPU compute path if you suspect GPU driver issues.
        </p>

        <div className="grid-2 mt-2">
          <div>
            <label>Backend</label>
            <select
              value={settings.computeBackend}
              onChange={e => update({ computeBackend: e.target.value as 'gpu' | 'cpu' })}
            >
              <option value="gpu">GPU (WGPU — recommended)</option>
              <option value="cpu">CPU only</option>
            </select>
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            <label>Show device info</label>
            <label style={{
              display: 'flex', alignItems: 'center', gap: 8,
              textTransform: 'none', letterSpacing: 0, color: 'var(--text)',
            }}>
              <input
                type="checkbox"
                style={{ width: 'auto' }}
                checked={settings.showDeviceInfo}
                onChange={e => update({ showDeviceInfo: e.target.checked })}
              />
              Surface backend / device info during training
            </label>
          </div>
        </div>

        {settings.showDeviceInfo && (
          <div className="status mt-2">
            Training uses{' '}
            <code>burn::backend::wgpu::WgpuDevice::default()</code>. On first
            run the device may take a moment to compile the shader pipeline.
            Subsequent runs reuse the cached compute kernels.
          </div>
        )}
      </div>

      <div className="card">
        <h3>About this update</h3>
        <p>
          NeuralCabin migrated its tensor math and autodiff from a from-scratch
          implementation to the <a href="https://burn.dev" target="_blank"
          rel="noreferrer">Burn</a> framework with the WGPU backend. The
          training loop now runs on your GPU when one is available, and falls
          back gracefully otherwise. The save-state format is unchanged — old
          workspaces load cleanly into the new engine.
        </p>
      </div>

      <div className="flex mt-2">
        <button onClick={onSave}>Save settings</button>
        <button className="secondary" onClick={onReset}>Reset to defaults</button>
      </div>
    </div>
  )
}

/** `<input type="color">` insists on a 6-char hex value. Coerce loose inputs
 *  (#fff, missing #, lower-case) into something it'll accept. */
function normalizeForColorInput(s: string): string {
  let v = s.trim().replace(/^#/, '')
  if (v.length === 3) v = v.split('').map(c => c + c).join('')
  if (!/^[0-9a-fA-F]{6}$/.test(v)) return '#d97757'
  return `#${v.toLowerCase()}`
}
