// Small settings module: the source of truth lives in localStorage so the
// values survive across app restarts. The UI reads/writes through these
// helpers, and `applySettings` pushes the values onto the document's CSS
// custom properties so the rest of the app picks them up automatically.

export type ComputeBackend = 'gpu' | 'cpu'

export interface AppSettings {
  /** CSS color string used as the primary accent (buttons, highlights, …). */
  primaryColor: string
  /** Hover-state colour, derived from `primaryColor` by default. */
  primaryHover: string
  /** App background base colour. Elevated surfaces are derived from this. */
  backgroundColor: string
  /**
   * Preferred compute backend for training.
   *
   * `gpu` routes through WGPU (the default — Burn falls back to integrated GPU
   * or CPU compute automatically if a discrete GPU is unavailable).
   *
   * `cpu` is a hint for the user that they explicitly want to disable GPU
   * usage; the engine respects this on the next training run.
   */
  computeBackend: ComputeBackend
  /** Show GPU device info on the Settings tab. */
  showDeviceInfo: boolean
}

const STORAGE_KEY = 'neuralcabin.settings.v1'

const DEFAULTS: AppSettings = {
  primaryColor: '#d97757',
  primaryHover: '#c8633e',
  backgroundColor: '#1a1815',
  computeBackend: 'gpu',
  showDeviceInfo: true,
}

export function loadSettings(): AppSettings {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    if (!raw) return { ...DEFAULTS }
    const parsed = JSON.parse(raw) as Partial<AppSettings>
    return { ...DEFAULTS, ...parsed }
  } catch {
    return { ...DEFAULTS }
  }
}

export function saveSettings(s: AppSettings): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(s))
}

export function resetSettings(): AppSettings {
  localStorage.removeItem(STORAGE_KEY)
  return { ...DEFAULTS }
}

/**
 * Push the user's settings into the document's CSS custom properties so the
 * existing stylesheet picks them up automatically. Called on app boot and
 * whenever the Settings tab saves a change.
 */
export function applySettings(s: AppSettings): void {
  const root = document.documentElement
  root.style.setProperty('--accent', s.primaryColor)
  root.style.setProperty('--accent-hover', s.primaryHover)
  // Derive the soft/bg variants from the same hue so the dependent UI keeps
  // its translucent feel without the user having to set three colours.
  const { r, g, b } = parseHex(s.primaryColor)
  root.style.setProperty('--accent-soft', `rgba(${r}, ${g}, ${b}, 0.12)`)
  root.style.setProperty('--accent-bg', `rgba(${r}, ${g}, ${b}, 0.08)`)

  // Background palette — base + two elevated shades + a recessed input shade.
  // We shift brightness in linear steps so any user-picked base produces a
  // coherent set of surfaces, the same way the default warm-dark palette did.
  const base = parseHex(s.backgroundColor)
  root.style.setProperty('--bg',        rgbStr(base))
  root.style.setProperty('--bg-elev-1', rgbStr(shiftBrightness(base,  8)))
  root.style.setProperty('--bg-elev-2', rgbStr(shiftBrightness(base, 16)))
  root.style.setProperty('--bg-input',  rgbStr(shiftBrightness(base, -4)))
}

function rgbStr({ r, g, b }: { r: number; g: number; b: number }): string {
  return `rgb(${r}, ${g}, ${b})`
}

function shiftBrightness(c: { r: number; g: number; b: number }, delta: number) {
  const clamp = (v: number) => Math.max(0, Math.min(255, v))
  return { r: clamp(c.r + delta), g: clamp(c.g + delta), b: clamp(c.b + delta) }
}

/** Parse a #rrggbb / #rgb string into rgb components. Falls back to the
 *  default accent if the input isn't well-formed. */
function parseHex(hex: string): { r: number; g: number; b: number } {
  const m = hex.trim().match(/^#([0-9a-fA-F]{3}|[0-9a-fA-F]{6})$/)
  if (!m) return { r: 217, g: 119, b: 87 }
  let s = m[1]
  if (s.length === 3) s = s.split('').map(c => c + c).join('')
  return {
    r: parseInt(s.slice(0, 2), 16),
    g: parseInt(s.slice(2, 4), 16),
    b: parseInt(s.slice(4, 6), 16),
  }
}

/** Auto-derive a slightly darker hover colour from a base hex string. */
export function deriveHover(hex: string): string {
  const { r, g, b } = parseHex(hex)
  const darken = (c: number) => Math.max(0, Math.round(c * 0.88))
  const hh = (n: number) => n.toString(16).padStart(2, '0')
  return `#${hh(darken(r))}${hh(darken(g))}${hh(darken(b))}`
}
