import { useMemo } from 'react'

/**
 * Generic neuron-by-neuron neural-network visualization.
 *
 * Accepts a per-layer activation array (including the input layer) and renders
 * each neuron as a colored circle, with light edges drawn between layers.
 * Designed to work for any layered network the engine exposes activations for;
 * plugins can reuse this component for their custom network types.
 */
export interface NetworkVizProps {
  /** Human-readable label for each layer (e.g. "input", "Linear (2 -> 8)"). */
  layerNames: string[]
  /** activations[layer][neuron] — values are colored on a diverging scale. */
  activations: number[][]
  /** Optional max neurons rendered per layer. Above this we sample + label. */
  maxNeuronsPerLayer?: number
  /** Optional explicit labels for input neurons (e.g. "x0", "x1"). */
  inputLabels?: string[]
  /** Optional explicit labels for output neurons (e.g. class names). */
  outputLabels?: string[]
  height?: number
}

const NODE_R = 12
const COL_GAP = 110

function color(v: number): string {
  // Diverging palette: red for negative, blue for positive, near-white at 0.
  const t = Math.max(-1, Math.min(1, v))
  if (t >= 0) {
    const a = Math.round(t * 255)
    return `rgb(${255 - a},${255 - Math.round(a * 0.4)},255)`
  } else {
    const a = Math.round(-t * 255)
    return `rgb(255,${255 - a},${255 - a})`
  }
}

function normalizeRow(row: number[]): number[] {
  if (row.length === 0) return row
  let max = 0
  for (const v of row) max = Math.max(max, Math.abs(v))
  if (max < 1e-9) return row.map(() => 0)
  return row.map(v => v / max)
}

export default function NetworkViz({
  layerNames, activations,
  maxNeuronsPerLayer = 24,
  inputLabels, outputLabels,
  height = 380,
}: NetworkVizProps) {
  const { layout, width } = useMemo(() => {
    const layout = activations.map((row, li) => {
      const n = row.length
      const truncated = n > maxNeuronsPerLayer
      const shown = truncated ? maxNeuronsPerLayer : n
      const normalized = normalizeRow(row)
      const vGap = (height - 40) / Math.max(shown, 1)
      const positions: { x: number; y: number; raw: number; norm: number; idx: number }[] = []
      for (let i = 0; i < shown; i++) {
        const realIdx = truncated
          ? Math.round((i / (shown - 1 || 1)) * (n - 1))
          : i
        positions.push({
          x: 40 + li * COL_GAP,
          y: 20 + vGap * (i + 0.5),
          raw: row[realIdx] ?? 0,
          norm: normalized[realIdx] ?? 0,
          idx: realIdx,
        })
      }
      return { name: layerNames[li] ?? `layer ${li}`, positions, truncated, total: n }
    })
    const width = 40 + activations.length * COL_GAP + 40
    return { layout, width }
  }, [activations, layerNames, maxNeuronsPerLayer, height])

  if (activations.length === 0) {
    return <div className="muted">No activations to display.</div>
  }

  return (
    <div style={{ overflow: 'auto', border: '1px solid var(--border)', borderRadius: 'var(--radius)' }}>
      <svg width={width} height={height} style={{ display: 'block', background: 'var(--bg-elev-1)' }}>
        {/* Edges between adjacent layers (thin, fixed opacity — full weights would be too dense). */}
        {layout.slice(0, -1).map((layer, li) =>
          layer.positions.map((a, i) =>
            layout[li + 1].positions.map((b, j) => (
              <line
                key={`${li}-${i}-${j}`}
                x1={a.x} y1={a.y} x2={b.x} y2={b.y}
                stroke="var(--border)" strokeWidth={0.4} opacity={0.5}
              />
            )),
          ),
        )}
        {/* Neurons */}
        {layout.map((layer, li) => (
          <g key={li}>
            <text
              x={layer.positions[0]?.x ?? 0}
              y={12}
              fontSize={11}
              fill="var(--text-muted)"
              textAnchor="middle"
            >
              {layer.name}{layer.truncated ? ` (${layer.total})` : ''}
            </text>
            {layer.positions.map((p, i) => {
              const label =
                li === 0 && inputLabels?.[p.idx] !== undefined
                  ? inputLabels[p.idx]
                  : li === layout.length - 1 && outputLabels?.[p.idx] !== undefined
                  ? outputLabels[p.idx]
                  : null
              return (
                <g key={i}>
                  <circle
                    cx={p.x} cy={p.y} r={NODE_R}
                    fill={color(p.norm)}
                    stroke="var(--border)" strokeWidth={1}
                  >
                    <title>{`${layer.name} · neuron ${p.idx}: ${p.raw.toFixed(4)}`}</title>
                  </circle>
                  <text
                    x={p.x} y={p.y + 3}
                    fontSize={9}
                    fill="#222"
                    textAnchor="middle"
                  >
                    {p.raw.toFixed(2)}
                  </text>
                  {label && (
                    <text
                      x={p.x} y={p.y + NODE_R + 11}
                      fontSize={10}
                      fill="var(--text-muted)"
                      textAnchor="middle"
                    >
                      {label}
                    </text>
                  )}
                </g>
              )
            })}
          </g>
        ))}
      </svg>
    </div>
  )
}
