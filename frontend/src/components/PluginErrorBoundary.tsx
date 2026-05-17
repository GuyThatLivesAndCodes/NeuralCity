import { Component, type ReactNode } from 'react'

/**
 * Error boundary used to wrap plugin-contributed UI. Without it, a single
 * uncaught render error in a plugin component takes the entire app down to a
 * blank screen — the host can't tell the user what happened. With it, we
 * surface the error inline and keep the rest of the app functional.
 */
interface Props {
  children: ReactNode
  fallbackTitle?: string
}
interface State {
  error: Error | null
}

export default class PluginErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: unknown) {
    console.error('[plugin error]', error, info)
  }

  reset = () => this.setState({ error: null })

  componentDidUpdate(prevProps: Props) {
    // If the children identity changes (e.g. user switched networks), clear
    // the captured error so the new subtree gets a fresh attempt.
    if (prevProps.children !== this.props.children && this.state.error) {
      this.setState({ error: null })
    }
  }

  render() {
    if (this.state.error) {
      return (
        <div className="status error" style={{ whiteSpace: 'pre-wrap' }}>
          <strong>{this.props.fallbackTitle ?? 'Plugin UI crashed'}</strong>
          <div style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            {String(this.state.error?.message ?? this.state.error)}
          </div>
          <button className="secondary" style={{ marginTop: 8 }} onClick={this.reset}>
            Retry
          </button>
        </div>
      )
    }
    return this.props.children
  }
}
