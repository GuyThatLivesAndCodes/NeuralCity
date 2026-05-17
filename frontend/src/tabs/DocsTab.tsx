import { Network } from '../api'

export default function DocsTab({ networks: list }: { networks: Network[] }) {
  const ff = list.filter(n => n.kind === 'feedforward')
  const nt = list.filter(n => n.kind === 'next_token')
  const trained = list.filter(n => n.trained).length

  return (
    <div className="tab-content">
      <h2>Documentation</h2>
      <p className="muted">
        Honest reference for what NeuralCabin does and how its pieces fit
        together. The numbers below are live — they reflect the current state
        of your workspace.
      </p>

      <div className="card">
        <h3>Workspace status</h3>
        <table>
          <tbody>
            <tr><th>Total networks</th><td>{list.length}</td></tr>
            <tr><th>Feed-forward</th><td>{ff.length}</td></tr>
            <tr><th>Next-token</th><td>{nt.length}</td></tr>
            <tr><th>Trained at least once</th><td>{trained}</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>What this app actually does</h3>
        <p>
          NeuralCabin trains small neural networks entirely in Rust, with no
          Python, no PyTorch, and no remote services. Training and inference
          run on your GPU via the <a href="https://burn.dev" target="_blank"
          rel="noreferrer">Burn</a> framework with the WGPU backend — on
          laptops and low-end PCs that falls back gracefully to integrated
          GPUs or CPU compute. The engine supports:
        </p>
        <ul style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li><code>Linear</code> (fully-connected) layers and five activations
              (<code>identity</code>, <code>relu</code>, <code>sigmoid</code>, <code>tanh</code>, <code>softmax</code>).</li>
          <li>Automatic differentiation provided by Burn's autodiff backend.</li>
          <li>Two losses: <code>MeanSquaredError</code> and softmax + <code>CrossEntropy</code>.</li>
          <li>Four optimizers: <code>Adam</code>, <code>AdamW</code>, <code>LAMB</code>, and <code>SGD</code> (with optional momentum).</li>
          <li>Char-level and word-level tokenizers for text networks.</li>
        </ul>
        <p className="mt-2 muted">
          <strong>What this app is NOT:</strong> there is no transformer,
          no self-attention, no convolution, no RNN. Sequence modeling is a
          fully-connected MLP that consumes a sliding window of one-hot tokens
          and predicts the next token. It works for tiny corpora — don’t expect
          GPT-quality output.
        </p>
      </div>

      <div className="card">
        <h3>Workflow</h3>
        <ol style={{ paddingLeft: 20, lineHeight: 1.8 }}>
          <li><strong>Networks tab</strong>: design the architecture. Pick
              feed-forward (numeric in/out) or next-token (text). The hidden-layer
              spec field accepts a comma-separated list of dims and activations,
              e.g. <code>64,relu,32,relu</code>.</li>
          <li><strong>Corpus tab</strong>: attach training data. The form changes
              with the network kind:
              <ul style={{ paddingLeft: 20, marginTop: 6 }}>
                <li>Feed-forward: paste or upload CSV — each row is{' '}
                    <code>in_dim + out_dim</code> comma-separated numbers.</li>
                <li>Next-token, <em>pretraining</em>: bulk-upload one or more
                    <code>.txt</code> files. The vocabulary is built automatically.</li>
                <li>Next-token, <em>fine-tuning</em>: provide input/output pairs
                    in the editor or import a JSON file.</li>
              </ul></li>
          <li><strong>Vocabulary tab</strong>: inspect the vocabulary that the
              corpus produced for next-token networks. Export to JSON.</li>
          <li><strong>Training tab</strong>: pick epochs, batch size, optimizer,
              learning rate, and (for fine-tuning) whether to mask user tokens.
              Loss curves stream live as the model trains.</li>
          <li><strong>Inference tab</strong>: feed inputs (or a prompt) to a
              trained network. For next-token networks you also get per-token
              probabilities for transparency. For feed-forward networks (and
              plugin-managed network types) the inference view also renders the
              <strong> network itself</strong> — every neuron's activation after
              the most recent forward pass, color-coded by magnitude.</li>
          <li><strong>Plugins tab</strong>: install, enable, or upload plugins
              that contribute brand-new network types. Plugin types appear in
              the same Type dropdown as the built-in ones on the Networks tab.</li>
        </ol>
      </div>

      <div className="card">
        <h3>Network visualization (Inference tab)</h3>
        <p>
          For any feed-forward network — built-in or plugin-contributed — the
          Inference tab can render the network as a column of neurons per
          layer, colored by their activation after the most recent forward
          pass. Hover a neuron to see its exact value. Layers wider than a
          handful of neurons are sampled so the view stays readable.
        </p>
        <p className="muted small mt-1">
          Backed by a Tauri command <code>infer_with_activations</code> that
          runs the same forward path used by training but captures the output
          of every layer. Next-token / transformer networks don't surface the
          visualization yet because their inference streams tokens.
        </p>
      </div>

      <div className="card">
        <h3>Plugins</h3>
        <p>
          Plugins extend NeuralCabin with new network types. A plugin owns the
          create form, the corpus UI, and the inference UI for its types —
          under the hood it usually creates a normal feed-forward network and
          stores its own metadata per network, but to the rest of the app the
          new type is a first-class citizen alongside feed-forward,
          next-token, and transformer.
        </p>
        <p>
          A plugin is a JS module whose default export is a plugin object:
        </p>
        <pre style={{
          background: 'var(--bg-input)', padding: 12, borderRadius: 'var(--radius)',
          border: '1px solid var(--border)', overflow: 'auto', fontSize: 12,
        }}>{`export default {
  id:      'com.example.my-plugin',
  name:    'My Plugin',
  version: '0.1.0',
  description: 'Adds a new network type.',
  networkTypes: [{
    id: 'my-type',
    label: 'My Type',
    description: 'What this type is good for.',
    CreateForm,            // React component: builds + creates a network
    CorpusUI:    CorpusUI, // optional: replaces the default Corpus tab
    InferenceUI: InferUI,  // optional: replaces the default Inference tab
  }],
}`}</pre>
        <p className="mt-2">
          Plugins are loaded as ES modules into the same JS realm as the host
          (no sandbox), so installing a plugin is a trust decision — only
          install plugins you actually trust.
        </p>
        <p className="muted small mt-1">
          A future curated <strong>NeuralCabin Marketplace</strong>, backed by
          Cloudflare, will host signed plugins for one-click install. The
          Plugins tab already shows the placeholder.
        </p>
      </div>

      <div className="card">
        <h3>Built-in plugin: Image Classification</h3>
        <p>
          Ships enabled out of the box and adds an <strong>Image Classification</strong>
          type to the Networks Type dropdown. Configurable image size (X / Y),
          color mode (grayscale or RGB), hidden-layer spec, output activation,
          and seed — under the hood it's a feed-forward MLP with{' '}
          <code>X · Y · channels</code> inputs and one output neuron per class.
        </p>
        <ul style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li><strong>Corpus:</strong> add text class labels, then attach
              samples by uploading images (auto-resized to the network's input
              dims) or by drawing on the built-in canvas. The canvas adapts
              to grayscale vs. RGB and exposes a color picker in RGB mode.</li>
          <li><strong>Output-dim drift:</strong> if you add more classes than
              the network was created with, the plugin rebuilds the underlying
              network at save time and migrates your samples — unless the
              network has already been trained, in which case it refuses to
              silently discard your weights.</li>
          <li><strong>Training:</strong> uses the standard Training tab — it's
              just a feed-forward model.</li>
          <li><strong>Inference:</strong> draw an image or upload one and the
              plugin predicts a class. Enable <em>real-time inference</em> to
              run a forward pass on every stroke (queued so it never piles up),
              with the network-viz updating live.</li>
        </ul>
        <p className="muted small mt-1">
          Per-network samples are stored in IndexedDB rather than localStorage,
          because RGB samples (e.g. 64×64×3 ≈ 12k floats) blow past the
          localStorage quota fast. Class lists, dims, and hyperparams stay in
          localStorage where they fit comfortably.
        </p>
      </div>

      <div className="card">
        <h3>Network types</h3>
        <div className="grid-2">
          <div>
            <h4>Feed-forward</h4>
            <p>Numeric vectors in, numeric vectors out. Use for regression
              (continuous targets) or classification (one-hot targets with a
              softmax / cross-entropy combo).</p>
            <p className="muted small mt-1">
              Examples: XOR, polynomial regression, small tabular classifiers.
            </p>
          </div>
          <div>
            <h4>Next-token prediction</h4>
            <p>Text in, text out — modelled as predicting the next token from a
              fixed-size window of preceding tokens. The window is one-hot
              encoded then fed through dense layers to <code>vocab_size</code> logits.</p>
            <p className="muted small mt-1">
              Two stages: <em>pretraining</em> on free-form text, then
              optional <em>fine-tuning</em> on input/output pairs.
            </p>
          </div>
        </div>
      </div>

      <div className="card">
        <h3>Fine-tuning format</h3>
        <p>
          The fine-tuning JSON format used by the Corpus tab is a flat array of
          objects, each with an <code>input</code> and <code>output</code> string:
        </p>
        <pre style={{
          background: 'var(--bg-input)', padding: 12, borderRadius: 'var(--radius)',
          border: '1px solid var(--border)', overflow: 'auto', fontSize: 12,
        }}>{`[
  { "input": "what is 2 + 2?",   "output": "4" },
  { "input": "what is 3 + 5?",   "output": "8" },
  { "input": "is the sky blue?", "output": "yes" }
]`}</pre>
        <p className="mt-1">
          Internally each pair is encoded as{' '}
          <code>&lt;user&gt; input &lt;eos&gt; &lt;assistant&gt; output &lt;eos&gt;</code>{' '}
          and one training example is emitted per token transition. With{' '}
          <em>mask user tokens</em> on (the default), the model is only scored
          on producing the assistant output — so it learns exactly when its
          turn starts and when to stop.
        </p>
      </div>

      <div className="card">
        <h3>Reserved vocabulary tokens</h3>
        <table>
          <thead><tr><th>ID</th><th>Token</th><th>Meaning</th></tr></thead>
          <tbody>
            <tr><td><code>0</code></td><td><code>&lt;pad&gt;</code></td><td>Left-padding for short contexts.</td></tr>
            <tr><td><code>1</code></td><td><code>&lt;unk&gt;</code></td><td>Anything not in the trained vocab.</td></tr>
            <tr><td><code>2</code></td><td><code>&lt;bos&gt;</code></td><td>Beginning of sequence (reserved).</td></tr>
            <tr><td><code>3</code></td><td><code>&lt;eos&gt;</code></td><td>End of a turn; halts generation when sampled.</td></tr>
            <tr><td><code>4</code></td><td><code>&lt;user&gt;</code></td><td>Start of a user message in fine-tune / chat mode.</td></tr>
            <tr><td><code>5</code></td><td><code>&lt;assistant&gt;</code></td><td>Start of an assistant reply in fine-tune / chat mode.</td></tr>
          </tbody>
        </table>
        <p className="muted small mt-1">
          Fine-tuning pairs are encoded as{' '}
          <code>&lt;user&gt; input &lt;eos&gt; &lt;assistant&gt; output &lt;eos&gt;</code>,
          and the chat inference prompt is wrapped the same way so the model
          knows when to start its reply and when to stop.
        </p>
      </div>

      <div className="card">
        <h3>Hidden-layer mini-language</h3>
        <p>
          The hidden-layer field on the Networks form takes a comma-separated
          list. Numbers become Linear layers (with the given output dim);
          words become activations.
        </p>
        <table>
          <thead><tr><th>Spec</th><th>Resulting layers</th></tr></thead>
          <tbody>
            <tr><td><code>8,tanh</code></td><td>Linear → 8 · tanh</td></tr>
            <tr><td><code>64,relu,32,relu</code></td><td>Linear → 64 · relu · Linear → 32 · relu</td></tr>
            <tr><td><code>(empty)</code></td><td>Direct projection from input to output (no hidden layer)</td></tr>
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>How training actually runs</h3>
        <ol style={{ paddingLeft: 20, lineHeight: 1.7 }}>
          <li>The frontend sends a <code>StartTrainingRequest</code> over Tauri IPC.</li>
          <li>The backend builds an <code>(X, Y)</code> tensor pair from the
              attached corpus — sliding windows for next-token networks, raw
              rows for feed-forward.</li>
          <li>An <code>Optimizer</code> is constructed sized to the model’s
              actual parameter shapes.</li>
          <li>For each epoch the example indices are shuffled (deterministic
              from <code>seed</code>) and processed in batches.</li>
          <li>Each batch runs a real autograd pass: the loss is built into a
              fresh tape, gradients flow backward, and the optimizer updates
              weights in place.</li>
          <li>After every epoch, a <code>training_update</code> event is emitted
              with the mean batch loss.</li>
          <li>When the run finishes, the network is marked as <code>trained</code>
              and inference becomes meaningful.</li>
        </ol>
      </div>
    </div>
  )
}
