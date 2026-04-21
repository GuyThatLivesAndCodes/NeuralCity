window.NC_DOCS = [
  {
    id: 'welcome',
    title: 'Welcome',
    body: `
<h1>Welcome to NeuralCity</h1>
<p>NeuralCity is a self-contained neural network platform built entirely in plain JavaScript — no NumPy, no TensorFlow, no PyTorch. Every tensor operation, every gradient, every optimizer step runs on readable source code you can inspect, modify, and learn from. The full engine is available at <a href="https://github.com/GuyThatLivesAndCodes/NeuralCity" target="_blank">github.com/GuyThatLivesAndCodes/NeuralCity</a>.</p>
<p>Select a topic on the left to explore how the engine works, how to design and train networks, and how to use NeuralScript to automate experiments.</p>
<h2>Quick start</h2>
<ol>
  <li>Click <code>+ New</code> in the sidebar and choose a template. <b>XOR Classifier</b> trains in under a second.</li>
  <li>Open the <b>Train</b> tab and click <b>Start Training</b>.</li>
  <li>Switch to <b>Inference</b> to run predictions against the trained model.</li>
  <li>Switch to <b>API</b> to serve the model over HTTP on your local network.</li>
</ol>
`
  },
  {
    id: 'how-it-works',
    title: 'How networks learn',
    body: `
<h1>How a neural network learns</h1>
<p>A neural network is a composition of mathematical functions with <b>trainable parameters</b> (weights and biases). Training is the process of repeatedly measuring prediction error, then adjusting every parameter in the direction that reduces it.</p>

<h2>1. Forward pass</h2>
<p>An input vector <code>x</code> flows through each layer in sequence. A <b>Linear</b> layer computes <code>y = x·W + b</code>. An <b>Activation</b> such as <code>ReLU</code> applies a non-linearity — for example, clamping negatives to zero. With enough layers, the network can approximate arbitrary functions.</p>

<h2>2. Loss computation</h2>
<p>The network's output is compared to the target to produce a scalar error value. For classification, NeuralCity uses <b>softmax cross-entropy</b>: logits are converted to probabilities, then the negative log-probability of the correct class is returned. For regression, <b>mean squared error</b> is used.</p>

<h2>3. Backward pass (autograd)</h2>
<p>NeuralCity's engine implements reverse-mode automatic differentiation. Each operation records its inputs during the forward pass. Starting from the loss, the engine traverses the computation graph in reverse and applies the chain rule to compute <code>∂loss/∂W</code> and <code>∂loss/∂b</code> for every parameter. Each operator — matmul, relu, softmax, and others — implements its own local derivative.</p>

<h2>4. Optimizer step</h2>
<p>Once every parameter has a gradient, the optimizer updates them. Plain SGD subtracts a scaled gradient: <code>W ← W − lr · ∂loss/∂W</code>. <b>Adam</b> maintains running averages of past gradients and their squares, which stabilizes training and often converges faster than SGD on its own.</p>

<h2>5. Iteration</h2>
<p>Repeating this process over thousands of mini-batches drives the loss down. That's the full training loop.</p>
`
  },
  {
    id: 'layers',
    title: 'Layers reference',
    body: `
<h1>Layers</h1>
<h2>Linear</h2>
<p>Computes <code>y = x·W + b</code>. Input shape <code>[B, in]</code>, output shape <code>[B, out]</code>. The fundamental building block of most network architectures.</p>
<h2>Activation</h2>
<table>
<tr><th>Name</th><th>Formula</th><th>When to use</th></tr>
<tr><td>relu</td><td>max(0, x)</td><td>Default choice for hidden layers in most architectures.</td></tr>
<tr><td>leakyRelu</td><td>max(αx, x)</td><td>Use when standard ReLU produces dead neurons (gradient = 0 for many units).</td></tr>
<tr><td>tanh</td><td>(e^x − e^−x)/(e^x + e^−x)</td><td>Smooth and bounded. Works well in small regressors.</td></tr>
<tr><td>sigmoid</td><td>1 / (1 + e^−x)</td><td>Outputs a value in (0, 1). Suited for binary classification outputs.</td></tr>
<tr><td>gelu</td><td>0.5·x·(1 + tanh(√(2/π)(x + 0.044715x³)))</td><td>Smooth approximation with better gradient flow. Common in language model architectures.</td></tr>
<tr><td>softmax</td><td>e^xᵢ / Σe^xⱼ</td><td>Converts a vector of logits into a probability distribution summing to 1.</td></tr>
</table>
<h2>Dropout</h2>
<p>During training, a random fraction <code>p</code> of activations is set to zero and the rest are scaled by <code>1/(1−p)</code> to maintain expected magnitude. This discourages co-adaptation between units and reduces overfitting. Dropout is automatically disabled during inference.</p>
<h2>Embedding</h2>
<p>Maps integer token IDs to dense vectors of length <code>dim</code> via a learned lookup table. Used as the input layer of language models to convert discrete tokens into continuous representations.</p>
`
  },
  {
    id: 'data',
    title: 'Training data',
    body: `
<h1>Training data formats</h1>
<p>Each network type expects a specific JSON structure in its <b>Training Data</b> tab.</p>
<h2>Classifier</h2>
<pre><code>[
  { "input": [0, 0], "label": 0 },
  { "input": [0, 1], "label": 1 },
  ...
]</code></pre>
<h2>Regressor</h2>
<pre><code>[
  { "input": [x1, x2], "output": [y1, y2] },
  ...
]</code></pre>
<h2>Character LM (free text)</h2>
<p>Supply either a single <code>{ "text": "..." }</code> object or an array of such objects. The model builds a character-level vocabulary from the full corpus. Larger and more varied corpora produce richer language models.</p>

<h2>Chat Assistant (user/assistant pairs)</h2>
<p>A chat model is a character LM trained on structured dialogue. Any of the following JSON shapes are accepted inside a <code>samples</code> array.</p>
<p><b>Single-turn pairs:</b></p>
<pre><code>{
  "samples": [
    { "user": "Hello, I need help", "assistant": "Of course — what's up?" },
    { "user": "What can you do?",   "assistant": "I can discuss anything you teach me." }
  ]
}</code></pre>
<p><b>Multi-turn (messages array):</b></p>
<pre><code>{ "samples": [
  { "messages": [
      { "role": "system",    "content": "You are concise." },
      { "role": "user",      "content": "Hi" },
      { "role": "assistant", "content": "Hello!" }
  ] }
] }</code></pre>
<p><b>Alternating turns:</b></p>
<pre><code>{ "samples": [
  { "conversation": [
      { "user": "Hi" },
      { "assistant": "Hey!" },
      { "user": "Tell me more" },
      { "assistant": "Sure — about what?" }
  ] }
] }</code></pre>
<p>Internally, NeuralCity flattens every conversation into a single tagged stream:</p>
<pre><code>&lt;|user|&gt;Hello, I need help&lt;|end|&gt;&lt;|assistant|&gt;Of course — what's up?&lt;|end|&gt;</code></pre>
<p>At inference time, incoming messages are wrapped in the same tags. Generation stops as soon as the model emits <code>&lt;|end|&gt;</code>, so the response contains only the assistant turn.</p>
<p><b>Tip:</b> Increase <b>contextLen</b> (under Architecture) so multiple back-and-forth turns fit within the model's window. A value of <code>96</code>–<code>256</code> is appropriate for most conversational data; use higher values if responses are paragraph-length.</p>
<h3>Why multi-turn training samples matter</h3>
<p>A model trained exclusively on single-turn exchanges only learns to respond to fresh prompts. It never learns to continue an ongoing thread. To produce a model that can hold context across turns, include samples with multiple exchanges — use the <code>messages</code> shape with 2, 4, or 6+ alternating turns per sample. The trainer flattens each conversation into a tag-delimited stream, so the model learns the full <code>user → assistant → user → assistant</code> pattern rather than just the opening reply.</p>
`
  },
  {
    id: 'chat-inference',
    title: 'Chatting with a model',
    body: `
<h1>Multi-turn inference</h1>
<p>Once a network is trained on chat data, the <b>Inference</b> tab becomes a full chat interface with conversation history, a persistent system prompt field, and a <b>Reset chat</b> button. Each message is encoded together with all prior turns, so the model has access to earlier context when generating a reply.</p>

<h2>How history is encoded</h2>
<p>Each generation request flattens the full conversation into the role-tagged format the model was trained on:</p>
<pre><code>&lt;|system|&gt;Be concise.&lt;|end|&gt;
&lt;|user|&gt;Hi&lt;|end|&gt;&lt;|assistant|&gt;Hello!&lt;|end|&gt;
&lt;|user|&gt;How are you?&lt;|end|&gt;&lt;|assistant|&gt;</code></pre>
<p>The model continues from the trailing <code>&lt;|assistant|&gt;</code> token. Generation stops when <code>&lt;|end|&gt;</code> is emitted, and only the assistant's reply is returned to the UI.</p>

<h2>History truncation</h2>
<p>When the conversation exceeds <code>contextLen</code>, the oldest turns are dropped first. Turns are never split mid-way. The system prompt (if set) is always preserved at the start of the context window, and the current user message is always preserved at the end. Long conversations degrade gracefully: the model retains the system prompt and the most recent turns while discarding earlier history.</p>

<h2>Programmatic access</h2>
<p>From the <b>Script</b> tab or any external client, pass an explicit history object to <code>predict</code>:</p>
<pre><code>predict(thisNet(), {
  history: [
    { role: "user",      content: "Hi" },
    { role: "assistant", content: "Hello!" }
  ],
  prompt: "How are you?",
  system: "Be concise.",
  maxTokens: 200,
  temperature: 0.7
})</code></pre>
<p>The <code>messages</code> field (OpenAI-compatible) is accepted as an alias for <code>history</code>.</p>
`
  },
  {
    id: 'script',
    title: 'NeuralScript language',
    body: `
<h1>NeuralScript</h1>
<p>NeuralScript is a lightweight scripting language built into NeuralCity for running experiments programmatically. It uses <code>do</code>/<code>end</code> blocks instead of curly braces and requires no semicolons.</p>

<h2>Variables and expressions</h2>
<pre><code>let x = 10
let name = "hello"
set x = x + 1
print x</code></pre>

<h2>Control flow</h2>
<pre><code>if x &gt; 0 do
  print "positive"
else do
  print "non-positive"
end

while x &gt; 0 do
  set x = x - 1
end

for i = 0 to 9 do
  print i
end</code></pre>

<h2>Functions</h2>
<pre><code>fn square(n) do
  return n * n
end
print square(6)</code></pre>

<h2>Neural API</h2>
<p>The following standard library functions are available in every script:</p>
<table>
<tr><th>Function</th><th>Description</th></tr>
<tr><td><code>build(spec)</code></td><td>Construct a model from an architecture specification.</td></tr>
<tr><td><code>await(train(spec, data, opts))</code></td><td>Train a model inline. Returns <code>{ state, metrics }</code>.</td></tr>
<tr><td><code>predict(network, input)</code></td><td>Run inference on a network object.</td></tr>
<tr><td><code>thisNet()</code></td><td>Returns the currently selected network.</td></tr>
<tr><td><code>range</code>, <code>len</code>, <code>push</code>, <code>str</code>, <code>num</code>, <code>keys</code>, <code>values</code></td><td>Collection utilities.</td></tr>
<tr><td><code>abs</code>, <code>min</code>, <code>max</code>, <code>sqrt</code>, <code>exp</code>, <code>log</code>, <code>sin</code>, <code>cos</code>, <code>floor</code>, <code>ceil</code>, <code>round</code>, <code>random</code></td><td>Math utilities.</td></tr>
</table>
<h2>Example</h2>
<pre><code>let spec = {
  kind: "classifier",
  inputDim: 2, outputDim: 2,
  hidden: [8], activation: "relu",
  classes: ["false", "true"]
}
let data = { samples: [
  { input: [0,0], label: 0 },
  { input: [0,1], label: 1 },
  { input: [1,0], label: 1 },
  { input: [1,1], label: 0 }
] }
let opts = { optimizer: "adam", learningRate: 0.05, batchSize: 4, epochs: 200, seed: 42 }
let result = await(train(spec, data, opts))
print "training complete"
print result.metrics[len(result.metrics) - 1]</code></pre>
`
  },
  {
    id: 'api',
    title: 'HTTP API',
    body: `
<h1>Serving models over HTTP</h1>
<p>Any trained network can be served on a local HTTP port. Open the <b>API</b> tab, select a network, choose a port (or leave it as <code>0</code> for automatic assignment), and click <b>Start</b>.</p>

<h2>Endpoints</h2>
<table>
<tr><th>Route</th><th>Method</th><th>Description</th></tr>
<tr><td><code>/</code> or <code>/info</code></td><td>GET</td><td>Returns network metadata, input specification, and — for chat models — the available request fields.</td></tr>
<tr><td><code>/predict</code></td><td>POST</td><td>Stateless inference. Accepts a JSON body matching the network's input specification and returns the prediction.</td></tr>
<tr><td><code>/chat</code></td><td>POST</td><td><b>Chat models only.</b> Stateful. The server maintains a conversation thread keyed by <code>sessionId</code>.</td></tr>
<tr><td><code>/chat/reset</code></td><td>POST</td><td>Clears the conversation history for a given session.</td></tr>
</table>

<h2>Examples</h2>
<p><b>Classifier or Regressor:</b></p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input":[0,1]}'</code></pre>
<p><b>Character LM (single-shot completion):</b></p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{"prompt":"the ","maxTokens":80,"temperature":1.0}'</code></pre>

<h2>Multi-turn chat</h2>
<p><b>Stateless</b> — you manage conversation history on the client side:</p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "history": [
      {"role":"user","content":"Hi"},
      {"role":"assistant","content":"Hello!"}
    ],
    "prompt": "How are you?",
    "system": "Be concise.",
    "maxTokens": 200,
    "temperature": 0.7
  }'</code></pre>
<p><b>Stateful</b> — the server maintains the conversation thread:</p>
<pre><code># First turn — the server assigns a sessionId
curl -X POST http://localhost:PORT/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message":"Hi","system":"Be concise."}'
# → {"sessionId":"session-…","reply":"…","history":[…]}

# Subsequent turns — pass the same sessionId
curl -X POST http://localhost:PORT/chat \\
  -H "Content-Type: application/json" \\
  -d '{"sessionId":"session-…","message":"How are you?"}'

# Clear the session when finished
curl -X POST http://localhost:PORT/chat/reset \\
  -H "Content-Type: application/json" \\
  -d '{"sessionId":"session-…"}'</code></pre>
<p>Sessions expire after one hour of inactivity. The server enforces a limit of 256 concurrent sessions per model and 64 turns per session. All session data is held in memory — nothing is written to disk, and sessions are cleared when the API server stops.</p>

<p>To allow other devices on the same network to call your model, share your local IP address. It is shown in the application's status bar.</p>
`
  },
  {
    id: 'encryption',
    title: 'Encryption',
    body: `
<h1>Encrypting a network</h1>
<p>Any saved network can be encrypted at rest with a passphrase. The weights and tokenizer are serialized into a single bundle, then encrypted with AES-256-GCM using a key derived from your passphrase via scrypt. NeuralCity never stores the passphrase. If it is lost, the weights are unrecoverable by design.</p>
<p>To enable encryption, open the <b>Editor</b> tab, toggle <b>Encryption</b>, enter a passphrase, and save. Training and inference on an encrypted network will prompt you to decrypt it first.</p>
<h2>Why this matters</h2>
<p>Models trained on personal or sensitive data carry real exposure if device storage is compromised. Encryption at rest ensures that a stolen or lost machine does not also mean stolen weights.</p>
`
  },
  {
    id: 'philosophy',
    title: 'Why from scratch?',
    body: `
<h1>No frameworks. No abstraction tax.</h1>
<p>Production frameworks are built for scale and generality, which means they sit behind tens of thousands of lines of abstraction, kernel code, and dispatch logic. That depth is appropriate for large-scale deployment. For understanding what a network is actually doing — and for small to medium models where that abstraction provides no benefit — it's just noise.</p>
<p>The NeuralCity engine (see <a href="https://github.com/GuyThatLivesAndCodes/NeuralCity" target="_blank">the source</a>) is approximately <b>2,000 lines of plain JavaScript</b>. You can open <code>src/engine/tensor.js</code>, read the <code>matmul</code> function, and trace every multiplication. You can modify a gradient computation, add a new activation function, or swap in a different optimizer. Nothing is hidden.</p>
<h2>Trade-offs</h2>
<ul>
  <li><b>No GPU acceleration.</b> All computation runs on CPU. Performance is sufficient for models up to a few million parameters; larger models will be slow.</li>
  <li><b>No distributed training.</b> Single machine, single process. This is intentional.</li>
  <li><b>No built-in Transformer blocks.</b> The primitives are all present, and NeuralScript can compose them, but there is no out-of-the-box Transformer layer.</li>
</ul>
<h2>What you get in return</h2>
<ul>
  <li><b>Readable source.</b> Every operator fits on one screen. Every gradient is traceable.</li>
  <li><b>Reproducible runs.</b> Seeded randomness covers initialization, batch shuffling, and sampling.</li>
  <li><b>Local data.</b> Nothing leaves your machine unless you explicitly start the API server.</li>
</ul>
`
  }
];
