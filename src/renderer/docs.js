window.NB_DOCS = [
  {
    id: 'welcome',
    title: 'Welcome',
    body: `
<h1>Welcome to NeuralCabin</h1>
<p>NeuralCabin is a hybrid Electron/Rust neural network workbench — build, train, and deploy networks from the ground up with no external ML frameworks. It supports three training paradigms: supervised gradient descent, deep Q-Learning (DQN) for reinforcement learning, and neuroevolution (Selective Reproduction) for gradient-free evolutionary training.</p>
<p>The compute engine is a two-layer hybrid: a pure-JavaScript autograd system handles the backward pass and graph construction; an optional Rust native binding (<code>native/rust-engine/</code>) accelerates the forward compute with parallelised CPU kernels. The active backend is selected automatically by <code>src/engine/tensor.js</code> — the JS fallback is always available if the Rust binding is absent.</p>
<p>Select a topic from the left sidebar to explore the engine internals, learn how to design and train networks, use Q-Learning or neuroevolution, understand the training data formats, use the production-grade HTTP API, script experiments with NeuralScript, and more.</p>

<h2>Quick start</h2>
<ol>
  <li>Click <code>+ New</code> in the sidebar and choose a template. <strong>XOR Classifier</strong> trains in under a second and is a good first sanity check.</li>
  <li>Open the <strong>Train</strong> tab, paste or verify your training data, and click <strong>Start Training</strong>. The live loss chart updates every epoch.</li>
  <li>Switch to <strong>Inference</strong> to run predictions — or, for chat models, to have a full multi-turn conversation with the trained model.</li>
  <li>Open the <strong>API</strong> tab to serve the model over HTTP so other applications or devices on your local network can query it.</li>
  <li>Use the <strong>Script</strong> tab to automate experiments with NeuralScript — loop over hyperparameters, build and train networks programmatically, and print results.</li>
</ol>

<h2>Application layout</h2>
<table>
<tr><th>Tab</th><th>Purpose</th></tr>
<tr><td><strong>Networks</strong></td><td>Sidebar list of saved networks. Create, rename, duplicate, or delete networks here.</td></tr>
<tr><td><strong>Editor</strong></td><td>Configure architecture (layer sizes, activation, dropout, embedding dim), set a seed, and toggle encryption. Changes here do not apply to an already-trained network until it is retrained.</td></tr>
<tr><td><strong>Train</strong></td><td>Paste training data, set optimizer options (epochs, learning rate, batch size), and start or stop training. The live chart shows loss per epoch.</td></tr>
<tr><td><strong>Inference / Chat</strong></td><td>Run predictions against the trained model. Chat models show a full conversation UI with history, system prompt, and a Reset button.</td></tr>
<tr><td><strong>API</strong></td><td>Start and stop the local HTTP server, choose a port, and view usage examples for the active network type.</td></tr>
<tr><td><strong>Script</strong></td><td>Write and run NeuralScript programs. Useful for hyperparameter sweeps and automated training pipelines.</td></tr>
<tr><td><strong>Docs</strong></td><td>This documentation.</td></tr>
</table>

<h2>Templates</h2>
<p>NeuralCabin ships with five built-in templates to get you started immediately:</p>
<table>
<tr><th>Template</th><th>Type</th><th>Description</th></tr>
<tr><td>XOR Classifier</td><td>Classifier</td><td>Learns the XOR function. Trains in &lt;1 second. Use as a smoke test for any engine change.</td></tr>
<tr><td>2D Spiral</td><td>Classifier</td><td>Two interleaved spiral arms — a classic non-linearly-separable benchmark requiring deeper networks.</td></tr>
<tr><td>Sine Regressor</td><td>Regressor</td><td>Approximates <code>sin(x)</code> on [0, 2π]. Good for exploring regression architecture choices.</td></tr>
<tr><td>Tiny Char LM</td><td>Character LM</td><td>A minimal character-level language model. Feed it a text corpus, sample completions.</td></tr>
<tr><td>Code Predictor</td><td>Character LM</td><td>A char LM preconfigured for small code snippets. Demonstrates using structured text as a corpus.</td></tr>
</table>
`
  },
  {
    id: 'how-it-works',
    title: 'How networks learn',
    body: `
<h1>How a neural network learns</h1>
<p>A neural network is a composition of mathematical functions with <strong>trainable parameters</strong> — weights and biases stored as floating-point numbers inside each layer. Training is the process of repeatedly measuring how wrong the network's predictions are, then nudging every parameter a small amount in the direction that reduces that error. Repeat this thousands of times and the network converges on parameters that generalize to unseen inputs.</p>

<h2>1. Forward pass</h2>
<p>An input vector <code>x</code> flows through each layer in sequence. A <strong>Linear</strong> layer computes <code>y = x·W + b</code> — a matrix multiply followed by a bias add. An <strong>Activation</strong> function such as <code>ReLU</code> then applies a non-linearity (for ReLU, clamping every negative value to zero). Stacking enough Linear + Activation pairs gives the network the capacity to approximate arbitrary continuous functions — this is the universal approximation property.</p>
<p>In NeuralCabin, every operation during the forward pass records its input tensors on the output tensor's <code>_parents</code> list and stores a <code>_backward</code> function that knows how to propagate gradients backward through that specific operation. This builds the <strong>computation graph</strong> dynamically — there is no separate graph-compilation step.</p>

<h2>2. Loss computation</h2>
<p>After the forward pass, the network's output is compared to the ground-truth target to produce a single scalar number called the <strong>loss</strong>. A lower loss means better predictions. Two loss functions are implemented:</p>
<ul>
  <li><strong>Softmax cross-entropy</strong> — used for classification. The raw output logits are converted to a probability distribution via softmax, then the negative log-probability of the correct class is taken. The gradient of this loss with respect to the logits has a clean closed form: <code>∂L/∂logit_j = p_j − 1{j=y}</code>, meaning the gradient is just the predicted probability minus 1 at the correct class index.</li>
  <li><strong>Mean squared error (MSE)</strong> — used for regression. Computes the average of squared differences between predicted and target outputs: <code>L = mean((ŷ − y)²)</code>.</li>
</ul>

<h2>3. Backward pass (autograd)</h2>
<p>NeuralCabin implements <strong>reverse-mode automatic differentiation</strong> (backpropagation). Calling <code>loss.backward()</code> traverses the computation graph in reverse topological order and applies the chain rule at each node to accumulate <code>∂loss/∂W</code> and <code>∂loss/∂b</code> for every trainable parameter.</p>
<p>The traversal is implemented in <code>Tensor.backward()</code>:</p>
<ol>
  <li>Build the topological order by recursively walking <code>_parents</code> (DFS, deduplication by <code>id</code>).</li>
  <li>Seed the loss tensor's own <code>grad</code> with 1 (since <code>∂loss/∂loss = 1</code>).</li>
  <li>Iterate in reverse order, calling each node's stored <code>_backward()</code> function. That function reads <code>out.grad</code> and accumulates into each parent's <code>grad</code> using <code>+=</code> (to handle shared nodes correctly).</li>
</ol>
<p>Each operator — <code>matmul</code>, <code>relu</code>, <code>gelu</code>, <code>softmax</code>, <code>embedding</code>, and others — implements its own local derivative. For the JS path, you can read every one of them in <code>src/engine/tensor-js.js</code>; <code>src/engine/tensor.js</code> is now the backend selector.</p>

<h2>4. Optimizer step</h2>
<p>After backward, every parameter has a populated <code>.grad</code> array. The optimizer uses these gradients to update the parameters. NeuralCabin ships ten optimizers (see the <strong>Optimizers</strong> reference section for full details):</p>
<ul>
  <li><strong>SGD</strong> — momentum-based gradient descent; simple and predictable baseline.</li>
  <li><strong>Adam</strong> — adaptive per-parameter learning rates via first and second gradient moments; the standard default for most tasks.</li>
  <li><strong>AdamW</strong> — Adam with decoupled weight decay; preferred over Adam when regularisation matters.</li>
  <li><strong>RAdam</strong> — Rectified Adam; warms up the adaptive rate automatically so manual warm-up schedules are unnecessary.</li>
  <li><strong>Lion</strong> — sign-only update; lower memory than Adam and often faster on large transformers.</li>
  <li><strong>Adafactor</strong> — factored second-moment estimate; O(r+c) memory instead of O(r·c) for large embedding tables.</li>
  <li><strong>AdamW 8-bit</strong> — AdamW with block-quantised moment buffers; near-identical convergence at half the memory.</li>
  <li><strong>LAMB</strong> — Adam with a per-layer trust ratio; designed for very large batch sizes.</li>
  <li><strong>LARS</strong> — SGD with per-layer local learning rates; standard in distributed image training.</li>
  <li><strong>Ranger</strong> — RAdam combined with Lookahead; slow weights stabilise training and often improve final quality.</li>
</ul>
<p>All optimizer state (moments, slow weights, step counters) is saved alongside the network weights and restored automatically when training resumes. After the optimizer updates the weights, gradients are zeroed via <code>zeroGrad()</code> before the next batch begins.</p>

<h2>5. The training loop</h2>
<p>NeuralCabin's trainer repeats the following for each epoch:</p>
<ol>
  <li>Shuffle the dataset (using the seeded RNG if a seed is set).</li>
  <li>Slice into mini-batches of the configured <code>batchSize</code>.</li>
  <li>For each batch: zero gradients → forward pass → loss → backward → optimizer step.</li>
  <li>Record the mean epoch loss and emit it to the live chart.</li>
</ol>
<p>This is the complete training loop. Nothing more is needed to train the models NeuralCabin supports.</p>

<h2>Reproducibility and seeding</h2>
<p>Setting a <strong>seed</strong> in the Editor tab makes every stochastic step deterministic: weight initialization (Kaiming / scaled Gaussian via Box-Muller), dataset shuffle order, dropout masks, and character-LM temperature sampling all use the same Mulberry32 seeded RNG. Two runs with the same seed, architecture, data, and hyperparameters produce byte-identical results. Omit the seed for non-deterministic runs.</p>
`
  },
  {
    id: 'rust-backend',
    title: 'Rust backend',
    body: `
<h1>Rust backend</h1>
<p>NeuralCabin ships with an optional native Rust acceleration layer. When available, it replaces the JS implementations of all compute-heavy tensor operations (matmul, activations, loss functions, optimizers) with parallelised Rust kernels while keeping the JavaScript autograd graph fully intact.</p>

<h2>Backend selection</h2>
<p><code>src/engine/tensor.js</code> is the runtime backend selector. At startup it reads the <code>NEURALCABIN_ENGINE_BACKEND</code> environment variable:</p>
<table>
<tr><th>Value</th><th>Behaviour</th></tr>
<tr><td><code>auto</code> (default)</td><td>Tries to load the Rust binding. Falls back to JS silently if the binding is absent or incompatible.</td></tr>
<tr><td><code>js</code></td><td>Forces the pure-JavaScript engine unconditionally.</td></tr>
<tr><td><code>rust</code></td><td>Requires the Rust binding. Throws on startup if the binding cannot be loaded.</td></tr>
</table>
<p>To override the binding path (useful during development):</p>
<pre><code>NEURALCABIN_NATIVE_BINDING=/absolute/path/to/neuralcabin_node.node</code></pre>

<h2>Building the Rust engine</h2>
<p>Requires Rust stable ≥ 1.78 and Cargo.</p>
<pre><code># Verify the Rust crate compiles cleanly (no .node output)
npm run engine:check:rust

# Build the N-API .node binding (release mode)
npm run engine:build:rust</code></pre>
<p>On Windows the output is <code>neuralcabin-node.win32-x64-msvc.node</code>. On Linux it is <code>neuralcabin-node.linux-x64-gnu.node</code>. The loader tries each name automatically.</p>

<h2>What Rust accelerates</h2>
<p>All operations below delegate to Rust when the binding is loaded. The autograd backward pass and graph traversal always run in JavaScript.</p>
<table>
<tr><th>Module</th><th>Operations</th></tr>
<tr><td><code>cpu.rs</code> — tensor ops</td><td>add, sub, mul, mul_scalar, matmul, relu, leaky_relu, tanh, sigmoid, gelu (+ tcache), softmax, softmax_cross_entropy (+ probs cache), mse_loss, dropout (+ mask), embedding, sum_all, randn, has_nan_or_inf</td></tr>
<tr><td><code>layers.rs</code> — layer passes</td><td>linear_forward, embedding_forward, embedding_backward (scatter-add), sequential_forward_inference</td></tr>
<tr><td><code>optim.rs</code> — optimizers</td><td>sgd_step, adam_step, adamw_step, clip_grad_norm</td></tr>
<tr><td><code>rl.rs</code> — Q-Learning</td><td>ReplayBuffer, epsilon_greedy, compute_td_targets, dqn_loss, dqn_huber_loss, soft_update_target</td></tr>
<tr><td><code>neuroevolution.rs</code> — Selective Reproduction</td><td>mutate, crossover_uniform/single_point/arithmetic, tournament/roulette/truncation_select, evolve_generation, fitness_stats</td></tr>
</table>

<h2>Performance notes</h2>
<ul>
  <li>All CPU kernels are parallelised with <strong>Rayon</strong>. Worker count matches available CPU cores.</li>
  <li>Matmul and activation kernels use an <strong>AVX2 FMA</strong> inner loop when supported, scalar otherwise.</li>
  <li>The Mulberry32 PRNG in Rust matches the JS <code>rngFromSeed</code> exactly, so switching backends does not break reproducible runs.</li>
  <li>For tiny networks (fewer than ~1 000 parameters) the JS engine is often faster due to N-API call overhead. The Rust path pays off at medium-to-large scale.</li>
</ul>

<h2>JS/Rust hybrid pattern</h2>
<p>The N-API wrapper (<code>native/rust-engine/neuralcabin-node/index.js</code>) spreads the full JS tensor API and then overrides individual compute functions:</p>
<pre><code>// What happens when gelu(x) is called with the Rust backend:
// 1. JS calls native.geluOp(x.data, x.shape)          ← Rust compute
// 2. Rust returns { data, shape, tcache }
// 3. JS builds a Tensor and stores tcache in _backward ← autograd in JS
// 4. backward() uses tcache instead of re-computing tanh</code></pre>
<p>Every op not yet implemented in Rust automatically falls back to the JS implementation — no configuration required.</p>

<h2>Checking the active backend</h2>
<p>The Training log shows the backend when training starts:</p>
<pre><code>training started (backend=rust/binding-loaded)</code></pre>
<p>You can also retrieve backend metadata via IPC:</p>
<pre><code>// In renderer or NeuralScript:
const info = await window.nb.training.backendInfo();
// { mode: 'rust', reason: 'binding-loaded', version: '0.2.0', threads: 8 }</code></pre>
`
  },
  {
    id: 'tensor-engine',
    title: 'Tensor engine internals',
    body: `
<h1>Tensor engine internals</h1>
<p>NeuralCabin supports two tensor backends: a pure-JavaScript implementation (<code>src/engine/tensor-js.js</code>) and a native Rust path (<code>native/rust-engine/</code>). <code>src/engine/tensor.js</code> selects the backend at runtime (see the <strong>Rust backend</strong> topic for build and configuration details). The autograd system (graph construction, backward pass) always runs in JavaScript regardless of which backend is active.</p>

<h2>The Tensor class</h2>
<p>A <code>Tensor</code> holds three things:</p>
<ul>
  <li><code>data</code> — a <code>Float32Array</code> of values stored in row-major order.</li>
  <li><code>shape</code> — a plain JS array of integers, e.g. <code>[4, 8]</code> for a 4×8 matrix.</li>
  <li><code>grad</code> — a <code>Float32Array</code> of the same size as <code>data</code>, populated during backward. <code>null</code> until <code>ensureGrad()</code> or <code>backward()</code> is called.</li>
</ul>
<p>Two autograd fields are also stored on each tensor produced by an operation: <code>_backward</code> (the local gradient function) and <code>_parents</code> (the input tensors that produced it). Together these form the <strong>dynamic computation graph</strong>.</p>

<h2>Constructors</h2>
<table>
<tr><th>Function</th><th>Description</th></tr>
<tr><td><code>tensor(shape, data, requiresGrad)</code></td><td>Wrap an existing array. Copies into a <code>Float32Array</code> if needed.</td></tr>
<tr><td><code>zeros(shape)</code></td><td>All-zero tensor.</td></tr>
<tr><td><code>ones(shape)</code></td><td>All-one tensor.</td></tr>
<tr><td><code>randn(shape, rng)</code></td><td>Standard-normal tensor via Box-Muller transform. Pass a seeded RNG for reproducibility.</td></tr>
<tr><td><code>rngFromSeed(seed)</code></td><td>Returns a Mulberry32 RNG function from an integer seed. Cheap, fast, and good enough for ML use.</td></tr>
</table>

<h2>matmul</h2>
<p><code>matmul(a, b)</code> — matrix multiply with shape <code>[B,K] × [K,N] → [B,N]</code>. This is the dominant operation in training and has been carefully tuned. The loop order is <strong>i-k-j</strong> rather than the naive i-j-k, which means the inner loop sweeps contiguously through both the output row and the B row — enabling V8's JIT to elide bounds checks and achieve near-linear memory access patterns. The backward pass computes <code>dA = dOut · Bᵀ</code> and <code>dB = Aᵀ · dOut</code>, each also using the i-k-j loop order.</p>

<h2>add and sub (with bias broadcast)</h2>
<p><code>add(a, b)</code> handles two shapes: same-shape elementwise addition, and the common case of a 2D activation <code>[B, N]</code> plus a 1D bias <code>[N]</code>. In the bias-add case, the backward sums the upstream gradient across the batch dimension to produce <code>∂L/∂bias</code>. <code>sub(a, b)</code> is the elementwise subtraction equivalent with the appropriate sign flip in the backward.</p>

<h2>Activation ops</h2>
<p>Each activation stores whatever it needs from the forward pass to compute the backward cheaply:</p>
<ul>
  <li><strong>relu</strong> — stores a binary mask (where input was positive). Backward multiplies upstream gradient by the mask.</li>
  <li><strong>leakyRelu</strong> — same as relu but non-zero gradient for negative inputs (slope <code>α = 0.01</code>).</li>
  <li><strong>tanh</strong> — stores the output values. Backward uses <code>1 - out²</code>, avoiding a second <code>Math.tanh</code> call.</li>
  <li><strong>sigmoid</strong> — numerically stable: uses <code>exp(x)/(1+exp(x))</code> for negative x to avoid overflow. Backward uses <code>s(1-s)</code> from cached output.</li>
  <li><strong>gelu</strong> — stores the inner tanh value in a <code>Float32Array</code> cache during forward. Backward computes the full derivative <code>0.5(1+t) + 0.5x·(1-t²)·c·(1+3·0.044715·x²)</code> without any additional transcendental calls.</li>
  <li><strong>softmax</strong> — fused max+exp+normalize in two row sweeps. Backward uses the identity <code>∂L/∂z_j = p_j(∂L/∂p_j − Σ_k p_k · ∂L/∂p_k)</code>, computed via a dot product over the row.</li>
</ul>

<h2>Embedding</h2>
<p><code>embedding(weights, ids)</code> — lookup table mapping integer token IDs to dense row vectors. The forward is a row-copy using <code>TypedArray.set</code> with a subarray view, which V8 compiles to a <code>memmove</code>. The backward scatter-adds upstream gradients back into the weight rows for all token IDs in the batch (IDs can repeat, so accumulation is required).</p>

<h2>Dropout</h2>
<p><code>dropout(a, p, training, rng)</code> — during training, samples a Bernoulli mask per element (probability <code>p</code> of zeroing) and scales surviving values by <code>1/(1-p)</code> (inverted dropout). During inference (<code>training=false</code>), the op is an identity — no mask is applied. The backward simply multiplies the upstream gradient by the same mask.</p>

<h2>Loss functions</h2>
<p><strong>softmaxCrossEntropy(logits, labels)</strong> — fused softmax + NLL loss. Computes log-sum-exp in a numerically stable way, returns a scalar mean loss over the batch, and stores the softmax probabilities for the backward. The backward gradient at logit <code>[i,j]</code> is <code>(p_ij − 1{j=y_i}) / B</code>. In the Rust path, the probs cache is returned alongside the loss and captured in the JS backward closure to avoid recomputing softmax.</p>
<p><strong>mseLoss(a, b)</strong> — mean squared error. Backward gradient at index <code>i</code> is <code>2(a_i − b_i) / N</code>.</p>

<h2>Additional ops</h2>
<p>Beyond the core set documented above, the engine also exposes:</p>
<ul>
  <li><strong>sumAll(a)</strong> — reduces the entire tensor to a scalar. Used internally for gradient norm computation.</li>
  <li><strong>randn(shape, rng)</strong> — standard-normal initialization via Box-Muller transform. The same function exists in the Rust core using Mulberry32 PRNG, producing identical values for the same seed.</li>
  <li><strong>hasNanOrInf(a)</strong> — fast NaN/Inf guard. The Rust path scans in parallel with early exit. Used to detect exploding gradients and unstable training before saving a checkpoint.</li>
  <li><strong>mulScalar(a, s)</strong> — broadcast scalar multiply. Used in gradient clipping and learning rate schedules.</li>
</ul>

<h2>Extending the engine</h2>
<p>Adding a new operation follows a consistent pattern:</p>
<ol>
  <li>Allocate the output tensor with <code>requiresGrad = a.requiresGrad || b.requiresGrad</code>.</li>
  <li>Compute the forward values into <code>out.data</code>. Cache any values needed for backward.</li>
  <li>Set <code>out._parents = [a, b, ...]</code>.</li>
  <li>Set <code>out._backward = () => { /* accumulate into a.grad, b.grad via += */ }</code>.</li>
  <li>Return <code>out</code>.</li>
</ol>
<p>The <code>+=</code> in backward is critical — it allows the same tensor to appear in multiple places in the graph without incorrectly overwriting earlier gradient contributions.</p>
`
  },
  {
    id: 'layers',
    title: 'Layers reference',
    body: `
<h1>Layers reference</h1>
<p>Layers are the building blocks you combine in the Editor tab to define a network's architecture. Each layer type wraps one or more tensor operations and exposes a set of trainable parameters.</p>

<h2>Linear</h2>
<p>Computes <code>y = x·W + b</code>. Input shape <code>[B, in_features]</code>, output shape <code>[B, out_features]</code>. Weights are initialized with Kaiming uniform scaling: <code>W ~ Uniform(−√(1/in), +√(1/in))</code>, which maintains activation variance across layers when followed by ReLU. Biases are initialized to zero.</p>
<p>This is the fundamental trainable layer. Almost every architecture uses one or more Linear layers, usually followed by an activation.</p>

<h2>Activation functions</h2>
<table>
<tr><th>Name</th><th>Formula</th><th>Notes</th></tr>
<tr><td><code>relu</code></td><td>max(0, x)</td><td>Default for hidden layers. Fast, sparse. Can produce dead neurons if learning rate is too high.</td></tr>
<tr><td><code>leakyRelu</code></td><td>max(αx, x), α=0.01</td><td>Use when standard ReLU causes training stagnation due to dead units. Gradient is never exactly zero.</td></tr>
<tr><td><code>tanh</code></td><td>(eˣ−e⁻ˣ)/(eˣ+e⁻ˣ)</td><td>Smooth, bounded output in (−1, 1). Works well in small regressors. Can saturate for large inputs.</td></tr>
<tr><td><code>sigmoid</code></td><td>1/(1+e⁻ˣ)</td><td>Output in (0, 1). Use only for binary output heads. Saturates and suffers from vanishing gradients in deep stacks.</td></tr>
<tr><td><code>gelu</code></td><td>0.5x·(1+tanh(√(2/π)(x+0.044715x³)))</td><td>Smooth non-linearity with near-zero suppression. Common in language model architectures. Slightly slower than ReLU.</td></tr>
<tr><td><code>softmax</code></td><td>eˣⁱ/Σeˣʲ</td><td>Converts logit vector to probability distribution. Use only on the final output layer for classification — never in hidden layers.</td></tr>
</table>

<h2>Dropout</h2>
<p>Randomly sets a fraction <code>p</code> of activations to zero during training and scales the rest by <code>1/(1−p)</code>. The network learns to be robust to missing units, reducing overfitting. Common values are <code>p = 0.1</code> to <code>0.5</code> — start at 0.1 or 0.2 for small networks. Dropout is automatically disabled at inference time; no manual flag is needed.</p>
<p><strong>Where to place Dropout:</strong> Insert it after activation layers in the body of the network, not on the input or immediately before the final output. For language models, placing a dropout after the embedding layer is conventional.</p>

<h2>Embedding</h2>
<p>Maps integer token IDs to dense learned vectors of size <code>embDim</code>. The weight matrix has shape <code>[vocabSize, embDim]</code> and is initialized with small random values. Used as the first layer of character LMs and chat models to convert discrete token indices into continuous representations the rest of the network can process.</p>
<p>The embedding lookup is equivalent to a one-hot multiplied by the weight matrix, but implemented as a direct row-copy for efficiency.</p>

<h2>Architecture guidelines</h2>
<table>
<tr><th>Task</th><th>Suggested architecture</th></tr>
<tr><td>Simple classifier (XOR, iris-scale)</td><td>Linear(in, 8–16) → ReLU → Linear(16, out)</td></tr>
<tr><td>2D spiral / moderately non-linear</td><td>2–3 hidden layers, 32–64 units, ReLU or GELU</td></tr>
<tr><td>Regression (sine, polynomial)</td><td>2 hidden layers, 16–64 units, tanh or ReLU</td></tr>
<tr><td>Character LM (small corpus)</td><td>Embedding(vocab, 32) → Linear(32·ctx, 128) → GELU → Linear(128, vocab)</td></tr>
<tr><td>Chat assistant</td><td>Embedding(vocab, 64) → 2× Linear(64·ctx, 256) → GELU → Linear(256, vocab); contextLen 96–256</td></tr>
</table>
<p>These are starting points. The right architecture depends on the complexity of the data. When in doubt, start smaller than you think you need and increase capacity only if training loss stalls at an unacceptably high value.</p>
`
  },
  {
    id: 'optimizers',
    title: 'Optimizers',
    body: `
<h1>Optimizers</h1>
<p>The optimizer updates the network's weights after each backward pass using the accumulated gradients. NeuralCabin provides ten optimizers. All optimizer state (momentum buffers, variance estimates, slow weights) is saved alongside the model weights so training can resume without a warmup bump.</p>

<h2>SGD — Stochastic Gradient Descent</h2>
<p>The simplest baseline. Subtracts a scaled gradient from each weight, with optional momentum:</p>
<pre><code>v ← momentum·v + g
W ← W − lr·v</code></pre>
<p>Predictable and interpretable. Requires careful lr tuning. Use when you want to study raw gradient dynamics or as a reference baseline.</p>

<h2>Adam — Adaptive Moment Estimation</h2>
<p>Maintains per-parameter first moment <code>m</code> (gradient EMA) and second moment <code>v</code> (squared gradient EMA) with bias correction:</p>
<pre><code>m̂ = m / (1−β₁ᵗ),  v̂ = v / (1−β₂ᵗ)
W ← W − lr · m̂ / (√v̂ + ε)</code></pre>
<p>Robust to lr choice. Good default for most tasks. Weight decay in Adam is coupled to the adaptive rate — use AdamW if you want true L2 decay.</p>

<h2>AdamW — Decoupled Weight Decay</h2>
<p>Identical to Adam but weight decay is applied directly to the weights rather than being mixed into the gradient. This is the correct way to implement L2 regularization with adaptive optimizers:</p>
<pre><code>W ← (1 − lr·λ)·W − lr · m̂ / (√v̂ + ε)</code></pre>
<p><strong>Recommended default for language models and any task where you want regularization.</strong> Default <code>weightDecay = 0.01</code>.</p>

<h2>RAdam — Rectified Adam</h2>
<p>Computes the variance of the adaptive learning rate analytically each step. When the approximated SMA length <code>ρ_t &gt; 5</code> (variance is tractable), applies a rectification factor; otherwise falls back to SGD-with-momentum. This eliminates the bad initial variance that can destabilize Adam in the first few hundred steps:</p>
<pre><code>ρ_t = ρ_∞ − 2t·β₂ᵗ/(1−β₂ᵗ)
if ρ_t > 5: W ← W − lr·rect·m̂/(√(v/bc₂)+ε)
else:       W ← W − lr·m̂</code></pre>
<p>Good when you find Adam unstable early in training. No warmup schedule needed.</p>

<h2>Ranger — RAdam + Lookahead</h2>
<p>Combines RAdam (inner optimizer) with Lookahead (outer averaging). Every <code>k=6</code> inner steps, Lookahead interpolates slow weights toward fast weights and resets fast ← slow:</p>
<pre><code>slow_w += α · (fast_w − slow_w);  fast_w = slow_w</code></pre>
<p>This smooths the trajectory through the loss landscape, often improving final accuracy. A strong general-purpose choice. Default <code>lookaheadK=6, lookaheadAlpha=0.5</code>.</p>

<h2>Lion — EvoLved Sign Optimizer</h2>
<p>Uses only the sign of the momentum interpolation — no second moment. Half the optimizer state of Adam:</p>
<pre><code>c = β₁·m + (1−β₁)·g         # update signal
W ← (1−lr·λ)·W − lr·sign(c) # weight decay + sign step
m ← β₂·m + (1−β₂)·g         # update momentum</code></pre>
<p>Requires a much smaller learning rate than Adam (3–10× smaller, e.g. <code>1e-4</code>). Strong on language tasks. Memory-efficient for large models. Default <code>lr=1e-4, β₁=0.9, β₂=0.99</code>.</p>

<h2>Adafactor — Factored Second Moment</h2>
<p>For 2D weight matrices, factors the second moment V into row and column statistics instead of storing a full per-element V matrix — O(r+c) memory instead of O(r·c). For bias/1D parameters, uses a standard scalar second moment. Includes gradient clipping via RMS threshold:</p>
<pre><code>V̂[i,j] ≈ Vr[i]·Vc[j] / mean(Vr)
u = g / √V̂;  W ← W − (lr/max(1,rms(u)/d))·u</code></pre>
<p>Ideal for large embedding tables or wide weight matrices where Adam's second moment buffer is a memory bottleneck. Default <code>clipThreshold=1.0</code>.</p>

<h2>AdamW 8-bit</h2>
<p>Identical update rule to AdamW, but stores the <code>m</code> and <code>v</code> buffers as 8-bit integers with block-wise float32 scales (block size 256). Reduces optimizer state memory ~4× at the cost of slight quantization error in the momentum estimates. Recommended for large charLM models where RAM is a concern.</p>

<h2>LAMB — Layer-wise Adaptive Moments</h2>
<p>Adam update scaled by a per-parameter trust ratio <code>‖w‖/‖r‖</code>, where <code>r</code> is the Adam update plus L2 term. Keeps the effective step size proportional to the weight norm and is designed for large batch training:</p>
<pre><code>r = m̂/(√v̂+ε) + λ·w
trust = ‖w‖/‖r‖  (or 1 if either is 0)
W ← W − lr·trust·r</code></pre>
<p>Use when training with very large effective batch sizes (many workers) or when Adam produces inconsistent gradient scales across layers.</p>

<h2>LARS — Layer-wise Adaptive Rate Scaling</h2>
<p>Extends SGD with momentum by computing a per-parameter local learning rate via a trust ratio. Prevents large gradient norms in some layers from overwhelming small norms in others:</p>
<pre><code>local_lr = lr·η·‖w‖/(‖g‖ + λ·‖w‖)
v ← momentum·v + local_lr·(g + λ·w)
W ← W − v</code></pre>
<p>Best for convex optimization or when distributing training across many machines with very large batches. Default <code>eta=1e-3</code> (trust coefficient).</p>

<h2>Optimizer state and resuming</h2>
<p>All optimizer state is saved to disk alongside the model weights. When you click <strong>Continue training</strong>, the optimizer picks up from exactly where it left off — including momentum buffers and step counters. Changing the optimizer type between sessions resets the state (the type mismatch is detected automatically).</p>

<h2>Hyperparameter quick reference</h2>
<table>
<tr><th>Optimizer</th><th>lr</th><th>Key params</th><th>Best for</th></tr>
<tr><td>Adam</td><td>1e-3</td><td>β₁=0.9, β₂=0.999</td><td>General default</td></tr>
<tr><td>AdamW</td><td>1e-3</td><td>weightDecay=0.01</td><td>LM / regularized tasks</td></tr>
<tr><td>RAdam</td><td>1e-3</td><td>same as Adam</td><td>Unstable early training</td></tr>
<tr><td>Ranger</td><td>1e-3</td><td>k=6, α=0.5</td><td>Strong general-purpose</td></tr>
<tr><td>Lion</td><td>1e-4</td><td>β₁=0.9, β₂=0.99</td><td>Memory-efficient LM</td></tr>
<tr><td>Adafactor</td><td>1e-3</td><td>clipThreshold=1</td><td>Large embeddings</td></tr>
<tr><td>AdamW 8-bit</td><td>1e-3</td><td>weightDecay=0.01</td><td>RAM-constrained large models</td></tr>
<tr><td>LAMB</td><td>1e-3</td><td>weightDecay=0.01</td><td>Large effective batch size</td></tr>
<tr><td>LARS</td><td>0.1</td><td>eta=1e-3, momentum=0.9</td><td>Very large batch / distributed</td></tr>
<tr><td>SGD</td><td>0.01</td><td>momentum=0.9</td><td>Baseline / interpretability</td></tr>
</table>
`
  },
  {
    id: 'rl',
    title: 'Q-Learning / DQN',
    body: `
<h1>Q-Learning and Deep Q-Networks</h1>
<p>NeuralCabin includes a complete Deep Q-Network (DQN) implementation for reinforcement learning tasks. The agent learns to choose actions that maximise long-term reward by approximating a Q-value function with a neural network.</p>

<h2>Core concepts</h2>
<ul>
  <li><strong>Q-value</strong> — Q(s, a) is the expected discounted future reward for taking action a in state s and then following the optimal policy.</li>
  <li><strong>Bellman target</strong> — r + γ · max_a' Q_target(s', a'), where γ (gamma) is the discount factor. The online network trains toward this target.</li>
  <li><strong>Experience replay</strong> — transitions (s, a, r, s', done) are stored in a circular buffer and sampled randomly to break temporal correlations.</li>
  <li><strong>Target network</strong> — a separate copy of the network updated slowly (soft update or periodic hard copy) to stabilise training.</li>
  <li><strong>ε-greedy policy</strong> — with probability ε pick a random action (explore), otherwise pick the action with highest Q-value (exploit). ε decays over training.</li>
</ul>

<h2>DQNAgent quick start</h2>
<pre><code>const { DQNAgent } = require('./src/engine/rl');

const agent = new DQNAgent({
  architecture: {
    kind:      'classifier',
    inputDim:  4,     // observation space dimension
    outputDim: 2,     // number of discrete actions
    hidden:    [64, 64]
  },
  gamma:            0.99,   // discount factor
  lr:               1e-3,
  batchSize:        64,
  epsilonStart:     1.0,
  epsilonEnd:       0.05,
  epsilonDecay:     0.995,  // multiplicative decay per trainStep
  targetUpdateFreq: 100,    // hard-sync every N trainSteps
  seed:             42,
});

// Episode loop
let state = env.reset();
for (let step = 0; step &lt; maxSteps; step++) {
  const action              = agent.selectAction(state);          // ε-greedy
  const { nextState, reward, done } = env.step(action);
  agent.observe(state, action, reward, nextState, done);          // store transition
  const loss                = agent.trainStep();                  // null until buffer ready
  if (done) break;
  state = nextState;
}</code></pre>

<h2>Soft target sync</h2>
<p>In addition to the periodic hard-copy (controlled by <code>targetUpdateFreq</code>), you can Polyak-average the target network each step:</p>
<pre><code>agent.softSyncTarget(0.005); // θ_target ← (1−τ)·θ_target + τ·θ_online</code></pre>
<p>Soft updates with τ ≈ 0.005 are smoother than periodic hard copies and preferred in most modern DQN variants.</p>

<h2>Saving and loading</h2>
<pre><code>const snapshot = agent.toJSON();

const agent2 = DQNAgent.fromJSON(snapshot, { lr: 1e-3, seed: 42 });</code></pre>

<h2>ReplayBuffer (standalone)</h2>
<pre><code>const { ReplayBuffer } = require('./src/engine/rl');
const buf = new ReplayBuffer(10000, 4, 1); // capacity, stateDim, actionDim

buf.push(state, action, reward, nextState, done);
const batch = buf.sample(64, seed);
// batch: { states, actions, rewards, nextStates, dones }</code></pre>

<h2>DQNAgent constructor options</h2>
<table>
<tr><th>Option</th><th>Default</th><th>Description</th></tr>
<tr><td><code>architecture</code></td><td>required</td><td>Network spec (kind, inputDim, outputDim, hidden[])</td></tr>
<tr><td><code>gamma</code></td><td>0.99</td><td>Discount factor for future rewards</td></tr>
<tr><td><code>lr</code></td><td>1e-3</td><td>Learning rate for the online Q-network</td></tr>
<tr><td><code>batchSize</code></td><td>64</td><td>Transitions sampled per trainStep</td></tr>
<tr><td><code>epsilonStart</code></td><td>1.0</td><td>Initial ε (fully random at start)</td></tr>
<tr><td><code>epsilonEnd</code></td><td>0.05</td><td>Minimum ε after decay</td></tr>
<tr><td><code>epsilonDecay</code></td><td>0.995</td><td>Multiplicative decay per trainStep</td></tr>
<tr><td><code>targetUpdateFreq</code></td><td>100</td><td>Hard-sync target every N trainSteps (0 = never)</td></tr>
<tr><td><code>replayCapacity</code></td><td>10000</td><td>Maximum transitions in the replay buffer</td></tr>
<tr><td><code>seed</code></td><td>null</td><td>RNG seed for deterministic action selection and buffer sampling</td></tr>
</table>

<h2>Rust acceleration</h2>
<p>When the Rust backend is active, the following RL operations run natively:</p>
<ul>
  <li><code>epsilon_greedy</code> — action selection from Q-value array</li>
  <li><code>compute_td_targets</code> — vectorised Bellman targets: r + γ · max Q(s')</li>
  <li><code>dqn_loss</code> / <code>dqn_huber_loss</code> — MSE or Huber loss over chosen-action Q values + per-sample gradient</li>
  <li><code>soft_update_target</code> — Polyak averaging over the full parameter vector</li>
  <li><code>ReplayBuffer.sample</code> — random mini-batch sampling with Mulberry32 seed</li>
</ul>
`
  },
  {
    id: 'neuroevolution',
    title: 'Neuroevolution',
    body: `
<h1>Neuroevolution — Selective Reproduction</h1>
<p>Neuroevolution trains neural networks without gradient descent. A population of networks evolves over generations: high-fitness individuals survive and reproduce; low-fitness ones are replaced. NeuralCabin implements a full evolutionary loop with three selection strategies, three crossover strategies, Gaussian or uniform mutation, and configurable elitism.</p>

<h2>When to use neuroevolution</h2>
<ul>
  <li>The reward signal is <strong>non-differentiable</strong> — e.g. a game score, win/loss outcome, or a physics simulation you can only evaluate, not differentiate through.</li>
  <li>The search space is <strong>multi-modal</strong> — gradient descent can get stuck in local minima; evolutionary algorithms explore broader regions.</li>
  <li>You want to train <strong>without a loss function</strong> — just define a fitness function that scores each network's behaviour as a number.</li>
</ul>

<h2>Population quick start</h2>
<pre><code>const { Population } = require('./src/engine/neuroevolution');

const pop = new Population({
  architecture: { kind: 'classifier', inputDim: 4, outputDim: 2, hidden: [32, 32] },
  size:        50,    // number of individuals
  eliteCount:  5,     // top-N survive unchanged each generation
  pMutate:     0.1,   // per-weight mutation probability
  mutationStd: 0.02,  // Gaussian mutation standard deviation
  tournamentK: 3,     // tournament pool size
  seed:        42,
});

for (let gen = 0; gen &lt; 200; gen++) {
  // Score each individual — fitnessFn(model, index) → number (higher = better)
  pop.evaluate((model, i) => runEpisode(model));

  const stats = pop.evolve();
  // stats: { min, max, mean, std }
}

const best = pop.getBest();</code></pre>

<h2>Async convenience wrapper</h2>
<pre><code>const { evolveNetwork } = require('./src/engine/neuroevolution');

const result = await evolveNetwork(
  pop,
  200,                                        // max generations
  async (model) => runEpisodeAsync(model),
  {
    onGeneration: ({ generation, stats }) => console.log(generation, stats.max),
    shouldStop:   () => earlyStopFlag,        // optional early-exit predicate
  }
);
// result: { best, generation, stats }</code></pre>

<h2>Selection strategies</h2>
<table>
<tr><th>Strategy</th><th>Description</th><th>Best for</th></tr>
<tr><td>Tournament (default)</td><td>Pick k random individuals; the fittest becomes a parent. k controls selection pressure.</td><td>Most tasks. k = 3–5 is a good starting point.</td></tr>
<tr><td>Roulette</td><td>Probability of selection ∝ fitness (fitness-proportionate).</td><td>When fitnesses are positive and well-spread. Fails if fitness is zero or negative.</td></tr>
<tr><td>Truncation</td><td>Only the top-N% can become parents.</td><td>High-pressure selection; converges faster but reduces diversity.</td></tr>
</table>

<h2>Crossover strategies</h2>
<table>
<tr><th>Strategy</th><th>Description</th></tr>
<tr><td>Uniform (default)</td><td>Each weight is taken from parent 1 or parent 2 with equal probability (coin flip per weight).</td></tr>
<tr><td>Single-point</td><td>A random split divides the weight vector; one half comes from each parent.</td></tr>
<tr><td>Arithmetic</td><td>Weighted average: child = α·p1 + (1−α)·p2. Produces smooth interpolation.</td></tr>
</table>

<h2>Mutation</h2>
<ul>
  <li><strong>Gaussian</strong> (default) — each selected weight ← w + N(0, mutationStd). Standard deviation controls magnitude.</li>
  <li><strong>Uniform</strong> — each selected weight ← w + Uniform(−scale, +scale). Configured via <code>mutate_uniform</code> in the Rust core.</li>
</ul>

<h2>Elitism</h2>
<p>The top <code>eliteCount</code> individuals by fitness are copied to the next generation unchanged, before selection/crossover/mutation runs. This guarantees the best solution never regresses. Values of 1–5 are typical; higher values reduce diversity.</p>

<h2>Saving and loading</h2>
<pre><code>const snapshot = pop.toJSON();
// { architecture, size, eliteCount, pMutate, mutationStd, seed, individuals: [...] }

const restored = Population.fromJSON(snapshot);</code></pre>

<h2>Population constructor options</h2>
<table>
<tr><th>Option</th><th>Default</th><th>Description</th></tr>
<tr><td><code>architecture</code></td><td>required</td><td>Network spec shared by all individuals</td></tr>
<tr><td><code>size</code></td><td>50</td><td>Population size</td></tr>
<tr><td><code>eliteCount</code></td><td>2</td><td>Top-N copied unchanged each generation</td></tr>
<tr><td><code>pMutate</code></td><td>0.05</td><td>Per-weight mutation probability</td></tr>
<tr><td><code>mutationStd</code></td><td>0.02</td><td>Gaussian mutation standard deviation</td></tr>
<tr><td><code>tournamentK</code></td><td>3</td><td>Tournament pool size</td></tr>
<tr><td><code>seed</code></td><td>null</td><td>RNG seed for reproducible evolution</td></tr>
</table>

<h2>Rust acceleration</h2>
<p>When the Rust backend is active, <code>pop.evolve()</code> runs the entire generation in Rust — selection, crossover, mutation, and elitism — as a single vectorised call over the flattened population parameter array. This is significantly faster than the JS fallback for large populations.</p>
`
  },
  {
    id: 'tokenizers',
    title: 'Tokenization',
    body: `
<h1>Tokenization for Language Models</h1>
<p>For Character LM and Chat networks, the tokenizer converts raw text to a sequence of integer IDs before training, and converts IDs back to text during inference. The choice of tokenization strategy directly affects vocab size, context efficiency, and model capacity. Select the tokenizer in the <strong>Editor</strong> tab under Architecture.</p>

<h2>Character — one token per character</h2>
<p>Every unique Unicode character in the corpus becomes one token. A typical English corpus produces a vocab of 60–120 tokens.</p>
<ul>
  <li><strong>Context:</strong> <code>contextLen=32</code> covers 32 characters — about 5–8 words.</li>
  <li><strong>Pros:</strong> Tiny vocab, no out-of-vocabulary problem, perfect reconstruction.</li>
  <li><strong>Cons:</strong> The model must learn spelling, word boundaries, and grammar independently. Requires many steps per "semantic unit". Best for very small corpora or when character-level precision matters (e.g. code generation).</li>
</ul>

<h2>Word-part — BPE subword (default)</h2>
<p>Byte-Pair Encoding (Sennrich et al., 2016). Starts from a character vocabulary and iteratively merges the most frequent adjacent pair into a new token, up to a target vocabulary size (default: 512). The resulting vocabulary contains common character sequences, syllables, prefixes, and whole words.</p>
<ul>
  <li><strong>Context:</strong> <code>contextLen=32</code> covers roughly 32 subword tokens — about 20–30 words depending on the corpus.</li>
  <li><strong>Pros:</strong> Balances vocab size and context efficiency. Handles rare words gracefully by falling back to sub-character merges. Most LLMs (GPT, BERT, LLaMA) use BPE variants.</li>
  <li><strong>Cons:</strong> Vocab and merges are corpus-specific — changing the corpus forces a rebuild of the tokenizer and model.</li>
  <li><strong>Best for:</strong> Most language model tasks. The recommended default.</li>
</ul>

<h2>Word — one token per whitespace-separated token</h2>
<p>Splits text on <code>/\S+|\s+/</code> (alternating runs of non-space and space characters). Each unique word and whitespace run is a distinct token.</p>
<ul>
  <li><strong>Context:</strong> <code>contextLen=16</code> covers 16 words — a meaningful semantic window for short conversations or sentences.</li>
  <li><strong>Pros:</strong> Simple, interpretable, efficient context use per word.</li>
  <li><strong>Cons:</strong> Vocabulary can be very large for diverse corpora; any word not seen at training time is silently dropped at inference. Best for small, controlled vocabulary corpora.</li>
</ul>

<h2>Changing tokenizer type</h2>
<p>Changing the tokenizer resets the vocabulary and requires retraining from scratch — the saved weights are invalidated automatically when you save the changed architecture. The old trained model is incompatible because the embedding table rows (one per token ID) no longer correspond to the same tokens.</p>

<h2>Extending vocabulary</h2>
<p>When you continue training with a corpus that contains tokens not in the saved vocabulary:</p>
<ul>
  <li><strong>Character:</strong> New characters are appended to the vocab (existing IDs stay stable). The model is rebuilt to fit the new vocab size — existing weights may lose a few epochs of performance.</li>
  <li><strong>Word:</strong> New words are appended. Same rebuild behavior.</li>
  <li><strong>Word-part (BPE):</strong> BPE merges cannot be extended incrementally; the tokenizer is rebuilt from the full new corpus. Model rebuilds from scratch.</li>
</ul>
`
  },
  {
    id: 'data',
    title: 'Training data',
    body: `
<h1>Training data formats</h1>
<p>Each network type expects a specific JSON structure. Paste your training data into the <strong>Train</strong> tab's data field. All formats are plain JSON.</p>

<h2>Classifier</h2>
<p>An array of <code>{ input, label }</code> objects. <code>input</code> is a numeric array with length equal to the network's <code>inputDim</code>. <code>label</code> is an integer class index starting at 0.</p>
<pre><code>[
  { "input": [0, 0], "label": 0 },
  { "input": [0, 1], "label": 1 },
  { "input": [1, 0], "label": 1 },
  { "input": [1, 1], "label": 0 }
]</code></pre>
<p>Class names are defined in the Editor tab's <strong>Classes</strong> field (comma-separated). The number of classes must match the network's <code>outputDim</code>.</p>

<h2>Regressor</h2>
<p>An array of <code>{ input, output }</code> objects. Both are numeric arrays. Their lengths must match <code>inputDim</code> and <code>outputDim</code> respectively.</p>
<pre><code>[
  { "input": [0.0],  "output": [0.000] },
  { "input": [0.5],  "output": [0.479] },
  { "input": [1.57], "output": [1.000] },
  { "input": [3.14], "output": [0.001] }
]</code></pre>

<h2>Character LM (free text)</h2>
<p>A single object with a <code>text</code> string, or an array of such objects for a multi-source corpus. The model builds a character-level vocabulary automatically from the full corpus — every unique character becomes a token.</p>
<pre><code>{ "text": "The quick brown fox jumps over the lazy dog." }</code></pre>
<p>Or as a multi-source corpus:</p>
<pre><code>[
  { "text": "First document content..." },
  { "text": "Second document content..." }
]</code></pre>
<p>Larger and more varied corpora produce richer, more coherent language models. The model trains on overlapping windows of length <code>contextLen</code> extracted from the flattened corpus text.</p>

<h2>Chat Assistant — single-turn pairs</h2>
<p>The simplest chat format: a <code>samples</code> array of <code>{ user, assistant }</code> pairs.</p>
<pre><code>{
  "samples": [
    { "user": "Hello, I need help.", "assistant": "Of course — what's up?" },
    { "user": "What can you do?",    "assistant": "I can discuss anything you teach me." },
    { "user": "Thanks!",             "assistant": "Anytime." }
  ]
}</code></pre>

<h2>Chat Assistant — multi-turn (messages array)</h2>
<p>For teaching the model to maintain context across a conversation, include samples with multiple alternating turns. An optional <code>system</code> role sets a persistent persona or constraint.</p>
<pre><code>{
  "samples": [
    { "messages": [
        { "role": "system",    "content": "You are a concise assistant." },
        { "role": "user",      "content": "Hi" },
        { "role": "assistant", "content": "Hello! How can I help?" },
        { "role": "user",      "content": "What's 2 + 2?" },
        { "role": "assistant", "content": "4." }
    ] }
  ]
}</code></pre>

<h2>Chat Assistant — alternating conversation array</h2>
<pre><code>{
  "samples": [
    { "conversation": [
        { "user": "Hi" },
        { "assistant": "Hey!" },
        { "user": "Tell me something interesting." },
        { "assistant": "Did you know honey never spoils?" }
    ] }
  ]
}</code></pre>

<h2>Internal encoding</h2>
<p>All three chat formats are normalized at training time into a single role-tagged flat stream:</p>
<pre><code>&lt;|system|&gt;You are a concise assistant.&lt;|end|&gt;
&lt;|user|&gt;Hi&lt;|end|&gt;&lt;|assistant|&gt;Hello! How can I help?&lt;|end|&gt;
&lt;|user|&gt;What's 2 + 2?&lt;|end|&gt;&lt;|assistant|&gt;4.&lt;|end|&gt;</code></pre>
<p>At inference time, incoming messages are wrapped in the same tags. Generation stops as soon as the model emits <code>&lt;|end|&gt;</code>, so only the assistant's reply is returned.</p>

<h2>Why multi-turn samples matter</h2>
<p>A model trained only on single-turn pairs learns to respond to fresh prompts but never learns to continue an ongoing thread. To build a model that holds context, include samples with 2, 4, or 6+ alternating turns. The trainer flattens each into the tag-delimited stream, so the model sees the full <code>user → assistant → user → assistant</code> pattern during training rather than only isolated Q&A pairs.</p>
<p><strong>Tip:</strong> Set <code>contextLen</code> (in the Editor) to at least 2× the expected average conversation length in characters. A value of <code>96</code>–<code>256</code> covers most short conversations; use <code>512</code>+ for paragraph-length responses or long dialogue chains.</p>

<h2>Dataset size guidance</h2>
<table>
<tr><th>Task</th><th>Minimum samples</th><th>Recommended</th></tr>
<tr><td>Classifier (simple, 2 class)</td><td>8–20</td><td>100+</td></tr>
<tr><td>Classifier (complex, 5+ class)</td><td>50</td><td>500+</td></tr>
<tr><td>Regressor</td><td>20</td><td>200+</td></tr>
<tr><td>Character LM</td><td>~500 chars</td><td>10 000+ chars</td></tr>
<tr><td>Chat assistant</td><td>30 pairs</td><td>200–600+ pairs, mix of single and multi-turn</td></tr>
</table>
`
  },
  {
    id: 'chat-inference',
    title: 'Chatting with a model',
    body: `
<h1>Multi-turn inference</h1>
<p>Once a network has been trained on chat data, the <strong>Inference</strong> tab becomes a full multi-turn chat interface with a conversation history panel, a persistent system prompt field, temperature and max-tokens controls, and a <strong>Reset chat</strong> button. Each message is encoded together with all prior turns so the model has access to earlier context when generating a reply.</p>

<h2>How history is encoded</h2>
<p>Before each generation call, the full conversation is flattened into the same role-tagged format the model was trained on:</p>
<pre><code>&lt;|system|&gt;Be concise.&lt;|end|&gt;
&lt;|user|&gt;Hi&lt;|end|&gt;&lt;|assistant|&gt;Hello!&lt;|end|&gt;
&lt;|user|&gt;How are you?&lt;|end|&gt;&lt;|assistant|&gt;</code></pre>
<p>The model sees this as its context and continues from the trailing <code>&lt;|assistant|&gt;</code>. Generation proceeds character by character using temperature-scaled softmax sampling, stopping when the model emits the <code>&lt;|end|&gt;</code> token or when <code>maxTokens</code> is reached.</p>

<h2>History truncation</h2>
<p>When the conversation grows longer than <code>contextLen</code> characters, the oldest turns are dropped first. Rules:</p>
<ul>
  <li>The system prompt (if set) is always preserved at the start.</li>
  <li>Turns are never split in the middle — whole turns are dropped.</li>
  <li>The current user message is always preserved at the end.</li>
</ul>
<p>This means long conversations degrade gracefully: the model retains the most recent context and the framing system prompt while discarding older history it can no longer fit.</p>

<h2>Generation parameters</h2>
<table>
<tr><th>Parameter</th><th>Effect</th><th>Typical range</th></tr>
<tr><td><code>temperature</code></td><td>Scales logits before softmax sampling. Higher = more random, lower = more deterministic (greedy at 0).</td><td>0.5–1.2 for chat; 0.3–0.7 for factual tasks</td></tr>
<tr><td><code>maxTokens</code></td><td>Hard cutoff on generated characters before <code>&lt;|end|&gt;</code>.</td><td>80–400 depending on expected response length</td></tr>
</table>

<h2>System prompt</h2>
<p>The system prompt is prepended as a <code>&lt;|system|&gt;...&lt;|end|&gt;</code> turn at the start of every context window. Use it to establish persona, tone, or constraints. Keep it short — it consumes context space on every turn. If the model was not trained on system prompts (no <code>system</code> role in training data), this field has no meaningful effect.</p>

<h2>Programmatic access from NeuralScript</h2>
<p>From the Script tab, pass an explicit history object to <code>predict</code>:</p>
<pre><code>predict(thisNet(), {
  history: [
    { role: "user",      content: "Hi" },
    { role: "assistant", content: "Hello!" }
  ],
  prompt:      "How are you?",
  system:      "Be concise.",
  maxTokens:   200,
  temperature: 0.7
})</code></pre>
<p>The <code>messages</code> field is accepted as an alias for <code>history</code> for OpenAI-API compatibility.</p>

<h2>Improving chat quality</h2>
<ul>
  <li><strong>More data, more turns.</strong> The single biggest lever. A model trained on 30 single-turn pairs will behave very differently from one trained on 400 multi-turn conversations.</li>
  <li><strong>Consistent formatting.</strong> Decide how the assistant phrases things and keep it consistent across samples. Contradictory training data confuses the model.</li>
  <li><strong>Increase contextLen.</strong> If the model seems to forget earlier parts of the conversation, increase <code>contextLen</code> in the Editor and retrain.</li>
  <li><strong>Lower temperature.</strong> If responses are incoherent or drifting, reduce temperature to 0.5–0.7.</li>
  <li><strong>More epochs.</strong> Character LMs and chat models often need 500–2000 epochs to converge on small datasets. If the loss is still decreasing, keep training.</li>
</ul>
`
  },
  {
    id: 'tokenizer',
    title: 'Tokenizer',
    body: `
<h1>Tokenizer</h1>
<p>NeuralCabin uses a <strong>character-level tokenizer</strong>. Every unique character in the training corpus becomes a token — there is no subword splitting or BPE. This keeps the implementation simple and the vocabulary fully interpretable, at the cost of longer sequence lengths compared to word-piece tokenizers.</p>

<h2>Vocabulary construction</h2>
<p>When a character LM or chat model is trained, the tokenizer scans the full training corpus and builds a sorted list of unique characters. Four special tokens are added automatically:</p>
<table>
<tr><th>Token</th><th>Purpose</th></tr>
<tr><td><code>&lt;|user|&gt;</code></td><td>Marks the start of a user turn in chat models.</td></tr>
<tr><td><code>&lt;|assistant|&gt;</code></td><td>Marks the start of an assistant turn.</td></tr>
<tr><td><code>&lt;|system|&gt;</code></td><td>Marks the start of a system prompt turn.</td></tr>
<tr><td><code>&lt;|end|&gt;</code></td><td>Marks the end of any turn. Generation stops here.</td></tr>
</table>
<p>The full vocabulary is serialized and saved with the model weights so the tokenizer state is always consistent between training and inference sessions.</p>

<h2>Append-only vocab extension</h2>
<p>If you retrain a network on data that contains new characters not present in the original training set, the tokenizer extends the vocabulary by appending new tokens. Existing token IDs are never reassigned, so previously learned embeddings remain valid. Only the embedding matrix is extended with new rows initialized to small random values.</p>

<h2>Encoding and decoding</h2>
<p>Encoding converts a string to an integer array: each character is looked up in the vocabulary map. Decoding is the reverse: each integer is mapped back to a character and the results are concatenated. Unknown characters at inference time (characters not in the vocabulary) are handled by a fallback to a special unknown token or skipped, depending on context.</p>

<h2>Practical implications of character-level tokenization</h2>
<ul>
  <li>A 200-character conversation flattens to ~200 tokens. Set <code>contextLen</code> accordingly.</li>
  <li>The model must learn spelling implicitly — it predicts one character at a time. Larger and more uniform corpora produce better spelling.</li>
  <li>Vocabulary size is typically 60–120 characters for English text. This is much smaller than typical subword vocabularies (30 000+), which reduces the embedding table size significantly.</li>
  <li>Character models can generate novel words the training data never contained, since they compose output character by character.</li>
</ul>
`
  },
  {
    id: 'script',
    title: 'NeuralScript language',
    body: `
<h1>NeuralScript</h1>
<p>NeuralScript is a lightweight scripting language built into NeuralCabin for running experiments programmatically. It uses <code>do</code>/<code>end</code> blocks for all compound statements, requires no semicolons, and supports first-class neural operations (<code>build</code>, <code>train</code>, <code>predict</code>) as standard library functions.</p>
<p>NeuralScript programs run in the <strong>Script</strong> tab. Output from <code>print</code> appears in the console panel below the editor. Async operations (training) must be awaited with <code>await()</code>.</p>

<h2>Variables</h2>
<pre><code>let x = 10
let name = "hello"
let flag = true
set x = x + 1
print x</code></pre>
<p><code>let</code> declares a new variable. <code>set</code> reassigns an existing one. Variables are dynamically typed.</p>

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

<h2>Objects and arrays</h2>
<pre><code>let obj = { key: "value", count: 3 }
let arr = [1, 2, 3]
print obj.key
print arr[0]</code></pre>

<h2>Neural API</h2>
<table>
<tr><th>Function</th><th>Description</th></tr>
<tr><td><code>build(spec)</code></td><td>Construct a model from an architecture specification object. Returns a network handle.</td></tr>
<tr><td><code>await(train(spec, data, opts))</code></td><td>Train a model. <code>spec</code> is the architecture, <code>data</code> is the training samples, <code>opts</code> holds optimizer options. Returns <code>{ state, metrics }</code>.</td></tr>
<tr><td><code>predict(network, input)</code></td><td>Run inference. <code>input</code> can be an array (classifier/regressor), a string (char LM), or a chat history object.</td></tr>
<tr><td><code>thisNet()</code></td><td>Returns the currently selected network in the UI. Useful for running inference on an already-trained model without rebuilding it.</td></tr>
</table>

<h2>Standard library utilities</h2>
<table>
<tr><th>Function</th><th>Description</th></tr>
<tr><td><code>range(n)</code></td><td>Returns <code>[0, 1, ..., n-1]</code>.</td></tr>
<tr><td><code>len(x)</code></td><td>Length of an array or string.</td></tr>
<tr><td><code>push(arr, val)</code></td><td>Appends a value to an array (mutates in place).</td></tr>
<tr><td><code>str(x)</code></td><td>Converts a value to a string.</td></tr>
<tr><td><code>num(x)</code></td><td>Parses a string to a number.</td></tr>
<tr><td><code>keys(obj)</code></td><td>Returns the keys of an object as an array.</td></tr>
<tr><td><code>values(obj)</code></td><td>Returns the values of an object as an array.</td></tr>
<tr><td><code>abs</code>, <code>min</code>, <code>max</code>, <code>sqrt</code>, <code>exp</code>, <code>log</code>, <code>sin</code>, <code>cos</code>, <code>floor</code>, <code>ceil</code>, <code>round</code>, <code>random</code></td><td>Standard math operations.</td></tr>
</table>

<h2>Full example — XOR hyperparameter sweep</h2>
<pre><code>let lrs = [0.01, 0.005, 0.001]
let bestLoss = 9999
let bestLr = 0

for i = 0 to len(lrs) - 1 do
  let lr = lrs[i]
  let spec = {
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
  let opts = {
    optimizer: "adam",
    learningRate: lr,
    batchSize: 4,
    epochs: 300,
    seed: 42
  }
  let result = await(train(spec, data, opts))
  let lastMetric = result.metrics[len(result.metrics) - 1]
  print "lr=" + str(lr) + "  loss=" + str(lastMetric.loss)
  if lastMetric.loss &lt; bestLoss do
    set bestLoss = lastMetric.loss
    set bestLr = lr
  end
end

print "Best lr: " + str(bestLr)
print "Best loss: " + str(bestLoss)</code></pre>

<h2>Inference from Script</h2>
<pre><code># Classifier
let net = thisNet()
let result = predict(net, [0, 1])
print result.label
print result.confidence

# Chat model
let reply = predict(net, {
  history: [
    { role: "user",      content: "Hi" },
    { role: "assistant", content: "Hello!" }
  ],
  prompt:      "What can you do?",
  maxTokens:   150,
  temperature: 0.8
})
print reply</code></pre>
`
  },
  {
    id: 'api',
    title: 'HTTP API',
    body: `
<h1>HTTP API</h1>
<p>Any trained network can be served over HTTP on your local machine. Open the <strong>API</strong> tab, select a network, choose a port (or leave it as <code>0</code> for automatic assignment), and click <strong>Start</strong>. Your local IP address is shown in the application status bar so other devices on the same network can reach the server.</p>

<h2>Endpoints</h2>
<table>
<tr><th>Route</th><th>Method</th><th>Description</th></tr>
<tr><td><code>/</code> or <code>/info</code></td><td>GET</td><td>Returns the network type, input specification, vocabulary (for LMs), and the list of accepted request fields.</td></tr>
<tr><td><code>/predict</code></td><td>POST</td><td>Stateless inference. Pass a JSON body matching the network's input format. For chat models, pass a full history each time.</td></tr>
<tr><td><code>/chat</code></td><td>POST</td><td><strong>Chat models only.</strong> Stateful endpoint — the server maintains conversation threads keyed by <code>sessionId</code>.</td></tr>
<tr><td><code>/chat/reset</code></td><td>POST</td><td>Clears the session history for a given <code>sessionId</code>.</td></tr>
<tr><td><code>/health</code></td><td>GET</td><td>Liveness probe. Always <code>200 OK</code>. Never auth-gated or rate-limited.</td></tr>
<tr><td><code>/metrics</code></td><td>GET</td><td>Prometheus text-format metrics (request count, error count, avg latency, uptime). Requires auth if enabled.</td></tr>
</table>

<h2>Classifier / Regressor</h2>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{"input": [0, 1]}'

# Response (classifier):
{ "label": "true", "labelIndex": 1, "confidence": 0.97, "probabilities": [0.03, 0.97] }

# Response (regressor):
{ "output": [0.479] }</code></pre>

<h2>Character LM (single-shot completion)</h2>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "prompt":      "the quick",
    "maxTokens":   80,
    "temperature": 1.0
  }'

# Response:
{ "completion": " brown fox jumps over" }</code></pre>

<h2>Chat — stateless (<code>/predict</code>)</h2>
<p>You manage the conversation history on the client side and send it with every request:</p>
<pre><code>curl -X POST http://localhost:PORT/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "history": [
      { "role": "user",      "content": "Hi" },
      { "role": "assistant", "content": "Hello!" }
    ],
    "prompt":      "How are you?",
    "system":      "Be concise.",
    "maxTokens":   200,
    "temperature": 0.7
  }'

# Response:
{ "reply": "I am well, thanks." }</code></pre>

<h2>Chat — stateful (<code>/chat</code>)</h2>
<p>The server keeps a conversation thread in memory, keyed by <code>sessionId</code>. You only send the new message each turn:</p>
<pre><code># First turn — server assigns a sessionId
curl -X POST http://localhost:PORT/chat \\
  -H "Content-Type: application/json" \\
  -d '{"message": "Hi", "system": "Be concise."}'

# Response:
{ "sessionId": "session-1f3a", "reply": "Hello!", "history": [...] }

# Subsequent turns — pass the same sessionId
curl -X POST http://localhost:PORT/chat \\
  -H "Content-Type: application/json" \\
  -d '{"sessionId": "session-1f3a", "message": "How are you?"}'

# Reset a session
curl -X POST http://localhost:PORT/chat/reset \\
  -H "Content-Type: application/json" \\
  -d '{"sessionId": "session-1f3a"}'</code></pre>

<h2>Session limits and lifecycle</h2>
<ul>
  <li>Sessions expire after <strong>1 hour</strong> of inactivity.</li>
  <li>Maximum <strong>256 concurrent sessions</strong> per model.</li>
  <li>Maximum <strong>64 turns</strong> per session (subsequent turns overwrite the oldest).</li>
  <li>All session data is held in memory — nothing is written to disk.</li>
  <li>Sessions are cleared when the API server stops.</li>
</ul>

<h2>Calling the API from JavaScript</h2>
<pre><code>async function chat(sessionId, message) {
  const res = await fetch('http://localhost:PORT/chat', {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ sessionId, message })
  });
  const data = await res.json();
  return { sessionId: data.sessionId, reply: data.reply };
}

// First call (no sessionId):
const first = await chat(null, 'Hello');
// { sessionId: 'session-...', reply: '...' }

// Continue the thread:
const second = await chat(first.sessionId, 'Tell me more.');</code></pre>

<h2>Authentication</h2>
<p>When the API server is started with an <code>authSecret</code>, all endpoints except <code>/health</code> require a Bearer token. Tokens are HS256-signed JWTs issued by NeuralCabin using Node's built-in <code>crypto</code> module — no external dependencies.</p>
<pre><code># Start with auth enabled (via API tab or IPC)
# authSecret: any string — keep it secret

# Issue a token from the app (via IPC: api:issue-token)
# curl with the token:
curl -X POST http://localhost:PORT/predict \\
  -H "Authorization: Bearer &lt;token&gt;" \\
  -H "Content-Type: application/json" \\
  -d '{"input": [0, 1]}'</code></pre>
<p>Tokens can be set to expire (e.g. 3600 for one hour) or last indefinitely. A request with a missing, malformed, or expired token receives <code>401 Unauthorized</code>.</p>
<p>When no <code>authSecret</code> is configured, the server is open — any request is accepted.</p>

<h2>Rate limiting</h2>
<p>By default the server allows <strong>120 requests per minute per IP address</strong> using a sliding-window algorithm. Requests that exceed the limit receive <code>429 Too Many Requests</code>. Configure the limit when starting the server:</p>
<pre><code># Start with a custom rate limit (30 req/min per IP)
# via IPC: api:start with opts.rateLimit = 30</code></pre>
<p>The <code>/health</code> endpoint is exempt from rate limiting.</p>

<h2>Health endpoint</h2>
<p><code>GET /health</code> — always returns <code>200 OK</code>, never rate-limited, never auth-gated. Use it as a liveness probe:</p>
<pre><code>curl http://localhost:PORT/health
{ "ok": true, "uptime": 42.3 }</code></pre>

<h2>Metrics endpoint</h2>
<p><code>GET /metrics</code> — returns Prometheus text-format metrics for the server instance. Requires auth if <code>authSecret</code> is set.</p>
<pre><code>curl -H "Authorization: Bearer &lt;token&gt;" http://localhost:PORT/metrics

# HELP neuralcabin_requests_total Total requests served
# TYPE neuralcabin_requests_total counter
neuralcabin_requests_total{network="my-net"} 1234

# HELP neuralcabin_errors_total Total error responses
# TYPE neuralcabin_errors_total counter
neuralcabin_errors_total{network="my-net"} 2

# HELP neuralcabin_latency_avg_ms Average request latency (ms)
# TYPE neuralcabin_latency_avg_ms gauge
neuralcabin_latency_avg_ms{network="my-net"} 3.7

# HELP neuralcabin_uptime_seconds Server uptime in seconds
# TYPE neuralcabin_uptime_seconds gauge
neuralcabin_uptime_seconds{network="my-net"} 3601.2</code></pre>

<h2>Security notes</h2>
<p>The API server binds to all network interfaces by default so devices on the same LAN can reach it. For internal use, rate limiting and no auth is usually sufficient. If you expose the server beyond your LAN, always enable <code>authSecret</code> and consider your OS firewall as a second layer of defence.</p>
`
  },
  {
    id: 'encryption',
    title: 'Encryption',
    body: `
<h1>Encrypting a network</h1>
<p>NeuralCabin supports optional AES-256-GCM encryption at rest for any saved network. The weights, optimizer state, and tokenizer are serialized into a single bundle, then encrypted using a key derived from your passphrase via <strong>scrypt</strong>. The passphrase is never stored — only the encrypted ciphertext and the scrypt salt and nonce are written to disk.</p>

<h2>Enabling encryption</h2>
<ol>
  <li>Open the <strong>Editor</strong> tab for the network you want to protect.</li>
  <li>Toggle <strong>Encryption</strong> to on.</li>
  <li>Enter a strong passphrase. There is no minimum length, but longer is better.</li>
  <li>Click <strong>Save</strong>. The network file on disk is now ciphertext.</li>
</ol>
<p>Subsequently loading the network will prompt for the passphrase to decrypt it before any operation (training, inference, serving) can proceed.</p>

<h2>Cryptographic details</h2>
<table>
<tr><th>Component</th><th>Details</th></tr>
<tr><td>Cipher</td><td>AES-256-GCM. Authenticated encryption — both confidentiality and integrity are protected. Any tampering with the ciphertext causes decryption to fail.</td></tr>
<tr><td>Key derivation</td><td>scrypt with N=16384, r=8, p=1. The output is a 32-byte key. scrypt is deliberately memory-hard to resist brute-force attacks.</td></tr>
<tr><td>Salt</td><td>16 bytes of cryptographically random data, unique per network. Stored alongside the ciphertext.</td></tr>
<tr><td>IV / nonce</td><td>12 bytes of random data, unique per save. Stored alongside the ciphertext.</td></tr>
</table>

<h2>Key loss is total loss</h2>
<p>NeuralCabin never stores or escrows the passphrase. If it is lost, the weights are unrecoverable by design. Back up the passphrase separately from the network file.</p>

<h2>When to use encryption</h2>
<p>Encryption at rest is worth enabling whenever a model has been trained on sensitive data — personal chat logs, confidential business content, or any corpus you would not want disclosed if the machine were lost or compromised. The overhead at load time (the scrypt derivation) is a one-time cost of roughly 100–500ms on modern hardware.</p>
`
  },
  {
    id: 'philosophy',
    title: 'Why from scratch?',
    body: `
<h1>No frameworks. No abstraction tax.</h1>
<p>Production ML frameworks — PyTorch, TensorFlow, JAX — are built for scale and generality. That means they sit behind tens of thousands of lines of kernel dispatch code, custom CUDA ops, and multi-level JIT compilation infrastructure. The depth is appropriate when you're training billion-parameter models on distributed GPU clusters. For understanding what a network is actually doing at the arithmetic level, or for small to medium models where none of that infrastructure provides any benefit, it's just noise standing between you and the computation.</p>

<p>The NeuralCabin engine (see <a href="https://github.com/GuyThatLivesAndCodes/NeuralCabin" target="_blank">the source</a>) is a hybrid JavaScript + Rust implementation. You can open <code>src/engine/tensor-js.js</code> and read the full JS autograd system in one sitting, while <code>native/rust-engine/neuralcabin-core/src/</code> contains the native kernel path. You can trace a gradient backward through a specific operation, verify the chain rule application by hand, and confirm it matches what the code computes. You can add a new activation function in ten lines. You can swap in a different optimizer and watch it change the loss curve. Nothing is hidden.</p>

<h2>What you trade away</h2>
<ul>
  <li><strong>No GPU path (yet).</strong> The Rust core runs on CPU with Rayon parallelism and AVX2 FMA. A WebGPU path is architecturally possible but not implemented.</li>
  <li><strong>No Transformer blocks out of the box.</strong> The primitives (matmul, embedding, softmax, gelu) are all present, and NeuralScript can compose them, but there is no pre-built attention layer.</li>
  <li><strong>No distributed training.</strong> Everything runs in a single Electron process on a single machine.</li>
  <li><strong>Mixed precision is available in the Rust dtype API but not exposed in the UI.</strong> FP16/BF16 dtypes are present in the Rust tensor type; the JS layer uses F32 throughout.</li>
</ul>

<h2>What you get in return</h2>
<ul>
  <li><strong>Readable source.</strong> Every operator fits on one screen. Every gradient is traceable. No dispatch tables, no kernel backends.</li>
  <li><strong>Reproducible runs.</strong> Seeded randomness covers weight initialization, batch shuffling, dropout, and sampling. Same seed + same data = same output, every time.</li>
  <li><strong>Local data.</strong> Nothing leaves your machine unless you explicitly start the HTTP API server. No cloud sync, no telemetry, no model upload.</li>
  <li><strong>Modifiable.</strong> If you want to experiment with a different optimizer, a custom loss function, or a novel activation — open the file and change it. There are no build steps, no generated files, no framework internals to fight.</li>
  <li><strong>Understandable.</strong> Building from scratch forces you to confront the actual mechanics. If you want to know why Adam converges faster than SGD, you can read exactly what Adam does differently and watch it happen in the loss curve.</li>
</ul>

<h2>Who this is for</h2>
<p>NeuralCabin is well suited for: learning how neural networks work at the implementation level; rapid prototyping of small models without framework setup overhead; training personal conversational models on local data with no cloud involvement; embedding a neural inference engine in an Electron application; and anyone who finds that black-box tools obscure more than they reveal.</p>
<p>It is not suited for: training large language models, computer vision at scale, production deployment requiring GPU throughput, or any task where PyTorch or JAX would provide meaningful performance or capability advantages. Use the right tool for the job.</p>
`
  },
  {
    id: 'architecture-tips',
    title: 'Architecture design tips',
    body: `
<h1>Architecture design tips</h1>
<p>Choosing a network architecture is partly science and partly empirical. These guidelines apply to the model sizes NeuralCabin is designed for.</p>

<h2>Start small, then scale up</h2>
<p>Resist the temptation to build a large network immediately. A 2-hidden-layer network with 32 units per layer will train fast, converge reliably, and tell you quickly whether the task is learnable. If training loss stalls at an unacceptably high value after full convergence, then add capacity. If training loss is good but the model behaves poorly at inference, the problem is usually data quality or quantity, not capacity.</p>

<h2>Depth vs. width</h2>
<p>For most tasks NeuralCabin handles, 2–3 hidden layers is sufficient. Adding more layers adds representational depth but also adds vanishing gradient risk and slower convergence. For classification of moderately non-linear data (2D spiral), go deeper (3 layers) before going wider. For regression over smooth functions, width matters more than depth.</p>

<h2>Activation choice</h2>
<ul>
  <li>Use <strong>ReLU</strong> as the default for classifiers and most regressors. It trains fast and rarely causes problems at NeuralCabin's scale.</li>
  <li>Use <strong>GELU</strong> for language models and chat assistants — it provides smoother gradients and is standard in LM architectures.</li>
  <li>Use <strong>tanh</strong> for small regressors where bounded output in the hidden layers is desirable.</li>
  <li>Never use <strong>softmax</strong> in a hidden layer. It is only appropriate as a final output activation for classification heads.</li>
</ul>

<h2>Dropout placement</h2>
<p>Dropout reduces overfitting on small datasets. Place it after the activation in each hidden layer, not on the input or on the final logit layer. A rate of <code>0.1</code>–<code>0.2</code> is a good starting point. Higher rates (0.3–0.5) can help if you have a small dataset and see a wide gap between training loss and validation behavior, but they also slow convergence.</p>

<h2>Language model sizing</h2>
<p>Character LMs are more sensitive to architecture than classifiers. The embedding dimension and hidden layer width interact with <code>contextLen</code>: the model effectively sees <code>contextLen × embDim</code> inputs after flattening, so small <code>embDim</code> with large <code>contextLen</code> can produce a very wide input layer. A balanced starting point:</p>
<pre><code>embDim:     32–64
contextLen: 64–128
hidden:     [128, 128]   (1–2 layers)
activation: gelu
dropout:    0.1</code></pre>
<p>For chat assistants, increase <code>contextLen</code> to 128–256 and <code>embDim</code> to 64. Expect training to take longer — budget 500–2000 epochs.</p>

<h2>Debugging a poorly training network</h2>
<table>
<tr><th>Symptom</th><th>Likely cause</th><th>Fix</th></tr>
<tr><td>Loss starts high and barely moves</td><td>Learning rate too low, or network too small</td><td>Increase learning rate; increase hidden size</td></tr>
<tr><td>Loss oscillates wildly or diverges</td><td>Learning rate too high</td><td>Reduce learning rate by 5–10×</td></tr>
<tr><td>Loss reaches a plateau much higher than expected</td><td>Insufficient capacity, or bad data</td><td>Add hidden units or layers; audit training data for errors</td></tr>
<tr><td>Loss goes to zero but inference is wrong</td><td>Overfitting, or data/inference mismatch</td><td>Add dropout; verify input preprocessing matches training</td></tr>
<tr><td>Chat model produces garbled output</td><td>contextLen too small, or too few training samples</td><td>Increase contextLen; add more multi-turn training data</td></tr>
</table>
`
  },
  {
    id: 'storage',
    title: 'Storage and persistence',
    body: `
<h1>Storage and persistence</h1>
<p>NeuralCabin saves all network state to the local filesystem. No data is sent to any remote server at any point.</p>

<h2>What is saved</h2>
<p>When a network is saved (either manually or after training), the following are serialized to a single JSON file:</p>
<ul>
  <li><strong>Architecture spec</strong> — layer sizes, activation, dropout rate, context length, embedding dimension.</li>
  <li><strong>Weights and biases</strong> — all trainable parameters as flat arrays.</li>
  <li><strong>Optimizer state</strong> — Adam's per-parameter <code>m</code>, <code>v</code>, and step count <code>t</code>. This means resuming training after a save is a true continuation, not a cold restart.</li>
  <li><strong>Tokenizer vocabulary</strong> — the character-to-ID mapping for LMs and chat models.</li>
  <li><strong>Training metadata</strong> — loss history, epoch count, seed.</li>
</ul>

<h2>File location</h2>
<p>Network files are stored in the application data directory, managed by Electron's <code>userData</code> path. On Windows this is typically <code>%APPDATA%\NeuralCabin\networks\</code>. You can open a network file in any text editor to inspect the raw JSON (or ciphertext, if encrypted).</p>

<h2>Encryption at rest</h2>
<p>See the <strong>Encryption</strong> topic for full details. Summary: when encryption is enabled, the serialized JSON is encrypted with AES-256-GCM before writing to disk. The passphrase is never stored.</p>

<h2>Exporting and importing</h2>
<p>Network files are portable — copy them to another machine running NeuralCabin and load them via the Networks sidebar. Architecture, weights, tokenizer, and optimizer state all transfer together.</p>

<h2>Resuming training</h2>
<p>Because Adam's optimizer state is persisted, you can stop training at epoch 500, close the application, reopen it, and continue from epoch 501 with the same optimizer momentum. The loss curve continues from where it left off. SGD has no persisted state (no momentum), so resuming with SGD is equivalent to a fresh optimizer start on the existing weights.</p>
`
  },
  {
    id: 'nbpl-intro',
    title: 'Plugin system (.nbpl)',
    body: `
<h1>The NeuralCabin Plugin System</h1>
<p>NeuralCabin's plugin system lets you extend the application with new model templates, custom training-data editors, and custom inference UIs — all delivered as a single portable file with the <code>.nbpl</code> extension. Plugins can run logic in the main (Node.js) process and render custom UI in the renderer process, communicating over Electron's IPC bridge.</p>

<h2>The .nbpl file format</h2>
<p>A <code>.nbpl</code> file is plain JSON. It requires no compression or binary packaging — open any <code>.nbpl</code> in a text editor to inspect or modify it.</p>
<pre><code>{
  "id":          "my-plugin",
  "name":        "My Plugin",
  "version":     "1.0.0",
  "description": "A short description shown in the Plugins tab.",
  "author":      "Your Name",
  "mainCode":    "/* Node.js source — runs in the main process */",
  "rendererCode":"/* Browser JS — evaluated in the renderer */",
}</code></pre>

<h2>Field reference</h2>
<table>
<tr><th>Field</th><th>Required</th><th>Description</th></tr>
<tr><td><code>id</code></td><td>Yes</td><td>Unique identifier. Must match <code>/^[a-z0-9_-]+$/i</code>. Used as the directory name and IPC namespace prefix.</td></tr>
<tr><td><code>name</code></td><td>No</td><td>Display name shown in the Plugins tab. Defaults to <code>id</code>.</td></tr>
<tr><td><code>version</code></td><td>No</td><td>SemVer string shown next to the plugin name. Defaults to <code>"0.0.0"</code>.</td></tr>
<tr><td><code>description</code></td><td>No</td><td>One-paragraph description shown in the Plugins tab.</td></tr>
<tr><td><code>author</code></td><td>No</td><td>Author name shown in the Plugins tab.</td></tr>
<tr><td><code>mainCode</code></td><td>No</td><td>A JavaScript string that will be written to <code>main.js</code> in the plugin directory and <code>require()</code>d in the main process. Must export <code>{ mainHandlers }</code> (see below).</td></tr>
<tr><td><code>rendererCode</code></td><td>No</td><td>A JavaScript string that will be written to <code>renderer.js</code> and evaluated in the renderer process via <code>new Function('api', code)(api)</code>. Receives the plugin <code>api</code> object as its sole argument.</td></tr>
</table>

<h2>How plugins are stored</h2>
<p>When you install a <code>.nbpl</code> file, the installer extracts the JSON fields and writes them to the user's application data directory:</p>
<pre><code>%APPDATA%/NeuralCabin/
└── plugins/
    └── my-plugin/
        ├── manifest.json    ← id, name, version, description, author
        ├── main.js          ← extracted mainCode
        └── renderer.js      ← extracted rendererCode</code></pre>
<p>The plugin directory layout mirrors the built-in <code>src/plugins/</code> directory in the NeuralCabin source tree. Bundled example plugins (such as the Chess plugin) are seeded into this directory on every app launch, ensuring the latest version is always present.</p>
<p>Networks created from a plugin template carry <code>arch.pluginKind</code> in their saved JSON. The Networks sidebar uses this to show a readable type label (e.g. "Self-Driving Car") instead of the generic "classifier" kind. Opening the Editor → Architecture tab for a plugin model shows the plugin's custom fields (registered via <code>api.registerArchFields()</code>) rather than the default inputDim / hidden-layers form.</p>

<h2>Plugin loading lifecycle</h2>
<ol>
  <li><strong>Startup</strong> — <code>PluginLoader.load(ipcMain)</code> scans every subdirectory of the plugins folder, reads <code>manifest.json</code> and <code>renderer.js</code>, and <code>require()</code>s <code>main.js</code>. Any IPC handlers exported from <code>mainHandlers</code> are registered at this point.</li>
  <li><strong>Renderer init</strong> — <code>initPlugins()</code> in the renderer calls <code>window.nb.plugins.list()</code>, iterates each plugin's <code>rendererCode</code>, and evaluates it via <code>new Function('api', code)(api)</code>. Templates, inference renderers, and train editors registered here become immediately usable.</li>
  <li><strong>Install / uninstall</strong> — Installs write files to disk; uninstalls remove them. Neither operation hot-reloads plugins. A manual restart is required to apply the changes.</li>
</ol>

<h2>IPC namespacing</h2>
<p>Each plugin's main-process handlers are registered under a namespaced channel: <code>plugin:&lt;id&gt;:&lt;channel&gt;</code>. For example, if the chess plugin exports <code>mainHandlers['chess:encodePosition']</code>, it is registered as <code>plugin:chess:chess:encodePosition</code>. From the renderer, call it via:</p>
<pre><code>nb.invoke('chess:encodePosition', fen)
// equivalent to: window.nb.plugins.invoke('chess', 'chess:encodePosition', fen)</code></pre>
<p>The <code>nb</code> object is the namespaced invoke helper provided to your renderer code by the plugin registry — it automatically prepends your plugin's ID.</p>

<h2>Security model</h2>
<p>Plugin <code>mainCode</code> runs with full Node.js access in the Electron main process — treat it like any application code you install. Plugin <code>rendererCode</code> is evaluated in the renderer with <code>unsafe-eval</code> allowed in the CSP; it has access to the renderer's <code>window</code> object but not to Node.js APIs directly. All main-process calls must go through the <code>nb.invoke()</code> bridge.</p>
`
  },
  {
    id: 'nbpl-dev-guide',
    title: 'Building a plugin',
    body: `
<h1>Building a NeuralCabin Plugin</h1>
<p>This guide walks through creating a complete plugin from scratch — defining main-process logic, writing a renderer UI, packaging it as a <code>.nbpl</code> file, and testing it in the app.</p>

<h2>1. Plan your plugin</h2>
<p>Every plugin can contribute up to five things:</p>
<ul>
  <li><strong>A model template</strong> — appears in the "New Network" modal under "Plugin Models". Defines the network architecture, training defaults, and initial training data.</li>
  <li><strong>Architecture fields</strong> — replaces the generic inputDim / hidden-layers form in the Editor tab with domain-specific settings (sensor count, grid size, debug flags, etc.). Use <code>api.registerArchFields()</code>.</li>
  <li><strong>Train settings overrides</strong> — relabels the standard learning-rate / batch-size / epochs panel with plugin-appropriate names (e.g. "Mutation std" instead of "Learning rate"). Use <code>api.registerTrainSettings()</code>.</li>
  <li><strong>A training editor</strong> — replaces the default JSON textarea on the Train tab for networks created from your template. Lets you build a live simulation UI, custom data-import widgets, etc.</li>
  <li><strong>An inference renderer</strong> — replaces the default inference UI for your network type. Use it to build a custom interactive interface for predictions or to visualise a live agent.</li>
</ul>
<p>You don't need all five. A plugin can register only a template, or any subset of the hooks.</p>

<h2>2. Write the main-process code</h2>
<p>Create <code>main.js</code>. This file is <code>require()</code>d in Node.js, so you have access to the full Node.js standard library.</p>
<pre><code>'use strict';

// main.js — main-process handlers for 'my-plugin'
function doSomethingExpensive(data) {
  // CPU-heavy work, file I/O, etc.
  return { result: data.length };
}

module.exports = {
  mainHandlers: {
    // Channel name is a free-form string; namespaced automatically by the loader.
    // Called from renderer via: nb.invoke('my-plugin:process', data)
    'my-plugin:process': (_, data) => doSomethingExpensive(data),

    // Handlers are plain async functions — return a Promise to resolve it.
    'my-plugin:asyncOp': async (_, id) => {
      const result = await someAsyncWork(id);
      return result;
    }
  }
};</code></pre>
<p>The first argument to every handler is the Electron <code>event</code> object (from <code>ipcMain.handle</code>). Your data arguments start from the second parameter. Return any JSON-serializable value.</p>

<h2>3. Write the renderer code</h2>
<p>Create <code>renderer.js</code>. This code is evaluated inside an IIFE <code>(function(api){ ... })(api)</code>, so any variable you declare is local. You receive one argument — the <code>api</code> object — through which you register all hooks.</p>
<pre><code>// renderer.js — evaluated in the Electron renderer process
(function (api) {
  'use strict';

  // ── 1. Register a template ─────────────────────────────────────────────
  api.registerTemplate({
    id:         'my-plugin',       // must be unique across all templates
    name:       'My Custom Model',
    kind:       'classifier',      // underlying engine kind
    pluginKind: 'my-plugin',       // key used by trainEditor / inferenceRenderer hooks
    desc:       'A brief description.',
    arch: {
      kind:      'classifier',
      pluginKind: 'my-plugin',
      inputDim:   16,
      outputDim:  8,
      hidden:     [64, 32],
      activation: 'relu',
      dropout:    0.1
    },
    training: {
      optimizer: 'adam', learningRate: 0.001,
      batchSize: 32, epochs: 50, seed: 42
    },
    trainingData: { samples: [] }
  });

  // ── 2. Register architecture fields (optional) ────────────────────────
  api.registerArchFields('my-plugin', {
    fields: [
      { id: 'numInputs',  label: 'Input count', type: 'number',
        default: 16, min: 1, max: 256, step: 1,
        hint: 'Changing this resets saved weights.' },
      { id: 'hidden',     label: 'Hidden layers', type: 'layers',     default: [64, 32] },
      { id: 'activation', label: 'Activation',    type: 'activation', default: 'relu' },
    ],
    computeDims: (arch) => ({ inputDim: arch.numInputs || 16, outputDim: 8 }),
  });

  // ── 3. Register a custom training editor ───────────────────────────────
  // 'my-plugin' matches arch.pluginKind above
  api.registerTrainEditor('my-plugin', function (root, network, nb) {
    root.innerHTML = '&lt;button id="my-upload-btn"&gt;Upload data&lt;/button&gt;';

    document.getElementById('my-upload-btn').addEventListener('click', async () => {
      const file = await window.nb.dialog.readTextFile({
        filters: [{ name: 'Text', extensions: ['txt'] }]
      });
      if (!file) return;

      // Call a main-process handler to parse the file
      const parsed = await nb.invoke('my-plugin:process', file.content);

      // Save samples back to the network
      await window.nb.networks.update(network.id, {
        trainingData: { samples: parsed.samples }
      });
    });
  });

  // ── 4. Register a custom inference renderer ────────────────────────────
  api.registerInferenceRenderer('my-plugin', function (root, network, nb) {
    const isTrained = !!(network.state || network.stateLocked);

    root.innerHTML = \`
      &lt;div class="panel"&gt;
        &lt;h2&gt;My Custom Inference&lt;/h2&gt;
        &lt;input id="my-input" type="text" placeholder="Enter input..." /&gt;
        &lt;button id="my-predict" \${!isTrained ? 'disabled' : ''}&gt;Predict&lt;/button&gt;
        &lt;div id="my-result"&gt;&lt;/div&gt;
      &lt;/div&gt;
    \`;

    document.getElementById('my-predict').addEventListener('click', async () => {
      const rawInput = document.getElementById('my-input').value;
      // Encode via main-process handler
      const encoded = await nb.invoke('my-plugin:process', rawInput);
      // Run inference through the engine
      const result = await window.nb.inference.run(network.id, { input: encoded.vector });
      document.getElementById('my-result').textContent = JSON.stringify(result);
    });
  });

})(api); // api is injected by the plugin registry</code></pre>

<h2>5. Package as a .nbpl file</h2>
<p>Read <code>main.js</code> and <code>renderer.js</code>, embed them as JSON strings, and write the combined manifest:</p>
<pre><code>// build-nbpl.js — run with: node build-nbpl.js
const fs = require('fs');

const mainCode     = fs.readFileSync('main.js', 'utf-8');
const rendererCode = fs.readFileSync('renderer.js', 'utf-8');

const nbpl = {
  id:          'my-plugin',
  name:        'My Custom Model',
  version:     '1.0.0',
  description: 'A brief description.',
  author:      'Your Name',
  mainCode,
  rendererCode
};

fs.writeFileSync('my-plugin.nbpl', JSON.stringify(nbpl, null, 2));
console.log('Wrote my-plugin.nbpl');</code></pre>
<p>Alternatively, build the JSON manually in any text editor — it is just a JSON object with two large string fields.</p>

<h2>6. Testing during development</h2>
<p>The fastest workflow avoids packaging entirely:</p>
<ol>
  <li>Create a directory <code>src/plugins/my-plugin/</code> in the NeuralCabin source tree.</li>
  <li>Add <code>manifest.json</code>, <code>main.js</code>, and <code>renderer.js</code> as separate files.</li>
  <li>The <code>seedBuiltins</code> method copies everything from <code>src/plugins/</code> to the user's plugin directory on every launch, so your changes are picked up automatically on restart.</li>
  <li>Launch with <code>npm start</code>. Your plugin is live after the renderer initializes.</li>
</ol>
<p>For production, package as a <code>.nbpl</code> and install via the Plugins tab → Install Plugin.</p>

<h2>Common pitfalls</h2>
<table>
<tr><th>Problem</th><th>Cause</th><th>Fix</th></tr>
<tr><td>Plugin not visible after install</td><td>Needs restart</td><td>Restart the app — plugin system loads at startup only</td></tr>
<tr><td>Inference renderer never called</td><td><code>pluginKind</code> mismatch</td><td>Ensure <code>arch.pluginKind</code> in the template matches the key passed to <code>registerInferenceRenderer()</code></td></tr>
<tr><td>Train editor not shown</td><td>Same mismatch</td><td>Same fix — <code>registerTrainEditor('my-plugin', ...)</code> key must equal <code>arch.pluginKind</code></td></tr>
<tr><td>Main handler returns undefined</td><td>Channel name wrong</td><td>Channel passed to <code>nb.invoke()</code> must exactly match a key in <code>mainHandlers</code></td></tr>
<tr><td>Error: "Plugin renderer failed"</td><td>JavaScript error in rendererCode</td><td>Open DevTools (launch with <code>--dev</code> flag) and check the console for the stack trace</td></tr>
</table>
`
  },
  {
    id: 'nbpl-api-ref',
    title: 'Plugin API reference',
    body: `
<h1>Plugin API Reference</h1>
<p>This page documents every API available to plugin code: the <code>api</code> object available in <code>rendererCode</code>, the <code>nb</code> invoke helper, the <code>window.nb</code> APIs accessible from renderer plugins, and the <code>mainHandlers</code> export format for <code>mainCode</code>.</p>

<h2>api object (rendererCode)</h2>
<p>The <code>api</code> object is the sole argument passed to your renderer code. It is the only sanctioned way to hook into the NeuralCabin UI from a plugin.</p>

<h3>api.registerTemplate(template)</h3>
<p>Adds a network template to the global template list. Templates appear in the "New Network" modal under the "Plugin Models" category.</p>
<table>
<tr><th>Field</th><th>Type</th><th>Description</th></tr>
<tr><td><code>id</code></td><td>string</td><td>Unique template ID. If a template with this ID already exists, it is silently skipped.</td></tr>
<tr><td><code>name</code></td><td>string</td><td>Display name in the New Network modal.</td></tr>
<tr><td><code>kind</code></td><td>string</td><td>Must match one of the engine's supported kinds: <code>'classifier'</code>, <code>'regressor'</code>, <code>'charLM'</code>, <code>'gpt'</code>.</td></tr>
<tr><td><code>pluginKind</code></td><td>string</td><td>The key used to look up the inference renderer and train editor. Must match the keys you pass to <code>registerInferenceRenderer()</code> and <code>registerTrainEditor()</code>.</td></tr>
<tr><td><code>desc</code></td><td>string</td><td>Short description shown below the template name.</td></tr>
<tr><td><code>arch</code></td><td>object</td><td>Architecture object saved to the network. Must include <code>kind</code>, <code>pluginKind</code>, <code>inputDim</code>, <code>outputDim</code>, <code>hidden[]</code>, <code>activation</code>, <code>dropout</code>.</td></tr>
<tr><td><code>training</code></td><td>object</td><td>Default training options: <code>optimizer</code>, <code>learningRate</code>, <code>batchSize</code>, <code>epochs</code>, <code>seed</code>.</td></tr>
<tr><td><code>trainingData</code></td><td>object</td><td>Initial training data. Typically <code>{ samples: [] }</code>.</td></tr>
</table>

<h3>api.registerTrainEditor(pluginKind, fn)</h3>
<p>Registers a custom training data editor for networks whose <code>arch.pluginKind</code> matches <code>pluginKind</code>. The editor replaces the default JSON textarea on the Train tab.</p>
<pre><code>api.registerTrainEditor('my-plugin', function (root, network, nb) {
  // root   — HTMLElement container to render into (empty on entry)
  // network — the full network object (id, name, architecture, trainingData, …)
  // nb     — the namespaced invoke helper for this plugin
});</code></pre>
<p>The <code>root</code> element is a <code>div</code> with id <code>plugin-train-editor</code>. Set <code>root.innerHTML</code> to render your UI. Use <code>window.nb.networks.update(network.id, patch)</code> to persist changes.</p>

<h3>api.registerInferenceRenderer(pluginKind, fn)</h3>
<p>Registers a custom inference UI for networks whose <code>arch.pluginKind</code> matches <code>pluginKind</code>. Replaces the entire Inference tab content for those networks.</p>
<pre><code>api.registerInferenceRenderer('my-plugin', function (root, network, nb) {
  // root   — HTMLElement to render into (cleared on entry)
  // network — full network object; check network.state to determine if trained
  // nb     — namespaced invoke helper
});</code></pre>
<p><strong>Note:</strong> The inference renderer is shown even for untrained networks. Check <code>!!(network.state || network.stateLocked)</code> to determine whether to enable AI-dependent buttons.</p>

<h3>api.registerTrainSettings(pluginKind, overrides)</h3>
<p>Customises how the standard Training settings panel (learning rate, batch size, epochs, seed, workers) is labelled and presented for networks of this plugin kind. Pass an object keyed by field name.</p>
<pre><code>api.registerTrainSettings('my-plugin', {
  lr:        { label: 'Mutation std',    hint: 'Gaussian mutation standard deviation' },
  bs:        { label: 'Population size', hint: 'Number of individuals per generation' },
  workers:   { hidden: true },   // hide a field entirely
  optimizer: { hidden: true },
  sectionHint: 'Neuroevolution settings — applied when the simulation starts.',
});</code></pre>
<p>Each entry may contain <code>label</code>, <code>hint</code>, and <code>hidden</code>. <code>sectionHint</code> is a special key that places a small note below the Training settings header.</p>

<h3>api.registerArchFields(pluginKind, spec)</h3>
<p>Registers plugin-specific architecture fields in the <strong>Editor</strong> tab → Architecture section for networks whose <code>arch.pluginKind</code> matches <code>pluginKind</code>. Replaces the generic inputDim / outputDim / hidden-layers form with a domain-specific panel. Commonly used to expose simulation parameters (grid size, sensor count, etc.) alongside the standard hidden-layer and activation controls.</p>
<pre><code>api.registerArchFields('my-plugin', {
  fields: [
    { id: 'numSensors',   label: 'Sensor count',  type: 'number',
      default: 9,   min: 1, max: 50, step: 1,
      hint: 'Changing this resets saved weights.' },
    { id: 'debugOverlay', label: 'Debug overlay', type: 'boolean',    default: false },
    { id: 'hidden',       label: 'Hidden layers', type: 'layers',     default: [64, 32] },
    { id: 'activation',   label: 'Activation',    type: 'activation', default: 'tanh' },
  ],
  computeDims: (arch) => ({
    inputDim:  arch.numSensors + 2,
    outputDim: 2,
  }),
});</code></pre>
<h4>Field descriptor properties</h4>
<table>
<tr><th>Property</th><th>Type</th><th>Description</th></tr>
<tr><td><code>id</code></td><td>string</td><td>Key written into <code>network.architecture</code>. Avoid <code>inputDim</code>, <code>outputDim</code> — use <code>computeDims</code> for those.</td></tr>
<tr><td><code>label</code></td><td>string</td><td>Human-readable label shown in the editor form.</td></tr>
<tr><td><code>type</code></td><td>string</td><td>One of <code>'number'</code>, <code>'boolean'</code>, <code>'layers'</code>, <code>'activation'</code>.</td></tr>
<tr><td><code>default</code></td><td>any</td><td>Value used when a network was created before this field existed (backward compatibility).</td></tr>
<tr><td><code>min</code> / <code>max</code> / <code>step</code></td><td>number</td><td>Range constraints for <code>type: 'number'</code> fields.</td></tr>
<tr><td><code>hint</code></td><td>string</td><td>Tooltip / help text rendered beneath the field.</td></tr>
</table>
<h4>Field types</h4>
<table>
<tr><th>type</th><th>Widget</th><th>Stored as</th></tr>
<tr><td><code>'number'</code></td><td>Number input with min / max / step</td><td><code>arch[id]</code> — number</td></tr>
<tr><td><code>'boolean'</code></td><td>Checkbox</td><td><code>arch[id]</code> — boolean</td></tr>
<tr><td><code>'layers'</code></td><td>Comma-separated sizes e.g. <code>64, 32, 16</code></td><td><code>arch.hidden</code> — number[]</td></tr>
<tr><td><code>'activation'</code></td><td>Dropdown: relu / tanh / sigmoid / linear / gelu</td><td><code>arch.activation</code> — string</td></tr>
</table>
<h4>computeDims callback</h4>
<p>The optional <code>computeDims(arch)</code> function receives the full <code>arch</code> object after each edit and returns <code>{ inputDim?, outputDim? }</code>. The app uses this to detect shape mismatches between the editor values and the saved weights — if the derived dims differ from what is stored, the user is offered the option to clear stale weights before the next training run. Always implement <code>computeDims</code> when your plugin's input or output size depends on a configurable parameter.</p>

<h3>api.invoke(channel, ...args)</h3>
<p>Calls a main-process handler registered under the plugin's own <code>mainHandlers</code>. This is a convenience alias for <code>window.nb.plugins.invoke(pluginId, channel, ...args)</code>.</p>
<pre><code>const result = await nb.invoke('my-plugin:doWork', payload);</code></pre>
<p>The channel string must exactly match a key in <code>mainHandlers</code>. Returns a Promise resolving to the handler's return value.</p>

<hr>

<h2>mainHandlers export (mainCode)</h2>
<p>Your <code>main.js</code> must export a <code>mainHandlers</code> object. Each key is an IPC channel name and each value is the handler function.</p>
<pre><code>module.exports = {
  mainHandlers: {
    // First arg is always the Electron IpcMainInvokeEvent (usually unused).
    // Remaining args are what the renderer passed to nb.invoke().
    'my-plugin:doWork': (event, payload) => {
      return processPayload(payload); // return value goes back to renderer
    },
    'my-plugin:fetchData': async (event, url) => {
      const data = await fetch(url).then(r => r.json());
      return data;
    }
  }
};</code></pre>
<p>All handlers are wrapped in <code>ipcMain.handle()</code> which supports async functions natively. Thrown errors are serialized and re-thrown in the renderer as rejected Promises.</p>

<hr>

<h2>window.nb APIs available to renderer plugins</h2>
<p>Plugin renderer code runs in the Electron renderer context and has access to the full <code>window.nb</code> bridge.</p>

<h3>window.nb.inference.run(networkId, input)</h3>
<p>Runs a forward pass through a trained network. Returns a Promise.</p>
<table>
<tr><th>input field</th><th>Used for</th><th>Type</th></tr>
<tr><td><code>input</code></td><td>classifier / regressor</td><td><code>number[]</code> of length <code>inputDim</code></td></tr>
<tr><td><code>prompt</code></td><td>charLM</td><td>string</td></tr>
<tr><td><code>history</code>, <code>prompt</code></td><td>gpt/chat</td><td>message array + string</td></tr>
</table>
<p>Return value for classifiers: <code>{ kind: 'classification', label, labelIndex, confidence, probs: number[] }</code>. The <code>probs</code> array has one entry per output class.</p>

<h3>window.nb.networks.update(networkId, patch)</h3>
<p>Merges <code>patch</code> into the saved network object. Use this to persist training data after parsing:</p>
<pre><code>await window.nb.networks.update(network.id, {
  trainingData: { samples: parsedSamples }
});</code></pre>

<h3>window.nb.dialog.readTextFile(options)</h3>
<p>Shows a native open-file dialog and returns the selected file's content as a string.</p>
<pre><code>const file = await window.nb.dialog.readTextFile({
  filters: [{ name: 'Text', extensions: ['txt', 'csv'] }]
});
if (file) {
  const { path, content } = file; // content is a UTF-8 string
}</code></pre>

<h3>window.nb.plugins.invoke(pluginId, channel, ...args)</h3>
<p>Low-level invoke. The <code>nb.invoke()</code> helper in your renderer code calls this automatically with your plugin's ID filled in. You can also call other plugins' handlers directly if needed.</p>

<hr>

<h2>Manifest fields quick reference</h2>
<table>
<tr><th>Field</th><th>Required</th><th>Constraints</th><th>Default</th></tr>
<tr><td><code>id</code></td><td>Yes</td><td><code>/^[a-z0-9_-]+$/i</code></td><td>—</td></tr>
<tr><td><code>name</code></td><td>No</td><td>Any string</td><td>Same as <code>id</code></td></tr>
<tr><td><code>version</code></td><td>No</td><td>SemVer string</td><td><code>"0.0.0"</code></td></tr>
<tr><td><code>description</code></td><td>No</td><td>Any string</td><td><code>""</code></td></tr>
<tr><td><code>author</code></td><td>No</td><td>Any string</td><td><code>""</code></td></tr>
<tr><td><code>mainCode</code></td><td>No</td><td>Valid JS; must export <code>{ mainHandlers }</code></td><td><code>""</code></td></tr>
<tr><td><code>rendererCode</code></td><td>No</td><td>Valid JS; receives <code>api</code> arg</td><td><code>""</code></td></tr>
</table>

<h2>Built-in example plugin — Chess</h2>
<p>The Chess Move Predictor plugin bundled with NeuralCabin (<code>src/plugins/chess/</code>) demonstrates all three hooks: a template registering a 73-input → 4096-output classifier, a training editor for parsing UCI game files, and an inference renderer with a full interactive chessboard. Browse <code>src/plugins/chess/main.js</code> and <code>renderer.js</code> for a complete real-world reference implementation.</p>
`
  },
  {
    id: 'contributing',
    title: 'Contributing',
    body: `
<h1>Contributing</h1>
<p>NeuralCabin is open-source and hosted at <a href="https://github.com/GuyThatLivesAndCodes/NeuralCabin" target="_blank">github.com/GuyThatLivesAndCodes/NeuralCabin</a>. Issues and pull requests are welcome.</p>

<h2>Repository layout</h2>
<pre><code>src/
  engine/      tensor.js (backend selector)
               tensor-js.js (pure-JS autograd engine)
               tensor-native-loader.js (N-API compatibility check)
               layers.js, optim.js, model.js, trainer.js
               rl.js (Q-Learning / DQN agent)
               neuroevolution.js (Selective Reproduction / Population)
               tokenizer.js, chat-format.js
  dsl/         lexer.js, parser.js, compiler.js, typecheck.js, interpreter.js
  main/        main.js, preload.js, storage.js,
               training-manager.js, api-server.js
  renderer/    index.html, styles.css, app.js,
               templates.js, docs.js
native/
  rust-engine/
    neuralcabin-core/   Rust library (cpu, layers, optim, rl, neuroevolution)
    neuralcabin-node/   N-API bridge (src/lib.rs + index.js)
  cpp-inference-server/ C++ inference scaffold
docs/          rust-migration.md
tests/         run-tests.js
assets/        icon.png, make-icon.js</code></pre>

<h2>Running from source</h2>
<pre><code>npm install
npm start               # launches the Electron app
npm test                # runs the engine test harness
npm run build:win       # builds NeuralCabin-Setup-x.x.x.exe
npm run engine:check:rust
npm run engine:build:rust
npm run server:build:cpp</code></pre>

<h2>Adding a new tensor op</h2>
<p>There are two paths depending on whether you want Rust acceleration:</p>
<p><strong>JS-only op</strong> (always available, no build step):</p>
<ol>
  <li>Open <code>src/engine/tensor-js.js</code>.</li>
  <li>Implement the forward pass, set <code>requiresGrad</code>, assign <code>out._parents</code> and <code>out._backward</code>. Use <code>+=</code> in all gradient accumulations.</li>
  <li>Export from the bottom of the file and add a test in <code>tests/run-tests.js</code>.</li>
</ol>
<p><strong>Rust-accelerated op</strong>:</p>
<ol>
  <li>Implement the kernel in <code>native/rust-engine/neuralcabin-core/src/cpu.rs</code>.</li>
  <li>Expose it as a <code>#[napi]</code> function in <code>native/rust-engine/neuralcabin-node/src/lib.rs</code>.</li>
  <li>Override the JS fallback in <code>native/rust-engine/neuralcabin-node/index.js</code>, capturing any backward cache (tcache, probs, mask) from the Rust result.</li>
  <li>Run <code>npm run engine:build:rust</code> and add a test case.</li>
</ol>

<h2>Adding a new layer type</h2>
<ol>
  <li>Open <code>src/engine/layers.js</code>.</li>
  <li>Implement a class with <code>forward(x, training)</code> and <code>parameters()</code> methods.</li>
  <li><code>parameters()</code> must return all trainable <code>Tensor</code> instances with <code>requiresGrad = true</code>.</li>
  <li>Register the layer type in the model builder (<code>src/engine/model.js</code>) and expose it in the Editor tab's architecture spec.</li>
</ol>

<h2>Adding a new template</h2>
<ol>
  <li>Open <code>src/renderer/templates.js</code>.</li>
  <li>Add an entry to the templates array with <code>id</code>, <code>name</code>, <code>spec</code> (architecture), <code>data</code> (training samples), and <code>opts</code> (optimizer defaults).</li>
  <li>The template will appear in the <code>+ New</code> dialog on next launch.</li>
</ol>

<h2>Code style</h2>
<ul>
  <li>Primary app code is plain ES5/ES6 JavaScript; native acceleration lives in Rust/C++ modules under <code>native/</code>.</li>
  <li>Keep the JS fallback engine dependency-light; native dependencies belong under <code>native/</code>.</li>
  <li>Comments on performance-sensitive sections (loop order, caching decisions) are expected and valued.</li>
  <li>Keep the engine small. If a feature belongs in application code, put it in <code>src/main/</code> or <code>src/renderer/</code>, not in the engine.</li>
</ul>
`
  }
];


