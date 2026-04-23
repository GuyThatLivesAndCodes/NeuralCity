window.NC_DOCS = [
  {
    id: 'welcome',
    title: 'Welcome',
    body: `
<h1>Welcome to NeuralCity</h1>
<p>NeuralCity is a self-contained neural network platform built entirely in plain JavaScript — no NumPy, no TensorFlow, no PyTorch. Every tensor operation, every gradient computation, and every optimizer step runs on readable source code you can inspect, modify, and learn from. The full engine is available at <a href="https://github.com/GuyThatLivesAndCodes/NeuralCity" target="_blank">github.com/GuyThatLivesAndCodes/NeuralCity</a>.</p>
<p>The engine is approximately <strong>2,000 lines of plain JavaScript</strong> — small enough to read in an afternoon, powerful enough to train real classifiers, regressors, character language models, and multi-turn chat assistants. There is no compilation step, no virtual environment, and no external runtime dependency beyond Node.js and Electron.</p>
<p>Select a topic from the left sidebar to explore the engine internals, learn how to design and train networks, understand the training data formats, use the HTTP API, script experiments with NeuralScript, and more.</p>

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
<p>NeuralCity ships with five built-in templates to get you started immediately:</p>
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
<p>In NeuralCity, every operation during the forward pass records its input tensors on the output tensor's <code>_parents</code> list and stores a <code>_backward</code> function that knows how to propagate gradients backward through that specific operation. This builds the <strong>computation graph</strong> dynamically — there is no separate graph-compilation step.</p>

<h2>2. Loss computation</h2>
<p>After the forward pass, the network's output is compared to the ground-truth target to produce a single scalar number called the <strong>loss</strong>. A lower loss means better predictions. Two loss functions are implemented:</p>
<ul>
  <li><strong>Softmax cross-entropy</strong> — used for classification. The raw output logits are converted to a probability distribution via softmax, then the negative log-probability of the correct class is taken. The gradient of this loss with respect to the logits has a clean closed form: <code>∂L/∂logit_j = p_j − 1{j=y}</code>, meaning the gradient is just the predicted probability minus 1 at the correct class index.</li>
  <li><strong>Mean squared error (MSE)</strong> — used for regression. Computes the average of squared differences between predicted and target outputs: <code>L = mean((ŷ − y)²)</code>.</li>
</ul>

<h2>3. Backward pass (autograd)</h2>
<p>NeuralCity implements <strong>reverse-mode automatic differentiation</strong> (backpropagation). Calling <code>loss.backward()</code> traverses the computation graph in reverse topological order and applies the chain rule at each node to accumulate <code>∂loss/∂W</code> and <code>∂loss/∂b</code> for every trainable parameter.</p>
<p>The traversal is implemented in <code>Tensor.backward()</code>:</p>
<ol>
  <li>Build the topological order by recursively walking <code>_parents</code> (DFS, deduplication by <code>id</code>).</li>
  <li>Seed the loss tensor's own <code>grad</code> with 1 (since <code>∂loss/∂loss = 1</code>).</li>
  <li>Iterate in reverse order, calling each node's stored <code>_backward()</code> function. That function reads <code>out.grad</code> and accumulates into each parent's <code>grad</code> using <code>+=</code> (to handle shared nodes correctly).</li>
</ol>
<p>Each operator — <code>matmul</code>, <code>relu</code>, <code>gelu</code>, <code>softmax</code>, <code>embedding</code>, and others — implements its own local derivative. You can read every one of them in <code>src/engine/tensor.js</code>.</p>

<h2>4. Optimizer step</h2>
<p>After backward, every parameter has a populated <code>.grad</code> array. The optimizer uses these gradients to update the parameters. NeuralCity ships ten optimizers (see the <strong>Optimizers</strong> reference section for full details):</p>
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
<p>NeuralCity's trainer repeats the following for each epoch:</p>
<ol>
  <li>Shuffle the dataset (using the seeded RNG if a seed is set).</li>
  <li>Slice into mini-batches of the configured <code>batchSize</code>.</li>
  <li>For each batch: zero gradients → forward pass → loss → backward → optimizer step.</li>
  <li>Record the mean epoch loss and emit it to the live chart.</li>
</ol>
<p>This is the complete training loop. Nothing more is needed to train the models NeuralCity supports.</p>

<h2>Reproducibility and seeding</h2>
<p>Setting a <strong>seed</strong> in the Editor tab makes every stochastic step deterministic: weight initialization (Kaiming / scaled Gaussian via Box-Muller), dataset shuffle order, dropout masks, and character-LM temperature sampling all use the same Mulberry32 seeded RNG. Two runs with the same seed, architecture, data, and hyperparameters produce byte-identical results. Omit the seed for non-deterministic runs.</p>
`
  },
  {
    id: 'tensor-engine',
    title: 'Tensor engine internals',
    body: `
<h1>Tensor engine internals</h1>
<p>All computation in NeuralCity is built on the <code>Tensor</code> class defined in <code>src/engine/tensor.js</code>. Understanding this class is the key to understanding everything else.</p>

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

<h2>add (with bias broadcast)</h2>
<p><code>add(a, b)</code> handles two shapes: same-shape elementwise addition, and the common case of a 2D activation <code>[B, N]</code> plus a 1D bias <code>[N]</code>. In the bias-add case, the backward sums the upstream gradient across the batch dimension to produce <code>∂L/∂bias</code> — this is the standard bias gradient accumulation.</p>

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
<p><strong>softmaxCrossEntropy(logits, labels)</strong> — fused softmax + NLL loss. Computes log-sum-exp in a numerically stable way, returns a scalar mean loss over the batch, and stores the softmax probabilities for the backward. The backward gradient at logit <code>[i,j]</code> is <code>(p_ij − 1{j=y_i}) / B</code>.</p>
<p><strong>mseLoss(a, b)</strong> — mean squared error. Backward gradient at index <code>i</code> is <code>2(a_i − b_i) / N</code>.</p>

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
<p>The optimizer updates the network's weights after each backward pass using the accumulated gradients. NeuralCity provides ten optimizers. All optimizer state (momentum buffers, variance estimates, slow weights) is saved alongside the model weights so training can resume without a warmup bump.</p>

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
<p>NeuralCity uses a <strong>character-level tokenizer</strong>. Every unique character in the training corpus becomes a token — there is no subword splitting or BPE. This keeps the implementation simple and the vocabulary fully interpretable, at the cost of longer sequence lengths compared to word-piece tokenizers.</p>

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
<p>NeuralScript is a lightweight scripting language built into NeuralCity for running experiments programmatically. It uses <code>do</code>/<code>end</code> blocks for all compound statements, requires no semicolons, and supports first-class neural operations (<code>build</code>, <code>train</code>, <code>predict</code>) as standard library functions.</p>
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

<h2>Security notes</h2>
<p>The API server binds to all network interfaces by default so devices on the same LAN can reach it. No authentication is implemented. Do not expose it to the public internet. If you need to restrict access, use your OS firewall to allow only specific IP addresses to reach the chosen port.</p>
`
  },
  {
    id: 'encryption',
    title: 'Encryption',
    body: `
<h1>Encrypting a network</h1>
<p>NeuralCity supports optional AES-256-GCM encryption at rest for any saved network. The weights, optimizer state, and tokenizer are serialized into a single bundle, then encrypted using a key derived from your passphrase via <strong>scrypt</strong>. The passphrase is never stored — only the encrypted ciphertext and the scrypt salt and nonce are written to disk.</p>

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
<p>NeuralCity never stores or escrows the passphrase. If it is lost, the weights are unrecoverable by design. Back up the passphrase separately from the network file.</p>

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

<p>The NeuralCity engine (see <a href="https://github.com/GuyThatLivesAndCodes/NeuralCity" target="_blank">the source</a>) is approximately <strong>2,000 lines of plain JavaScript</strong>. You can open <code>src/engine/tensor.js</code> and read the entire autograd system in one sitting. You can trace a gradient backward through a specific operation, verify the chain rule application by hand, and confirm it matches what the code computes. You can add a new activation function in ten lines. You can swap in a different optimizer and watch it change the loss curve. Nothing is hidden.</p>

<h2>What you trade away</h2>
<ul>
  <li><strong>No GPU acceleration.</strong> All computation runs on CPU in JavaScript. This is sufficient for networks up to a few million parameters — XOR to small char LMs. Larger models will be slow.</li>
  <li><strong>No distributed training.</strong> Single machine, single process. Intentional.</li>
  <li><strong>No automatic mixed precision, gradient checkpointing, or FSDP.</strong> Not needed at NeuralCity's scale.</li>
  <li><strong>No Transformer blocks out of the box.</strong> The primitives (matmul, embedding, softmax, gelu) are all present, and NeuralScript can compose them, but there is no pre-built attention layer.</li>
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
<p>NeuralCity is well suited for: learning how neural networks work at the implementation level; rapid prototyping of small models without framework setup overhead; training personal conversational models on local data with no cloud involvement; embedding a neural inference engine in an Electron application; and anyone who finds that black-box tools obscure more than they reveal.</p>
<p>It is not suited for: training large language models, computer vision at scale, production deployment requiring GPU throughput, or any task where PyTorch or JAX would provide meaningful performance or capability advantages. Use the right tool for the job.</p>
`
  },
  {
    id: 'architecture-tips',
    title: 'Architecture design tips',
    body: `
<h1>Architecture design tips</h1>
<p>Choosing a network architecture is partly science and partly empirical. These guidelines apply to the model sizes NeuralCity is designed for.</p>

<h2>Start small, then scale up</h2>
<p>Resist the temptation to build a large network immediately. A 2-hidden-layer network with 32 units per layer will train fast, converge reliably, and tell you quickly whether the task is learnable. If training loss stalls at an unacceptably high value after full convergence, then add capacity. If training loss is good but the model behaves poorly at inference, the problem is usually data quality or quantity, not capacity.</p>

<h2>Depth vs. width</h2>
<p>For most tasks NeuralCity handles, 2–3 hidden layers is sufficient. Adding more layers adds representational depth but also adds vanishing gradient risk and slower convergence. For classification of moderately non-linear data (2D spiral), go deeper (3 layers) before going wider. For regression over smooth functions, width matters more than depth.</p>

<h2>Activation choice</h2>
<ul>
  <li>Use <strong>ReLU</strong> as the default for classifiers and most regressors. It trains fast and rarely causes problems at NeuralCity's scale.</li>
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
<p>NeuralCity saves all network state to the local filesystem. No data is sent to any remote server at any point.</p>

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
<p>Network files are stored in the application data directory, managed by Electron's <code>userData</code> path. On Windows this is typically <code>%APPDATA%\NeuralCity\networks\</code>. You can open a network file in any text editor to inspect the raw JSON (or ciphertext, if encrypted).</p>

<h2>Encryption at rest</h2>
<p>See the <strong>Encryption</strong> topic for full details. Summary: when encryption is enabled, the serialized JSON is encrypted with AES-256-GCM before writing to disk. The passphrase is never stored.</p>

<h2>Exporting and importing</h2>
<p>Network files are portable — copy them to another machine running NeuralCity and load them via the Networks sidebar. Architecture, weights, tokenizer, and optimizer state all transfer together.</p>

<h2>Resuming training</h2>
<p>Because Adam's optimizer state is persisted, you can stop training at epoch 500, close the application, reopen it, and continue from epoch 501 with the same optimizer momentum. The loss curve continues from where it left off. SGD has no persisted state (no momentum), so resuming with SGD is equivalent to a fresh optimizer start on the existing weights.</p>
`
  },
  {
    id: 'contributing',
    title: 'Contributing',
    body: `
<h1>Contributing</h1>
<p>NeuralCity is open-source and hosted at <a href="https://github.com/GuyThatLivesAndCodes/NeuralCity" target="_blank">github.com/GuyThatLivesAndCodes/NeuralCity</a>. Issues and pull requests are welcome.</p>

<h2>Repository layout</h2>
<pre><code>src/
  engine/      tensor.js, layers.js, optim.js, model.js,
               tokenizer.js, chat-format.js, trainer.js
  dsl/         lexer.js, parser.js, interpreter.js
  main/        main.js, preload.js, storage.js,
               training-manager.js, api-server.js
  renderer/    index.html, styles.css, app.js,
               templates.js, docs.js
tests/         run-tests.js
assets/        icon.png, make-icon.js</code></pre>

<h2>Running from source</h2>
<pre><code>npm install
npm start          # launches the Electron app
npm test           # runs the engine test harness
npm run build:win  # builds NeuralCity-Setup-1.0.0.exe</code></pre>

<h2>Adding a new tensor op</h2>
<ol>
  <li>Open <code>src/engine/tensor.js</code>.</li>
  <li>Implement the forward pass into a new <code>Tensor</code> with <code>requiresGrad</code> set correctly.</li>
  <li>Assign <code>out._parents</code> and <code>out._backward</code>. Use <code>+=</code> in all gradient accumulations.</li>
  <li>Export from the bottom of the file.</li>
  <li>Add a test case in <code>tests/run-tests.js</code>. Verify the gradient numerically with finite differences if in doubt.</li>
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
  <li>Plain ES5/ES6 JavaScript. No TypeScript, no transpilation.</li>
  <li>No external runtime dependencies in the engine (<code>src/engine/</code>).</li>
  <li>Comments on performance-sensitive sections (loop order, caching decisions) are expected and valued.</li>
  <li>Keep the engine small. If a feature belongs in application code, put it in <code>src/main/</code> or <code>src/renderer/</code>, not in the engine.</li>
</ul>
`
  }
];
