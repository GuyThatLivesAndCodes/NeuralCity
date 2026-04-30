//! Reverse-mode automatic differentiation on a flat tape.
//!
//! A `Tape` records every forward operation. Each operation knows how to
//! propagate gradients to its parents during the backward sweep. Parameters
//! live outside the tape: callers build a fresh tape per forward/backward
//! pair, register parameter tensors as leaves, then read accumulated grads
//! after `backward(loss)`.

use crate::tensor::Tensor;

/// Handle to a value living inside a `Tape`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub struct Value(pub usize);

/// Operations the tape knows how to differentiate.
#[derive(Debug, Clone)]
pub enum Op {
    Leaf,
    Add(Value, Value),
    Sub(Value, Value),
    Mul(Value, Value),
    /// Matrix-multiplication (m x k) @ (k x n) -> (m x n).
    MatMul(Value, Value),
    /// Add a row-vector bias, broadcast across all rows of the input.
    AddBias { x: Value, b: Value },
    MulScalar(Value, f32),
    AddScalar(Value, f32),
    SumAll(Value),
    ReLU(Value),
    Sigmoid(Value),
    Tanh(Value),
    /// Mean-squared error against a fixed target (target is constant — no grad).
    MseLoss { pred: Value, target: Tensor },
    /// Softmax-cross-entropy of `logits` against fixed one-hot `target`.
    /// Returns a scalar averaged over rows (the batch).
    SoftmaxCrossEntropy { logits: Value, target: Tensor, softmax: Tensor },
}

#[derive(Default)]
pub struct Tape {
    pub values: Vec<Tensor>,
    pub grads: Vec<Option<Tensor>>,
    pub ops: Vec<Op>,
}

impl Tape {
    pub fn new() -> Self { Self::default() }

    pub fn len(&self) -> usize { self.values.len() }
    pub fn is_empty(&self) -> bool { self.values.is_empty() }

    /// Register a leaf value (an input or parameter).
    pub fn leaf(&mut self, t: Tensor) -> Value {
        let idx = self.values.len();
        self.values.push(t);
        self.grads.push(None);
        self.ops.push(Op::Leaf);
        Value(idx)
    }

    fn push(&mut self, t: Tensor, op: Op) -> Value {
        let idx = self.values.len();
        self.values.push(t);
        self.grads.push(None);
        self.ops.push(op);
        Value(idx)
    }

    pub fn value(&self, v: Value) -> &Tensor { &self.values[v.0] }
    pub fn grad(&self, v: Value) -> Option<&Tensor> { self.grads[v.0].as_ref() }

    // ---- forward ops -------------------------------------------------------------

    pub fn add(&mut self, a: Value, b: Value) -> Value {
        let out = self.values[a.0].add(&self.values[b.0]).expect("add shape");
        self.push(out, Op::Add(a, b))
    }

    pub fn sub(&mut self, a: Value, b: Value) -> Value {
        let out = self.values[a.0].sub(&self.values[b.0]).expect("sub shape");
        self.push(out, Op::Sub(a, b))
    }

    pub fn mul(&mut self, a: Value, b: Value) -> Value {
        let out = self.values[a.0].mul(&self.values[b.0]).expect("mul shape");
        self.push(out, Op::Mul(a, b))
    }

    pub fn matmul(&mut self, a: Value, b: Value) -> Value {
        let out = self.values[a.0].matmul(&self.values[b.0]).expect("matmul shape");
        self.push(out, Op::MatMul(a, b))
    }

    pub fn add_bias(&mut self, x: Value, b: Value) -> Value {
        let xt = &self.values[x.0];
        let bt = &self.values[b.0];
        let bb = bt.broadcast_rows(xt.rows()).expect("bias broadcast");
        let out = xt.add(&bb).expect("bias add");
        self.push(out, Op::AddBias { x, b })
    }

    pub fn mul_scalar(&mut self, a: Value, s: f32) -> Value {
        let out = self.values[a.0].mul_scalar(s);
        self.push(out, Op::MulScalar(a, s))
    }

    pub fn add_scalar(&mut self, a: Value, s: f32) -> Value {
        let out = self.values[a.0].add_scalar(s);
        self.push(out, Op::AddScalar(a, s))
    }

    pub fn sum_all(&mut self, a: Value) -> Value {
        let s = self.values[a.0].sum();
        self.push(Tensor::from_scalar(s), Op::SumAll(a))
    }

    pub fn relu(&mut self, a: Value) -> Value {
        let out = self.values[a.0].map(|x| if x > 0.0 { x } else { 0.0 });
        self.push(out, Op::ReLU(a))
    }

    pub fn sigmoid(&mut self, a: Value) -> Value {
        let out = self.values[a.0].map(|x| 1.0 / (1.0 + (-x).exp()));
        self.push(out, Op::Sigmoid(a))
    }

    pub fn tanh(&mut self, a: Value) -> Value {
        let out = self.values[a.0].map(|x| x.tanh());
        self.push(out, Op::Tanh(a))
    }

    pub fn mse_loss(&mut self, pred: Value, target: Tensor) -> Value {
        let p = &self.values[pred.0];
        assert_eq!(p.shape, target.shape, "mse: pred {:?} target {:?}", p.shape, target.shape);
        let n = p.data.len() as f32;
        let mut s = 0.0_f32;
        for (a, b) in p.data.iter().zip(&target.data) {
            let d = a - b;
            s += d * d;
        }
        let loss = s / n;
        self.push(Tensor::from_scalar(loss), Op::MseLoss { pred, target })
    }

    pub fn softmax_cross_entropy(&mut self, logits: Value, target: Tensor) -> Value {
        let l = &self.values[logits.0];
        assert_eq!(l.shape.len(), 2, "softmax_ce expects (batch, classes)");
        assert_eq!(l.shape, target.shape, "softmax_ce: logits {:?} target {:?}", l.shape, target.shape);
        let (rows, cols) = (l.rows(), l.cols());
        let mut sm = vec![0.0_f32; rows * cols];
        let mut loss = 0.0_f32;
        for i in 0..rows {
            // numerically-stable softmax row-wise
            let row = &l.data[i * cols..(i + 1) * cols];
            let mut m = f32::NEG_INFINITY;
            for &x in row { if x > m { m = x; } }
            let mut sum = 0.0_f32;
            for j in 0..cols {
                let e = (row[j] - m).exp();
                sm[i * cols + j] = e;
                sum += e;
            }
            for j in 0..cols {
                sm[i * cols + j] /= sum;
                let p = sm[i * cols + j].max(1e-12);
                loss -= target.data[i * cols + j] * p.ln();
            }
        }
        loss /= rows as f32;
        let softmax = Tensor::new(vec![rows, cols], sm);
        self.push(Tensor::from_scalar(loss), Op::SoftmaxCrossEntropy { logits, target, softmax })
    }

    // ---- backward ----------------------------------------------------------------

    fn accumulate(grads: &mut [Option<Tensor>], idx: usize, contrib: Tensor) {
        match &mut grads[idx] {
            Some(g) => g.add_inplace(&contrib).expect("grad shape"),
            None => grads[idx] = Some(contrib),
        }
    }

    pub fn backward(&mut self, loss: Value) {
        // Seed: dL/dL = 1.
        let scalar = self.values[loss.0].clone();
        assert_eq!(scalar.data.len(), 1, "backward expects a scalar loss, got shape {:?}", scalar.shape);
        self.grads[loss.0] = Some(Tensor::from_scalar(1.0));

        for i in (0..self.ops.len()).rev() {
            // No grad flowing into this node — skip.
            let g_out = match self.grads[i].clone() {
                Some(g) => g,
                None => continue,
            };
            let op = self.ops[i].clone();
            match op {
                Op::Leaf => { /* leaf — gradient stays in self.grads[i] */ }
                Op::Add(a, b) => {
                    Self::accumulate(&mut self.grads, a.0, g_out.clone());
                    Self::accumulate(&mut self.grads, b.0, g_out);
                }
                Op::Sub(a, b) => {
                    Self::accumulate(&mut self.grads, a.0, g_out.clone());
                    Self::accumulate(&mut self.grads, b.0, g_out.neg());
                }
                Op::Mul(a, b) => {
                    let av = self.values[a.0].clone();
                    let bv = self.values[b.0].clone();
                    Self::accumulate(&mut self.grads, a.0, g_out.mul(&bv).expect("mul grad"));
                    Self::accumulate(&mut self.grads, b.0, g_out.mul(&av).expect("mul grad"));
                }
                Op::MatMul(a, b) => {
                    // C = A @ B.  dA = dC @ B^T.  dB = A^T @ dC.
                    let av = self.values[a.0].clone();
                    let bv = self.values[b.0].clone();
                    let bt = bv.transpose().expect("matmul B^T");
                    let at = av.transpose().expect("matmul A^T");
                    Self::accumulate(&mut self.grads, a.0, g_out.matmul(&bt).expect("dA"));
                    Self::accumulate(&mut self.grads, b.0, at.matmul(&g_out).expect("dB"));
                }
                Op::AddBias { x, b } => {
                    Self::accumulate(&mut self.grads, x.0, g_out.clone());
                    let db = g_out.sum_rows().expect("bias sum_rows");
                    Self::accumulate(&mut self.grads, b.0, db);
                }
                Op::MulScalar(a, s) => {
                    Self::accumulate(&mut self.grads, a.0, g_out.mul_scalar(s));
                }
                Op::AddScalar(a, _) => {
                    Self::accumulate(&mut self.grads, a.0, g_out);
                }
                Op::SumAll(a) => {
                    let g = g_out.data[0];
                    let av_shape = self.values[a.0].shape.clone();
                    Self::accumulate(&mut self.grads, a.0, Tensor::filled(av_shape, g));
                }
                Op::ReLU(a) => {
                    let av = &self.values[a.0];
                    let mask = av.map(|x| if x > 0.0 { 1.0 } else { 0.0 });
                    Self::accumulate(&mut self.grads, a.0, g_out.mul(&mask).expect("relu grad"));
                }
                Op::Sigmoid(a) => {
                    let yv = self.values[i].clone();           // y = sigmoid(x)
                    let one_minus_y = yv.map(|y| 1.0 - y);
                    let dy = yv.mul(&one_minus_y).expect("sig").mul(&g_out).expect("sig*g");
                    Self::accumulate(&mut self.grads, a.0, dy);
                }
                Op::Tanh(a) => {
                    let yv = self.values[i].clone();           // y = tanh(x)
                    let dy = g_out.mul(&yv.map(|y| 1.0 - y * y)).expect("tanh grad");
                    Self::accumulate(&mut self.grads, a.0, dy);
                }
                Op::MseLoss { pred, target } => {
                    // L = mean((p - t)^2).  dL/dp = 2/N * (p - t) * dL.
                    let p = self.values[pred.0].clone();
                    let n = p.data.len() as f32;
                    let scale = 2.0 * g_out.data[0] / n;
                    let diff = p.sub(&target).expect("mse diff");
                    Self::accumulate(&mut self.grads, pred.0, diff.mul_scalar(scale));
                }
                Op::SoftmaxCrossEntropy { logits, target, softmax } => {
                    // dL/dlogits = (softmax - target) / batch_size * dL.
                    let batch = softmax.rows() as f32;
                    let diff = softmax.sub(&target).expect("ce diff");
                    Self::accumulate(&mut self.grads, logits.0, diff.mul_scalar(g_out.data[0] / batch));
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    /// Numerical-vs-analytic gradient check on a small linear+MSE graph.
    #[test]
    fn matmul_mse_grad_matches_numerical() {
        let w_init = vec![0.1_f32, -0.2, 0.3, 0.4, 0.5, -0.6];
        let x = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]); // (batch=2, in=2)
        let target = Tensor::new(vec![2, 3], vec![0.0, 1.0, 0.0, 1.0, 0.0, 0.0]);

        let analytic = {
            let mut tape = Tape::new();
            let xv = tape.leaf(x.clone());
            let wv = tape.leaf(Tensor::new(vec![2, 3], w_init.clone()));
            let y = tape.matmul(xv, wv);
            let loss = tape.mse_loss(y, target.clone());
            tape.backward(loss);
            tape.grad(wv).cloned().unwrap()
        };

        let eps = 1e-3_f32;
        for i in 0..w_init.len() {
            let mut wp = w_init.clone();
            let mut wm = w_init.clone();
            wp[i] += eps;
            wm[i] -= eps;
            let lp = forward_loss(&x, &wp, &target);
            let lm = forward_loss(&x, &wm, &target);
            let num = (lp - lm) / (2.0 * eps);
            let an = analytic.data[i];
            assert!((num - an).abs() < 1e-2, "grad mismatch at {i}: num={num} an={an}");
        }
    }

    fn forward_loss(x: &Tensor, w: &[f32], target: &Tensor) -> f32 {
        let mut tape = Tape::new();
        let xv = tape.leaf(x.clone());
        let wv = tape.leaf(Tensor::new(vec![2, 3], w.to_vec()));
        let y = tape.matmul(xv, wv);
        let loss = tape.mse_loss(y, target.clone());
        tape.value(loss).data[0]
    }

    #[test]
    fn sigmoid_backward_correct() {
        let mut tape = Tape::new();
        let x = tape.leaf(Tensor::new(vec![1, 3], vec![-1.0, 0.0, 1.0]));
        let y = tape.sigmoid(x);
        let s = tape.sum_all(y);
        tape.backward(s);
        let g = tape.grad(x).unwrap();
        // d/dx sum(σ) = σ(x)*(1-σ(x))
        for (i, xv) in [-1.0_f32, 0.0, 1.0].iter().enumerate() {
            let s = 1.0 / (1.0 + (-xv).exp());
            let exp = s * (1.0 - s);
            assert!((g.data[i] - exp).abs() < 1e-6);
        }
    }
}
