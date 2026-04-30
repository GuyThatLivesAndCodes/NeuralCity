//! Neural-network layer and model abstractions.
//!
//! A `Model` is a sequential stack of `Layer`s. Each forward pass builds a
//! fresh `Tape`, registers the model's parameters as leaves on it, and returns
//! the output handle plus the parameter handles in canonical order.

use crate::activations::Activation;
use crate::autograd::{Tape, Value};
use crate::loss::Loss;
use crate::optimizer::Optimizer;
use crate::tensor::{SplitMix64, Tensor};
use serde::{Deserialize, Serialize};

/// Declarative layer description — used for UI building & for serialisation.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LayerSpec {
    Linear { in_dim: usize, out_dim: usize },
    Activation(Activation),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LinearLayer {
    pub in_dim: usize,
    pub out_dim: usize,
    pub w: Tensor,   // (in_dim, out_dim)
    pub b: Tensor,   // (1, out_dim)
}

impl LinearLayer {
    pub fn new(in_dim: usize, out_dim: usize, rng: &mut SplitMix64) -> Self {
        let w = Tensor::xavier(vec![in_dim, out_dim], in_dim, out_dim, rng);
        let b = Tensor::zeros(vec![1, out_dim]);
        Self { in_dim, out_dim, w, b }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum Layer {
    Linear(LinearLayer),
    Activation(Activation),
}

impl Layer {
    pub fn forward_eager(&self, x: &Tensor) -> Tensor {
        match self {
            Layer::Linear(l) => {
                let z = x.matmul(&l.w).expect("linear matmul");
                let bb = l.b.broadcast_rows(z.rows()).expect("linear bias");
                z.add(&bb).expect("linear add")
            }
            Layer::Activation(a) => a.apply(x),
        }
    }

    /// Build forward into the tape. Pushes parameter Values into `param_values`
    /// in declaration order: `[w, b, ...]` per linear layer.
    pub fn forward_tape(&self, tape: &mut Tape, x: Value, param_values: &mut Vec<Value>) -> Value {
        match self {
            Layer::Linear(l) => {
                let w = tape.leaf(l.w.clone());
                let b = tape.leaf(l.b.clone());
                param_values.push(w);
                param_values.push(b);
                let z = tape.matmul(x, w);
                tape.add_bias(z, b)
            }
            Layer::Activation(a) => a.forward(tape, x),
        }
    }

    pub fn parameter_count(&self) -> usize {
        match self {
            Layer::Linear(l) => l.w.len() + l.b.len(),
            Layer::Activation(_) => 0,
        }
    }

    pub fn output_dim(&self, input_dim: usize) -> usize {
        match self {
            Layer::Linear(l) => l.out_dim,
            Layer::Activation(_) => input_dim,
        }
    }

    pub fn describe(&self) -> String {
        match self {
            Layer::Linear(l) => format!("Linear ({} -> {})", l.in_dim, l.out_dim),
            Layer::Activation(a) => format!("Activation: {}", a.name()),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Model {
    pub input_dim: usize,
    pub layers: Vec<Layer>,
    pub seed: u64,
}

impl Model {
    pub fn new(input_dim: usize) -> Self {
        Self { input_dim, layers: Vec::new(), seed: 0xC0FFEE }
    }

    pub fn from_specs(input_dim: usize, specs: &[LayerSpec], seed: u64) -> Self {
        let mut rng = SplitMix64::new(seed);
        let mut layers = Vec::with_capacity(specs.len());
        let mut cur_dim = input_dim;
        for spec in specs {
            match spec {
                LayerSpec::Linear { in_dim, out_dim } => {
                    assert_eq!(*in_dim, cur_dim,
                        "layer in_dim {in_dim} doesn't match running dim {cur_dim}");
                    layers.push(Layer::Linear(LinearLayer::new(*in_dim, *out_dim, &mut rng)));
                    cur_dim = *out_dim;
                }
                LayerSpec::Activation(a) => layers.push(Layer::Activation(*a)),
            }
        }
        Self { input_dim, layers, seed }
    }

    pub fn output_dim(&self) -> usize {
        let mut d = self.input_dim;
        for l in &self.layers { d = l.output_dim(d); }
        d
    }

    pub fn parameter_count(&self) -> usize {
        self.layers.iter().map(|l| l.parameter_count()).sum()
    }

    pub fn parameter_shapes(&self) -> Vec<Vec<usize>> {
        let mut out = Vec::new();
        for l in &self.layers {
            if let Layer::Linear(ll) = l {
                out.push(ll.w.shape.clone());
                out.push(ll.b.shape.clone());
            }
        }
        out
    }

    /// Mutable parameter references in canonical order (w0, b0, w1, b1, ...).
    pub fn parameters_mut(&mut self) -> Vec<&mut Tensor> {
        let mut out: Vec<&mut Tensor> = Vec::new();
        for l in &mut self.layers {
            if let Layer::Linear(ll) = l {
                out.push(&mut ll.w);
                out.push(&mut ll.b);
            }
        }
        out
    }

    /// Pure forward (no autograd). Used for inference / evaluation.
    pub fn predict(&self, input: &Tensor) -> Tensor {
        let mut x = input.clone();
        for l in &self.layers { x = l.forward_eager(&x); }
        x
    }

    /// One training step on a single (input, target) batch.
    /// Returns the scalar loss.
    pub fn train_step(
        &mut self,
        optimizer: &mut Optimizer,
        loss_kind: Loss,
        input: &Tensor,
        target: &Tensor,
    ) -> f32 {
        let mut tape = Tape::new();
        let xv = tape.leaf(input.clone());
        let mut param_values: Vec<Value> = Vec::new();
        let mut cur = xv;
        for layer in &self.layers {
            cur = layer.forward_tape(&mut tape, cur, &mut param_values);
        }
        let loss_v = loss_kind.build(&mut tape, cur, target);
        let loss_scalar = tape.value(loss_v).data[0];
        tape.backward(loss_v);

        // Gather gradients in the same order as parameters_mut().
        let grads: Vec<Option<Tensor>> =
            param_values.iter().map(|v| tape.grad(*v).cloned()).collect();
        let mut params = self.parameters_mut();
        optimizer.step(&mut params, &grads);
        loss_scalar
    }

    /// Evaluate loss on a (possibly held-out) batch without updating params.
    pub fn evaluate_loss(&self, loss_kind: Loss, input: &Tensor, target: &Tensor) -> f32 {
        let pred = self.predict(input);
        loss_kind.eval(&pred, target)
    }

    /// Classification accuracy: argmax of prediction against argmax of one-hot
    /// target. Both must be (batch, classes).
    pub fn accuracy(&self, input: &Tensor, target_onehot: &Tensor) -> f32 {
        let pred = self.predict(input);
        assert_eq!(pred.shape, target_onehot.shape);
        let (rows, cols) = (pred.rows(), pred.cols());
        let mut hits = 0;
        for i in 0..rows {
            let pr = &pred.data[i * cols..(i + 1) * cols];
            let tr = &target_onehot.data[i * cols..(i + 1) * cols];
            let pa = argmax(pr);
            let ta = argmax(tr);
            if pa == ta { hits += 1; }
        }
        hits as f32 / rows as f32
    }
}

fn argmax(row: &[f32]) -> usize {
    let mut best = 0_usize;
    let mut bv = f32::NEG_INFINITY;
    for (i, &v) in row.iter().enumerate() {
        if v > bv { bv = v; best = i; }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::optimizer::OptimizerKind;

    /// XOR is the canonical "non-linearly separable" problem.  A small MLP must
    /// learn it; we assert the model converges below loss < 0.05 within budget.
    #[test]
    fn mlp_learns_xor() {
        let specs = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 8 },
            LayerSpec::Activation(Activation::Tanh),
            LayerSpec::Linear { in_dim: 8, out_dim: 1 },
            LayerSpec::Activation(Activation::Sigmoid),
        ];
        let mut model = Model::from_specs(2, &specs, 42);
        let mut opt = Optimizer::new(
            OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
            &model.parameter_shapes(),
        );
        let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);
        let mut last = f32::INFINITY;
        for _ in 0..2000 {
            last = model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
        }
        assert!(last < 0.05, "XOR did not converge: loss={last}");
        let p = model.predict(&x);
        for (pi, ti) in p.data.iter().zip(&y.data) {
            assert!((pi - ti).abs() < 0.2, "XOR pred {pi} vs target {ti}");
        }
    }
}
