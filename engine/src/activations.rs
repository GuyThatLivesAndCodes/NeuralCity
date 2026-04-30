//! Activation functions used both at inference time (pure tensor ops) and as
//! tape nodes during training (see `autograd::Tape`).

use crate::autograd::{Tape, Value};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Activation {
    Identity,
    ReLU,
    Sigmoid,
    Tanh,
    /// Row-wise softmax. Use only as the last layer with `Loss::CrossEntropy`,
    /// or as a pure inference op.
    Softmax,
}

impl Activation {
    pub fn name(&self) -> &'static str {
        match self {
            Activation::Identity => "Identity",
            Activation::ReLU => "ReLU",
            Activation::Sigmoid => "Sigmoid",
            Activation::Tanh => "Tanh",
            Activation::Softmax => "Softmax",
        }
    }

    pub fn all() -> &'static [Activation] {
        &[
            Activation::Identity,
            Activation::ReLU,
            Activation::Sigmoid,
            Activation::Tanh,
            Activation::Softmax,
        ]
    }

    /// Pure-tensor forward (used for inference and Softmax inside the cross-entropy
    /// op).
    pub fn apply(&self, x: &Tensor) -> Tensor {
        match self {
            Activation::Identity => x.clone(),
            Activation::ReLU => x.map(|v| if v > 0.0 { v } else { 0.0 }),
            Activation::Sigmoid => x.map(|v| 1.0 / (1.0 + (-v).exp())),
            Activation::Tanh => x.map(|v| v.tanh()),
            Activation::Softmax => softmax_rows(x),
        }
    }

    /// Tape forward: registers an autograd node for this activation.
    /// Softmax during training is handled by `Tape::softmax_cross_entropy`,
    /// so this function panics if you ask it to insert a standalone Softmax node.
    pub fn forward(&self, tape: &mut Tape, x: Value) -> Value {
        match self {
            Activation::Identity => x,
            Activation::ReLU => tape.relu(x),
            Activation::Sigmoid => tape.sigmoid(x),
            Activation::Tanh => tape.tanh(x),
            Activation::Softmax => panic!(
                "Softmax cannot be used as an autograd activation node — pair it with \
                 Loss::CrossEntropy at the model output instead."
            ),
        }
    }
}

/// Numerically-stable row-wise softmax for a 2-D tensor.
pub fn softmax_rows(x: &Tensor) -> Tensor {
    assert_eq!(x.shape.len(), 2, "softmax_rows expects (batch, classes)");
    let (rows, cols) = (x.rows(), x.cols());
    let mut out = vec![0.0_f32; rows * cols];
    for i in 0..rows {
        let row = &x.data[i * cols..(i + 1) * cols];
        let mut m = f32::NEG_INFINITY;
        for &v in row { if v > m { m = v; } }
        let mut sum = 0.0_f32;
        for j in 0..cols {
            let e = (row[j] - m).exp();
            out[i * cols + j] = e;
            sum += e;
        }
        for j in 0..cols { out[i * cols + j] /= sum; }
    }
    Tensor::new(vec![rows, cols], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn softmax_rows_sums_to_one() {
        let x = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, -1.0, 0.0, 1.0]);
        let s = softmax_rows(&x);
        for i in 0..2 {
            let sum: f32 = s.data[i * 3..(i + 1) * 3].iter().sum();
            assert!((sum - 1.0).abs() < 1e-6);
        }
    }
}
