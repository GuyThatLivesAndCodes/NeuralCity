//! Loss functions, both as eager scalars (for inference / metrics) and as
//! autograd nodes (for training).

use crate::activations::softmax_rows;
use crate::autograd::{Tape, Value};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Loss {
    /// Mean-squared error. Pair with `Activation::Identity`, `Sigmoid`, or `Tanh`.
    MeanSquaredError,
    /// Softmax + categorical cross-entropy. The model's last layer should be
    /// `Activation::Identity` — softmax is applied internally for numerical
    /// stability.
    CrossEntropy,
}

impl Loss {
    pub fn name(&self) -> &'static str {
        match self {
            Loss::MeanSquaredError => "MeanSquaredError",
            Loss::CrossEntropy => "CrossEntropy",
        }
    }

    pub fn all() -> &'static [Loss] {
        &[Loss::MeanSquaredError, Loss::CrossEntropy]
    }

    /// Build the loss into the tape and return the scalar loss handle.
    pub fn build(&self, tape: &mut Tape, output: Value, target: &Tensor) -> Value {
        match self {
            Loss::MeanSquaredError => tape.mse_loss(output, target.clone()),
            Loss::CrossEntropy => tape.softmax_cross_entropy(output, target.clone()),
        }
    }

    /// Eager scalar — handy for evaluating a model without the autograd tape.
    pub fn eval(&self, output: &Tensor, target: &Tensor) -> f32 {
        match self {
            Loss::MeanSquaredError => {
                assert_eq!(output.shape, target.shape, "MSE shape mismatch");
                let n = output.data.len() as f32;
                let mut s = 0.0_f32;
                for (a, b) in output.data.iter().zip(&target.data) {
                    let d = a - b;
                    s += d * d;
                }
                s / n
            }
            Loss::CrossEntropy => {
                assert_eq!(output.shape, target.shape, "CE shape mismatch");
                let sm = softmax_rows(output);
                let rows = sm.rows() as f32;
                let mut s = 0.0_f32;
                for (p, t) in sm.data.iter().zip(&target.data) {
                    s -= t * p.max(1e-12).ln();
                }
                s / rows
            }
        }
    }
}
