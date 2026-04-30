//! Optimisers: parameter update rules. SGD (with optional momentum) and Adam,
//! both implemented from scratch.

use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

#[derive(Copy, Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OptimizerKind {
    Sgd { lr: f32, momentum: f32 },
    Adam { lr: f32, beta1: f32, beta2: f32, eps: f32 },
}

impl OptimizerKind {
    pub fn name(&self) -> &'static str {
        match self {
            OptimizerKind::Sgd { .. } => "SGD",
            OptimizerKind::Adam { .. } => "Adam",
        }
    }

    pub fn lr(&self) -> f32 {
        match *self {
            OptimizerKind::Sgd { lr, .. } | OptimizerKind::Adam { lr, .. } => lr,
        }
    }

    pub fn set_lr(&mut self, new_lr: f32) {
        match self {
            OptimizerKind::Sgd { lr, .. } => *lr = new_lr,
            OptimizerKind::Adam { lr, .. } => *lr = new_lr,
        }
    }
}

/// Per-parameter optimiser state.
pub struct Optimizer {
    pub kind: OptimizerKind,
    pub step: u64,
    pub momentum: Vec<Tensor>,    // SGD momentum buffers (or m for Adam)
    pub variance: Vec<Tensor>,    // Adam v buffers
}

impl Optimizer {
    /// Create an optimiser sized for `param_shapes` parameters in deterministic
    /// order (matching `Model::parameters_mut`).
    pub fn new(kind: OptimizerKind, param_shapes: &[Vec<usize>]) -> Self {
        let momentum = param_shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        let variance = param_shapes.iter().map(|s| Tensor::zeros(s.clone())).collect();
        Self { kind, step: 0, momentum, variance }
    }

    /// Apply one update step. `params` and `grads` must align in length and
    /// per-element shape.
    pub fn step(&mut self, params: &mut [&mut Tensor], grads: &[Option<Tensor>]) {
        self.step += 1;
        assert_eq!(params.len(), grads.len(), "param/grad length mismatch");
        assert_eq!(params.len(), self.momentum.len(), "optimiser was sized for {} params", self.momentum.len());

        match self.kind {
            OptimizerKind::Sgd { lr, momentum } => {
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    if momentum != 0.0 {
                        let m = &mut self.momentum[i];
                        // m = momentum*m + g
                        for (mv, gv) in m.data.iter_mut().zip(&g.data) {
                            *mv = momentum * *mv + *gv;
                        }
                        // p -= lr * m
                        p.axpy_inplace(-lr, m).expect("sgd step");
                    } else {
                        p.axpy_inplace(-lr, g).expect("sgd step");
                    }
                }
            }
            OptimizerKind::Adam { lr, beta1, beta2, eps } => {
                let t = self.step as f32;
                let bc1 = 1.0 - beta1.powf(t);
                let bc2 = 1.0 - beta2.powf(t);
                for (i, p) in params.iter_mut().enumerate() {
                    let g = match &grads[i] { Some(g) => g, None => continue };
                    let m = &mut self.momentum[i];
                    let v = &mut self.variance[i];
                    for ((mv, vv), gv) in m.data.iter_mut().zip(v.data.iter_mut()).zip(&g.data) {
                        *mv = beta1 * *mv + (1.0 - beta1) * *gv;
                        *vv = beta2 * *vv + (1.0 - beta2) * (*gv) * (*gv);
                    }
                    for (j, pv) in p.data.iter_mut().enumerate() {
                        let mhat = m.data[j] / bc1;
                        let vhat = v.data[j] / bc2;
                        *pv -= lr * mhat / (vhat.sqrt() + eps);
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sgd_descends_on_simple_quadratic() {
        // Minimise (x - 3)^2 with grad = 2*(x - 3).
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(OptimizerKind::Sgd { lr: 0.1, momentum: 0.0 }, &shapes);
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..200 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        assert!((x.data[0] - 3.0).abs() < 1e-3, "x converged to {}", x.data[0]);
    }

    #[test]
    fn adam_converges_too() {
        let shapes = vec![vec![1]];
        let mut opt = Optimizer::new(
            OptimizerKind::Adam { lr: 0.1, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
            &shapes,
        );
        let mut x = Tensor::new(vec![1], vec![0.0]);
        for _ in 0..500 {
            let g = Tensor::new(vec![1], vec![2.0 * (x.data[0] - 3.0)]);
            opt.step(&mut [&mut x], &[Some(g)]);
        }
        assert!((x.data[0] - 3.0).abs() < 5e-2, "x converged to {}", x.data[0]);
    }
}
