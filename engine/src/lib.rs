//! NeuralCabin engine: a from-scratch tensor + autograd + neural network library.
//!
//! Nothing external is used for math. `serde` is used solely for model persistence.

pub mod tensor;
pub mod autograd;
pub mod activations;
pub mod loss;
pub mod optimizer;
pub mod nn;
pub mod data;
pub mod persistence;

pub use activations::Activation;
pub use autograd::{Tape, Value};
pub use loss::Loss;
pub use nn::{Layer, LayerSpec, Model};
pub use optimizer::{Optimizer, OptimizerKind};
pub use tensor::Tensor;
