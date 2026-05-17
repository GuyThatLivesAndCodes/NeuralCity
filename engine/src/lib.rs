//! NeuralCabin engine: tensors, neural-network modules, optimizers, and
//! tokenization — now backed by the [Burn](https://burn.dev) framework so
//! training and inference can run on a GPU via the WGPU backend.
//!
//! The public API (`Tensor`, `Model`, `Loss`, `OptimizerKind`, etc.) is
//! intentionally CPU/Vec<f32> shaped so saved-state files keep their format
//! and so the Tauri layer doesn't need to know Burn types. The training step
//! converts to Burn tensors, runs forward/backward on the chosen device, and
//! writes the updated weights back out.

pub mod backend;
pub mod tensor;
pub mod activations;
pub mod loss;
pub mod optimizer;
pub mod nn;
pub mod data;
pub mod persistence;
pub mod tokenizer;
pub mod corpus;
pub mod transformer;

pub use activations::Activation;
pub use backend::{
    default_gpu_device, CpuAutodiffBackend, CpuBackend, GpuAutodiffBackend, GpuBackend, GpuDevice,
};
pub use loss::Loss;
pub use nn::{train_step_on_device, Layer, LayerSpec, Model};
pub use optimizer::{Optimizer, OptimizerKind};
pub use tensor::Tensor;
pub use tokenizer::{TokenizerMode, Vocabulary};
pub use transformer::{FrozenSnapshotGpu, InferenceSession, TrainerSession};
