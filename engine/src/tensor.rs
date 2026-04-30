//! Dense, row-major N-dimensional tensor backed by a `Vec<f32>`.
//!
//! All math is implemented inline. No BLAS, no `ndarray`, no SIMD intrinsics —
//! just straightforward Rust loops. Operations work on owned tensors and return
//! owned tensors so the autograd tape can track them cleanly.

// Plain index loops are clearer than enumerate-style iterators for the small
// dense numerical kernels in this file.
#![allow(clippy::needless_range_loop)]

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch { lhs: Vec<usize>, rhs: Vec<usize>, op: &'static str },
    BadReshape { from: Vec<usize>, to: Vec<usize> },
    BadMatmul { lhs: Vec<usize>, rhs: Vec<usize> },
    NotMatrix(Vec<usize>),
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { lhs, rhs, op } => {
                write!(f, "shape mismatch in {op}: {lhs:?} vs {rhs:?}")
            }
            TensorError::BadReshape { from, to } => {
                write!(f, "cannot reshape {from:?} -> {to:?} (element count differs)")
            }
            TensorError::BadMatmul { lhs, rhs } => {
                write!(f, "cannot matmul {lhs:?} x {rhs:?}")
            }
            TensorError::NotMatrix(s) => write!(f, "expected 2-D tensor, got {s:?}"),
        }
    }
}

impl std::error::Error for TensorError {}

impl Tensor {
    pub fn new(shape: Vec<usize>, data: Vec<f32>) -> Self {
        let n: usize = shape.iter().product();
        assert_eq!(n, data.len(), "data length {} does not match shape {:?}", data.len(), shape);
        Self { shape, data }
    }

    pub fn zeros(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![0.0; n], shape }
    }

    pub fn ones(shape: Vec<usize>) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![1.0; n], shape }
    }

    pub fn filled(shape: Vec<usize>, v: f32) -> Self {
        let n: usize = shape.iter().product();
        Self { data: vec![v; n], shape }
    }

    pub fn from_scalar(v: f32) -> Self {
        Self { shape: vec![1], data: vec![v] }
    }

    /// Initialise with a deterministic Xavier/Glorot-uniform draw using the supplied PRNG.
    pub fn xavier(shape: Vec<usize>, fan_in: usize, fan_out: usize, rng: &mut SplitMix64) -> Self {
        let limit = (6.0_f32 / (fan_in as f32 + fan_out as f32)).sqrt();
        let n: usize = shape.iter().product();
        let mut data = Vec::with_capacity(n);
        for _ in 0..n {
            let u: f32 = rng.next_f32() * 2.0 - 1.0;
            data.push(u * limit);
        }
        Self { shape, data }
    }

    pub fn len(&self) -> usize { self.data.len() }
    pub fn is_empty(&self) -> bool { self.data.is_empty() }
    pub fn rank(&self) -> usize { self.shape.len() }

    pub fn rows(&self) -> usize {
        debug_assert_eq!(self.shape.len(), 2, "rows() requires a 2-D tensor");
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        debug_assert_eq!(self.shape.len(), 2, "cols() requires a 2-D tensor");
        self.shape[1]
    }

    fn ensure_same_shape(&self, other: &Tensor, op: &'static str) -> Result<(), TensorError> {
        if self.shape != other.shape {
            return Err(TensorError::ShapeMismatch {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
                op,
            });
        }
        Ok(())
    }

    // --- element-wise operations --------------------------------------------------

    pub fn add(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.ensure_same_shape(other, "add")?;
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a + b).collect();
        Ok(Tensor { shape: self.shape.clone(), data })
    }

    pub fn sub(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.ensure_same_shape(other, "sub")?;
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a - b).collect();
        Ok(Tensor { shape: self.shape.clone(), data })
    }

    pub fn mul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        self.ensure_same_shape(other, "mul")?;
        let data = self.data.iter().zip(&other.data).map(|(a, b)| a * b).collect();
        Ok(Tensor { shape: self.shape.clone(), data })
    }

    pub fn add_scalar(&self, v: f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x + v).collect() }
    }
    pub fn mul_scalar(&self, v: f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x * v).collect() }
    }
    pub fn div_scalar(&self, v: f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|x| x / v).collect() }
    }

    /// In-place add — used by optimisers.
    pub fn add_inplace(&mut self, other: &Tensor) -> Result<(), TensorError> {
        self.ensure_same_shape(other, "add_inplace")?;
        for (a, b) in self.data.iter_mut().zip(&other.data) { *a += *b; }
        Ok(())
    }

    /// In-place axpy: self += alpha * other. Used by SGD/Adam.
    pub fn axpy_inplace(&mut self, alpha: f32, other: &Tensor) -> Result<(), TensorError> {
        self.ensure_same_shape(other, "axpy")?;
        for (a, b) in self.data.iter_mut().zip(&other.data) { *a += alpha * *b; }
        Ok(())
    }

    pub fn neg(&self) -> Tensor { self.mul_scalar(-1.0) }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor {
        Tensor { shape: self.shape.clone(), data: self.data.iter().map(|&x| f(x)).collect() }
    }

    pub fn sum(&self) -> f32 { self.data.iter().sum() }
    pub fn mean(&self) -> f32 { self.sum() / self.data.len() as f32 }

    // --- shape ops ---------------------------------------------------------------

    pub fn reshape(&self, new_shape: Vec<usize>) -> Result<Tensor, TensorError> {
        let n: usize = new_shape.iter().product();
        if n != self.data.len() {
            return Err(TensorError::BadReshape { from: self.shape.clone(), to: new_shape });
        }
        Ok(Tensor { shape: new_shape, data: self.data.clone() })
    }

    /// 2-D transpose.
    pub fn transpose(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::NotMatrix(self.shape.clone()));
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0_f32; r * c];
        for i in 0..r {
            for j in 0..c {
                out[j * r + i] = self.data[i * c + j];
            }
        }
        Ok(Tensor { shape: vec![c, r], data: out })
    }

    // --- matrix multiplication ---------------------------------------------------

    /// 2-D matmul: (m x k) * (k x n) = (m x n).
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(TensorError::BadMatmul {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
            });
        }
        let (m, k) = (self.shape[0], self.shape[1]);
        let (k2, n) = (other.shape[0], other.shape[1]);
        if k != k2 {
            return Err(TensorError::BadMatmul {
                lhs: self.shape.clone(),
                rhs: other.shape.clone(),
            });
        }
        let mut out = vec![0.0_f32; m * n];
        // ikj loop order — better cache behaviour than ijk for row-major.
        for i in 0..m {
            for kk in 0..k {
                let a = self.data[i * k + kk];
                if a == 0.0 { continue; }
                let row_b = kk * n;
                let row_o = i * n;
                for j in 0..n {
                    out[row_o + j] += a * other.data[row_b + j];
                }
            }
        }
        Ok(Tensor { shape: vec![m, n], data: out })
    }

    /// Sum a (rows x cols) tensor down rows -> shape (1, cols).
    pub fn sum_rows(&self) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 {
            return Err(TensorError::NotMatrix(self.shape.clone()));
        }
        let (r, c) = (self.shape[0], self.shape[1]);
        let mut out = vec![0.0_f32; c];
        for i in 0..r {
            for j in 0..c {
                out[j] += self.data[i * c + j];
            }
        }
        Ok(Tensor { shape: vec![1, c], data: out })
    }

    /// Broadcast a (1, cols) bias vector across `rows` rows.
    pub fn broadcast_rows(&self, rows: usize) -> Result<Tensor, TensorError> {
        if self.shape.len() != 2 || self.shape[0] != 1 {
            return Err(TensorError::NotMatrix(self.shape.clone()));
        }
        let c = self.shape[1];
        let mut out = Vec::with_capacity(rows * c);
        for _ in 0..rows {
            out.extend_from_slice(&self.data);
        }
        Ok(Tensor { shape: vec![rows, c], data: out })
    }
}

// -------------------------------------------------------------------------------------
// Tiny deterministic PRNG (SplitMix64 -> f32 in [0,1)). Pure self-contained Rust.
// We don't pull in `rand` because the engine must be dependency-free for math/init.
// -------------------------------------------------------------------------------------

#[derive(Clone, Debug)]
pub struct SplitMix64 { state: u64 }

impl SplitMix64 {
    pub fn new(seed: u64) -> Self { Self { state: seed.wrapping_add(0x9E37_79B9_7F4A_7C15) } }

    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    pub fn next_f32(&mut self) -> f32 {
        // 24 high bits -> [0, 1).
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }

    /// Box–Muller standard normal sample.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.next_f32().clamp(1e-7, 1.0 - 1e-7);
        let u2 = self.next_f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matmul_basic() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let b = Tensor::new(vec![3, 2], vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]);
        let c = a.matmul(&b).unwrap();
        assert_eq!(c.shape, vec![2, 2]);
        // [[58,64],[139,154]]
        assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
    }

    #[test]
    fn transpose_roundtrip() {
        let a = Tensor::new(vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let t = a.transpose().unwrap();
        assert_eq!(t.shape, vec![3, 2]);
        assert_eq!(t.transpose().unwrap(), a);
    }

    #[test]
    fn elementwise_ops() {
        let a = Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]);
        let b = Tensor::ones(vec![2, 2]);
        assert_eq!(a.add(&b).unwrap().data, vec![2.0, 3.0, 4.0, 5.0]);
        assert_eq!(a.sub(&b).unwrap().data, vec![0.0, 1.0, 2.0, 3.0]);
        assert_eq!(a.mul_scalar(2.0).data, vec![2.0, 4.0, 6.0, 8.0]);
    }

    #[test]
    fn broadcast_and_sum_rows() {
        let bias = Tensor::new(vec![1, 3], vec![10.0, 20.0, 30.0]);
        let b = bias.broadcast_rows(2).unwrap();
        assert_eq!(b.shape, vec![2, 3]);
        assert_eq!(b.sum_rows().unwrap().data, vec![20.0, 40.0, 60.0]);
    }
}
