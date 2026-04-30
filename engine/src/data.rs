//! Tiny dataset utilities. Includes a CSV loader (no external CSV crate),
//! one-hot encoding, batched iteration, and a couple of canned datasets used
//! by the UI as default examples.

use crate::tensor::{SplitMix64, Tensor};
use std::fs;
use std::io;
use std::path::Path;

#[derive(Clone, Debug)]
pub struct Dataset {
    pub feature_names: Vec<String>,
    pub label_names: Vec<String>,
    pub features: Tensor,    // (n_samples, n_features)
    pub labels: Tensor,      // (n_samples, n_label_dims)
    pub task: TaskKind,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum TaskKind {
    Regression,
    Classification { num_classes: usize },
}

impl Dataset {
    pub fn n(&self) -> usize { self.features.rows() }
    pub fn n_features(&self) -> usize { self.features.cols() }
    pub fn n_outputs(&self) -> usize { self.labels.cols() }

    /// Select rows by index — returns owned `(features, labels)` tensors.
    pub fn batch(&self, indices: &[usize]) -> (Tensor, Tensor) {
        let f_cols = self.features.cols();
        let l_cols = self.labels.cols();
        let mut f = Vec::with_capacity(indices.len() * f_cols);
        let mut l = Vec::with_capacity(indices.len() * l_cols);
        for &i in indices {
            f.extend_from_slice(&self.features.data[i * f_cols..(i + 1) * f_cols]);
            l.extend_from_slice(&self.labels.data[i * l_cols..(i + 1) * l_cols]);
        }
        (
            Tensor::new(vec![indices.len(), f_cols], f),
            Tensor::new(vec![indices.len(), l_cols], l),
        )
    }

    /// Split off the last `frac` of rows as a held-out set.
    pub fn train_test_split(&self, test_frac: f32) -> (Dataset, Dataset) {
        let n = self.n();
        let n_test = ((n as f32) * test_frac).round() as usize;
        let n_train = n - n_test;
        let f_cols = self.features.cols();
        let l_cols = self.labels.cols();
        let train_f = Tensor::new(
            vec![n_train, f_cols],
            self.features.data[..n_train * f_cols].to_vec(),
        );
        let train_l = Tensor::new(
            vec![n_train, l_cols],
            self.labels.data[..n_train * l_cols].to_vec(),
        );
        let test_f = Tensor::new(
            vec![n_test, f_cols],
            self.features.data[n_train * f_cols..].to_vec(),
        );
        let test_l = Tensor::new(
            vec![n_test, l_cols],
            self.labels.data[n_train * l_cols..].to_vec(),
        );
        (
            Dataset {
                feature_names: self.feature_names.clone(),
                label_names: self.label_names.clone(),
                features: train_f,
                labels: train_l,
                task: self.task,
            },
            Dataset {
                feature_names: self.feature_names.clone(),
                label_names: self.label_names.clone(),
                features: test_f,
                labels: test_l,
                task: self.task,
            },
        )
    }
}

/// Tiny CSV loader: comma-separated, optional header row. The last column is
/// treated as the label by default. For classification you may pass
/// `num_classes = Some(k)` to one-hot encode the integer label column.
pub fn load_csv<P: AsRef<Path>>(
    path: P,
    has_header: bool,
    num_classes: Option<usize>,
) -> io::Result<Dataset> {
    let raw = fs::read_to_string(path)?;
    parse_csv(&raw, has_header, num_classes)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

pub fn parse_csv(
    raw: &str,
    has_header: bool,
    num_classes: Option<usize>,
) -> Result<Dataset, String> {
    let mut lines = raw.lines().filter(|l| !l.trim().is_empty());

    let mut feature_names: Vec<String> = Vec::new();
    let mut label_name = "target".to_string();
    if has_header {
        let header = lines.next().ok_or("empty file")?;
        let cols: Vec<&str> = header.split(',').map(|s| s.trim()).collect();
        if cols.len() < 2 { return Err("need at least 2 columns".into()); }
        for c in &cols[..cols.len() - 1] { feature_names.push(c.to_string()); }
        label_name = cols.last().unwrap().to_string();
    }

    let mut rows_f: Vec<Vec<f32>> = Vec::new();
    let mut rows_l: Vec<f32> = Vec::new();
    for (li, line) in lines.enumerate() {
        let cols: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
        if cols.len() < 2 {
            return Err(format!("row {} has only {} columns", li, cols.len()));
        }
        let mut feats = Vec::with_capacity(cols.len() - 1);
        for c in &cols[..cols.len() - 1] {
            feats.push(c.parse::<f32>().map_err(|e| format!("row {li}: {e}"))?);
        }
        rows_f.push(feats);
        rows_l.push(
            cols.last()
                .unwrap()
                .parse::<f32>()
                .map_err(|e| format!("row {li}: {e}"))?,
        );
    }
    if rows_f.is_empty() { return Err("no data rows".into()); }
    let n_features = rows_f[0].len();
    if !feature_names.iter().all(|s| !s.is_empty()) || feature_names.len() != n_features {
        feature_names = (0..n_features).map(|i| format!("x{i}")).collect();
    }

    let n = rows_f.len();
    let mut features = Vec::with_capacity(n * n_features);
    for r in &rows_f { features.extend_from_slice(r); }

    let (labels, label_names, task) = match num_classes {
        Some(k) => {
            let mut onehot = vec![0.0_f32; n * k];
            let mut names = Vec::with_capacity(k);
            for c in 0..k { names.push(format!("class_{c}")); }
            for (i, v) in rows_l.iter().enumerate() {
                let cls = *v as usize;
                if cls >= k {
                    return Err(format!("row {i} label {cls} out of range for {k} classes"));
                }
                onehot[i * k + cls] = 1.0;
            }
            (
                Tensor::new(vec![n, k], onehot),
                names,
                TaskKind::Classification { num_classes: k },
            )
        }
        None => (
            Tensor::new(vec![n, 1], rows_l),
            vec![label_name],
            TaskKind::Regression,
        ),
    };

    Ok(Dataset {
        feature_names,
        label_names,
        features: Tensor::new(vec![n, n_features], features),
        labels,
        task,
    })
}

/// Shuffle indices in-place using the given RNG (Fisher–Yates).
pub fn shuffle_indices(indices: &mut [usize], rng: &mut SplitMix64) {
    for i in (1..indices.len()).rev() {
        let j = (rng.next_u64() as usize) % (i + 1);
        indices.swap(i, j);
    }
}

// -------------------------------------------------------------------------------------
// Canned datasets shipped with the app — no internet or filesystem access required.
// -------------------------------------------------------------------------------------

/// Classic XOR: 2 inputs, 1 binary output.
pub fn xor() -> Dataset {
    Dataset {
        feature_names: vec!["a".into(), "b".into()],
        label_names: vec!["xor".into()],
        features: Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]),
        labels: Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]),
        task: TaskKind::Regression,
    }
}

/// 2-D spiral: 3 interleaved arms, 100 points per arm. A common toy
/// classification benchmark that requires non-linear capacity.
pub fn spirals(points_per_class: usize, classes: usize, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let n = points_per_class * classes;
    let mut feats = Vec::with_capacity(n * 2);
    let mut labels = vec![0.0_f32; n * classes];
    let mut feat_idx = 0;
    for c in 0..classes {
        for i in 0..points_per_class {
            let r = i as f32 / points_per_class as f32;
            let t = c as f32 * 4.0 + 4.0 * r + 0.2 * rng.next_normal();
            let x = r * t.sin();
            let y = r * t.cos();
            feats.push(x);
            feats.push(y);
            labels[feat_idx * classes + c] = 1.0;
            feat_idx += 1;
        }
    }
    let mut name_iter = (0..classes).map(|c| format!("class_{c}"));
    Dataset {
        feature_names: vec!["x".into(), "y".into()],
        label_names: name_iter.by_ref().collect(),
        features: Tensor::new(vec![n, 2], feats),
        labels: Tensor::new(vec![n, classes], labels),
        task: TaskKind::Classification { num_classes: classes },
    }
}

/// Noisy sine wave: 1-D regression target = sin(2πx) + N(0, σ²).
pub fn sine(n: usize, noise: f32, seed: u64) -> Dataset {
    let mut rng = SplitMix64::new(seed);
    let mut feats = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);
    for i in 0..n {
        let x = i as f32 / n as f32;
        feats.push(x);
        let y = (2.0 * std::f32::consts::PI * x).sin() + noise * rng.next_normal();
        labels.push(y);
    }
    Dataset {
        feature_names: vec!["x".into()],
        label_names: vec!["sin(2πx)+ε".into()],
        features: Tensor::new(vec![n, 1], feats),
        labels: Tensor::new(vec![n, 1], labels),
        task: TaskKind::Regression,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_csv_roundtrip() {
        let raw = "a,b,t\n1,2,0\n3,4,1\n5,6,0\n";
        let d = parse_csv(raw, true, None).unwrap();
        assert_eq!(d.n(), 3);
        assert_eq!(d.n_features(), 2);
        assert_eq!(d.feature_names, vec!["a", "b"]);
        assert_eq!(d.labels.data, vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn parse_csv_onehot() {
        let raw = "a,b,t\n0,0,0\n1,0,1\n0,1,2\n";
        let d = parse_csv(raw, true, Some(3)).unwrap();
        assert_eq!(d.labels.shape, vec![3, 3]);
        assert_eq!(&d.labels.data[..3], &[1.0, 0.0, 0.0]);
        assert_eq!(&d.labels.data[3..6], &[0.0, 1.0, 0.0]);
        assert_eq!(&d.labels.data[6..], &[0.0, 0.0, 1.0]);
    }
}
