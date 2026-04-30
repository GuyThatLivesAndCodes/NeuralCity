//! Background training thread.
//!
//! The UI keeps a [`TrainerHandle`]. The trainer thread owns the model and
//! optimiser, runs the training loop, and pushes per-epoch metrics into the
//! shared [`TrainingState`]. The UI samples that state every frame.

use neuralcabin_engine::data::{shuffle_indices, Dataset, TaskKind};
use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
use neuralcabin_engine::tensor::SplitMix64;
use neuralcabin_engine::{Loss, Model};
use std::sync::mpsc::{self, Sender, TryRecvError};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct TrainingConfig {
    pub epochs: usize,
    pub batch_size: usize,
    pub optimizer: OptimizerKind,
    pub loss: Loss,
    pub validation_frac: f32,
    pub seed: u64,
}

#[derive(Default, Debug, Clone)]
pub struct TrainingState {
    pub running: bool,
    pub paused: bool,
    pub stopped: bool,
    pub epoch: usize,
    pub total_epochs: usize,
    pub last_loss: f32,
    pub last_val_loss: Option<f32>,
    pub last_accuracy: Option<f32>,
    pub loss_history: Vec<f32>,
    pub val_loss_history: Vec<f32>,
    pub accuracy_history: Vec<f32>,
    pub error: Option<String>,
    pub elapsed_secs: f32,
}

#[derive(Debug)]
pub enum TrainerCmd {
    Stop,
    Pause(bool),
    SetLr(f32),
}

pub struct TrainerHandle {
    pub state: Arc<Mutex<TrainingState>>,
    pub cmd_tx: Sender<TrainerCmd>,
    join: Option<JoinHandle<Model>>,
}

impl TrainerHandle {
    pub fn pause(&self, paused: bool) { let _ = self.cmd_tx.send(TrainerCmd::Pause(paused)); }
    pub fn stop(&self)              { let _ = self.cmd_tx.send(TrainerCmd::Stop); }
    pub fn set_lr(&self, lr: f32)   { let _ = self.cmd_tx.send(TrainerCmd::SetLr(lr)); }

    pub fn is_finished(&self) -> bool {
        self.join.as_ref().is_none_or(|j| j.is_finished())
    }

    /// Block until training thread finishes and recover the trained model.
    pub fn join(mut self) -> Option<Model> {
        self.join.take().and_then(|j| j.join().ok())
    }
}

pub fn spawn(model: Model, dataset: Arc<Dataset>, cfg: TrainingConfig) -> TrainerHandle {
    let state = Arc::new(Mutex::new(TrainingState {
        running: true,
        total_epochs: cfg.epochs,
        ..Default::default()
    }));
    let (cmd_tx, cmd_rx) = mpsc::channel();
    let state_t = Arc::clone(&state);
    let join = thread::Builder::new()
        .name("neuralcabin-trainer".into())
        .spawn(move || run_loop(model, dataset, cfg, state_t, cmd_rx))
        .expect("spawn trainer");
    TrainerHandle { state, cmd_tx, join: Some(join) }
}

fn run_loop(
    mut model: Model,
    dataset: Arc<Dataset>,
    mut cfg: TrainingConfig,
    state: Arc<Mutex<TrainingState>>,
    cmd_rx: mpsc::Receiver<TrainerCmd>,
) -> Model {
    let mut rng = SplitMix64::new(cfg.seed);
    let (train, val) = dataset.train_test_split(cfg.validation_frac);
    let n_train = train.n();
    let mut optimizer = Optimizer::new(cfg.optimizer, &model.parameter_shapes());
    let mut indices: Vec<usize> = (0..n_train).collect();
    let started = Instant::now();
    let mut paused = false;
    let do_classification = matches!(dataset.task, TaskKind::Classification { .. });

    'outer: for epoch in 0..cfg.epochs {
        // Drain commands.
        loop {
            match cmd_rx.try_recv() {
                Ok(TrainerCmd::Stop) => break 'outer,
                Ok(TrainerCmd::Pause(p)) => paused = p,
                Ok(TrainerCmd::SetLr(lr)) => {
                    optimizer.kind.set_lr(lr);
                    cfg.optimizer.set_lr(lr);
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => break 'outer,
            }
        }
        // Honour pause without busy-spinning.
        while paused {
            thread::sleep(Duration::from_millis(50));
            match cmd_rx.try_recv() {
                Ok(TrainerCmd::Stop) => break 'outer,
                Ok(TrainerCmd::Pause(p)) => paused = p,
                Ok(TrainerCmd::SetLr(lr)) => {
                    optimizer.kind.set_lr(lr);
                    cfg.optimizer.set_lr(lr);
                }
                Err(TryRecvError::Empty) => continue,
                Err(TryRecvError::Disconnected) => break 'outer,
            }
        }

        shuffle_indices(&mut indices, &mut rng);
        let mut sum_loss = 0.0_f32;
        let mut n_batches = 0_usize;
        let bs = cfg.batch_size.max(1).min(n_train.max(1));
        for chunk in indices.chunks(bs) {
            let (xb, yb) = train.batch(chunk);
            let l = model.train_step(&mut optimizer, cfg.loss, &xb, &yb);
            if !l.is_finite() {
                let mut s = state.lock().unwrap();
                s.error = Some(format!("loss diverged to {l} at epoch {epoch}"));
                s.running = false;
                s.stopped = true;
                return model;
            }
            sum_loss += l;
            n_batches += 1;
        }
        let train_loss = sum_loss / n_batches.max(1) as f32;
        let val_loss = if val.n() > 0 {
            Some(model.evaluate_loss(cfg.loss, &val.features, &val.labels))
        } else {
            None
        };
        let accuracy = if do_classification && val.n() > 0 {
            Some(model.accuracy(&val.features, &val.labels))
        } else if do_classification {
            Some(model.accuracy(&train.features, &train.labels))
        } else {
            None
        };
        {
            let mut s = state.lock().unwrap();
            s.epoch = epoch + 1;
            s.last_loss = train_loss;
            s.last_val_loss = val_loss;
            s.last_accuracy = accuracy;
            s.loss_history.push(train_loss);
            if let Some(v) = val_loss { s.val_loss_history.push(v); }
            if let Some(a) = accuracy { s.accuracy_history.push(a); }
            s.elapsed_secs = started.elapsed().as_secs_f32();
        }
    }

    let mut s = state.lock().unwrap();
    s.running = false;
    s.stopped = true;
    s.elapsed_secs = started.elapsed().as_secs_f32();
    drop(s);
    model
}
