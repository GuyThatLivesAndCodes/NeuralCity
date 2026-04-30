//! egui application — the workbench shell.

use crate::plot::{scatter_2d, LinePlot};
use crate::trainer::{self, TrainerHandle, TrainingConfig, TrainingState};
use eframe::CreationContext;
use egui::{Color32, RichText};
use neuralcabin_engine::data::{self, Dataset, TaskKind};
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::OptimizerKind;
use neuralcabin_engine::persistence::{self, ModelFile};
use neuralcabin_engine::tensor::Tensor;
use neuralcabin_engine::{Activation, Loss};
use std::sync::Arc;
use std::time::Duration;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum DatasetChoice { Xor, Spirals, Sine, Csv }

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
enum OptChoice { Sgd, Adam }

pub struct NeuralCabinApp {
    // Architecture
    input_dim: usize,
    layer_specs: Vec<LayerSpec>,
    pending_linear_units: usize,
    pending_activation: Activation,
    seed: u64,
    model: Option<Model>,
    build_message: Option<String>,

    // Dataset
    dataset_choice: DatasetChoice,
    spirals_classes: usize,
    spirals_per_class: usize,
    sine_n: usize,
    sine_noise: f32,
    csv_path: String,
    csv_has_header: bool,
    csv_num_classes: String,
    dataset: Option<Arc<Dataset>>,
    dataset_message: Option<String>,

    // Training
    loss_choice: Loss,
    opt_choice: OptChoice,
    learning_rate: f32,
    momentum: f32,
    beta1: f32,
    beta2: f32,
    epsilon: f32,
    epochs: usize,
    batch_size: usize,
    val_frac: f32,
    trainer: Option<TrainerHandle>,
    last_state: TrainingState,
    use_log_y: bool,

    // Inference
    inference_inputs: Vec<f32>,
    inference_output: Option<Tensor>,

    // Persistence
    model_path: String,
    persistence_message: Option<String>,
}

impl NeuralCabinApp {
    pub fn new(_cc: &CreationContext<'_>) -> Self {
        let mut app = Self {
            input_dim: 2,
            layer_specs: vec![
                LayerSpec::Linear { in_dim: 2, out_dim: 16 },
                LayerSpec::Activation(Activation::Tanh),
                LayerSpec::Linear { in_dim: 16, out_dim: 1 },
                LayerSpec::Activation(Activation::Sigmoid),
            ],
            pending_linear_units: 8,
            pending_activation: Activation::ReLU,
            seed: 0x5EED,
            model: None,
            build_message: None,

            dataset_choice: DatasetChoice::Xor,
            spirals_classes: 3,
            spirals_per_class: 100,
            sine_n: 200,
            sine_noise: 0.05,
            csv_path: String::new(),
            csv_has_header: true,
            csv_num_classes: String::new(),
            dataset: None,
            dataset_message: None,

            loss_choice: Loss::MeanSquaredError,
            opt_choice: OptChoice::Adam,
            learning_rate: 0.05,
            momentum: 0.9,
            beta1: 0.9,
            beta2: 0.999,
            epsilon: 1e-8,
            epochs: 1000,
            batch_size: 32,
            val_frac: 0.2,
            trainer: None,
            last_state: TrainingState::default(),
            use_log_y: true,

            inference_inputs: vec![0.0, 0.0],
            inference_output: None,

            model_path: "model.json".into(),
            persistence_message: None,
        };
        app.load_dataset();
        app.build_model();
        app
    }

    fn current_optimizer(&self) -> OptimizerKind {
        match self.opt_choice {
            OptChoice::Sgd => OptimizerKind::Sgd { lr: self.learning_rate, momentum: self.momentum },
            OptChoice::Adam => OptimizerKind::Adam {
                lr: self.learning_rate,
                beta1: self.beta1,
                beta2: self.beta2,
                eps: self.epsilon,
            },
        }
    }

    fn build_model(&mut self) {
        // Validate the layer chain dimensions.
        let mut cur = self.input_dim;
        for (i, spec) in self.layer_specs.iter().enumerate() {
            if let LayerSpec::Linear { in_dim, out_dim: _ } = spec {
                if *in_dim != cur {
                    self.build_message = Some(format!(
                        "Layer {} expects in_dim={in_dim}, but running dim is {cur}. \
                         Fix dimensions and rebuild.", i + 1
                    ));
                    self.model = None;
                    return;
                }
            }
            if let LayerSpec::Linear { out_dim, .. } = spec { cur = *out_dim; }
        }
        let model = Model::from_specs(self.input_dim, &self.layer_specs, self.seed);
        self.inference_inputs = vec![0.0; self.input_dim];
        self.build_message = Some(format!(
            "Model built: {} layers, {} parameters, output dim {}.",
            model.layers.len(),
            model.parameter_count(),
            model.output_dim()
        ));
        self.model = Some(model);
    }

    fn load_dataset(&mut self) {
        let result: Result<Dataset, String> = (|| match self.dataset_choice {
            DatasetChoice::Xor => Ok(data::xor()),
            DatasetChoice::Spirals => Ok(data::spirals(self.spirals_per_class, self.spirals_classes, self.seed)),
            DatasetChoice::Sine => Ok(data::sine(self.sine_n, self.sine_noise, self.seed)),
            DatasetChoice::Csv => {
                if self.csv_path.trim().is_empty() {
                    return Err("Provide a CSV file path.".into());
                }
                let nc = self.csv_num_classes.trim();
                let num_classes = if nc.is_empty() {
                    None
                } else {
                    Some(nc.parse::<usize>().map_err(|e| format!("num_classes: {e}"))?)
                };
                data::load_csv(&self.csv_path, self.csv_has_header, num_classes)
                    .map_err(|e| format!("CSV load failed: {e}"))
            }
        })();
        match result {
            Ok(ds) => {
                self.dataset_message = Some(format!(
                    "Loaded {} samples, {} features, {} outputs ({}).",
                    ds.n(), ds.n_features(), ds.n_outputs(),
                    match ds.task {
                        TaskKind::Regression => "regression".into(),
                        TaskKind::Classification { num_classes } => format!("classification, {num_classes} classes"),
                    }
                ));
                self.dataset = Some(Arc::new(ds));
            }
            Err(e) => {
                self.dataset_message = Some(format!("Error: {e}"));
                self.dataset = None;
            }
        }
    }

    fn start_training(&mut self) {
        let Some(model) = self.model.clone() else {
            self.build_message = Some("Build a model first.".into());
            return;
        };
        let Some(dataset) = self.dataset.clone() else {
            self.dataset_message = Some("Load a dataset first.".into());
            return;
        };
        if model.input_dim != dataset.n_features() {
            self.build_message = Some(format!(
                "Model input_dim={} but dataset has {} features.",
                model.input_dim, dataset.n_features()
            ));
            return;
        }
        if model.output_dim() != dataset.n_outputs() {
            self.build_message = Some(format!(
                "Model output_dim={} but dataset has {} output dims.",
                model.output_dim(), dataset.n_outputs()
            ));
            return;
        }
        let cfg = TrainingConfig {
            epochs: self.epochs,
            batch_size: self.batch_size,
            optimizer: self.current_optimizer(),
            loss: self.loss_choice,
            validation_frac: self.val_frac.clamp(0.0, 0.9),
            seed: self.seed,
        };
        self.last_state = TrainingState { running: true, total_epochs: self.epochs, ..Default::default() };
        self.trainer = Some(trainer::spawn(model, dataset, cfg));
    }

    fn poll_trainer(&mut self) {
        if let Some(handle) = &self.trainer {
            if let Ok(s) = handle.state.lock() { self.last_state = s.clone(); }
            if handle.is_finished() {
                let h = self.trainer.take().unwrap();
                if let Some(trained) = h.join() {
                    self.model = Some(trained);
                    self.build_message = Some("Training finished — model updated.".into());
                }
            }
        }
    }

    fn predict(&mut self) {
        let Some(model) = &self.model else { return; };
        if model.input_dim != self.inference_inputs.len() {
            self.inference_inputs.resize(model.input_dim, 0.0);
        }
        let input = Tensor::new(vec![1, model.input_dim], self.inference_inputs.clone());
        let mut out = model.predict(&input);
        if self.loss_choice == Loss::CrossEntropy {
            // For CE-trained models the raw output is logits — show probabilities.
            out = neuralcabin_engine::activations::softmax_rows(&out);
        }
        self.inference_output = Some(out);
    }

    fn save_model(&mut self) {
        let Some(model) = &self.model else {
            self.persistence_message = Some("No model to save.".into()); return;
        };
        let file = ModelFile::wrap(model.clone(), Some(self.loss_choice), Some(self.current_optimizer()));
        match persistence::save(&self.model_path, &file) {
            Ok(()) => self.persistence_message = Some(format!("Saved -> {}", self.model_path)),
            Err(e) => self.persistence_message = Some(format!("Save failed: {e}")),
        }
    }

    fn load_model(&mut self) {
        match persistence::load(&self.model_path) {
            Ok(f) => {
                self.input_dim = f.model.input_dim;
                self.layer_specs = f.model.layers.iter().map(|l| match l {
                    neuralcabin_engine::nn::Layer::Linear(ll) =>
                        LayerSpec::Linear { in_dim: ll.in_dim, out_dim: ll.out_dim },
                    neuralcabin_engine::nn::Layer::Activation(a) => LayerSpec::Activation(*a),
                }).collect();
                if let Some(l) = f.loss { self.loss_choice = l; }
                if let Some(o) = f.optimizer { self.set_optimizer(o); }
                self.inference_inputs = vec![0.0; f.model.input_dim];
                self.model = Some(f.model);
                self.persistence_message = Some(format!("Loaded <- {}", self.model_path));
            }
            Err(e) => self.persistence_message = Some(format!("Load failed: {e}")),
        }
    }

    fn set_optimizer(&mut self, kind: OptimizerKind) {
        match kind {
            OptimizerKind::Sgd { lr, momentum } => {
                self.opt_choice = OptChoice::Sgd;
                self.learning_rate = lr;
                self.momentum = momentum;
            }
            OptimizerKind::Adam { lr, beta1, beta2, eps } => {
                self.opt_choice = OptChoice::Adam;
                self.learning_rate = lr;
                self.beta1 = beta1;
                self.beta2 = beta2;
                self.epsilon = eps;
            }
        }
    }
}

impl eframe::App for NeuralCabinApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // Keep the UI responsive while training.
        if self.trainer.is_some() {
            ctx.request_repaint_after(Duration::from_millis(75));
        }
        self.poll_trainer();

        egui::TopBottomPanel::top("top").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.heading("NeuralCabin");
                ui.label(RichText::new("pure-Rust neural network workbench").italics().weak());
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.label(format!("v{}", env!("CARGO_PKG_VERSION")));
                });
            });
        });

        egui::SidePanel::left("architecture").default_width(320.0).show(ctx, |ui| {
            self.architecture_panel(ui);
        });
        egui::SidePanel::right("inference").default_width(320.0).show(ctx, |ui| {
            self.inference_panel(ui);
        });
        egui::CentralPanel::default().show(ctx, |ui| {
            self.training_panel(ui);
        });
    }
}

impl NeuralCabinApp {
    fn architecture_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Architecture");
        ui.separator();

        ui.horizontal(|ui| {
            ui.label("Input dim:");
            let mut dim = self.input_dim as i32;
            if ui.add(egui::DragValue::new(&mut dim).range(1..=4096)).changed() {
                self.input_dim = dim.max(1) as usize;
                if let Some(LayerSpec::Linear { in_dim, .. }) = self.layer_specs.first_mut() {
                    *in_dim = self.input_dim;
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Init seed:");
            let mut s = self.seed as i64;
            if ui.add(egui::DragValue::new(&mut s)).changed() { self.seed = s as u64; }
        });

        ui.add_space(6.0);
        ui.label(RichText::new("Layers (top → bottom):").strong());
        let mut to_remove: Option<usize> = None;
        let mut current_dim = self.input_dim;
        for (i, spec) in self.layer_specs.iter_mut().enumerate() {
            ui.group(|ui| {
                ui.horizontal(|ui| {
                    ui.label(format!("{}.", i + 1));
                    match spec {
                        LayerSpec::Linear { in_dim, out_dim } => {
                            *in_dim = current_dim;
                            ui.label("Linear");
                            ui.label(format!("in={in_dim}"));
                            ui.label("→ out=");
                            let mut o = *out_dim as i32;
                            if ui.add(egui::DragValue::new(&mut o).range(1..=4096)).changed() {
                                *out_dim = o.max(1) as usize;
                            }
                            current_dim = *out_dim;
                        }
                        LayerSpec::Activation(a) => {
                            ui.label("Activation:");
                            egui::ComboBox::from_id_salt(("act", i))
                                .selected_text(a.name())
                                .show_ui(ui, |ui| {
                                    for opt in Activation::all() {
                                        ui.selectable_value(a, *opt, opt.name());
                                    }
                                });
                        }
                    }
                    if ui.small_button("✕").clicked() { to_remove = Some(i); }
                });
            });
        }
        if let Some(i) = to_remove { self.layer_specs.remove(i); }

        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("+ Linear").clicked() {
                let in_dim = current_dim;
                self.layer_specs.push(LayerSpec::Linear { in_dim, out_dim: self.pending_linear_units });
            }
            ui.label("units:");
            let mut u = self.pending_linear_units as i32;
            if ui.add(egui::DragValue::new(&mut u).range(1..=4096)).changed() {
                self.pending_linear_units = u.max(1) as usize;
            }
        });
        ui.horizontal(|ui| {
            if ui.button("+ Activation").clicked() {
                self.layer_specs.push(LayerSpec::Activation(self.pending_activation));
            }
            egui::ComboBox::from_id_salt("pending_act")
                .selected_text(self.pending_activation.name())
                .show_ui(ui, |ui| {
                    for opt in Activation::all() {
                        ui.selectable_value(&mut self.pending_activation, *opt, opt.name());
                    }
                });
        });

        ui.add_space(8.0);
        if ui.add_sized([ui.available_width(), 28.0], egui::Button::new("Build / Reset Model")).clicked() {
            self.build_model();
        }
        if let Some(msg) = &self.build_message {
            let color = if msg.starts_with("Layer ") || msg.starts_with("Model input")
                || msg.starts_with("Model output") { Color32::LIGHT_RED } else { Color32::LIGHT_GREEN };
            ui.colored_label(color, msg);
        }
        if let Some(model) = &self.model {
            ui.separator();
            ui.label(RichText::new("Compiled model").strong());
            for (i, l) in model.layers.iter().enumerate() {
                ui.label(format!("  {}. {}", i + 1, l.describe()));
            }
            ui.label(format!("Parameters: {}", model.parameter_count()));
        }

        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("Persistence").strong());
        ui.horizontal(|ui| {
            ui.label("Path:");
            ui.add(egui::TextEdit::singleline(&mut self.model_path).desired_width(200.0));
        });
        ui.horizontal(|ui| {
            if ui.button("Save").clicked() { self.save_model(); }
            if ui.button("Load").clicked() { self.load_model(); }
        });
        if let Some(m) = &self.persistence_message { ui.label(m); }
    }

    fn training_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Training");
        ui.separator();
        let training = self.trainer.is_some();

        ui.horizontal_wrapped(|ui| {
            ui.label("Dataset:");
            for (label, choice) in [
                ("XOR", DatasetChoice::Xor),
                ("Spirals", DatasetChoice::Spirals),
                ("Sine", DatasetChoice::Sine),
                ("CSV", DatasetChoice::Csv),
            ] {
                if ui.selectable_label(self.dataset_choice == choice, label).clicked() {
                    self.dataset_choice = choice;
                }
            }
            if ui.button("Reload").clicked() { self.load_dataset(); }
        });

        match self.dataset_choice {
            DatasetChoice::Spirals => {
                ui.horizontal(|ui| {
                    ui.label("Classes:");
                    let mut c = self.spirals_classes as i32;
                    if ui.add(egui::DragValue::new(&mut c).range(2..=10)).changed() {
                        self.spirals_classes = c.max(2) as usize;
                    }
                    ui.label("per class:");
                    let mut p = self.spirals_per_class as i32;
                    if ui.add(egui::DragValue::new(&mut p).range(10..=2000)).changed() {
                        self.spirals_per_class = p.max(10) as usize;
                    }
                });
            }
            DatasetChoice::Sine => {
                ui.horizontal(|ui| {
                    ui.label("N:");
                    let mut n = self.sine_n as i32;
                    if ui.add(egui::DragValue::new(&mut n).range(10..=10_000)).changed() {
                        self.sine_n = n.max(10) as usize;
                    }
                    ui.label("noise σ:");
                    ui.add(egui::DragValue::new(&mut self.sine_noise).speed(0.01).range(0.0..=2.0));
                });
            }
            DatasetChoice::Csv => {
                ui.horizontal(|ui| {
                    ui.label("Path:");
                    ui.add(egui::TextEdit::singleline(&mut self.csv_path).desired_width(360.0));
                });
                ui.horizontal(|ui| {
                    ui.checkbox(&mut self.csv_has_header, "Has header");
                    ui.label("num_classes (blank = regression):");
                    ui.add(egui::TextEdit::singleline(&mut self.csv_num_classes).desired_width(60.0));
                });
            }
            _ => {}
        }
        if let Some(m) = &self.dataset_message { ui.label(RichText::new(m).weak()); }

        ui.separator();
        ui.horizontal_wrapped(|ui| {
            ui.label("Loss:");
            for l in Loss::all() {
                if ui.selectable_label(self.loss_choice == *l, l.name()).clicked() {
                    self.loss_choice = *l;
                }
            }
            ui.add_space(20.0);
            ui.label("Optimizer:");
            ui.selectable_value(&mut self.opt_choice, OptChoice::Sgd, "SGD");
            ui.selectable_value(&mut self.opt_choice, OptChoice::Adam, "Adam");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Learning rate:");
            if ui.add(egui::DragValue::new(&mut self.learning_rate).speed(0.001).range(1e-6..=10.0)).changed() {
                if let Some(h) = &self.trainer { h.set_lr(self.learning_rate); }
            }
            match self.opt_choice {
                OptChoice::Sgd => {
                    ui.label("momentum:");
                    ui.add(egui::DragValue::new(&mut self.momentum).speed(0.01).range(0.0..=0.999));
                }
                OptChoice::Adam => {
                    ui.label("β₁:");
                    ui.add(egui::DragValue::new(&mut self.beta1).speed(0.001).range(0.0..=0.999));
                    ui.label("β₂:");
                    ui.add(egui::DragValue::new(&mut self.beta2).speed(0.0001).range(0.0..=0.99999));
                }
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Epochs:");
            let mut e = self.epochs as i32;
            if ui.add(egui::DragValue::new(&mut e).range(1..=1_000_000)).changed() {
                self.epochs = e.max(1) as usize;
            }
            ui.label("Batch size:");
            let mut b = self.batch_size as i32;
            if ui.add(egui::DragValue::new(&mut b).range(1..=4096)).changed() {
                self.batch_size = b.max(1) as usize;
            }
            ui.label("Validation frac:");
            ui.add(egui::DragValue::new(&mut self.val_frac).speed(0.01).range(0.0..=0.9));
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if !training {
                if ui.add(egui::Button::new(RichText::new("▶ Train").strong())).clicked() {
                    self.start_training();
                }
            } else {
                let paused = self.last_state.paused;
                if ui.button(if paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    if let Some(h) = &self.trainer { h.pause(!paused); }
                    self.last_state.paused = !paused;
                }
                if ui.button("⏹ Stop").clicked() {
                    if let Some(h) = &self.trainer { h.stop(); }
                }
            }
            ui.checkbox(&mut self.use_log_y, "log-scale loss");
        });

        ui.separator();
        let s = &self.last_state;
        ui.horizontal_wrapped(|ui| {
            ui.label(format!("epoch {}/{}", s.epoch, s.total_epochs));
            ui.separator();
            ui.label(format!("loss = {:.5}", s.last_loss));
            if let Some(v) = s.last_val_loss {
                ui.separator();
                ui.label(format!("val loss = {:.5}", v));
            }
            if let Some(a) = s.last_accuracy {
                ui.separator();
                ui.label(format!("accuracy = {:.2}%", a * 100.0));
            }
            ui.separator();
            ui.label(format!("elapsed = {:.1}s", s.elapsed_secs));
        });
        if let Some(e) = &s.error {
            ui.colored_label(Color32::LIGHT_RED, e);
        }

        ui.add_space(8.0);
        let series_owned: Vec<(String, Vec<f32>, Color32)> = vec![
            ("train".into(), s.loss_history.clone(), Color32::from_rgb(52, 152, 219)),
            ("val".into(), s.val_loss_history.clone(), Color32::from_rgb(231, 76, 60)),
        ];
        let series_ref: Vec<(&str, &[f32], Color32)> = series_owned.iter()
            .map(|(n, v, c)| (n.as_str(), v.as_slice(), *c))
            .collect();
        LinePlot {
            title: "Loss",
            series: series_ref,
            log_y: self.use_log_y,
            min_height: 200.0,
        }.show(ui);

        if let Some(a) = s.accuracy_history.last().copied() {
            ui.add_space(4.0);
            let acc = [("accuracy".to_string(), s.accuracy_history.clone(),
                Color32::from_rgb(46, 204, 113))];
            let acc_ref: Vec<(&str, &[f32], Color32)> = acc.iter()
                .map(|(n, v, c)| (n.as_str(), v.as_slice(), *c)).collect();
            LinePlot { title: "Validation accuracy", series: acc_ref, log_y: false, min_height: 140.0 }.show(ui);
            let _ = a;
        }

        // Optional 2-D scatter when the dataset is 2-D classification.
        if let Some(ds) = &self.dataset {
            if ds.n_features() == 2 {
                if let TaskKind::Classification { num_classes } = ds.task {
                    ui.add_space(8.0);
                    ui.label(RichText::new("Dataset (2-D scatter)").strong());
                    let mut points = Vec::with_capacity(ds.n());
                    for i in 0..ds.n() {
                        let x = ds.features.data[i * 2];
                        let y = ds.features.data[i * 2 + 1];
                        let row = &ds.labels.data[i * num_classes..(i + 1) * num_classes];
                        let mut best = 0;
                        let mut bv = f32::NEG_INFINITY;
                        for (k, v) in row.iter().enumerate() { if *v > bv { bv = *v; best = k; } }
                        points.push((x, y, best));
                    }
                    scatter_2d(ui, &points, 220.0);
                }
            }
        }
    }

    fn inference_panel(&mut self, ui: &mut egui::Ui) {
        ui.heading("Inference");
        ui.separator();
        let Some(model) = &self.model else {
            ui.label("(build a model first)");
            return;
        };
        if self.inference_inputs.len() != model.input_dim {
            self.inference_inputs.resize(model.input_dim, 0.0);
        }
        ui.label(format!("Inputs ({}):", model.input_dim));
        egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
            for (i, v) in self.inference_inputs.iter_mut().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(format!("x{i}:"));
                    ui.add(egui::DragValue::new(v).speed(0.05));
                });
            }
        });
        ui.add_space(6.0);
        if ui.add_sized([ui.available_width(), 28.0], egui::Button::new("Predict")).clicked() {
            self.predict();
        }
        ui.add_space(8.0);
        if let Some(out) = &self.inference_output {
            ui.label(RichText::new("Output").strong());
            for (i, v) in out.data.iter().enumerate() {
                ui.label(format!("y{i}: {v:.6}"));
            }
            // For classification, show argmax.
            if out.data.len() > 1 {
                let mut best = 0; let mut bv = f32::NEG_INFINITY;
                for (i, v) in out.data.iter().enumerate() { if *v > bv { bv = *v; best = i; } }
                ui.add_space(4.0);
                ui.label(RichText::new(format!("argmax = class {best}  (p = {bv:.3})")).strong());
            }
        }

        ui.add_space(12.0);
        ui.separator();
        ui.label(RichText::new("Tips").strong());
        ui.label(RichText::new(
            "• Use Tanh + Sigmoid + MSE for tiny binary tasks like XOR.\n\
             • For classification (Spirals, MNIST-like), end with Linear → Identity \
               and choose Loss = CrossEntropy. Softmax is applied internally for \
               numerical stability and reported here as probabilities."
        ).weak());
    }
}
