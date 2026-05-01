//! Multiple-network store. The "active" network drives every other tab.

use crate::corpus::{Corpus, CorpusTemplate};
use crate::trainer::{TrainerHandle, TrainingState};
use crate::vocab::Vocab;
use neuralcabin_engine::nn::{LayerSpec, Model};
use neuralcabin_engine::optimizer::OptimizerKind;
use neuralcabin_engine::tensor::Tensor;
use neuralcabin_engine::{Activation, Loss};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum NetworkKind {
    Simplex,
    NextTokenGen,
    Plugin { plugin_id: String, type_name: String },
}

impl NetworkKind {
    pub fn label(&self) -> String {
        match self {
            NetworkKind::Simplex => "Simplex (MLP)".into(),
            NetworkKind::NextTokenGen => "Next-Token Generation".into(),
            NetworkKind::Plugin { plugin_id, type_name } => {
                format!("Plugin · {plugin_id} / {type_name}")
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum OptChoice { Sgd, Adam }

impl OptChoice {
    #[allow(dead_code)]
    pub fn name(&self) -> &'static str {
        match self { OptChoice::Sgd => "SGD", OptChoice::Adam => "Adam" }
    }
}

pub struct NetworkInstance {
    pub id: u64,
    pub name: String,
    pub kind: NetworkKind,

    // Architecture.
    pub input_dim: usize,
    pub layer_specs: Vec<LayerSpec>,
    pub seed: u64,
    pub model: Option<Model>,
    pub build_message: Option<String>,
    pub pending_linear_units: usize,
    pub pending_activation: Activation,

    // Corpus + vocab.
    pub corpus: Corpus,
    pub vocab: Vocab,

    // Training hyperparameters.
    pub loss_choice: Loss,
    pub opt_choice: OptChoice,
    pub learning_rate: f32,
    pub momentum: f32,
    pub beta1: f32,
    pub beta2: f32,
    pub epsilon: f32,
    pub epochs: usize,
    pub batch_size: usize,
    pub val_frac: f32,
    pub trainer: Option<TrainerHandle>,
    pub last_state: TrainingState,
    pub use_log_y: bool,

    // Inference (numeric).
    pub inference_inputs: Vec<f32>,
    pub inference_output: Option<Tensor>,
    pub realtime_inference: bool,

    // Inference (text).
    pub prompt: String,
    pub generated: String,
    pub temperature: f32,
    pub max_tokens: usize,

    // Persistence.
    pub persistence_message: Option<String>,
}

impl NetworkInstance {
    pub fn new_simplex(id: u64, name: String, seed: u64) -> Self {
        let layers = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 16 },
            LayerSpec::Activation(Activation::Tanh),
            LayerSpec::Linear { in_dim: 16, out_dim: 1 },
            LayerSpec::Activation(Activation::Sigmoid),
        ];
        let mut me = Self::common(id, name, NetworkKind::Simplex, 2, layers, seed);
        me.corpus.template = CorpusTemplate::Xor;
        me.loss_choice = Loss::MeanSquaredError;
        me
    }

    pub fn new_next_token(id: u64, name: String, seed: u64) -> Self {
        let layers = vec![
            LayerSpec::Linear { in_dim: 1, out_dim: 32 },
            LayerSpec::Activation(Activation::Tanh),
            LayerSpec::Linear { in_dim: 32, out_dim: 1 },
            LayerSpec::Activation(Activation::Identity),
        ];
        let mut me = Self::common(id, name, NetworkKind::NextTokenGen, 1, layers, seed);
        me.corpus.template = CorpusTemplate::Text;
        me.loss_choice = Loss::CrossEntropy;
        me
    }

    pub fn new_plugin(id: u64, name: String, plugin_id: String, type_name: String, seed: u64) -> Self {
        let layers = vec![
            LayerSpec::Linear { in_dim: 2, out_dim: 16 },
            LayerSpec::Activation(Activation::ReLU),
            LayerSpec::Linear { in_dim: 16, out_dim: 1 },
        ];
        let mut me = Self::common(
            id,
            name,
            NetworkKind::Plugin { plugin_id, type_name },
            2,
            layers,
            seed,
        );
        me.corpus.template = CorpusTemplate::Custom;
        me
    }

    fn common(
        id: u64,
        name: String,
        kind: NetworkKind,
        input_dim: usize,
        layer_specs: Vec<LayerSpec>,
        seed: u64,
    ) -> Self {
        Self {
            id,
            name,
            kind,
            input_dim,
            layer_specs,
            seed,
            model: None,
            build_message: None,
            pending_linear_units: 8,
            pending_activation: Activation::ReLU,
            corpus: Corpus::default(),
            vocab: Vocab::default(),
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
            inference_inputs: vec![0.0; input_dim],
            inference_output: None,
            realtime_inference: false,
            prompt: String::new(),
            generated: String::new(),
            temperature: 0.8,
            max_tokens: 64,
            persistence_message: None,
        }
    }

    pub fn current_optimizer(&self) -> OptimizerKind {
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

    pub fn build_model(&mut self) {
        let mut cur = self.input_dim;
        for (i, spec) in self.layer_specs.iter().enumerate() {
            if let LayerSpec::Linear { in_dim, .. } = spec {
                if *in_dim != cur {
                    self.build_message = Some(format!(
                        "Layer {} expects in_dim={in_dim}, but running dim is {cur}.",
                        i + 1
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
            "Built {} layers · {} parameters · output dim {}.",
            model.layers.len(), model.parameter_count(), model.output_dim()
        ));
        self.model = Some(model);
    }

    pub fn set_optimizer(&mut self, kind: OptimizerKind) {
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

    /// Adjust the input layer's `in_dim` and the architecture's `input_dim`
    /// to match a new value (typically when the vocab grew).
    pub fn set_input_dim(&mut self, new_in: usize) {
        self.input_dim = new_in.max(1);
        if let Some(LayerSpec::Linear { in_dim, .. }) = self.layer_specs.first_mut() {
            *in_dim = self.input_dim;
        }
    }

    /// Adjust the final layer's `out_dim` (used to keep next-token-gen output
    /// width in sync with vocab size).
    pub fn set_output_dim(&mut self, new_out: usize) {
        for spec in self.layer_specs.iter_mut().rev() {
            if let LayerSpec::Linear { out_dim, .. } = spec {
                *out_dim = new_out.max(1);
                return;
            }
        }
    }
}

#[derive(Default)]
pub struct NetworkStore {
    pub list: Vec<NetworkInstance>,
    pub active: Option<u64>,
    pub next_id: u64,
}

impl NetworkStore {
    pub fn add(&mut self, mut net: NetworkInstance) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        net.id = id;
        net.build_model();
        self.list.push(net);
        if self.active.is_none() { self.active = Some(id); }
        id
    }

    pub fn remove(&mut self, id: u64) {
        self.list.retain(|n| n.id != id);
        if self.active == Some(id) {
            self.active = self.list.first().map(|n| n.id);
        }
    }

    pub fn select(&mut self, id: u64) {
        if self.list.iter().any(|n| n.id == id) {
            self.active = Some(id);
        }
    }

    pub fn active(&self) -> Option<&NetworkInstance> {
        self.active.and_then(|id| self.list.iter().find(|n| n.id == id))
    }

    pub fn active_mut(&mut self) -> Option<&mut NetworkInstance> {
        let id = self.active?;
        self.list.iter_mut().find(|n| n.id == id)
    }

    pub fn iter(&self) -> impl Iterator<Item = &NetworkInstance> { self.list.iter() }
}
