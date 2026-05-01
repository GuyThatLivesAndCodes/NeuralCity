//! egui application — tabbed workbench shell.
//!
//! The app is organised around a `NetworkStore` (multiple networks) and a tab
//! selector. The active network is displayed in a dropdown at the top of the
//! window; switching networks instantly reloads weights and reflects the new
//! state in every tab.

use crate::corpus::{Corpus, CorpusTemplate};
use crate::docs;
use crate::networks::{NetworkInstance, NetworkKind, NetworkStore, OptChoice};
use crate::paths;
use crate::plot::{scatter_2d, LinePlot};
use crate::plugins::PluginRegistry;
use crate::theme;
use crate::trainer::{self, TrainingConfig, TrainingState};
use crate::vocab::{Vocab, VocabMode};
use eframe::CreationContext;
use egui::{Color32, RichText};
use neuralcabin_engine::data::TaskKind;
use neuralcabin_engine::nn::LayerSpec;
use neuralcabin_engine::persistence::{self, ModelFile};
use neuralcabin_engine::tensor::{SplitMix64, Tensor};
use neuralcabin_engine::{Activation, Loss};
use std::time::Duration;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Tab {
    Docs,
    Networks,
    Corpus,
    Vocab,
    Training,
    Inference,
    Plugins,
}

impl Tab {
    pub const ALL: [Tab; 7] = [
        Tab::Docs, Tab::Networks, Tab::Corpus, Tab::Vocab,
        Tab::Training, Tab::Inference, Tab::Plugins,
    ];
    pub fn label(&self) -> &'static str {
        match self {
            Tab::Docs => "Docs",
            Tab::Networks => "Networks",
            Tab::Corpus => "Corpus",
            Tab::Vocab => "Vocab",
            Tab::Training => "Training",
            Tab::Inference => "Inference",
            Tab::Plugins => "Plugins",
        }
    }
}

/// "Create Network" dialog state — when `open` is true a panel is rendered
/// below the network list asking for kind & name.
#[derive(Default)]
struct CreateDialog {
    open: bool,
    name: String,
    kind_idx: usize,
    base_options: Vec<NetworkKind>,
}

pub struct NeuralCabinApp {
    tab: Tab,
    store: NetworkStore,
    plugins: PluginRegistry,
    create: CreateDialog,
    docs_section: usize,
    theme_applied: bool,
    rng: SplitMix64,
    pending_delete: Option<u64>,
    pending_load: Option<std::path::PathBuf>,
}

impl NeuralCabinApp {
    pub fn new(_cc: &CreationContext<'_>) -> Self {
        let mut app = Self {
            tab: Tab::Networks,
            store: NetworkStore::default(),
            plugins: PluginRegistry::default(),
            create: CreateDialog::default(),
            docs_section: 0,
            theme_applied: false,
            rng: SplitMix64::new(0xA11CE),
            pending_delete: None,
            pending_load: None,
        };
        // Seed with a starter Simplex XOR network so the workbench has
        // something to look at on first launch.
        let mut starter = NetworkInstance::new_simplex(0, "xor-mlp".into(), 0x5EED);
        starter.corpus.template = CorpusTemplate::Xor;
        starter.corpus.build_numeric(starter.seed);
        app.store.add(starter);
        app
    }

    fn rebuild_create_options(&mut self) {
        let mut opts = vec![NetworkKind::Simplex, NetworkKind::NextTokenGen];
        for (pid, ty) in self.plugins.all_network_types() {
            opts.push(NetworkKind::Plugin { plugin_id: pid, type_name: ty });
        }
        self.create.base_options = opts;
        if self.create.kind_idx >= self.create.base_options.len() {
            self.create.kind_idx = 0;
        }
    }

    /// Whether the active network's Vocab tab should be enabled.
    fn vocab_enabled(&self) -> bool {
        match self.store.active().map(|n| &n.kind) {
            Some(NetworkKind::NextTokenGen) => true,
            Some(NetworkKind::Plugin { plugin_id, .. }) => self
                .plugins
                .find_by_id(plugin_id)
                .map(|p| p.manifest.manages_vocab)
                .unwrap_or(false),
            _ => false,
        }
    }

    fn poll_active_trainer(&mut self) {
        let Some(net) = self.store.active_mut() else { return };
        if let Some(handle) = &net.trainer {
            if let Ok(s) = handle.state.lock() { net.last_state = s.clone(); }
            if handle.is_finished() {
                let h = net.trainer.take().unwrap();
                if let Some(trained) = h.join() {
                    net.model = Some(trained);
                    net.build_message = Some("Training finished — model updated.".into());
                }
            }
        }
    }

    fn any_training(&self) -> bool {
        self.store.list.iter().any(|n| n.trainer.is_some())
    }
}

impl eframe::App for NeuralCabinApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        if !self.theme_applied {
            theme::apply(ctx);
            self.theme_applied = true;
        }
        if let Some(id) = self.pending_delete.take() {
            self.store.remove(id);
        }
        if let Some(path) = self.pending_load.take() {
            if let Some(net) = self.store.active_mut() {
                load_active(net, &path);
            }
        }
        if self.any_training() {
            ctx.request_repaint_after(Duration::from_millis(75));
        }
        self.poll_active_trainer();
        self.rebuild_create_options();

        self.top_bar(ctx);
        self.tab_strip(ctx);
        egui::CentralPanel::default().show(ctx, |ui| {
            match self.tab {
                Tab::Docs => self.docs_tab(ui),
                Tab::Networks => self.networks_tab(ui),
                Tab::Corpus => self.corpus_tab(ui),
                Tab::Vocab => self.vocab_tab(ui),
                Tab::Training => self.training_tab(ui),
                Tab::Inference => self.inference_tab(ui),
                Tab::Plugins => self.plugins_tab(ui),
            }
        });
    }
}

// ===== TOP BAR & TAB STRIP =====

impl NeuralCabinApp {
    fn top_bar(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top").exact_height(46.0).show(ctx, |ui| {
            ui.horizontal_centered(|ui| {
                ui.add_space(4.0);
                ui.label(
                    RichText::new("NeuralCabin")
                        .color(theme::TEXT)
                        .size(18.0)
                        .strong(),
                );
                ui.add_space(8.0);
                ui.label(
                    RichText::new("pure-Rust workbench")
                        .color(theme::TEXT_FAINT)
                        .italics()
                        .size(11.5),
                );
                ui.add_space(20.0);

                // Active-network dropdown — LM-Studio-style.
                let active_label = self.store.active()
                    .map(|n| format!("◆ {}  ·  {}", n.name, n.kind.label()))
                    .unwrap_or_else(|| "◇ no network".into());
                egui::ComboBox::from_id_salt("active_net_top")
                    .width(300.0)
                    .selected_text(RichText::new(active_label).color(theme::TEXT))
                    .show_ui(ui, |ui| {
                        let ids: Vec<(u64, String, String)> = self.store.iter()
                            .map(|n| (n.id, n.name.clone(), n.kind.label())).collect();
                        for (id, name, kind) in ids {
                            let label = format!("{name}  ·  {kind}");
                            let selected = self.store.active == Some(id);
                            if ui.selectable_label(selected, label).clicked() {
                                self.store.select(id);
                            }
                        }
                        ui.separator();
                        if ui.button("＋ New network…").clicked() {
                            self.tab = Tab::Networks;
                            self.create.open = true;
                        }
                    });

                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    ui.add_space(6.0);
                    ui.label(RichText::new(format!("v{}", env!("CARGO_PKG_VERSION")))
                        .color(theme::TEXT_FAINT).size(11.0));
                    ui.add_space(12.0);
                    let alive = if self.any_training() { 1.0 } else { 0.0 };
                    let label = if alive > 0.0 { "training" } else { "idle" };
                    theme::pulse_dot(ui, alive, label);
                });
            });
        });
    }

    fn tab_strip(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("tab_strip").exact_height(40.0).show(ctx, |ui| {
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                for tab in Tab::ALL {
                    let mut text = RichText::new(tab.label()).size(13.5);
                    let enabled = match tab {
                        Tab::Vocab => self.vocab_enabled(),
                        _ => true,
                    };
                    if !enabled { text = text.color(theme::TEXT_FAINT); }
                    let selected = self.tab == tab;
                    if selected { text = text.color(theme::ACCENT).strong(); }
                    let resp = ui.add_enabled(
                        enabled,
                        egui::Button::new(text)
                            .frame(false)
                            .min_size(egui::vec2(0.0, 28.0)),
                    );
                    if resp.clicked() { self.tab = tab; }
                    if selected {
                        let r = resp.rect;
                        ui.painter().line_segment(
                            [
                                egui::pos2(r.left() + 6.0, r.bottom()),
                                egui::pos2(r.right() - 6.0, r.bottom()),
                            ],
                            egui::Stroke::new(2.0, theme::ACCENT),
                        );
                    }
                    ui.add_space(2.0);
                }
            });
        });
    }
}

// ===== DOCS TAB =====

impl NeuralCabinApp {
    fn docs_tab(&mut self, ui: &mut egui::Ui) {
        let sections = docs::sections();
        egui::SidePanel::left("docs_nav")
            .resizable(true)
            .default_width(220.0)
            .show_inside(ui, |ui| {
                theme::section_heading(ui, "Sections");
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (i, s) in sections.iter().enumerate() {
                        let selected = self.docs_section == i;
                        let mut text = RichText::new(s.title).size(13.0);
                        if selected { text = text.color(theme::ACCENT).strong(); }
                        if ui.add(
                            egui::Button::new(text)
                                .frame(false)
                                .min_size(egui::vec2(ui.available_width(), 22.0)),
                        ).clicked() {
                            self.docs_section = i;
                        }
                    }
                });
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            let s = &sections[self.docs_section.min(sections.len() - 1)];
            ui.label(RichText::new(s.title).size(22.0).strong());
            theme::hairline(ui);
            ui.add_space(6.0);
            egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
                for line in s.body.lines() {
                    if line.is_empty() {
                        ui.add_space(6.0);
                    } else if line.starts_with("- ") {
                        ui.horizontal(|ui| {
                            ui.add_space(8.0);
                            ui.label(RichText::new("•").color(theme::TEXT_WEAK));
                            ui.label(RichText::new(line.trim_start_matches("- ")).color(theme::TEXT));
                        });
                    } else if line.starts_with("```") {
                        // skip fence markers — rendered inline by the next/prev block
                    } else if line == "------------" || line == "---------------" || line == "-------" {
                        theme::hairline(ui);
                    } else if line.ends_with(':')
                        && line.len() < 60
                        && line.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    {
                        ui.add_space(4.0);
                        ui.label(RichText::new(line).color(theme::TEXT).strong());
                    } else {
                        ui.label(RichText::new(line).color(theme::TEXT));
                    }
                }
            });
        });
    }
}

// ===== NETWORKS TAB =====

impl NeuralCabinApp {
    fn networks_tab(&mut self, ui: &mut egui::Ui) {
        egui::SidePanel::left("networks_list")
            .resizable(true)
            .default_width(240.0)
            .show_inside(ui, |ui| {
                ui.add_space(4.0);
                if ui
                    .add(
                        egui::Button::new(RichText::new("＋  Create Network").strong())
                            .min_size(egui::vec2(ui.available_width(), 28.0)),
                    )
                    .clicked()
                {
                    self.create.open = true;
                    self.create.name = format!("net-{}", self.store.next_id);
                }
                ui.add_space(6.0);
                theme::hairline(ui);
                ui.add_space(4.0);
                let active = self.store.active;
                let entries: Vec<(u64, String, String)> = self
                    .store
                    .iter()
                    .map(|n| (n.id, n.name.clone(), n.kind.label()))
                    .collect();
                let mut to_select: Option<u64> = None;
                egui::ScrollArea::vertical().show(ui, |ui| {
                    for (id, name, kind) in entries {
                        let selected = active == Some(id);
                        let mut text = RichText::new(format!("◆  {name}")).size(13.5);
                        if selected { text = text.color(theme::ACCENT).strong(); }
                        let avail = ui.available_width();
                        if ui
                            .add(
                                egui::Button::new(text)
                                    .frame(selected)
                                    .min_size(egui::vec2(avail, 24.0)),
                            )
                            .clicked()
                        {
                            to_select = Some(id);
                        }
                        ui.label(
                            RichText::new(format!("    {kind}"))
                                .color(theme::TEXT_WEAK)
                                .size(11.0),
                        );
                        ui.add_space(2.0);
                    }
                });
                if let Some(id) = to_select { self.store.select(id); }

                if self.create.open {
                    ui.add_space(8.0);
                    theme::hairline(ui);
                    ui.add_space(4.0);
                    self.create_dialog(ui);
                }
            });
        egui::CentralPanel::default().show_inside(ui, |ui| {
            self.network_editor(ui);
        });
    }

    fn create_dialog(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("New network").color(theme::TEXT).strong());
        ui.horizontal(|ui| {
            ui.label("Name:");
            ui.add(egui::TextEdit::singleline(&mut self.create.name).desired_width(160.0));
        });
        ui.label(RichText::new("Type:").color(theme::TEXT_WEAK).size(11.5));
        let kinds = self.create.base_options.clone();
        for (i, kind) in kinds.iter().enumerate() {
            let selected = self.create.kind_idx == i;
            let mut text = RichText::new(format!("• {}", kind.label())).size(12.5);
            if selected { text = text.color(theme::ACCENT).strong(); }
            if ui
                .add(
                    egui::Button::new(text)
                        .frame(selected)
                        .min_size(egui::vec2(ui.available_width(), 22.0)),
                )
                .clicked()
            {
                self.create.kind_idx = i;
            }
        }
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("Create").clicked() {
                let name = if self.create.name.trim().is_empty() {
                    format!("net-{}", self.store.next_id)
                } else {
                    self.create.name.trim().to_string()
                };
                let seed = self.rng.next_u64();
                let kind = self
                    .create
                    .base_options
                    .get(self.create.kind_idx)
                    .cloned()
                    .unwrap_or(NetworkKind::Simplex);
                let net = match kind {
                    NetworkKind::Simplex => NetworkInstance::new_simplex(0, name, seed),
                    NetworkKind::NextTokenGen => NetworkInstance::new_next_token(0, name, seed),
                    NetworkKind::Plugin { plugin_id, type_name } => {
                        NetworkInstance::new_plugin(0, name, plugin_id, type_name, seed)
                    }
                };
                let new_id = self.store.add(net);
                self.store.select(new_id);
                self.create.open = false;
                self.create.name.clear();
            }
            if ui.button("Cancel").clicked() {
                self.create.open = false;
            }
        });
    }

    fn network_editor(&mut self, ui: &mut egui::Ui) {
        let active_id = self.store.active;
        let saved = paths::list_saved_networks().unwrap_or_default();
        let storage_path = paths::networks_dir()
            .map(|p| p.display().to_string())
            .unwrap_or_else(|e| format!("(unavailable: {e})"));
        let pending_delete = &mut self.pending_delete;
        let pending_load = &mut self.pending_load;
        let Some(net) = self.store.active_mut() else {
            ui.add_space(40.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("No network selected.").color(theme::TEXT_WEAK));
                ui.add_space(8.0);
                ui.label(RichText::new("Use ＋ Create Network on the left.").color(theme::TEXT_FAINT));
            });
            return;
        };
        ui.horizontal(|ui| {
            ui.label(RichText::new(&net.name).size(20.0).strong());
            ui.label(RichText::new(net.kind.label()).color(theme::TEXT_WEAK));
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui
                    .button(RichText::new("✕  Delete network").color(theme::TEXT_WEAK))
                    .on_hover_text("Remove this network from the workspace")
                    .clicked()
                {
                    if let Some(id) = active_id { *pending_delete = Some(id); }
                }
            });
        });
        theme::hairline(ui);
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Rename:");
            ui.add(egui::TextEdit::singleline(&mut net.name).desired_width(220.0));
            ui.add_space(12.0);
            ui.label("Init seed:");
            let mut s = net.seed as i64;
            if ui.add(egui::DragValue::new(&mut s)).changed() { net.seed = s as u64; }
        });

        ui.horizontal(|ui| {
            ui.label("Input dim:");
            let mut dim = net.input_dim as i32;
            if ui.add(egui::DragValue::new(&mut dim).range(1..=4096)).changed() {
                net.set_input_dim(dim.max(1) as usize);
            }
        });

        ui.add_space(6.0);
        ui.label(RichText::new("Layers (top → bottom)").color(theme::TEXT).strong());
        let mut to_remove: Option<usize> = None;
        let mut current_dim = net.input_dim;
        for (i, spec) in net.layer_specs.iter_mut().enumerate() {
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
                            if ui.add(egui::DragValue::new(&mut o).range(1..=8192)).changed() {
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
        if let Some(i) = to_remove { net.layer_specs.remove(i); }

        ui.horizontal(|ui| {
            if ui.button("＋ Linear").clicked() {
                let in_dim = current_dim;
                net.layer_specs.push(LayerSpec::Linear { in_dim, out_dim: net.pending_linear_units });
            }
            ui.label("units:");
            let mut u = net.pending_linear_units as i32;
            if ui.add(egui::DragValue::new(&mut u).range(1..=8192)).changed() {
                net.pending_linear_units = u.max(1) as usize;
            }
            ui.add_space(8.0);
            if ui.button("＋ Activation").clicked() {
                net.layer_specs.push(LayerSpec::Activation(net.pending_activation));
            }
            egui::ComboBox::from_id_salt("pending_act")
                .selected_text(net.pending_activation.name())
                .show_ui(ui, |ui| {
                    for opt in Activation::all() {
                        ui.selectable_value(&mut net.pending_activation, *opt, opt.name());
                    }
                });
        });

        ui.add_space(6.0);
        if ui
            .add_sized(
                [ui.available_width(), 30.0],
                egui::Button::new(RichText::new("Build / Reset Model").strong()),
            )
            .clicked()
        {
            net.build_model();
        }
        if let Some(msg) = &net.build_message {
            let color = if msg.starts_with("Layer ") || msg.starts_with("Model ")
            { theme::DANGER } else { theme::TEXT_WEAK };
            ui.colored_label(color, msg);
        }
        if let Some(model) = &net.model {
            ui.add_space(4.0);
            ui.label(RichText::new("Compiled model").color(theme::TEXT).strong());
            for (i, l) in model.layers.iter().enumerate() {
                ui.label(RichText::new(format!("  {}. {}", i + 1, l.describe()))
                    .color(theme::TEXT_WEAK).monospace());
            }
            ui.label(RichText::new(format!("Parameters: {}", model.parameter_count()))
                .color(theme::TEXT_WEAK));
        }

        ui.add_space(10.0);
        theme::section_heading(ui, "Persistence");
        let stem = paths::sanitize_filename(&net.name);
        ui.label(
            RichText::new(format!("File: {stem}.json"))
                .color(theme::TEXT_WEAK)
                .size(11.5),
        );
        ui.label(
            RichText::new(format!("Location: {storage_path}"))
                .color(theme::TEXT_FAINT)
                .size(11.0),
        );
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui
                .button(RichText::new("💾  Save").strong())
                .on_hover_text("Save this network into the app data folder")
                .clicked()
            {
                save_active(net);
            }
            let load_label = if saved.is_empty() {
                "Load… (none saved)".to_string()
            } else {
                "Load saved network…".to_string()
            };
            egui::ComboBox::from_id_salt("load_saved_pick")
                .selected_text(load_label)
                .show_ui(ui, |ui| {
                    if saved.is_empty() {
                        ui.label(
                            RichText::new("(no saved networks yet)")
                                .color(theme::TEXT_FAINT),
                        );
                    } else {
                        for (name, path) in &saved {
                            if ui.selectable_label(false, name).clicked() {
                                *pending_load = Some(path.clone());
                            }
                        }
                    }
                });
        });
        if let Some(m) = &net.persistence_message {
            ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
        }
    }
}

fn save_active(net: &mut NetworkInstance) {
    let Some(model) = &net.model else {
        net.persistence_message = Some("No model to save.".into());
        return;
    };
    let dir = match paths::networks_dir() {
        Ok(d) => d,
        Err(e) => {
            net.persistence_message = Some(format!("appdata error: {e}"));
            return;
        }
    };
    let stem = paths::sanitize_filename(&net.name);
    let path = dir.join(format!("{stem}.json"));
    let file = ModelFile::wrap(model.clone(), Some(net.loss_choice), Some(net.current_optimizer()));
    match persistence::save(&path, &file) {
        Ok(()) => net.persistence_message = Some(format!("Saved → {}", path.display())),
        Err(e) => net.persistence_message = Some(format!("Save failed: {e}")),
    }
}

fn load_active(net: &mut NetworkInstance, path: &std::path::Path) {
    match persistence::load(path) {
        Ok(f) => {
            net.input_dim = f.model.input_dim;
            net.layer_specs = f
                .model
                .layers
                .iter()
                .map(|l| match l {
                    neuralcabin_engine::nn::Layer::Linear(ll) =>
                        LayerSpec::Linear { in_dim: ll.in_dim, out_dim: ll.out_dim },
                    neuralcabin_engine::nn::Layer::Activation(a) => LayerSpec::Activation(*a),
                })
                .collect();
            if let Some(l) = f.loss { net.loss_choice = l; }
            if let Some(o) = f.optimizer { net.set_optimizer(o); }
            net.inference_inputs = vec![0.0; f.model.input_dim];
            net.model = Some(f.model);
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if !stem.is_empty() { net.name = stem.to_string(); }
            }
            net.persistence_message = Some(format!("Loaded ← {}", path.display()));
        }
        Err(e) => net.persistence_message = Some(format!("Load failed: {e}")),
    }
}

// ===== CORPUS TAB =====

impl NeuralCabinApp {
    fn corpus_tab(&mut self, ui: &mut egui::Ui) {
        let Some(net) = self.store.active_mut() else {
            empty_state(ui, "No network selected — create one on the Networks tab.");
            return;
        };
        ui.label(RichText::new("Corpus").size(20.0).strong());
        theme::hairline(ui);
        ui.add_space(6.0);
        match &net.kind {
            NetworkKind::NextTokenGen => corpus_text_panel(ui, &mut net.corpus, &net.vocab),
            NetworkKind::Simplex => corpus_numeric_panel(ui, &mut net.corpus, net.seed, false),
            NetworkKind::Plugin { .. } => {
                ui.label(RichText::new(
                    "Plugin networks default to numeric corpora — pick a template:",
                ).color(theme::TEXT_WEAK));
                corpus_numeric_panel(ui, &mut net.corpus, net.seed, true);
            }
        }
    }
}

fn corpus_numeric_panel(ui: &mut egui::Ui, corpus: &mut Corpus, seed: u64, allow_text: bool) {
    ui.horizontal_wrapped(|ui| {
        ui.label("Template:");
        for tpl in Corpus::numeric_templates() {
            if ui.selectable_label(corpus.template == *tpl, tpl.name()).clicked() {
                corpus.template = *tpl;
            }
        }
        if allow_text
            && ui.selectable_label(corpus.template == CorpusTemplate::Text, "Text").clicked()
        {
            corpus.template = CorpusTemplate::Text;
        }
    });
    ui.add_space(4.0);
    match corpus.template {
        CorpusTemplate::Spirals => {
            ui.horizontal(|ui| {
                ui.label("Classes:");
                let mut c = corpus.spirals_classes as i32;
                if ui.add(egui::DragValue::new(&mut c).range(2..=10)).changed() {
                    corpus.spirals_classes = c.max(2) as usize;
                }
                ui.label("per class:");
                let mut p = corpus.spirals_per_class as i32;
                if ui.add(egui::DragValue::new(&mut p).range(10..=2000)).changed() {
                    corpus.spirals_per_class = p.max(10) as usize;
                }
            });
        }
        CorpusTemplate::Sine => {
            ui.horizontal(|ui| {
                ui.label("N:");
                let mut n = corpus.sine_n as i32;
                if ui.add(egui::DragValue::new(&mut n).range(10..=10_000)).changed() {
                    corpus.sine_n = n.max(10) as usize;
                }
                ui.label("noise σ:");
                ui.add(egui::DragValue::new(&mut corpus.sine_noise).speed(0.01).range(0.0..=2.0));
            });
        }
        CorpusTemplate::Csv => {
            ui.horizontal(|ui| {
                ui.label("Path:");
                ui.add(egui::TextEdit::singleline(&mut corpus.csv_path).desired_width(360.0));
            });
            ui.horizontal(|ui| {
                ui.checkbox(&mut corpus.csv_has_header, "Has header");
                ui.label("num_classes (blank = regression):");
                ui.add(egui::TextEdit::singleline(&mut corpus.csv_num_classes).desired_width(60.0));
            });
        }
        CorpusTemplate::Custom => {
            ui.horizontal(|ui| {
                ui.label("input dim:");
                let mut a = corpus.custom_input_dim as i32;
                if ui.add(egui::DragValue::new(&mut a).range(1..=128)).changed() {
                    corpus.custom_input_dim = a.max(1) as usize;
                }
                ui.label("output dim:");
                let mut b = corpus.custom_output_dim as i32;
                if ui.add(egui::DragValue::new(&mut b).range(1..=128)).changed() {
                    corpus.custom_output_dim = b.max(1) as usize;
                }
                ui.checkbox(&mut corpus.custom_classification, "one-hot classification");
            });
            theme::caption(
                ui,
                "Each line is one row: comma-separated values, inputs first then outputs. \
                 Blank lines and lines starting with # are ignored.",
            );
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut corpus.custom_rows)
                        .desired_width(f32::INFINITY)
                        .desired_rows(10)
                        .code_editor(),
                );
            });
        }
        CorpusTemplate::Xor | CorpusTemplate::Text => {}
    }
    ui.add_space(6.0);
    if ui.button("Build / Reload corpus").clicked() {
        if matches!(corpus.template, CorpusTemplate::Text) {
            corpus.message = Some("Switch to a numeric template.".into());
        } else {
            corpus.build_numeric(seed);
        }
    }
    if let Some(m) = &corpus.message {
        ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
    }
    if let Some(ds) = &corpus.dataset {
        ui.add_space(4.0);
        if ds.n_features() == 2 {
            if let TaskKind::Classification { num_classes } = ds.task {
                ui.label(RichText::new("2-D scatter preview").color(theme::TEXT_WEAK));
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

fn corpus_text_panel(ui: &mut egui::Ui, corpus: &mut Corpus, vocab: &Vocab) {
    corpus.template = CorpusTemplate::Text;
    ui.horizontal(|ui| {
        ui.label("Context size:");
        let mut c = corpus.context_size as i32;
        if ui.add(egui::DragValue::new(&mut c).range(1..=64)).changed() {
            corpus.context_size = c.max(1) as usize;
        }
        ui.label(RichText::new(format!("vocab = {} tokens", vocab.len())).color(theme::TEXT_WEAK));
    });
    ui.add_space(4.0);
    ui.label(RichText::new("Corpus body — paste documentation, chats, source, anything.")
        .color(theme::TEXT_WEAK).size(11.5));
    egui::ScrollArea::vertical().max_height(260.0).show(ui, |ui| {
        ui.add(
            egui::TextEdit::multiline(&mut corpus.text_body)
                .desired_width(f32::INFINITY)
                .desired_rows(12)
                .code_editor(),
        );
    });
    ui.add_space(4.0);
    ui.horizontal(|ui| {
        ui.label("Append file:");
        ui.add(egui::TextEdit::singleline(&mut corpus.upload_path).desired_width(360.0));
        if ui.button("Upload").clicked() {
            let path = corpus.upload_path.clone();
            if let Err(e) = corpus.upload_text_file(&path) {
                corpus.message = Some(format!("Upload failed: {e}"));
            }
        }
    });
    if !corpus.text_paths.is_empty() {
        ui.label(RichText::new(format!("Files: {}", corpus.text_paths.join(", ")))
            .color(theme::TEXT_FAINT).size(11.0));
    }
    ui.add_space(6.0);
    ui.horizontal(|ui| {
        if ui.button("Re-tokenise").clicked() {
            corpus.retokenise(vocab);
        }
        if ui.button("Build training set").clicked() {
            if let Err(e) = corpus.build_text_dataset(vocab) {
                corpus.message = Some(format!("Build failed: {e}"));
            }
        }
    });
    if let Some(m) = &corpus.message {
        ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
    }
}

// ===== VOCAB TAB =====

impl NeuralCabinApp {
    fn vocab_tab(&mut self, ui: &mut egui::Ui) {
        let allowed = self.vocab_enabled();
        let Some(net) = self.store.active_mut() else {
            empty_state(ui, "No network selected.");
            return;
        };
        ui.label(RichText::new("Vocabulary").size(20.0).strong());
        theme::hairline(ui);
        if !allowed {
            ui.add_space(20.0);
            ui.vertical_centered(|ui| {
                ui.label(RichText::new("Vocab is only available for next-token-generation networks.")
                    .color(theme::TEXT_WEAK));
                ui.add_space(4.0);
                ui.label(RichText::new("(or for plugin networks whose manifest sets manages_vocab = true)")
                    .color(theme::TEXT_FAINT).size(11.0));
            });
            return;
        }
        let plugin_managed = matches!(&net.kind, NetworkKind::Plugin { .. });
        if plugin_managed {
            ui.label(RichText::new("Plugin-managed vocab — the active plugin owns this tab.")
                .color(theme::TEXT_WEAK));
            ui.add_space(4.0);
        }

        ui.horizontal(|ui| {
            ui.label("Auto-generate from corpus:");
            let mut new_mode: Option<VocabMode> = None;
            for m in VocabMode::all() {
                if ui.selectable_label(net.vocab.mode == *m, m.name()).clicked() {
                    new_mode = Some(*m);
                }
            }
            if let Some(m) = new_mode {
                net.vocab.auto_generate(&net.corpus.text_body, m);
                if matches!(net.kind, NetworkKind::NextTokenGen) {
                    let v = net.vocab.len().max(1);
                    let in_dim = v * net.corpus.context_size.max(1);
                    net.set_input_dim(in_dim);
                    net.set_output_dim(v);
                }
            }
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if ui.button("Wipe").clicked() { net.vocab.clear(); }
            ui.add(
                egui::TextEdit::singleline(&mut net.vocab.draft_token)
                    .hint_text("token to add")
                    .desired_width(180.0),
            );
            if ui.button("Add").clicked() {
                let t = net.vocab.draft_token.trim().to_string();
                if !t.is_empty() {
                    net.vocab.add_unique(t);
                    net.vocab.draft_token.clear();
                }
            }
        });
        ui.horizontal(|ui| {
            ui.label("Upload (one token per line):");
            ui.add(egui::TextEdit::singleline(&mut net.vocab.upload_path).desired_width(280.0));
            if ui.button("Load").clicked() {
                let path = net.vocab.upload_path.clone();
                if let Err(e) = net.vocab.load_file(&path) {
                    net.vocab.message = Some(format!("Load failed: {e}"));
                }
            }
        });

        if let Some(m) = &net.vocab.message {
            ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
        }
        ui.add_space(8.0);
        ui.label(RichText::new(format!("{} tokens", net.vocab.len())).color(theme::TEXT_WEAK));
        egui::ScrollArea::vertical().auto_shrink([false; 2]).show(ui, |ui| {
            let mut to_remove: Option<usize> = None;
            for (i, tok) in net.vocab.tokens.iter().enumerate() {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(format!("{i:>5}  ")).color(theme::TEXT_FAINT).monospace());
                    let display = if tok.is_empty() { "<empty>".to_string() } else { tok.replace('\n', "\\n") };
                    ui.label(RichText::new(display).monospace().color(theme::TEXT));
                    if i > 0 && ui.small_button("✕").clicked() { to_remove = Some(i); }
                });
            }
            if let Some(i) = to_remove { net.vocab.tokens.remove(i); }
        });
    }
}

// ===== TRAINING TAB =====

impl NeuralCabinApp {
    fn training_tab(&mut self, ui: &mut egui::Ui) {
        let Some(net) = self.store.active_mut() else {
            empty_state(ui, "No network selected.");
            return;
        };
        ui.label(RichText::new("Training").size(20.0).strong());
        theme::hairline(ui);
        ui.add_space(4.0);
        let training = net.trainer.is_some();

        ui.horizontal_wrapped(|ui| {
            ui.label("Loss:");
            for l in Loss::all() {
                if ui.selectable_label(net.loss_choice == *l, l.name()).clicked() {
                    net.loss_choice = *l;
                }
            }
            ui.add_space(20.0);
            ui.label("Optimiser:");
            ui.selectable_value(&mut net.opt_choice, OptChoice::Sgd, "SGD");
            ui.selectable_value(&mut net.opt_choice, OptChoice::Adam, "Adam");
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Learning rate:");
            if ui.add(egui::DragValue::new(&mut net.learning_rate).speed(0.001).range(1e-6..=10.0)).changed() {
                if let Some(h) = &net.trainer { h.set_lr(net.learning_rate); }
            }
            match net.opt_choice {
                OptChoice::Sgd => {
                    ui.label("momentum:");
                    ui.add(egui::DragValue::new(&mut net.momentum).speed(0.01).range(0.0..=0.999));
                }
                OptChoice::Adam => {
                    ui.label("β₁:");
                    ui.add(egui::DragValue::new(&mut net.beta1).speed(0.001).range(0.0..=0.999));
                    ui.label("β₂:");
                    ui.add(egui::DragValue::new(&mut net.beta2).speed(0.0001).range(0.0..=0.99999));
                }
            }
        });

        ui.horizontal_wrapped(|ui| {
            ui.label("Epochs:");
            let mut e = net.epochs as i32;
            if ui.add(egui::DragValue::new(&mut e).range(1..=1_000_000)).changed() {
                net.epochs = e.max(1) as usize;
            }
            ui.label("Batch size:");
            let mut b = net.batch_size as i32;
            if ui.add(egui::DragValue::new(&mut b).range(1..=4096)).changed() {
                net.batch_size = b.max(1) as usize;
            }
            ui.label("Validation frac:");
            ui.add(egui::DragValue::new(&mut net.val_frac).speed(0.01).range(0.0..=0.9));
        });

        ui.add_space(6.0);
        ui.horizontal(|ui| {
            if !training {
                if ui
                    .add(egui::Button::new(RichText::new("▶  Train").strong())
                        .min_size(egui::vec2(110.0, 28.0)))
                    .clicked()
                {
                    start_training(net);
                }
            } else {
                let paused = net.last_state.paused;
                if ui.button(if paused { "▶ Resume" } else { "⏸ Pause" }).clicked() {
                    if let Some(h) = &net.trainer { h.pause(!paused); }
                    net.last_state.paused = !paused;
                }
                if ui.button("⏹ Stop").clicked() {
                    if let Some(h) = &net.trainer { h.stop(); }
                }
            }
            ui.checkbox(&mut net.use_log_y, "log-scale loss");
            theme::pulse_dot(ui, if training { 1.0 } else { 0.0 }, "");
        });

        if let Some(m) = &net.build_message {
            ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
        }

        ui.add_space(6.0);
        let s = &net.last_state;
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
            ui.colored_label(theme::DANGER, e);
        }

        ui.add_space(8.0);
        let series_owned: Vec<(String, Vec<f32>, Color32)> = vec![
            ("train".into(), s.loss_history.clone(), theme::TEXT),
            ("val".into(), s.val_loss_history.clone(), theme::TEXT_WEAK),
        ];
        let series_ref: Vec<(&str, &[f32], Color32)> = series_owned
            .iter()
            .map(|(n, v, c)| (n.as_str(), v.as_slice(), *c))
            .collect();
        LinePlot {
            title: "Loss",
            series: series_ref,
            log_y: net.use_log_y,
            min_height: 220.0,
        }
        .show(ui);

        if !s.accuracy_history.is_empty() {
            ui.add_space(4.0);
            let acc = [(
                "accuracy".to_string(),
                s.accuracy_history.clone(),
                theme::TEXT,
            )];
            let acc_ref: Vec<(&str, &[f32], Color32)> = acc
                .iter()
                .map(|(n, v, c)| (n.as_str(), v.as_slice(), *c))
                .collect();
            LinePlot {
                title: "Validation accuracy",
                series: acc_ref,
                log_y: false,
                min_height: 140.0,
            }
            .show(ui);
        }
    }
}

fn start_training(net: &mut NetworkInstance) {
    if net.model.is_none() {
        net.build_message = Some("Build a model first.".into());
        return;
    }
    if net.corpus.dataset.is_none() {
        // For text networks try to build the dataset on the fly.
        if matches!(net.kind, NetworkKind::NextTokenGen) {
            let vocab_clone = net.vocab.clone();
            if let Err(e) = net.corpus.build_text_dataset(&vocab_clone) {
                net.build_message = Some(format!("Cannot start: {e}"));
                return;
            }
        } else {
            net.corpus.build_numeric(net.seed);
            if net.corpus.dataset.is_none() {
                net.build_message = Some("Cannot start: load a corpus first.".into());
                return;
            }
        }
    }
    let model = net.model.clone().unwrap();
    let dataset = net.corpus.dataset.clone().unwrap();
    if model.input_dim != dataset.n_features() {
        net.build_message = Some(format!(
            "Model input_dim={} but dataset has {} features. Rebuild the model.",
            model.input_dim, dataset.n_features()
        ));
        return;
    }
    if model.output_dim() != dataset.n_outputs() {
        net.build_message = Some(format!(
            "Model output_dim={} but dataset has {} output dims.",
            model.output_dim(), dataset.n_outputs()
        ));
        return;
    }
    let cfg = TrainingConfig {
        epochs: net.epochs,
        batch_size: net.batch_size,
        optimizer: net.current_optimizer(),
        loss: net.loss_choice,
        validation_frac: net.val_frac.clamp(0.0, 0.9),
        seed: net.seed,
    };
    net.last_state = TrainingState {
        running: true,
        total_epochs: net.epochs,
        ..Default::default()
    };
    net.trainer = Some(trainer::spawn(model, dataset, cfg));
}

// ===== INFERENCE TAB =====

impl NeuralCabinApp {
    fn inference_tab(&mut self, ui: &mut egui::Ui) {
        let active_kind = self.store.active().map(|n| n.kind.clone());
        let plugin_managed = match &active_kind {
            Some(NetworkKind::Plugin { plugin_id, .. }) => self
                .plugins
                .find_by_id(plugin_id)
                .map(|p| p.manifest.manages_inference)
                .unwrap_or(false),
            _ => false,
        };
        let Some(net) = self.store.active_mut() else {
            empty_state(ui, "No network selected.");
            return;
        };
        ui.label(RichText::new("Inference").size(20.0).strong());
        theme::hairline(ui);
        ui.add_space(4.0);
        if net.model.is_none() {
            ui.label(RichText::new("Build the model first (Networks tab).")
                .color(theme::TEXT_WEAK));
            return;
        }
        match &net.kind {
            NetworkKind::Simplex => simplex_inference(ui, net),
            NetworkKind::NextTokenGen => text_inference(ui, net),
            NetworkKind::Plugin { plugin_id, type_name } => {
                if plugin_managed {
                    ui.label(RichText::new(format!(
                        "Plugin '{plugin_id}' manages this tab (type: {type_name})."
                    )).color(theme::TEXT_WEAK));
                    ui.add_space(4.0);
                    theme::caption(
                        ui,
                        "Falling back to Simplex inputs until the plugin runtime is wired in.",
                    );
                }
                simplex_inference(ui, net);
            }
        }
    }
}

fn simplex_inference(ui: &mut egui::Ui, net: &mut NetworkInstance) {
    let Some(model) = &net.model else { return; };
    if net.inference_inputs.len() != model.input_dim {
        net.inference_inputs.resize(model.input_dim, 0.0);
    }
    ui.horizontal(|ui| {
        ui.label(format!("Inputs ({}):", model.input_dim));
        ui.checkbox(&mut net.realtime_inference, "Real-time");
    });
    let mut changed = false;
    egui::ScrollArea::vertical().max_height(280.0).show(ui, |ui| {
        for (i, v) in net.inference_inputs.iter_mut().enumerate() {
            ui.horizontal(|ui| {
                ui.label(format!("x{i}:"));
                if ui.add(egui::DragValue::new(v).speed(0.05)).changed() { changed = true; }
            });
        }
    });
    ui.add_space(4.0);
    let predict_clicked = ui
        .add_sized(
            [ui.available_width(), 28.0],
            egui::Button::new(RichText::new("Predict").strong()),
        )
        .clicked();
    if predict_clicked || (net.realtime_inference && changed) {
        run_simplex_predict(net);
    }
    if let Some(out) = &net.inference_output {
        ui.add_space(8.0);
        ui.label(RichText::new("Output").color(theme::TEXT).strong());
        for (i, v) in out.data.iter().enumerate() {
            ui.label(RichText::new(format!("y{i}: {v:.6}")).monospace());
        }
        if out.data.len() > 1 {
            let mut best = 0;
            let mut bv = f32::NEG_INFINITY;
            for (i, v) in out.data.iter().enumerate() { if *v > bv { bv = *v; best = i; } }
            ui.add_space(4.0);
            ui.label(
                RichText::new(format!("argmax = class {best}  (p = {bv:.3})"))
                    .color(theme::TEXT)
                    .strong(),
            );
        }
    }
    if net.realtime_inference {
        ui.ctx().request_repaint_after(Duration::from_millis(80));
    }
}

fn run_simplex_predict(net: &mut NetworkInstance) {
    let Some(model) = &net.model else { return };
    let input = Tensor::new(vec![1, model.input_dim], net.inference_inputs.clone());
    let mut out = model.predict(&input);
    if net.loss_choice == Loss::CrossEntropy {
        out = neuralcabin_engine::activations::softmax_rows(&out);
    }
    net.inference_output = Some(out);
}

fn text_inference(ui: &mut egui::Ui, net: &mut NetworkInstance) {
    ui.horizontal(|ui| {
        ui.label("Temperature:");
        ui.add(egui::DragValue::new(&mut net.temperature).speed(0.01).range(0.05..=4.0));
        ui.label("Max tokens:");
        let mut m = net.max_tokens as i32;
        if ui.add(egui::DragValue::new(&mut m).range(1..=4096)).changed() {
            net.max_tokens = m.max(1) as usize;
        }
        ui.label(RichText::new(format!("vocab = {}", net.vocab.len())).color(theme::TEXT_WEAK));
    });
    ui.label(RichText::new("Prompt").color(theme::TEXT_WEAK).size(11.5));
    egui::ScrollArea::vertical().max_height(140.0).show(ui, |ui| {
        ui.add(
            egui::TextEdit::multiline(&mut net.prompt)
                .desired_width(f32::INFINITY)
                .desired_rows(4),
        );
    });
    ui.add_space(4.0);
    if ui
        .add_sized(
            [ui.available_width(), 28.0],
            egui::Button::new(RichText::new("Generate").strong()),
        )
        .clicked()
    {
        match generate_text(net) {
            Ok(out) => net.generated = out,
            Err(e) => net.generated = format!("[error] {e}"),
        }
    }
    ui.add_space(6.0);
    ui.label(RichText::new("Output").color(theme::TEXT).strong());
    egui::ScrollArea::vertical()
        .auto_shrink([false; 2])
        .show(ui, |ui| {
            ui.label(RichText::new(&net.generated).monospace().color(theme::TEXT));
        });
}

fn generate_text(net: &mut NetworkInstance) -> Result<String, String> {
    let Some(model) = &net.model else { return Err("no model".into()); };
    if net.vocab.is_empty() { return Err("vocab is empty".into()); }
    let v = net.vocab.len();
    let ctx = net.corpus.context_size.max(1);
    if model.input_dim != ctx * v {
        return Err(format!(
            "model input_dim={} but ctx*vocab={} ({} * {})",
            model.input_dim, ctx * v, ctx, v
        ));
    }
    if model.output_dim() != v {
        return Err(format!("model output_dim={} but vocab={}", model.output_dim(), v));
    }
    let mut history: Vec<usize> = net.vocab.encode(&net.prompt);
    if history.is_empty() { history.push(0); }
    let mut rng = SplitMix64::new(net.seed.wrapping_add(history.len() as u64).wrapping_add(0xBEEF));
    let mut out_tokens: Vec<usize> = Vec::new();
    let temp = net.temperature.max(1e-3);
    for _ in 0..net.max_tokens {
        // Build the input: one-hot of last `ctx` tokens (left-pad with <unk>=0).
        let mut input = vec![0.0_f32; ctx * v];
        for k in 0..ctx {
            let idx = if history.len() + k >= ctx {
                history[history.len() + k - ctx]
            } else {
                0
            };
            let id = idx.min(v - 1);
            input[k * v + id] = 1.0;
        }
        let logits = model.predict(&Tensor::new(vec![1, ctx * v], input));
        // temperature-scaled softmax
        let mut row: Vec<f32> = logits.data.iter().map(|x| x / temp).collect();
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        for r in row.iter_mut() { *r = (*r - max).exp(); }
        let sum: f32 = row.iter().sum();
        if sum <= 0.0 || !sum.is_finite() { break; }
        for r in row.iter_mut() { *r /= sum; }
        // Sample
        let u = rng.next_f32();
        let mut acc = 0.0_f32;
        let mut chosen = 0;
        for (i, p) in row.iter().enumerate() {
            acc += *p;
            if u <= acc { chosen = i; break; }
        }
        out_tokens.push(chosen);
        history.push(chosen);
    }
    // Decode: concatenate token strings. Char/subword tokens already have no separator.
    let mut out = net.prompt.clone();
    out.push('|');
    for id in out_tokens {
        if id < net.vocab.tokens.len() {
            out.push_str(&net.vocab.tokens[id]);
        }
    }
    Ok(out)
}

// ===== PLUGINS TAB =====

impl NeuralCabinApp {
    fn plugins_tab(&mut self, ui: &mut egui::Ui) {
        ui.label(RichText::new("Plugins").size(20.0).strong());
        theme::hairline(ui);
        ui.add_space(4.0);
        ui.horizontal(|ui| {
            if ui.button("📖 Open Plugin docs").clicked() {
                self.tab = Tab::Docs;
                let target_id = "plugins";
                if let Some((idx, _)) = docs::sections().iter().enumerate().find(|(_, s)| s.id == target_id) {
                    self.docs_section = idx;
                }
            }
        });
        ui.add_space(4.0);

        ui.horizontal(|ui| {
            ui.label("Path:");
            ui.add(
                egui::TextEdit::singleline(&mut self.plugins.upload_path)
                    .desired_width(420.0)
                    .hint_text("path to .zip / .json / folder"),
            );
            if ui.button("Install").clicked() {
                let path = self.plugins.upload_path.clone();
                self.plugins.install(&path);
            }
        });
        if let Some(m) = &self.plugins.message {
            ui.label(RichText::new(m).color(theme::TEXT_WEAK).size(11.5));
        }
        ui.add_space(6.0);

        egui::SidePanel::left("plugins_list_inner")
            .resizable(true)
            .default_width(240.0)
            .show_inside(ui, |ui| {
                theme::section_heading(ui, "Installed");
                let names: Vec<String> = self
                    .plugins
                    .plugins
                    .iter()
                    .map(|p| p.manifest.name.clone())
                    .collect();
                if names.is_empty() {
                    ui.label(RichText::new("No plugins installed.").color(theme::TEXT_FAINT));
                }
                let selected = self.plugins.selected;
                let mut want_select: Option<usize> = None;
                let mut want_delete: Option<usize> = None;
                for (i, name) in names.iter().enumerate() {
                    ui.horizontal(|ui| {
                        let mut text = RichText::new(format!("◆ {name}")).size(13.0);
                        if selected == Some(i) { text = text.color(theme::ACCENT).strong(); }
                        if ui
                            .add(
                                egui::Button::new(text)
                                    .frame(selected == Some(i))
                                    .min_size(egui::vec2(ui.available_width() - 28.0, 24.0)),
                            )
                            .clicked()
                        {
                            want_select = Some(i);
                        }
                        if ui.small_button("✕").clicked() { want_delete = Some(i); }
                    });
                }
                if let Some(i) = want_select { self.plugins.selected = Some(i); }
                if let Some(i) = want_delete { self.plugins.remove(i); }
            });

        egui::CentralPanel::default().show_inside(ui, |ui| {
            let Some(idx) = self.plugins.selected else {
                empty_state(ui, "Select a plugin on the left to configure it.");
                return;
            };
            if idx >= self.plugins.plugins.len() {
                self.plugins.selected = None;
                return;
            }
            let entry = &mut self.plugins.plugins[idx];
            ui.label(RichText::new(&entry.manifest.name).size(20.0).strong());
            ui.label(RichText::new(format!(
                "{} · v{} · by {}",
                entry.manifest.id,
                if entry.manifest.version.is_empty() { "?" } else { &entry.manifest.version },
                if entry.manifest.author.is_empty() { "unknown" } else { &entry.manifest.author },
            )).color(theme::TEXT_WEAK));
            theme::hairline(ui);
            if !entry.manifest.description.is_empty() {
                ui.label(RichText::new(&entry.manifest.description).color(theme::TEXT));
                ui.add_space(4.0);
            }
            ui.label(RichText::new(format!(
                "Network types: {}",
                if entry.manifest.network_types.is_empty() {
                    "(none)".into()
                } else {
                    entry.manifest.network_types.join(", ")
                }
            )).color(theme::TEXT_WEAK));
            ui.label(RichText::new(format!(
                "Manages vocab: {}    ·    Manages inference: {}",
                entry.manifest.manages_vocab, entry.manifest.manages_inference
            )).color(theme::TEXT_WEAK));
            ui.label(RichText::new(format!("Source: {}", entry.source_path.display()))
                .color(theme::TEXT_FAINT).size(11.0));

            ui.add_space(8.0);
            theme::section_heading(ui, "Settings (JSON)");
            egui::ScrollArea::vertical().max_height(220.0).show(ui, |ui| {
                ui.add(
                    egui::TextEdit::multiline(&mut entry.settings_json)
                        .desired_width(f32::INFINITY)
                        .desired_rows(8)
                        .code_editor(),
                );
            });
            ui.add_space(4.0);
            if ui.button("Validate JSON").clicked() {
                match serde_json::from_str::<serde_json::Value>(&entry.settings_json) {
                    Ok(_) => self.plugins.message = Some("Settings JSON is valid.".into()),
                    Err(e) => self.plugins.message = Some(format!("Invalid JSON: {e}")),
                }
            }
        });
    }
}

fn empty_state(ui: &mut egui::Ui, msg: &str) {
    ui.add_space(40.0);
    ui.vertical_centered(|ui| {
        ui.label(RichText::new(msg).color(theme::TEXT_WEAK));
    });
}
