//! NeuralCabin UI — an egui-based desktop workbench.
//!
//! Public surface: [`run`] launches the application; everything else is an
//! internal implementation detail.

mod app;
mod trainer;
mod plot;

pub use app::NeuralCabinApp;

/// Launch the desktop application. Blocks until the window is closed.
pub fn run() -> eframe::Result<()> {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("NeuralCabin")
            .with_inner_size([1200.0, 780.0])
            .with_min_inner_size([900.0, 600.0]),
        ..Default::default()
    };
    eframe::run_native(
        "NeuralCabin",
        native_options,
        Box::new(|cc| Ok(Box::new(NeuralCabinApp::new(cc)))),
    )
}
