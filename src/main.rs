//! NeuralCabin entry point.
//!
//! Default behaviour launches the interactive workbench. Pass `--xor-demo`
//! for a headless training-and-inference smoke test that exercises the engine
//! without opening a window — useful for CI environments without a display.

use std::process::ExitCode;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--xor-demo" || a == "--demo") {
        match xor_demo() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("xor demo failed: {e}");
                ExitCode::FAILURE
            }
        }
    } else if args.iter().any(|a| a == "--help" || a == "-h") {
        println!("{}", help_text());
        ExitCode::SUCCESS
    } else {
        match neuralcabin_ui::run() {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => {
                eprintln!("UI failed to start: {e}\n\
                          Run with `--xor-demo` for a headless smoke test.");
                ExitCode::FAILURE
            }
        }
    }
}

fn help_text() -> &'static str {
    "NeuralCabin — pure-Rust neural network workbench.\n\n\
     Usage:\n\
     \tneuralcabin              Launch the interactive workbench (default).\n\
     \tneuralcabin --xor-demo   Train an MLP on XOR headlessly and print results.\n\
     \tneuralcabin --help       Show this help."
}

fn xor_demo() -> Result<(), Box<dyn std::error::Error>> {
    use neuralcabin_engine::nn::{LayerSpec, Model};
    use neuralcabin_engine::optimizer::{Optimizer, OptimizerKind};
    use neuralcabin_engine::tensor::Tensor;
    use neuralcabin_engine::{Activation, Loss};

    let specs = vec![
        LayerSpec::Linear { in_dim: 2, out_dim: 8 },
        LayerSpec::Activation(Activation::Tanh),
        LayerSpec::Linear { in_dim: 8, out_dim: 1 },
        LayerSpec::Activation(Activation::Sigmoid),
    ];
    let mut model = Model::from_specs(2, &specs, 42);
    let mut opt = Optimizer::new(
        OptimizerKind::Adam { lr: 0.05, beta1: 0.9, beta2: 0.999, eps: 1e-8 },
        &model.parameter_shapes(),
    );
    let x = Tensor::new(vec![4, 2], vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
    let y = Tensor::new(vec![4, 1], vec![0.0, 1.0, 1.0, 0.0]);
    println!("Training XOR MLP...");
    let mut last = f32::INFINITY;
    for epoch in 1..=2000 {
        last = model.train_step(&mut opt, Loss::MeanSquaredError, &x, &y);
        if epoch % 200 == 0 {
            println!("  epoch {epoch:>5}  loss = {last:.6}");
        }
    }
    println!("Final loss = {last:.6}");
    let pred = model.predict(&x);
    println!("Predictions:");
    for (i, p) in pred.data.iter().enumerate() {
        let xi = &x.data[i * 2..(i + 1) * 2];
        let ti = y.data[i];
        println!("  XOR({}, {}) -> {p:.4}  (target {ti})", xi[0], xi[1]);
    }
    if last > 0.05 {
        return Err(format!("convergence check failed: final loss {last} > 0.05").into());
    }
    Ok(())
}
