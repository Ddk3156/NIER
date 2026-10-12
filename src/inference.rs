use crate::model::loader::load_model_from_file;
use crate::tensor::Tensor;
use serde::Deserialize;
use std::env;
use std::fs;
use std::process;

use crate::layers::dense::Dense;
use crate::model::sequential::{Layer, Sequential};
use std::hint::black_box;
use std::time::Instant;

/// Input JSON payload format.
#[derive(Debug, Deserialize)]
struct InputConfig {
    input: Vec<f32>,
}

/// Runs the CLI inference and benchmarking workflows.
pub fn run_cli() {
    let args: Vec<String> = env::args().collect();

    if args.len() == 2 && args[1] == "benchmark" {
        if let Err(e) = run_benchmarks() {
            eprintln!("Benchmark Error: {}", e);
            process::exit(1);
        }
        return;
    }

    if args.len() < 4 || args[1] != "infer" {
        eprintln!(
            "Usage:\n  {} infer <model_path> <input_path>\n  {} benchmark",
            args[0], args[0]
        );
        process::exit(1);
    }

    let model_path = &args[2];
    let input_path = &args[3];

    if let Err(e) = execute_inference(model_path, input_path) {
        eprintln!("Inference Error: {}", e);
        process::exit(1);
    }
}

/// Helper function performing file reads, tensor conversion, and forward pass.
pub fn execute_inference(model_path: &str, input_path: &str) -> Result<(), String> {
    // 1. Load Model from disk
    let model = load_model_from_file(model_path)?;

    // 2. Load and Parse Input JSON from disk
    let input_content = fs::read_to_string(input_path)
        .map_err(|e| format!("Failed to read input file '{}': {}", input_path, e))?;

    let input_config: InputConfig = serde_json::from_str(&input_content)
        .map_err(|e| format!("Invalid input JSON definition: {}", e))?;

    if input_config.input.is_empty() {
        return Err("Input vector cannot be empty".to_string());
    }

    // 3. Convert vector into 2D Tensor [1, input.len()]
    let input_len = input_config.input.len();
    let input_tensor = Tensor::new(input_config.input, vec![1, input_len])?;

    // 4. Run Forward Pass
    let output = model.forward(&input_tensor)?;

    // 5. Print Prediction Results
    println!("Inference Output:");
    println!("Shape: {:?}", output.shape());
    println!("Data: {:?}", output_data(&output));

    Ok(())
}

/// Helper to extract data slice from output tensor for printing.
fn output_data(tensor: &Tensor) -> Vec<f32> {
    let mut data = Vec::with_capacity(tensor.numel());
    if tensor.shape().len() == 2 {
        let rows = tensor.shape()[0];
        let cols = tensor.shape()[1];
        for r in 0..rows {
            for c in 0..cols {
                data.push(tensor.get_2d(r, c).unwrap_or(0.0));
            }
        }
    } else {
        for i in 0..tensor.numel() {
            data.push(tensor.get(i).unwrap_or(0.0));
        }
    }
    data
}

// ============================================================================
// NIER Benchmarking
// ============================================================================

/// Constructs a deterministic feedforward neural network for benchmarking.
fn build_benchmark_model(dimensions: &[usize]) -> Result<Sequential, String> {
    if dimensions.len() < 2 || dimensions.iter().any(|&d| d == 0) {
        return Err("Model dimensions must contain at least two positive values.".into());
    }

    let mut model = Sequential::new();

    for (layer_index, pair) in dimensions.windows(2).enumerate() {
        let input_size = pair[0];
        let output_size = pair[1];

        // Deterministic weights: reproducible across benchmark runs.
        let weights: Vec<f32> = (0..input_size * output_size)
            .map(|i| (((i * 17 + 13) % 101) as f32 - 50.0) / 5000.0)
            .collect();

        let bias = vec![0.01_f32; output_size];

        let weights_tensor = Tensor::new(weights, vec![input_size, output_size])?;

        let bias_tensor = Tensor::new(bias, vec![1, output_size])?;

        let dense = Dense::new(weights_tensor, bias_tensor)?;
        model.add(Layer::Dense(dense));

        // Apply ReLU after every Dense layer except the final one.
        if layer_index + 1 < dimensions.len() - 1 {
            model.add(Layer::ReLU);
        }
    }

    Ok(model)
}

/// Measures forward-pass latency for representative network sizes.
fn run_benchmarks() -> Result<(), String> {
    let cases: [(&str, &[usize], usize); 3] = [
        ("Small", &[32, 64, 10], 5_000),
        ("Medium", &[128, 256, 64, 10], 1_000),
        ("Large", &[512, 512, 256, 10], 200),
    ];

    const WARMUP_ITERATIONS: usize = 50;
    const MEASUREMENT_ROUNDS: usize = 7;

    println!("\n==================================================");
    println!("       NIER — Neural Inference Engine");
    println!("              Inference Benchmark");
    println!("==================================================");
    println!(
        "{:<10} {:<24} {:>14} {:>16}",
        "Model", "Architecture", "Median (us)", "Throughput/s"
    );
    println!("{}", "-".repeat(68));

    for (name, dimensions, iterations) in cases {
        let model = build_benchmark_model(dimensions)?;

        let input_size = dimensions[0];

        let input_data: Vec<f32> = (0..input_size)
            .map(|i| ((i % 23) as f32 - 11.0) / 10.0)
            .collect();

        let input = Tensor::new(input_data, vec![1, input_size])?;

        // Warm up before collecting timing measurements.
        for _ in 0..WARMUP_ITERATIONS {
            let output = model.forward(black_box(&input))?;
            black_box(output);
        }

        let mut latencies_us = Vec::with_capacity(MEASUREMENT_ROUNDS);

        for _ in 0..MEASUREMENT_ROUNDS {
            let start = Instant::now();

            for _ in 0..iterations {
                let output = model.forward(black_box(&input))?;
                black_box(output);
            }

            let elapsed_seconds = start.elapsed().as_secs_f64();

            let latency_us = elapsed_seconds * 1_000_000.0 / iterations as f64;

            latencies_us.push(latency_us);
        }

        // Median is less sensitive to occasional scheduling interruptions.
        latencies_us.sort_by(f64::total_cmp);

        let median_us = latencies_us[MEASUREMENT_ROUNDS / 2];
        let throughput = 1_000_000.0 / median_us;

        let architecture = dimensions
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(" -> ");

        println!(
            "{:<10} {:<24} {:>14.3} {:>16.2}",
            name, architecture, median_us, throughput
        );
    }

    println!("{}", "-".repeat(68));
    println!("Batch size: 1 | Warmup: 50 iterations per model");
    println!("Measurement rounds: 7 per model");
    println!("Metric: median forward-pass latency and derived throughput");
    println!("Note: timings include layer and tensor allocation overhead.");
    println!("==================================================\n");

    Ok(())
}

// ============================================================================
// Integration Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_execute_inference_end_to_end() {
        let res = execute_inference("src/examples/simple.json", "src/examples/input.json");
        assert!(res.is_ok());
    }

    #[test]
    fn test_execute_inference_nonexistent_files() {
        assert!(execute_inference("nonexistent_model.json", "src/examples/input.json").is_err());
        assert!(execute_inference("src/examples/simple.json", "nonexistent_input.json").is_err());
    }
}
