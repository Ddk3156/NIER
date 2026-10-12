# NIER — Neural Inference Engine in Rust

A lightweight neural network inference engine built from scratch in Rust.

NIER explores neural network inference at the systems level, implementing tensor operations, neural network layers, sequential execution, JSON model loading, and performance benchmarking.

## Features

- Tensor operations and matrix multiplication
- Dense layers with weights and bias
- ReLU and numerically stable Softmax activations
- Sequential neural network execution
- JSON model loading and validation
- Command-line inference
- Python-based correctness testing
- Baseline versus optimized matrix multiplication benchmarks

## Architecture

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the detailed architecture.

## Build

    cargo build --release

## Run Inference

    cargo run --release -- infer src/examples/simple.json src/examples/input.json

## Run Tests

    cargo test

## Python Correctness Tests

    python3 src/benchmark/correctness.py

## Performance Results

| Network | Baseline | Optimized | Speedup |
|---|---:|---:|---:|
| Small | 3.618 µs | 0.929 µs | 3.89x |
| Medium | 83.575 µs | 6.880 µs | 12.15x |
| Large | 725.294 µs | 41.631 µs | 17.42x |

These measurements apply to the tested workloads on the development machine. They are not comparisons against external inference frameworks.

## Limitations

- CPU-based inference
- Limited supported layer types
- Not intended to replace production inference frameworks

## License

No license has been specified. Reuse and redistribution permissions should not be assumed.
