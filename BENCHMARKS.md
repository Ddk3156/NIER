# NIER Benchmark Results

## Purpose

These benchmarks compare baseline and optimized matrix multiplication implementations within NIER.

The optimized implementation changes the loop order from i-j-k to i-k-j to improve contiguous memory access.

## Results

| Network | Baseline median | Optimized median | Speedup | Latency reduction |
|---|---:|---:|---:|---:|
| Small (32 -> 64 -> 10) | 3.618 us | 0.929 us | 3.89x | 74.3% |
| Medium (128 -> 256 -> 64 -> 10) | 83.575 us | 6.880 us | 12.15x | 91.8% |
| Large (512 -> 512 -> 256 -> 10) | 725.294 us | 41.631 us | 17.42x | 94.3% |

## Methodology

- Baseline and optimized implementations were measured separately.
- Each reported value is the median of three run-level results.
- The benchmark uses fixed network architectures.
- Measurements include the model forward pass and associated tensor/layer overhead.
- CLI startup and JSON parsing are excluded from the timed measurement.

## Interpretation

The optimized implementation achieved up to 17.42x lower measured latency on the tested workloads.

Results depend on hardware, workload, and build configuration. These measurements do not establish performance superiority over other inference frameworks.

## Reproduction

Build the release binary:

    cargo build --release

Run the benchmark:

    cargo run --release -- benchmark
