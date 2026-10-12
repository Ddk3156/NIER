use crate::layers::dense::Dense;
use crate::model::model_format::{LayerConfig, ModelConfig};
use crate::model::sequential::{Layer, Sequential};
use crate::tensor::Tensor;
use std::fs;
use std::path::Path;

pub const SUPPORTED_FORMAT_VERSION: u32 = 1;

/// Loads a `Sequential` model from a JSON file path.
pub fn load_model_from_file<P: AsRef<Path>>(path: P) -> Result<Sequential, String> {
    let content = fs::read_to_string(&path)
        .map_err(|e| format!("Failed to read model file '{:?}': {}", path.as_ref(), e))?;
    load_model_from_str(&content)
}

/// Parses a model JSON string into a validated `Sequential` model.
pub fn load_model_from_str(json_str: &str) -> Result<Sequential, String> {
    let config: ModelConfig = serde_json::from_str(json_str)
        .map_err(|e| format!("Invalid JSON model definition: {}", e))?;

    if config.format_version != SUPPORTED_FORMAT_VERSION {
        return Err(format!(
            "Unsupported format_version {}. Expected version {}",
            config.format_version, SUPPORTED_FORMAT_VERSION
        ));
    }

    let mut model = Sequential::new();

    for (idx, layer_cfg) in config.layers.into_iter().enumerate() {
        match layer_cfg {
            LayerConfig::Dense { weights, bias } => {
                let dense_layer = parse_dense_layer(weights, bias)
                    .map_err(|e| format!("Error in layer {}: {}", idx, e))?;
                model.add(Layer::Dense(dense_layer));
            }
            LayerConfig::Relu => {
                model.add(Layer::ReLU);
            }
            LayerConfig::Softmax => {
                model.add(Layer::Softmax);
            }
        }
    }

    // System Validation: Verify inter-layer feature compatibility across the pipeline
    validate_layer_compatibility(&model)?;

    Ok(model)
}

/// Validates that output feature dimensions of consecutive Dense layers match across activations.
fn validate_layer_compatibility(model: &Sequential) -> Result<(), String> {
    let mut last_out_features: Option<(usize, usize)> = None; // (layer_index, out_features)

    // Inspect layers held inside the Sequential model
    for (idx, layer) in model.layers().iter().enumerate() {
        if let Layer::Dense(dense) = layer {
            let in_features = dense.weights().shape()[0];
            let out_features = dense.weights().shape()[1];

            if let Some((prev_idx, prev_out_features)) = last_out_features {
                if prev_out_features != in_features {
                    return Err(format!(
                        "Feature dimension mismatch between Layer {} (outputs {} features) and Layer {} (expects {} features)",
                        prev_idx, prev_out_features, idx, in_features
                    ));
                }
            }

            last_out_features = Some((idx, out_features));
        }
    }

    Ok(())
}

/// Validates semantic dimensions and parses nested weight arrays into row-major Tensor format.
fn parse_dense_layer(weights_2d: Vec<Vec<f32>>, bias_1d: Vec<f32>) -> Result<Dense, String> {
    let rows = weights_2d.len();
    if rows == 0 {
        return Err("Weights matrix cannot be empty (0 rows)".to_string());
    }

    let cols = weights_2d[0].len();
    if cols == 0 {
        return Err("Weights row cannot be empty (0 columns)".to_string());
    }

    if bias_1d.is_empty() {
        return Err("Bias vector cannot be empty".to_string());
    }

    if bias_1d.len() != cols {
        return Err(format!(
            "Bias length ({}) does not match weight output columns ({})",
            bias_1d.len(),
            cols
        ));
    }

    // Convert row-major nested arrays into flattened vector
    let mut flattened_weights = Vec::with_capacity(rows * cols);
    for row in weights_2d {
        if row.len() != cols {
            return Err(format!(
                "Inconsistent weight row length: expected {} columns, got {}",
                cols,
                row.len()
            ));
        }
        flattened_weights.extend(row);
    }

    let weight_tensor = Tensor::new(flattened_weights, vec![rows, cols])?;
    let bias_len = bias_1d.len();
    let bias_tensor = Tensor::new(bias_1d, vec![1, bias_len])?;

    Dense::new(weight_tensor, bias_tensor)
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;
    use std::fs::File;
    use std::io::Write;

    #[test]
    fn test_load_valid_model_and_run_inference() {
        let json_data = r#"{
            "format_version": 1,
            "layers": [
                {
                    "type": "dense",
                    "weights": [[0.2, 0.8], [0.5, -0.3]],
                    "bias": [0.1, 0.2]
                },
                {
                    "type": "relu"
                },
                {
                    "type": "dense",
                    "weights": [[0.4], [0.7]],
                    "bias": [0.05]
                }
            ]
        }"#;

        let model = load_model_from_str(json_data).unwrap();
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();

        let output = model.forward(&input).unwrap();
        assert_eq!(output.shape(), &[1, 1]);

        let val = output.get_2d(0, 0).unwrap();
        assert!((val - 0.85).abs() < 1e-5);
    }

    #[test]
    fn test_load_from_file_valid_and_missing() {
        let json_data = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": [[1.0]], "bias": [0.0] }]
        }"#;

        // 1. Create temporary file
        let mut temp_path = env::temp_dir();
        temp_path.push("nier_test_valid_model.json");

        let mut file = File::create(&temp_path).unwrap();
        file.write_all(json_data.as_bytes()).unwrap();

        // 2. Load model from valid file
        let model_res = load_model_from_file(&temp_path);
        assert!(model_res.is_ok());

        // Cleanup
        let _ = fs::remove_file(&temp_path);

        // 3. Test non-existent file path
        let nonexistent_path = env::temp_dir().join("nonexistent_model_12345.json");
        let err = load_model_from_file(nonexistent_path).unwrap_err();
        assert!(err.contains("Failed to read model file"));
    }

    #[test]
    fn test_reject_incompatible_consecutive_dense_layers() {
        // Dense 1 outputs 3 features, but Dense 2 expects 4 features
        let json_data = r#"{
            "format_version": 1,
            "layers": [
                {
                    "type": "dense",
                    "weights": [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
                    "bias": [0.1, 0.1, 0.1]
                },
                {
                    "type": "relu"
                },
                {
                    "type": "dense",
                    "weights": [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]],
                    "bias": [0.1, 0.1]
                }
            ]
        }"#;

        let err = load_model_from_str(json_data).unwrap_err();
        assert!(err.contains("Feature dimension mismatch between Layer 0 (outputs 3 features) and Layer 2 (expects 4 features)"));
    }

    #[test]
    fn test_reject_empty_weight_matrix_or_rows() {
        let json_empty_matrix = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": [], "bias": [0.1] }]
        }"#;
        assert!(load_model_from_str(json_empty_matrix).is_err());

        let json_empty_row = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": [[]], "bias": [] }]
        }"#;
        assert!(load_model_from_str(json_empty_row).is_err());
    }

    #[test]
    fn test_reject_empty_bias_vector() {
        let json_data = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": [[1.0, 2.0]], "bias": [] }]
        }"#;
        let err = load_model_from_str(json_data).unwrap_err();
        assert!(err.contains("Bias vector cannot be empty"));
    }
}
