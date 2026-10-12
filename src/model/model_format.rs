use serde::{Deserialize, Serialize};

/// Outer model schema for serialized NIER model files.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelConfig {
    pub format_version: u32,
    pub layers: Vec<LayerConfig>,
}

/// Enum representing serialized layer variants in JSON.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "lowercase", deny_unknown_fields)]
pub enum LayerConfig {
    Dense {
        weights: Vec<Vec<f32>>,
        bias: Vec<f32>,
    },
    Relu,
    Softmax,
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deserialize_valid_contract_example_json() {
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
                },
                {
                    "type": "softmax"
                }
            ]
        }"#;

        let config: ModelConfig = serde_json::from_str(json_data).unwrap();

        assert_eq!(config.format_version, 1);
        assert_eq!(config.layers.len(), 4);
    }

    #[test]
    fn test_reject_unknown_layer_type() {
        let json = r#"{ "format_version": 1, "layers": [{ "type": "sigmoid" }] }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }

    #[test]
    fn test_reject_missing_format_version() {
        let json = r#"{ "layers": [{ "type": "relu" }] }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }

    #[test]
    fn test_reject_missing_dense_weights() {
        let json = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "bias": [0.1] }]
        }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }

    #[test]
    fn test_reject_missing_dense_bias() {
        let json = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": [[1.0]] }]
        }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }

    #[test]
    fn test_reject_wrong_weights_type() {
        let json = r#"{
            "format_version": 1,
            "layers": [{ "type": "dense", "weights": "invalid", "bias": [0.1] }]
        }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }

    #[test]
    fn test_reject_unknown_fields() {
        let json = r#"{
            "format_version": 1,
            "unknown_field": true,
            "layers": []
        }"#;
        assert!(serde_json::from_str::<ModelConfig>(json).is_err());
    }
}
