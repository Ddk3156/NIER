use crate::tensor::Tensor;

/// A Dense (Fully Connected) neural network layer.
#[derive(Debug, Clone, PartialEq)]
pub struct Dense {
    weights: Tensor,
    bias: Tensor,
}

impl Dense {
    /// Creates a new Dense layer from weights and bias tensors.
    ///
    /// Validation:
    /// - Weights must be a 2D matrix [in_features, out_features]
    /// - Bias shape must match weights output dimension [1, out_features]
    pub fn new(weights: Tensor, bias: Tensor) -> Result<Self, String> {
        if weights.shape().len() != 2 {
            return Err(format!(
                "Dense weights must be 2D, but got shape {:?}",
                weights.shape()
            ));
        }

        if bias.shape().len() != 2 || bias.shape()[0] != 1 {
            return Err(format!(
                "Dense bias must be 2D with shape [1, out_features], but got shape {:?}",
                bias.shape()
            ));
        }

        let out_features = weights.shape()[1];
        if bias.shape()[1] != out_features {
            return Err(format!(
                "Dimension mismatch: weights output dimension ({}) does not match bias dimension ({})",
                out_features,
                bias.shape()[1]
            ));
        }

        Ok(Self { weights, bias })
    }

    /// Performs the forward operation: output = input * weights + bias
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        // Step 1: Matrix multiplication (input * weights)
        let linear = input.matmul(&self.weights)?;

        // Step 2: Add bias
        linear.add(&self.bias)
    }

    /// Accessor for weights.
    pub fn weights(&self) -> &Tensor {
        &self.weights
    }

    /// Accessor for bias.
    pub fn bias(&self) -> &Tensor {
        &self.bias
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dense_forward_basic() {
        // Input: [1, 2] (shape [1, 2])
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();

        // Weights: [[3, 4], [5, 6]] (shape [2, 2])
        let weights = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();

        // Bias: [1, 1] (shape [1, 2])
        let bias = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();

        let layer = Dense::new(weights, bias).unwrap();
        let output = layer.forward(&input).unwrap();

        // [1, 2] * [[3, 4], [5, 6]] = [13, 16]
        // [13, 16] + [1, 1] = [14, 17]
        assert_eq!(output.shape(), &[1, 2]);
        assert_eq!(output.get_2d(0, 0).unwrap(), 14.0);
        assert_eq!(output.get_2d(0, 1).unwrap(), 17.0);
    }

    #[test]
    fn test_dense_output_shape() {
        // Input: [1, 3]
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        // Weights: [3, 2]
        let weights = Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap();
        // Bias: [1, 2]
        let bias = Tensor::new(vec![0.5, 0.5], vec![1, 2]).unwrap();

        let layer = Dense::new(weights, bias).unwrap();
        let output = layer.forward(&input).unwrap();

        assert_eq!(output.shape(), &[1, 2]);
    }

    #[test]
    fn test_dense_dimension_mismatch() {
        // Weights output feature dimension is 2
        let weights = Tensor::new(vec![1.0; 6], vec![3, 2]).unwrap();
        // Bias feature dimension is 3 (incompatible)
        let bias = Tensor::new(vec![1.0; 3], vec![1, 3]).unwrap();

        assert!(Dense::new(weights, bias).is_err());
    }

    #[test]
    fn test_dense_input_unchanged() {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();
        let weights = Tensor::new(vec![3.0, 4.0, 5.0, 6.0], vec![2, 2]).unwrap();
        let bias = Tensor::new(vec![1.0, 1.0], vec![1, 2]).unwrap();

        let layer = Dense::new(weights, bias).unwrap();
        let _ = layer.forward(&input).unwrap();

        // Verify input remains untouched
        assert_eq!(input.get_2d(0, 0).unwrap(), 1.0);
        assert_eq!(input.get_2d(0, 1).unwrap(), 2.0);
    }
}
