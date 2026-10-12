use crate::layers::activations::{relu, softmax};
use crate::layers::dense::Dense;
use crate::tensor::Tensor;

/// Represents the layer operations supported by NIER.
#[derive(Debug, Clone, PartialEq)]
pub enum Layer {
    Dense(Dense),
    ReLU,
    Softmax,
}

/// Sequential container that executes layers in order.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Sequential {
    layers: Vec<Layer>,
}

impl Sequential {
    /// Creates a new empty Sequential model.
    pub fn new() -> Self {
        Self { layers: Vec::new() }
    }

    /// Appends a layer to the model pipeline.
    pub fn add(&mut self, layer: Layer) {
        self.layers.push(layer);
    }

    /// Executes the forward pass through all layers sequentially.
    ///
    /// Empty models return the input unchanged (identity operation).
    pub fn forward(&self, input: &Tensor) -> Result<Tensor, String> {
        let mut current_output = input.clone();

        for layer in &self.layers {
            current_output = match layer {
                Layer::Dense(dense) => dense.forward(&current_output)?,
                Layer::ReLU => relu(&current_output)?,
                Layer::Softmax => softmax(&current_output)?,
            };
        }

        Ok(current_output)
    }

    /// Returns a slice of the layers inside the model.
    pub fn layers(&self) -> &[Layer] {
        &self.layers
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequential_forward_execution_and_values() {
        // Build network:
        // Input [1, 2]
        // Dense 1: W1 = [[1, 0, -1], [0, 1, 1]], b1 = [0, 0, 0]
        //   -> [1, 2] x W1 = [1, 2, 1]
        // ReLU -> [1, 2, 1]
        // Dense 2: W2 = [[1, -1], [0, 1], [1, 0]], b2 = [0, 0]
        //   -> [1, 2, 1] x W2 = [2, 1]
        // Softmax -> softmax([2, 1])

        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();

        let w1 = Tensor::new(vec![1.0, 0.0, -1.0, 0.0, 1.0, 1.0], vec![2, 3]).unwrap();
        let b1 = Tensor::new(vec![0.0, 0.0, 0.0], vec![1, 3]).unwrap();
        let dense1 = Dense::new(w1, b1).unwrap();

        let w2 = Tensor::new(vec![1.0, -1.0, 0.0, 1.0, 1.0, 0.0], vec![3, 2]).unwrap();
        let b2 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();
        let dense2 = Dense::new(w2, b2).unwrap();

        let mut model = Sequential::new();
        model.add(Layer::Dense(dense1));
        model.add(Layer::ReLU);
        model.add(Layer::Dense(dense2));
        model.add(Layer::Softmax);

        let output = model.forward(&input).unwrap();

        // Check output shape
        assert_eq!(output.shape(), &[1, 2]);

        // Expected output for softmax([2.0, 1.0]):
        // e^2 / (e^2 + e^1) approx 0.7310586
        // e^1 / (e^2 + e^1) approx 0.2689414
        let p0 = output.get_2d(0, 0).unwrap();
        let p1 = output.get_2d(0, 1).unwrap();

        assert!((p0 - 0.7310586).abs() < 1e-4);
        assert!((p1 - 0.2689414).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_validity() {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();

        let w1 = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b1 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();

        let mut model = Sequential::new();
        model.add(Layer::Dense(Dense::new(w1, b1).unwrap()));
        model.add(Layer::Softmax);

        let output = model.forward(&input).unwrap();

        let p0 = output.get_2d(0, 0).unwrap();
        let p1 = output.get_2d(0, 1).unwrap();

        assert!(p0 >= 0.0 && p0 <= 1.0);
        assert!(p1 >= 0.0 && p1 <= 1.0);
        assert!(((p0 + p1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_error_propagation_on_incompatible_input() {
        // Input has 3 features [1, 3] but Dense expects 2 features [2, 2]
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();

        let w1 = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b1 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();

        let mut model = Sequential::new();
        model.add(Layer::Dense(Dense::new(w1, b1).unwrap()));

        assert!(model.forward(&input).is_err());
    }

    #[test]
    fn test_input_preservation() {
        let input = Tensor::new(vec![1.0, 2.0], vec![1, 2]).unwrap();

        let w1 = Tensor::new(vec![1.0, 0.0, 0.0, 1.0], vec![2, 2]).unwrap();
        let b1 = Tensor::new(vec![0.0, 0.0], vec![1, 2]).unwrap();

        let mut model = Sequential::new();
        model.add(Layer::Dense(Dense::new(w1, b1).unwrap()));

        let _ = model.forward(&input).unwrap();

        // Verify original input was untouched
        assert_eq!(input.shape(), &[1, 2]);
        assert_eq!(input.get_2d(0, 0).unwrap(), 1.0);
        assert_eq!(input.get_2d(0, 1).unwrap(), 2.0);
    }

    #[test]
    fn test_empty_model_identity_behavior() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![1, 3]).unwrap();
        let model = Sequential::new();

        let output = model.forward(&input).unwrap();

        assert_eq!(output, input);
    }
}
