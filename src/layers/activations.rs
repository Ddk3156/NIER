use crate::tensor::Tensor;

/// Applies the Rectified Linear Unit (ReLU) activation function element-wise.
///
/// For each element x:
/// - x < 0  => 0.0
/// - x >= 0 => x
///
/// Returns a new Tensor with the exact same shape, leaving the input unchanged.
///
/*
pub fn relu(input: &Tensor) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(input.numel());

    for idx in 0..input.numel() {
        let value = if input.shape().len() == 1 {
            // 1D tensor
            // TODO: use a general element accessor once Tensor exposes one
            return Err("ReLU currently requires a 2D tensor".to_string());
        } else if input.shape().len() == 2 {
            let cols = input.shape()[1];
            let row = idx / cols;
            let col = idx % cols;

            input.get_2d(row, col)?
        } else {
            return Err(format!(
                "ReLU currently supports only 1D or 2D tensors, got rank {}",
                input.shape().len()
            ));
        };

        data.push(value.max(0.0));
    }

    Tensor::new(data, input.shape().to_vec())
}
*/

pub fn relu(input: &Tensor) -> Result<Tensor, String> {
    let mut data = Vec::with_capacity(input.numel()); //numel = number of elements, returns the no. of elements in the tensor

    for index in 0..input.numel() {
        let value = input.get(index)?;
        data.push(value.max(0.0));
    }

    Tensor::new(data, input.shape().to_vec())
}

/// Applies the numerically stable Softmax activation function.
///
/// Supports 1D tensors `[n]` and 2D tensors `[batch_size, n]`.
/// Computes softmax independently across each row/vector:
/// formula: p_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
pub fn softmax(input: &Tensor) -> Result<Tensor, String> {
    let rank = input.shape().len();
    let numel = input.numel();

    if numel == 0 {
        return Err("Cannot apply softmax to an empty tensor".to_string());
    }

    match rank {
        1 => {
            let n = input.shape()[0];
            let mut elements = Vec::with_capacity(n);
            for i in 0..n {
                elements.push(input.get(i)?);
            }

            let max_val = elements.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = elements.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exps: f32 = exps.iter().sum();
            let probabilities: Vec<f32> = exps.iter().map(|&e| e / sum_exps).collect();

            Tensor::new(probabilities, input.shape().to_vec())
        }
        2 => {
            let rows = input.shape()[0];
            let cols = input.shape()[1];
            let mut probabilities = Vec::with_capacity(rows * cols);

            for r in 0..rows {
                let mut row_vals = Vec::with_capacity(cols);
                for c in 0..cols {
                    row_vals.push(input.get_2d(r, c)?);
                }

                let max_val = row_vals.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = row_vals.iter().map(|&x| (x - max_val).exp()).collect();
                let sum_exps: f32 = exps.iter().sum();

                for e in exps {
                    probabilities.push(e / sum_exps);
                }
            }

            Tensor::new(probabilities, input.shape().to_vec())
        }
        _ => Err(format!(
            "Softmax currently supports only 1D or 2D tensors, got rank {}",
            rank
        )),
    }
}
// ============================================================================
// Unit Tests (Add to existing mod tests in activations.rs)
// ============================================================================

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor; // Adjust import path if needed based on your module tree

    #[test]
    fn test_relu_isolation() {
        // 1. Create input tensor with mixed values
        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();

        // 2. Run ReLU forward pass
        let output = relu(&input).unwrap();

        // 3. Assert shape and clamped values using public methods
        assert_eq!(output.shape(), &[2, 2]);
        assert_eq!(output.as_slice(), &[0.0, 2.0, 0.0, 4.0]);
    }

    #[test]
    fn test_relu_mixed_values() {
        let input = Tensor::new(vec![-2.0, -0.5, 0.0, 2.0, 5.0], vec![1, 5]).unwrap();

        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[1, 5]);
        assert_eq!(
            vec![
                result.get_2d(0, 0).unwrap(),
                result.get_2d(0, 1).unwrap(),
                result.get_2d(0, 2).unwrap(),
                result.get_2d(0, 3).unwrap(),
                result.get_2d(0, 4).unwrap(),
            ],
            vec![0.0, 0.0, 0.0, 2.0, 5.0]
        );
    }

    #[test]
    fn test_relu_all_positive() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(
            vec![
                result.get_2d(0, 0).unwrap(),
                result.get_2d(0, 1).unwrap(),
                result.get_2d(1, 0).unwrap(),
                result.get_2d(1, 1).unwrap(),
            ],
            vec![1.0, 2.0, 3.0, 4.0]
        );
    }

    #[test]
    fn test_relu_all_negative() {
        let input = Tensor::new(vec![-1.0, -2.0, -3.0, -4.0], vec![2, 2]).unwrap();
        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(
            vec![
                result.get_2d(0, 0).unwrap(),
                result.get_2d(0, 1).unwrap(),
                result.get_2d(1, 0).unwrap(),
                result.get_2d(1, 1).unwrap(),
            ],
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_relu_zeros() {
        let input = Tensor::new(vec![0.0, 0.0, 0.0, 0.0], vec![2, 2]).unwrap();
        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(
            vec![
                result.get_2d(0, 0).unwrap(),
                result.get_2d(0, 1).unwrap(),
                result.get_2d(1, 0).unwrap(),
                result.get_2d(1, 1).unwrap(),
            ],
            vec![0.0, 0.0, 0.0, 0.0]
        );
    }

    #[test]
    fn test_relu_2d_tensor() {
        // Matrix:
        // [-1.0,  2.0]
        // [-3.0,  4.0]
        let input = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![2, 2]).unwrap();
        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.get_2d(0, 0).unwrap(), 0.0);
        assert_eq!(result.get_2d(0, 1).unwrap(), 2.0);
        assert_eq!(result.get_2d(1, 0).unwrap(), 0.0);
        assert_eq!(result.get_2d(1, 1).unwrap(), 4.0);
    }

    #[test]
    fn test_relu_input_unchanged_and_shape_preserved() {
        let input = Tensor::new(vec![-5.0, 10.0], vec![1, 2]).unwrap();
        let result = relu(&input).unwrap();

        // Shape preserved
        assert_eq!(result.shape(), input.shape());

        // Original input untouched
        assert_eq!(input.get_2d(0, 0).unwrap(), -5.0);
        assert_eq!(input.get_2d(0, 1).unwrap(), 10.0);
    }

    #[test]
    fn test_relu_1d_tensor() {
        let input = Tensor::new(vec![-3.0, -1.0, 0.0, 2.0, 5.0], vec![5]).unwrap();

        let result = relu(&input).unwrap();

        assert_eq!(result.shape(), &[5]);

        assert_eq!(result.get(0).unwrap(), 0.0);
        assert_eq!(result.get(1).unwrap(), 0.0);
        assert_eq!(result.get(2).unwrap(), 0.0);
        assert_eq!(result.get(3).unwrap(), 2.0);
        assert_eq!(result.get(4).unwrap(), 5.0);
    }

    #[test]
    fn test_softmax_basic_probability() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let result = softmax(&input).unwrap();

        assert_eq!(result.shape(), &[3]);

        let sum: f32 = (0..3)
            .map(|i| {
                let prob = result.get(i).unwrap();
                assert!(
                    prob >= 0.0 && prob <= 1.0,
                    "Probability must be between 0 and 1"
                );
                prob
            })
            .sum();

        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_equal_values() {
        let input = Tensor::new(vec![2.0, 2.0, 2.0], vec![3]).unwrap();
        let result = softmax(&input).unwrap();

        let p0 = result.get(0).unwrap();
        let p1 = result.get(1).unwrap();
        let p2 = result.get(2).unwrap();

        let expected = 1.0 / 3.0;
        assert!((p0 - expected).abs() < 1e-4);
        assert!((p1 - expected).abs() < 1e-4);
        assert!((p2 - expected).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_dominant_value() {
        let input = Tensor::new(vec![1.0, 1.0, 10.0], vec![3]).unwrap();
        let result = softmax(&input).unwrap();

        let p0 = result.get(0).unwrap();
        let p1 = result.get(1).unwrap();
        let p2 = result.get(2).unwrap();

        assert!(
            p2 > 0.99,
            "Dominant value should take almost all probability weight"
        );
        assert!(p0 < 0.01);
        assert!(p1 < 0.01);
    }

    #[test]
    fn test_softmax_numerical_stability() {
        let input = Tensor::new(vec![1000.0, 1001.0, 1002.0], vec![3]).unwrap();
        let result = softmax(&input).unwrap();

        let p0 = result.get(0).unwrap();
        let p1 = result.get(1).unwrap();
        let p2 = result.get(2).unwrap();

        assert!(!p0.is_nan() && !p0.is_infinite());
        assert!(!p1.is_nan() && !p1.is_infinite());
        assert!(!p2.is_nan() && !p2.is_infinite());

        let sum = p0 + p1 + p2;
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_input_unchanged() {
        let input = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let _ = softmax(&input).unwrap();

        assert_eq!(input.shape(), &[3]);
        assert_eq!(input.get(0).unwrap(), 1.0);
        assert_eq!(input.get(1).unwrap(), 2.0);
        assert_eq!(input.get(2).unwrap(), 3.0);
    }

    #[test]
    fn test_softmax_2d_single_row() {
        // Input shape [1, 2]
        let input = Tensor::new(vec![2.0, 1.0], vec![1, 2]).unwrap();
        let result = softmax(&input).unwrap();

        assert_eq!(result.shape(), &[1, 2]);

        let p0 = result.get_2d(0, 0).unwrap();
        let p1 = result.get_2d(0, 1).unwrap();

        assert!((p0 - 0.7310586).abs() < 1e-4);
        assert!((p1 - 0.2689414).abs() < 1e-4);
        assert!(((p0 + p1) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_softmax_2d_multiple_rows() {
        // Input shape [2, 3] -> 2 independent rows
        let input = Tensor::new(
            vec![
                1.0, 2.0, 3.0, // Row 0
                2.0, 2.0, 2.0, // Row 1
            ],
            vec![2, 3],
        )
        .unwrap();

        let result = softmax(&input).unwrap();
        assert_eq!(result.shape(), &[2, 3]);

        // Row 0 sum check
        let sum_row0: f32 = (0..3).map(|c| result.get_2d(0, c).unwrap()).sum();
        assert!((sum_row0 - 1.0).abs() < 1e-6);

        // Row 1 sum check & equality check
        let sum_row1: f32 = (0..3).map(|c| result.get_2d(1, c).unwrap()).sum();
        assert!((sum_row1 - 1.0).abs() < 1e-6);
        assert!((result.get_2d(1, 0).unwrap() - (1.0 / 3.0)).abs() < 1e-4);
    }

    #[test]
    fn test_softmax_2d_numerical_stability() {
        let input = Tensor::new(vec![1000.0, 1001.0], vec![1, 2]).unwrap();
        let result = softmax(&input).unwrap();

        let p0 = result.get_2d(0, 0).unwrap();
        let p1 = result.get_2d(0, 1).unwrap();

        assert!(!p0.is_nan() && !p0.is_infinite());
        assert!(!p1.is_nan() && !p1.is_infinite());
        assert!(((p0 + p1) - 1.0).abs() < 1e-6);
    }
}
