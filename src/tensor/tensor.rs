/// Fundamental Tensor data structure for NIER.
#[derive(Debug, Clone, PartialEq)]
pub struct Tensor {
    data: Vec<f32>,
    shape: Vec<usize>,
}

impl Tensor {
    /// Creates a new Tensor after validating that data length matches shape dimensions.
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> Result<Self, String> {
        let expected_len: usize = shape.iter().product();

        if data.len() != expected_len {
            return Err(format!(
                "Invalid shape: expected {} elements for shape {:?}, but got {}",
                expected_len,
                shape,
                data.len()
            ));
        }

        Ok(Self { data, shape })
    }

    /// Returns the shape of the tensor as a slice.
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    /// Returns the total number of numerical elements in the tensor.
    pub fn numel(&self) -> usize {
        self.data.len()
    }

    /// Retrieves an element at (row, column) for 2D tensors.
    pub fn get_2d(&self, row: usize, col: usize) -> Result<f32, String> {
        // Ensure the tensor is 2-dimensional
        if self.shape.len() != 2 {
            return Err(format!(
                "Expected 2D tensor, but tensor has rank {}",
                self.shape.len()
            ));
        }

        let rows = self.shape[0];
        let cols = self.shape[1];

        // Bounds check
        if row >= rows || col >= cols {
            return Err(format!(
                "Index ({}, {}) out of bounds for tensor with shape [{}, {}]",
                row, col, rows, cols
            ));
        }

        // Map 2D (row, col) into flat 1D vector index
        let flat_index = row * cols + col;
        Ok(self.data[flat_index])
    }

    /// Returns an element by its flat index.
    pub fn get(&self, index: usize) -> Result<f32, String> {
        if index >= self.data.len() {
            return Err(format!(
                "Index {} out of bounds for tensor with {} elements",
                index,
                self.data.len()
            ));
        }

        Ok(self.data[index])
    }

    /// Performs element-wise addition between two tensors.
    ///
    /// Both tensors must have the exact same shape.
    pub fn add(&self, other: &Tensor) -> Result<Tensor, String> {
        // 1. Validate that shapes match exactly
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch for addition: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        // 2. Add corresponding elements pairwise
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        // 3. Construct and return the new Tensor
        Tensor::new(data, self.shape.clone())
    }

    /// Performs element-wise multiplication (Hadamard product) between two tensors.
    ///
    /// Both tensors must have the exact same shape.
    pub fn mul(&self, other: &Tensor) -> Result<Tensor, String> {
        // 1. Validate that shapes match exactly
        if self.shape != other.shape {
            return Err(format!(
                "Shape mismatch for multiplication: {:?} vs {:?}",
                self.shape, other.shape
            ));
        }

        // 2. Multiply corresponding elements pairwise
        let data: Vec<f32> = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect();

        // 3. Construct and return the new Tensor
        Tensor::new(data, self.shape.clone())
    }

    /// Performs 2D matrix multiplication (A x B).
    ///
    /// Requires both tensors to be 2D with matching inner dimensions:
    /// self.shape = [m, n], other.shape = [n, p] -> result shape = [m, p].
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor, String> {
        // 1. Ensure both inputs are 2D
        if self.shape.len() != 2 || other.shape.len() != 2 {
            return Err(format!(
                "matmul requires 2D tensors, got ranks {} and {}",
                self.shape.len(),
                other.shape.len()
            ));
        }

        let m = self.shape[0];
        let n1 = self.shape[1];
        let n2 = other.shape[0];
        let p = other.shape[1];

        // 2. Validate inner dimensions (n1 == n2)
        if n1 != n2 {
            return Err(format!(
                "Incompatible matrix dimensions for matmul: [{m}, {n1}] x [{n2}, {p}]"
            ));
        }

        // 3. Compute C[i, j] = sum_k (A[i, k] * B[k, j])
        let mut result_data = Vec::with_capacity(m * p);

        for i in 0..m {
            for j in 0..p {
                let mut sum = 0.0;
                for k in 0..n1 {
                    let a_val = self.data[i * n1 + k];
                    let b_val = other.data[k * p + j];
                    sum += a_val * b_val;
                }
                result_data.push(sum);
            }
        }

        // 4. Construct result tensor with shape [m, p]
        Tensor::new(result_data, vec![m, p])
    }

    pub fn as_slice(&self) -> &[f32] {
        &self.data
    }
}

// ============================================================================
// Unit Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_tensor_creation() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]);
        assert!(tensor.is_ok());
    }

    #[test]
    fn test_invalid_shape_rejected() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0], vec![2, 3]);
        assert!(tensor.is_err());
    }

    #[test]
    fn test_correct_shape_returned() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(tensor.shape(), &[2, 3]);
    }

    #[test]
    fn test_correct_element_count() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        assert_eq!(tensor.numel(), 6);
    }

    #[test]
    fn test_correct_element_retrieved() {
        // [ 1.0, 2.0, 3.0 ]
        // [ 4.0, 5.0, 6.0 ]
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        assert_eq!(tensor.get_2d(0, 2), Ok(3.0));
        assert_eq!(tensor.get_2d(1, 0), Ok(4.0));
    }

    #[test]
    fn test_out_of_bounds_access_handled() {
        let tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        assert!(tensor.get_2d(10, 2).is_err());
        assert!(tensor.get_2d(0, 5).is_err());
    }

    #[test]
    fn test_add_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = a.add(&b).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data, vec![5.0, 7.0, 9.0]);
    }

    #[test]
    fn test_add_mismatched_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();

        assert!(a.add(&b).is_err());
    }

    #[test]
    fn test_add_2d_tensors() {
        // Matrix A:
        // [1.0, 2.0]
        // [3.0, 4.0]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        // Matrix B:
        // [10.0, 20.0]
        // [30.0, 40.0]
        let b = Tensor::new(vec![10.0, 20.0, 30.0, 40.0], vec![2, 2]).unwrap();

        let result = a.add(&b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data, vec![11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn test_mul_same_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let result = a.mul(&b).unwrap();

        assert_eq!(result.shape(), &[3]);
        assert_eq!(result.data, vec![4.0, 10.0, 18.0]);
    }

    #[test]
    fn test_mul_mismatched_shape() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0], vec![2]).unwrap();

        assert!(a.mul(&b).is_err());
    }

    #[test]
    fn test_mul_2d_tensors() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 0.5, 3.0, 0.25], vec![2, 2]).unwrap();

        let result = a.mul(&b).unwrap();

        assert_eq!(result.shape(), &[2, 2]);
        assert_eq!(result.data, vec![2.0, 1.0, 9.0, 1.0]);
    }

    #[test]
    fn test_matmul_2x2_by_2x2() {
        // A = [[1, 2], [3, 4]]
        // B = [[2, 0], [1, 2]]
        // C = [[4, 4], [10, 8]]
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();
        let b = Tensor::new(vec![2.0, 0.0, 1.0, 2.0], vec![2, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data, vec![4.0, 4.0, 10.0, 8.0]);

        // Verify inputs remain unchanged
        assert_eq!(a.data, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(b.data, vec![2.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_matmul_2x3_by_3x2() {
        // A = [[1, 2, 3],
        //      [4, 5, 6]]  (2x3)
        // B = [[7, 8],
        //      [9, 1],
        //      [2, 3]]     (3x2)
        // C = [[31, 19],
        //      [85, 55]]   (2x2)
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![7.0, 8.0, 9.0, 1.0, 2.0, 3.0], vec![3, 2]).unwrap();

        let c = a.matmul(&b).unwrap();

        assert_eq!(c.shape(), &[2, 2]);
        assert_eq!(c.data, vec![31.0, 19.0, 85.0, 55.0]);
    }

    #[test]
    fn test_matmul_incompatible_dimensions() {
        // [2, 3] x [2, 3] -> Inner dimensions (3 vs 2) mismatch
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();
        let b = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], vec![2, 3]).unwrap();

        assert!(a.matmul(&b).is_err());
    }

    #[test]
    fn test_matmul_non_2d_rejection() {
        let one_tensor = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let two_tensor = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]).unwrap();

        assert!(one_tensor.matmul(&two_tensor).is_err());
        assert!(two_tensor.matmul(&one_tensor).is_err());
    }

    #[test]
    fn test_add_inputs_unchanged() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let _ = a.add(&b).unwrap();

        // Verify neither a nor b was mutated
        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[3]);
    }

    #[test]
    fn test_mul_inputs_unchanged() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]).unwrap();
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]).unwrap();

        let _ = a.mul(&b).unwrap();

        // Verify neither a nor b was mutated
        assert_eq!(a.shape(), &[3]);
        assert_eq!(b.shape(), &[3]);
    }
}
