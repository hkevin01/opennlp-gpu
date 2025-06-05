/**
 * OpenCL kernels for matrix operations used in OpenNLP GPU acceleration.
 */

/**
 * Matrix multiplication kernel: C = A * B
 * 
 * @param A Input matrix A (m x k)
 * @param B Input matrix B (k x n)
 * @param C Output matrix C (m x n)
 * @param M Number of rows in A and C
 * @param N Number of columns in B and C
 * @param K Number of columns in A and rows in B
 */
__kernel void matrixMultiply(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int M,
    const int N,
    const int K)
{
    // Get global position in the output matrix
    const int row = get_global_id(0);  // Row in C
    const int col = get_global_id(1);  // Column in C
    
    // Check if we're within bounds
    if (row < M && col < N) {
        float sum = 0.0f;
        
        // Compute dot product of row of A and column of B
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Store the result
        C[row * N + col] = sum;
    }
}

/**
 * Matrix addition kernel: C = A + B
 * 
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param size Number of elements in the matrices
 */
__kernel void matrixAdd(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    // Get global position
    const int i = get_global_id(0);
    
    // Check if we're within bounds
    if (i < size) {
        C[i] = A[i] + B[i];
    }
}

/**
 * Element-wise matrix scalar multiplication: B = A * scalar
 * 
 * @param A Input matrix A
 * @param B Output matrix B
 * @param scalar Scalar value to multiply by
 * @param size Number of elements in the matrices
 */
__kernel void matrixScalarMultiply(
    __global const float* A,
    __global float* B,
    const float scalar,
    const int size)
{
    // Get global position
    const int i = get_global_id(0);
    
    // Check if we're within bounds
    if (i < size) {
        B[i] = A[i] * scalar;
    }
}

/**
 * Element-wise matrix subtraction: C = A - B
 * 
 * @param A Input matrix A
 * @param B Input matrix B
 * @param C Output matrix C
 * @param size Number of elements in the matrices
 */
__kernel void matrixSubtract(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int size)
{
    // Get global position
    const int i = get_global_id(0);
    
    // Check if we're within bounds
    if (i < size) {
        C[i] = A[i] - B[i];
    }
}

/**
 * Element-wise matrix-vector multiplication: C[i,j] = A[i,j] * B[j]
 * This is useful for scaling columns of a matrix by a vector.
 * 
 * @param A Input matrix A (m x n)
 * @param B Input vector B (n)
 * @param C Output matrix C (m x n)
 * @param m Number of rows in A and C
 * @param n Number of columns in A and C, and length of B
 */
__kernel void matrixColumnScale(
    __global const float* A,
    __global const float* B,
    __global float* C,
    const int m,
    const int n)
{
    // Get global position
    const int row = get_global_id(0);
    const int col = get_global_id(1);
    
    // Check if we're within bounds
    if (row < m && col < n) {
        C[row * n + col] = A[row * n + col] * B[col];
    }
}
