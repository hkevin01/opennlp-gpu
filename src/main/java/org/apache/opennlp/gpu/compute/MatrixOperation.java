package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Interface defining operations for matrix computations.
 * Implementations can provide CPU or GPU-accelerated versions.
 */
public interface MatrixOperation {
    
    /**
     * Gets the compute provider associated with this matrix operation.
     *
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Multiplies matrices A and B to produce matrix C.
     *
     * @param a matrix A as flattened array
     * @param b matrix B as flattened array
     * @param c result matrix C as flattened array
     * @param rowsA number of rows in matrix A
     * @param colsB number of columns in matrix B
     * @param sharedDim shared dimension (columns of A, rows of B)
     */
    void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim);
    
    /**
     * Adds matrices A and B to produce matrix C.
     *
     * @param a matrix A as flattened array
     * @param b matrix B as flattened array
     * @param c result matrix C as flattened array
     * @param elements number of elements in each matrix
     */
    void add(float[] a, float[] b, float[] c, int elements);
    
    /**
     * Subtracts matrix B from matrix A to produce matrix C.
     *
     * @param a matrix A as flattened array
     * @param b matrix B as flattened array
     * @param c result matrix C as flattened array
     * @param elements number of elements in each matrix
     */
    void subtract(float[] a, float[] b, float[] c, int elements);
    
    /**
     * Multiplies matrix A by a scalar value to produce matrix B.
     *
     * @param a input matrix A as flattened array
     * @param b result matrix B as flattened array
     * @param scalar the scalar value to multiply by
     * @param elements number of elements in the matrix
     */
    void scalarMultiply(float[] a, float[] b, float scalar, int elements);
    
    /**
     * Transposes matrix A to produce matrix B.
     *
     * @param a input matrix A as flattened array
     * @param b result matrix B as flattened array
     * @param rows number of rows in matrix A
     * @param cols number of columns in matrix A
     */
    void transpose(float[] a, float[] b, int rows, int cols);
    
    /**
     * Releases any resources used by this matrix operation.
     */
    void release();
}
