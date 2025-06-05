package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Interface for matrix operations that can be accelerated on different hardware.
 */
public interface MatrixOperation {
    
    /**
     * Perform matrix multiplication: C = A * B.
     *
     * @param a matrix A
     * @param b matrix B
     * @param c result matrix C
     * @param rowsA number of rows in A
     * @param colsB number of columns in B
     * @param sharedDim shared dimension (columns of A, rows of B)
     */
    void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim);
    
    /**
     * Perform matrix addition: C = A + B.
     *
     * @param a matrix A
     * @param b matrix B
     * @param c result matrix C
     * @param elements total number of elements
     */
    void add(float[] a, float[] b, float[] c, int elements);
    
    /**
     * Perform matrix subtraction: C = A - B.
     *
     * @param a matrix A
     * @param b matrix B
     * @param c result matrix C
     * @param elements total number of elements
     */
    void subtract(float[] a, float[] b, float[] c, int elements);
    
    /**
     * Perform matrix-scalar multiplication: B = A * scalar.
     *
     * @param a matrix A
     * @param b result matrix B
     * @param scalar scalar value
     * @param elements total number of elements
     */
    void scalarMultiply(float[] a, float[] b, float scalar, int elements);
    
    /**
     * Perform matrix transpose: B = A^T.
     *
     * @param a matrix A
     * @param b result matrix B
     * @param rows number of rows in A
     * @param cols number of columns in A
     */
    void transpose(float[] a, float[] b, int rows, int cols);
    
    /**
     * Get the compute provider used by this operation.
     *
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Release resources associated with this operation.
     */
    void release();
}
