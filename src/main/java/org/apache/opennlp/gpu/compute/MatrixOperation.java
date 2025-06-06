package org.apache.opennlp.gpu.compute;

/**
 * Interface for matrix operations.
 */
public interface MatrixOperation {
    
    /**
     * Multiply two matrices.
     * 
     * @param a matrix A
     * @param b matrix B
     * @param result the result matrix
     * @param m rows in A
     * @param n columns in B
     * @param k columns in A / rows in B
     */
    void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k);
    
    /**
     * Release resources.
     */
    void release();
}
