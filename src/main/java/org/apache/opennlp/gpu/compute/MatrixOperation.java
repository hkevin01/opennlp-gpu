package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Enhanced matrix operations interface for GPU acceleration
 * Supports both basic and advanced mathematical operations
 */
public interface MatrixOperation {
    
    /**
     * Get the compute provider for this matrix operation
     */
    ComputeProvider getProvider();
    
    /**
     * Release resources held by this matrix operation
     */
    void release();
    
    // Basic Matrix Operations
    void multiply(float[] a, float[] b, float[] result, int m, int n, int k);
    void transpose(float[] input, float[] output, int rows, int cols);
    void scalarMultiply(float[] input, float[] output, float scalar, int length);
    void add(float[] a, float[] b, float[] result, int size);
    void subtract(float[] a, float[] b, float[] result, int size);
    
    // Advanced Matrix Operations for ML
    void dotProduct(float[] a, float[] b, float[] result, int length);
    void vectorNorm(float[] input, float[] result, int length);
    void elementWiseMultiply(float[] a, float[] b, float[] result, int size);
    void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols);
    
    // Activation Functions (for neural networks)
    void sigmoid(float[] input, float[] result, int size);
    void tanh(float[] input, float[] result, int size);
    void relu(float[] input, float[] result, int size);
    void softmax(float[] input, float[] result, int size);
    
    // Statistical Operations
    void mean(float[] input, float[] result, int size);
    void variance(float[] input, float[] result, int size, float mean);
    void normalize(float[] input, float[] result, int size);
    
    // Utility Operations
    void copyArray(float[] source, float[] destination, int size);
    void fillArray(float[] array, float value, int size);
    void findMax(float[] input, int[] maxIndex, float[] maxValue, int size);
    void findMin(float[] input, int[] minIndex, float[] minValue, int size);
}
