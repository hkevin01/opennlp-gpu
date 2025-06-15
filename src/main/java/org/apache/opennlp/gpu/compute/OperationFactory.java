package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

public class OperationFactory {
    
    public static MatrixOperation createMatrixOperation() {
        return new DummyMatrixOperation();
    }
    
    public static MatrixOperation createMatrixOperation(ComputeProvider provider) {
        return new DummyMatrixOperation();
    }
    
    public static class DummyMatrixOperation implements MatrixOperation {
        @Override
        public ComputeProvider getProvider() { 
            return null; 
        }
        
        @Override
        public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) { 
            // CPU fallback implementation
            for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < sharedDim; k++) {
                        sum += a[i * sharedDim + k] * b[k * colsB + j];
                    }
                    c[i * colsB + j] = sum;
                }
            }
        }
        
        @Override
        public void add(float[] a, float[] b, float[] c, int elements) { 
            for (int i = 0; i < elements; i++) {
                c[i] = a[i] + b[i];
            }
        }
        
        @Override
        public void subtract(float[] a, float[] b, float[] c, int elements) { 
            for (int i = 0; i < elements; i++) {
                c[i] = a[i] - b[i];
            }
        }
        
        @Override
        public void scalarMultiply(float[] a, float[] b, float scalar, int elements) { 
            for (int i = 0; i < elements; i++) {
                b[i] = a[i] * scalar;
            }
        }
        
        @Override
        public void transpose(float[] a, float[] b, int rows, int cols) { 
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    b[j * rows + i] = a[i * cols + j];
                }
            }
        }
        
        @Override
        public void release() { 
            // No-op
        }
        
        @Override
        public void fillArray(float[] array, float value, int size) {
            // Dummy implementation
        }
        
        @Override
        public void copyArray(float[] source, float[] destination, int size) {
            // Dummy implementation
        }
        
        @Override
        public void findMax(float[] input, int[] maxIndex, float[] maxValue, int size) {
            // Dummy implementation
        }
        
        @Override
        public void findMin(float[] input, int[] minIndex, float[] minValue, int size) {
            // Dummy implementation
        }
        
        // Add all missing methods from MatrixOperation interface
        @Override
        public void dotProduct(float[] a, float[] b, float[] result, int length) {
            // Dummy implementation
        }
        
        @Override
        public void vectorNorm(float[] input, float[] result, int length) {
            // Dummy implementation
        }
        
        @Override
        public void elementWiseMultiply(float[] a, float[] b, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols) {
            // Dummy implementation
        }
        
        @Override
        public void sigmoid(float[] input, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void tanh(float[] input, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void relu(float[] input, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void softmax(float[] input, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void mean(float[] input, float[] result, int size) {
            // Dummy implementation
        }
        
        @Override
        public void variance(float[] input, float[] result, int size, float mean) {
            // Dummy implementation
        }
        
        @Override
        public void normalize(float[] input, float[] result, int size) {
            // Dummy implementation
        }
    }
}
