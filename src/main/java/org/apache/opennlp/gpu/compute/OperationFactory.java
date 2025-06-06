package org.apache.opennlp.gpu.compute;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating operations.
 */
public class OperationFactory {
    
    private static final Logger logger = LoggerFactory.getLogger(OperationFactory.class);
    
    /**
     * Create a matrix operation for the specified device.
     * 
     * @param deviceIndex the device index
     * @return the matrix operation
     */
    public MatrixOperation createMatrixOperation(int deviceIndex) {
        logger.info("Creating matrix operation for device index: {}", deviceIndex);
        
        // Simplified implementation - just return a dummy matrix operation
        return new DummyMatrixOperation();
    }
    
    /**
     * Dummy matrix operation implementation for testing.
     */
    private static class DummyMatrixOperation implements MatrixOperation {
        @Override
        public void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k) {
            // Simple CPU implementation of matrix multiplication
            for (int i = 0; i < m; i++) {
                for (int j = 0; j < n; j++) {
                    float sum = 0;
                    for (int x = 0; x < k; x++) {
                        sum += a[i * k + x] * b[x * n + j];
                    }
                    result[i * n + j] = sum;
                }
            }
        }
        
        @Override
        public void release() {
            // Nothing to release in dummy implementation
        }
    }
}
