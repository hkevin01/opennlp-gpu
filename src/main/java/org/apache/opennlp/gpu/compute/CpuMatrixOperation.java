package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * CPU-based implementation of matrix operations.
 * This class provides fallback implementations when GPU acceleration is not available.
 */
@Slf4j
@RequiredArgsConstructor
public class CpuMatrixOperation implements MatrixOperation {
    
    @Getter
    private final ComputeProvider provider;
    
    /**
     * Creates a new CPU matrix operation with the specified provider.
     *
     * @param provider the compute provider to use
     */
    public CpuMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
        logger.info("Initializing CPU matrix operations with provider: {}", provider.getName());
    }
    
    @Override
    public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) {
        logger.debug("CPU matrix multiply: {}x{} * {}x{}", rowsA, sharedDim, sharedDim, colsB);
        
        // Basic matrix multiplication algorithm
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
        logger.debug("CPU matrix add: {} elements", elements);
        
        for (int i = 0; i < elements; i++) {
            c[i] = a[i] + b[i];
        }
    }
    
    @Override
    public void subtract(float[] a, float[] b, float[] c, int elements) {
        logger.debug("CPU matrix subtract: {} elements", elements);
        
        for (int i = 0; i < elements; i++) {
            c[i] = a[i] - b[i];
        }
    }
    
    @Override
    public void scalarMultiply(float[] a, float[] b, float scalar, int elements) {
        logger.debug("CPU scalar multiply: {} elements by {}", elements, scalar);
        
        for (int i = 0; i < elements; i++) {
            b[i] = a[i] * scalar;
        }
    }
    
    @Override
    public void transpose(float[] a, float[] b, int rows, int cols) {
        logger.debug("CPU matrix transpose: {}x{}", rows, cols);
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[j * rows + i] = a[i * cols + j];
            }
        }
    }
    
    @Override
    public void release() {
        logger.info("Releasing CPU matrix operation resources");
        // No resources to release for CPU implementation
    }
}
