package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Interface for matrix operations.
 */
public interface MatrixOperation {
    
    /**
     * Gets the compute provider.
     *
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    // Add any other methods that are required by implementations
    // For example:
    // float[][] multiply(float[][] matrixA, float[][] matrixB);
    // float[][] add(float[][] matrixA, float[][] matrixB);
    // void release();
}
