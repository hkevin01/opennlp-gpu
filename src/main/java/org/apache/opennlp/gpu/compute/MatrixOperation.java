package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Interface for matrix operations.
 */
public interface MatrixOperation {
    /**
     * Gets the compute provider used by this operation.
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Releases resources used by this operation.
     */
    void release();
    
    // Add other methods needed by implementations
}
