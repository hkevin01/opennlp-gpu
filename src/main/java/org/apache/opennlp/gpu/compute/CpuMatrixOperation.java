package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.MatrixOperation;

/**
 * CPU implementation of matrix operations.
 */
public class CpuMatrixOperation implements MatrixOperation {
    
    private final ComputeProvider provider;
    
    /**
     * Creates a new CPU matrix operation.
     *
     * @param provider the compute provider
     */
    public CpuMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
    }
    
    // Implementation methods
}
