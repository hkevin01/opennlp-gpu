package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.FeatureExtractionOperation;

/**
 * CPU implementation of feature extraction operations.
 */
public class CpuFeatureExtractionOperation implements FeatureExtractionOperation {
    
    private final ComputeProvider provider;
    
    /**
     * Creates a new CPU feature extraction operation.
     *
     * @param provider the compute provider
     */
    public CpuFeatureExtractionOperation(ComputeProvider provider) {
        this.provider = provider;
    }
    
    // Implementation methods
}
