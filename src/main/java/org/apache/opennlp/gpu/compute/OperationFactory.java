package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating operations with the most suitable compute provider.
 */
public class OperationFactory {
    
    private static final Logger logger = LoggerFactory.getLogger(OperationFactory.class);
    
    private final ComputeProviderFactory providerFactory;
    
    /**
     * Creates a new operation factory.
     */
    public OperationFactory() {
        this.providerFactory = ComputeProviderFactory.getInstance();
    }
    
    /**
     * Creates a new operation factory with the specified provider factory.
     *
     * @param providerFactory the provider factory to use
     */
    public OperationFactory(ComputeProviderFactory providerFactory) {
        this.providerFactory = providerFactory;
    }
    
    /**
     * Create a matrix operation with the best available provider.
     *
     * @param matrixSize the size of matrices to be processed
     * @return a matrix operation
     */
    public MatrixOperation createMatrixOperation(int matrixSize) {
        ComputeProvider provider = providerFactory.getBestProvider("matrixOperations", matrixSize);
        
        if (provider == null) {
            logger.warn("No suitable provider found for matrix operations, using CPU fallback");
            provider = providerFactory.getProvider(ComputeProvider.Type.CPU);
        }
        
        if (provider == null) {
            throw new RuntimeException("No compute providers available");
        }
        
        logger.debug("Using provider {} for matrix operations", provider.getName());
        
        switch (provider.getType()) {
            case OPENCL:
                return new OpenClMatrixOperation(provider);
            case CUDA:
                // TODO: Implement CUDA matrix operations
                logger.warn("CUDA matrix operations not implemented yet, falling back to CPU");
                return new CpuMatrixOperation(provider);
            case CPU:
            default:
                return new CpuMatrixOperation(provider);
        }
    }
    
    /**
     * Create a feature extraction operation with the best available provider.
     *
     * @param dataSize the size of data to be processed
     * @return a feature extraction operation
     */
    public FeatureExtractionOperation createFeatureExtractionOperation(int dataSize) {
        ComputeProvider provider = providerFactory.getBestProvider("featureExtraction", dataSize);
        
        if (provider == null) {
            logger.warn("No suitable provider found for feature extraction, using CPU fallback");
            provider = providerFactory.getProvider(ComputeProvider.Type.CPU);
        }
        
        if (provider == null) {
            throw new RuntimeException("No compute providers available");
        }
        
        logger.debug("Using provider {} for feature extraction", provider.getName());
        
        switch (provider.getType()) {
            case OPENCL:
                return new OpenClFeatureExtractionOperation(provider);
            case CUDA:
                // TODO: Implement CUDA feature extraction
                // Implementation will be similar to:
                // return new CudaFeatureExtractionOperation(provider);
                // For now, fall back to CPU
                logger.warn("CUDA feature extraction not implemented yet, falling back to CPU");
                return new CpuFeatureExtractionOperation(provider);
            case CPU:
            default:
                return new CpuFeatureExtractionOperation(provider);
        }
    }
    
    /**
     * Get the provider factory used by this operation factory.
     *
     * @return the provider factory
     */
    public ComputeProviderFactory getProviderFactory() {
        return providerFactory;
    }
}
