package org.apache.opennlp.gpu.compute;

import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.opennlp.gpu.common.ComputeProvider;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.opennlp.gpu.common.FeatureExtractionAdapter;
import org.apache.opennlp.gpu.common.FeatureExtractionOperation;

/**
 * Factory for creating operations with the most suitable compute provider.
 */
public class OperationFactory {
    private static final Logger logger = LoggerFactory.getLogger(OperationFactory.class);
    
    @Getter
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
    public org.apache.opennlp.gpu.compute.MatrixOperation createMatrixOperation(int matrixSize) {
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
                return new CudaMatrixOperation(provider);
            case ROCM:
                return new RocmMatrixOperation(provider);
            case CPU:
            default:
                // Use fully qualified name to avoid ambiguity
                return new org.apache.opennlp.gpu.compute.CpuMatrixOperation(provider);
        }
    }
    
    /**
     * Create a feature extraction operation with the best available provider.
     *
     * @param dataSize the size of data to be processed
     * @return a feature extraction operation
     */
    public org.apache.opennlp.gpu.common.FeatureExtractionOperation createFeatureExtractionOperation(int dataSize) {
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
                return new CudaFeatureExtractionOperation(provider);
            case ROCM:
                return new RocmFeatureExtractionOperation(provider);
            case CPU:
            default:
                // Cast or wrap the CPU implementation to match the expected interface
                return new org.apache.opennlp.gpu.common.FeatureExtractionAdapter(new CpuFeatureExtractionOperation(provider), provider);
        }
    }
    
    public static FeatureExtractionOperation createFeatureExtractionOperation(ComputeProvider.Type type) {
        logger.info("Creating feature extraction operation for type: {}", type);
        ComputeProvider provider = ComputeProviderFactory.getInstance().getProvider(type);
        if (provider == null) {
            logger.error("No provider found for type: {}. Falling back to CPU.", type);
            provider = ComputeProviderFactory.getInstance().getProvider(ComputeProvider.Type.CPU);
            if (provider == null) {
                throw new IllegalStateException("CPU ComputeProvider is not available.");
            }
        }

        switch (type) {
            case CUDA:
                return new CudaFeatureExtractionOperation(provider);
            case ROCM:
                return new RocmFeatureExtractionOperation(provider);
            case OPENCL:
                logger.warn("OpenCL feature extraction not fully implemented, returning CPU fallback.");
                // Fall through to CPU
            case CPU:
            default:
                logger.debug("Creating CpuFeatureExtractionOperation and wrapping with FeatureExtractionAdapter");
                CpuFeatureExtractionOperation cpuOperation = new CpuFeatureExtractionOperation(provider);
                // Pass both required arguments to the adapter
                return new FeatureExtractionAdapter(cpuOperation, provider);
        }
    }
}
