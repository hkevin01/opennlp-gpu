package org.apache.opennlp.gpu.common;

/**
 * Interface for compute providers (CPU, GPU, etc.)
 * Provides abstraction for different computational backends
 */
public interface ComputeProvider {
    
    /**
     * Enumeration of supported compute provider types
     */
    enum Type {
        CPU,
        CUDA,
        OPENCL,
        ROCM
    }
    
    /**
     * Get the name of this compute provider
     */
    String getName();
    
    /**
     * Check if this provider is available and ready to use
     */
    boolean isAvailable();
    
    /**
     * Initialize the provider with the given configuration
     */
    void initialize(GpuConfig config);
    
    /**
     * Check if this is a GPU provider (as opposed to CPU)
     */
    default boolean isGpuProvider() {
        return false; // Default to CPU provider
    }
    
    /**
     * Get the type of this provider
     */
    Type getType();
    
    /**
     * Get the maximum memory available to this provider (in MB)
     */
    long getMaxMemoryMB();
    
    /**
     * Get the current memory usage (in MB)
     */
    long getCurrentMemoryUsageMB();
    
    /**
     * Check if the provider supports a specific operation
     */
    boolean supportsOperation(String operationType);
    
    /**
     * Get the resource manager for this provider
     */
    Object getResourceManager();
    
    /**
     * Matrix multiplication operation
     */
    void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k);
    
    /**
     * Matrix addition operation
     */
    void matrixAdd(float[] a, float[] b, float[] result, int size);
    
    /**
     * Matrix transpose operation
     */
    void matrixTranspose(float[] input, float[] output, int rows, int cols);
    
    /**
     * Extract features from text
     */
    void extractFeatures(String[] text, float[] features);
    
    /**
     * Compute TF-IDF scores
     */
    void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size);
    
    /**
     * Initialize the provider
     */
    void initialize();
    
    /**
     * Get performance characteristics of this provider
     */
    default ProviderCapabilities getCapabilities() {
        return new ProviderCapabilities();
    }
    
    /**
     * Cleanup and release resources
     */
    default void cleanup() {
        // Default implementation - no cleanup needed
    }
    
    /**
     * Simple capabilities container
     */
    class ProviderCapabilities {
        private boolean supportsParallelComputation = false;
        private boolean supportsGpuAcceleration = false;
        private int maxThreads = 1;
        
        public boolean supportsParallelComputation() { return supportsParallelComputation; }
        public boolean supportsGpuAcceleration() { return supportsGpuAcceleration; }
        public int getMaxThreads() { return maxThreads; }
        
        public void setSupportsParallelComputation(boolean supports) { this.supportsParallelComputation = supports; }
        public void setSupportsGpuAcceleration(boolean supports) { this.supportsGpuAcceleration = supports; }
        public void setMaxThreads(int threads) { this.maxThreads = threads; }
    }
}
