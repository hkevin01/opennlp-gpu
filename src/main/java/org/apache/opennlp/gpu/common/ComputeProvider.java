package org.apache.opennlp.gpu.common;

/**
 * Interface for GPU compute providers.
 */
public interface ComputeProvider {
    
    // Suppress warnings for unused enum constants (these values are part of the public API)
    @SuppressWarnings("unused")
    enum Type {
        CUDA,
        ROCM,
        OPENCL,
        CPU    // Add CPU type for the CPU compute provider
    }
    
    /**
     * Initialize the compute provider.
     * 
     * @return true if initialization was successful
     */
    boolean initialize();
    
    /**
     * Get the type of the compute provider.
     * 
     * @return the provider type
     */
    Type getType();
    
    /**
     * Check if the provider is available.
     * 
     * @return true if the provider is available
     */
    boolean isAvailable();
    
    /**
     * Get the name of the provider.
     * 
     * @return the provider name
     */
    String getName();
    
    /**
     * Release resources held by the provider.
     */
    void release();
    
    /**
     * Get a resource manager for this provider.
     * 
     * @return the resource manager
     */
    ResourceManager getResourceManager();
    
    /**
     * Check if the provider supports the specified operation.
     * 
     * @param operationName the name of the operation
     * @return true if the operation is supported
     */
    boolean supportsOperation(String operationName);
    
    /**
     * Get a performance score for the specified operation and data size.
     * 
     * @param operationName the name of the operation
     * @param dataSize the size of the data
     * @return a performance score
     */
    double getPerformanceScore(String operationName, int dataSize);
}
