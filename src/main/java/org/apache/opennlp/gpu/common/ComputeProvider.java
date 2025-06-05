package org.apache.opennlp.gpu.common;

/**
 * Interface for GPU compute providers.
 */
public interface ComputeProvider {
    
    /**
     * Enum defining the types of compute providers.
     */
    enum Type {
        CPU,
        CUDA,
        ROCM,
        OPENCL,
        AUTO;
        
        /**
         * Returns a string representation of the compute provider type.
         *
         * @return the string representation
         */
        @Override
        public String toString() {
            return name().toLowerCase();
        }
    }
    
    /**
     * Initialize the compute provider.
     * 
     * @return true if initialization was successful
     */
    boolean initialize();
    
    /**
     * Get the name of the compute provider.
     * 
     * @return the provider name
     */
    String getName();
    
    /**
     * Check if this provider is available on the system.
     * 
     * @return true if available
     */
    boolean isAvailable();
    
    /**
     * Get the type of this compute provider.
     * 
     * @return the provider type
     */
    Type getType();
    
    /**
     * Release resources used by this provider.
     */
    void release();
    
    /**
     * Check if this provider supports the specified operation.
     * 
     * @param operationName the name of the operation
     * @return true if the operation is supported
     */
    boolean supportsOperation(String operationName);
    
    /**
     * Get the performance score for an operation.
     * 
     * @param operationName the name of the operation
     * @param dataSize the size of the data to process
     * @return the performance score (higher is better)
     */
    double getPerformanceScore(String operationName, int dataSize);
    
    /**
     * Get the resource manager for this provider.
     * 
     * @return the resource manager
     */
    ResourceManager getResourceManager();
}
