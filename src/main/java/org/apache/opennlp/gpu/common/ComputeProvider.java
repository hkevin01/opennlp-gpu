package org.apache.opennlp.gpu.common;

/**
 * Interface for compute providers that can execute operations on different 
 * hardware backends (CPU, GPU via OpenCL, etc.)
 * 
 * This is the core interface of the abstraction layer, allowing different
 * hardware implementations to be used interchangeably.
 */
public interface ComputeProvider {
  
  /**
   * Enum representing the different types of compute providers.
   */
  enum Type {
    CPU,     // CPU-based fallback implementation
    OPENCL,  // OpenCL implementation via JOCL
    CUDA     // CUDA implementation (if available)
  }
  
  /**
   * Initialize the compute provider.
   * @return true if initialization was successful, false otherwise
   */
  boolean initialize();
  
  /**
   * Get the type of this compute provider.
   * @return the provider type
   */
  Type getType();
  
  /**
   * Check if this provider is available on the current system.
   * @return true if available, false otherwise
   */
  boolean isAvailable();
  
  /**
   * Get the name of this compute provider.
   * @return a human-readable name
   */
  String getName();
  
  /**
   * Get the compute capability level of this provider.
   * Higher values indicate more advanced capabilities.
   * @return a numeric value representing compute capability
   */
  int getComputeCapability();
  
  /**
   * Get a performance score for the given operation type and size.
   * This is used for automatic selection of the best provider.
   * 
   * @param operationType the type of operation to benchmark
   * @param problemSize the size of the problem (e.g., matrix dimensions)
   * @return a score where higher values indicate better performance
   */
  double getPerformanceScore(String operationType, int problemSize);
  
  /**
   * Get the resource manager associated with this provider.
   * @return the resource manager
   */
  ResourceManager getResourceManager();
  
  /**
   * Check if this provider supports a specific operation.
   * 
   * @param operationType the type of operation to check
   * @return true if the operation is supported, false otherwise
   */
  boolean supportsOperation(String operationType);
  
  /**
   * Release resources associated with this compute provider.
   */
  void release();
}
