package org.apache.opennlp.gpu.common;

/**
 * Interface for compute providers that can execute operations on different 
 * hardware backends (CPU, GPU via OpenCL, etc.)
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
   * Release resources associated with this compute provider.
   */
  void release();
}
