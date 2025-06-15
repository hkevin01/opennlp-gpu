package org.apache.opennlp.gpu.common;

/**
 * Enhanced interface for compute providers (CPU/GPU)
 */
public interface ComputeProvider {
    
    /**
     * Provider type enumeration
     */
    enum Type {
        CPU,
        OPENCL,
        CUDA,
        ROCM
    }
    
    boolean isGpuProvider();
    void cleanup();
    
    // Additional methods needed by existing code
    String getName();
    Type getType();
    boolean isAvailable();
    Object getResourceManager();
    
    // Matrix operation methods
    void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k);
    void matrixAdd(float[] a, float[] b, float[] result, int size);
    void matrixTranspose(float[] input, float[] output, int rows, int cols);
    
    // Feature extraction methods
    void extractFeatures(String[] text, float[] features);
    void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size);
    
    // Initialization and configuration
    void initialize();
    boolean supportsOperation(String operationType);
}
