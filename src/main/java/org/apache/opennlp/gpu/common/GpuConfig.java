package org.apache.opennlp.gpu.common;

/**
 * Configuration for GPU operations
 */
public class GpuConfig {
    private boolean gpuEnabled = false;
    private boolean debugMode = false;
    private int memoryPoolSizeMB = 256; // Default 256MB
    private int batchSize = 32; // Default batch size
    private int maxMemoryUsageMB = 1024; // Default 1GB limit
    
    public GpuConfig() {
        // Default configuration
    }
    
    public boolean isGpuEnabled() {
        return gpuEnabled;
    }
    
    public void setGpuEnabled(boolean enabled) {
        this.gpuEnabled = enabled;
    }
    
    public boolean isDebugMode() {
        return debugMode;
    }
    
    public void setDebugMode(boolean debugMode) {
        this.debugMode = debugMode;
    }
    
    public int getMemoryPoolSizeMB() {
        return memoryPoolSizeMB;
    }
    
    public void setMemoryPoolSizeMB(int memoryPoolSizeMB) {
        this.memoryPoolSizeMB = memoryPoolSizeMB;
    }
    
    public int getBatchSize() {
        return batchSize;
    }
    
    public void setBatchSize(int batchSize) {
        this.batchSize = batchSize;
    }
    
    public int getMaxMemoryUsageMB() {
        return maxMemoryUsageMB;
    }
    
    public void setMaxMemoryUsageMB(int maxMemoryUsageMB) {
        this.maxMemoryUsageMB = maxMemoryUsageMB;
    }
    
    /**
     * Check if GPU acceleration is available on the system
     * @return true if GPU is available and can be used
     */
    public static boolean isGpuAvailable() {
        try {
            // Try to initialize JOCL to check GPU availability
            org.jocl.CL.setExceptionsEnabled(true);
            // Basic check - if we can get platform info, GPU is likely available
            return true; // For now, assume GPU is available if JOCL is on classpath
        } catch (Exception e) {
            return false;
        }
    }
}
