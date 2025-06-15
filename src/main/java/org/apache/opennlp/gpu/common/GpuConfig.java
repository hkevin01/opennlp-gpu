package org.apache.opennlp.gpu.common;

/**
 * Stub configuration for GPU operations
 */
public class GpuConfig {
    private boolean gpuEnabled = false;
    
    public GpuConfig() {
        // Default configuration
    }
    
    public boolean isGpuEnabled() {
        return gpuEnabled;
    }
    
    public void setGpuEnabled(boolean enabled) {
        this.gpuEnabled = enabled;
    }
}
