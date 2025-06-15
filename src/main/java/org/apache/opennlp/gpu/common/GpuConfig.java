package org.apache.opennlp.gpu.common;

/**
 * Configuration for GPU operations
 */
public class GpuConfig {
    private boolean gpuEnabled = false;
    private boolean debugMode = false;
    
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
}
