/*
 * Copyright 2025 OpenNLP GPU Extension Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * 
 * This project is a third-party GPU acceleration extension for Apache OpenNLP.
 * It is not officially endorsed or maintained by the Apache Software Foundation.
 */
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
            // Simple check for development/testing
            // In production, this would check for actual GPU hardware/drivers
            String gpuProperty = System.getProperty("gpu.available", "false");
            return "true".equals(gpuProperty);
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Get GPU information for diagnostics
     * @return Map containing GPU information
     */
    public static java.util.Map<String, Object> getGpuInfo() {
        java.util.Map<String, Object> info = new java.util.HashMap<>();
        
        info.put("available", isGpuAvailable());
        info.put("vendor", System.getProperty("gpu.vendor", "Unknown"));
        info.put("device", System.getProperty("gpu.device", "Unknown"));
        info.put("driver_version", System.getProperty("gpu.driver", "Unknown"));
        info.put("memory_total", System.getProperty("gpu.memory.total", "0"));
        info.put("memory_free", System.getProperty("gpu.memory.free", "0"));
        
        return info;
    }
}
