package org.apache.opennlp.gpu.rocm;

import org.apache.opennlp.gpu.common.GpuDevice;

/**
 * Represents an AMD GPU device available through ROCm.
 * This class encapsulates information about an AMD GPU device.
 */
public class RocmDevice implements GpuDevice {
    private final String name;
    private final int deviceId;
    private long devicePtr;
    private String architecture = "Unknown";
    private long memoryBytes = 1024L * 1024L * 1024L; // 1GB default
    
    /**
     * Create a new ROCm device.
     *
     * @param name device name
     * @param deviceId device index
     * @param devicePtr native device pointer
     */
    public RocmDevice(String name, int deviceId, long devicePtr) {
        this.name = name != null ? name : "Unknown";
        this.deviceId = deviceId;
        this.devicePtr = devicePtr;
    }
    
    /**
     * Get the device name.
     *
     * @return the device name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the device index.
     *
     * @return the device index
     */
    public int getDeviceId() {
        return deviceId;
    }
    
    /**
     * Get the native device pointer.
     *
     * @return the device pointer
     */
    public long getDevicePtr() {
        return devicePtr;
    }
    
    /**
     * Get the amount of memory in megabytes.
     * 
     * @return the memory size in MB
     */
    @Override
    public long getMemoryMB() {
        // Implementation of the missing method
        return 1024; // Default 1GB, replace with actual implementation
    }
    
    /**
     * Get a short description of the device.
     * 
     * @return a string describing the device
     */
    public String getDescription() {
        return String.format("%s (%s) with %d MB", name, architecture, getMemoryMB());
    }
    
    /**
     * Check if this device has enough memory for the specified size.
     * 
     * @param requiredBytes the number of bytes required
     * @return true if the device has enough memory
     */
    public boolean hasEnoughMemory(long requiredBytes) {
        // Allow up to 80% of total memory to be used
        return requiredBytes <= (memoryBytes * 0.8);
    }
}
