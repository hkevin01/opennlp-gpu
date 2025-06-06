package org.apache.opennlp.gpu.common;

import java.util.ArrayList;
import java.util.List;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Represents a GPU device.
 */
public class GpuDevice {
    
    private static final Logger logger = LoggerFactory.getLogger(GpuDevice.class);
    
    private final String name;
    private final String vendor;
    private final long globalMemSize;
    private final int computeUnits;
    
    /**
     * Create a new GPU device.
     * 
     * @param name the device name
     * @param vendor the device vendor
     * @param globalMemSize the global memory size in bytes
     * @param computeUnits the number of compute units
     */
    public GpuDevice(String name, String vendor, long globalMemSize, int computeUnits) {
        this.name = name;
        this.vendor = vendor;
        this.globalMemSize = globalMemSize;
        this.computeUnits = computeUnits;
    }
    
    /**
     * Get the name of the device.
     * 
     * @return the device name
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the vendor of the device.
     * 
     * @return the device vendor
     */
    public String getVendor() {
        return vendor;
    }
    
    /**
     * Get the global memory size in bytes.
     * 
     * @return the global memory size
     */
    public long getGlobalMemSize() {
        return globalMemSize;
    }
    
    /**
     * Get the number of compute units.
     * 
     * @return the number of compute units
     */
    public int getComputeUnits() {
        return computeUnits;
    }
    
    /**
     * Get the amount of memory in megabytes.
     * 
     * @return the memory size in MB
     */
    public int getMemoryMB() {
        return (int)(globalMemSize / (1024 * 1024));
    }
    
    /**
     * Get a list of available GPU devices.
     * 
     * @return a list of available devices
     */
    public static List<GpuDevice> getAvailableDevices() {
        logger.info("Detecting available GPU devices");
        
        // Simplified implementation - in a real app, this would use JNI to query devices
        List<GpuDevice> devices = new ArrayList<>();
        
        // Add a dummy device for testing
        devices.add(new GpuDevice("NVIDIA GeForce RTX 3080", "NVIDIA", 10L * 1024 * 1024 * 1024, 68));
        
        return devices;
    }
    
    @Override
    public String toString() {
        return String.format("GpuDevice[name=%s, vendor=%s, memory=%d MB]", 
                name, vendor, getMemoryMB());
    }
}
