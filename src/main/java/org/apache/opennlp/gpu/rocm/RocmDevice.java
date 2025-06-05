package org.apache.opennlp.gpu.rocm;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NonNull;

/**
 * Represents an AMD GPU device available through ROCm.
 * This class encapsulates information about an AMD GPU device.
 */
@Data
@Builder
@AllArgsConstructor
public class RocmDevice {
    
    private final int deviceId;
    
    @NonNull
    private final String name;
    
    @NonNull
    private final String architecture;
    
    private final long memoryBytes;
    
    private final int computeUnits;
    
    /**
     * Get the amount of memory in megabytes.
     * 
     * @return the memory size in MB
     */
    public int getMemoryMB() {
        return (int)(memoryBytes / (1024 * 1024));
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
