package org.apache.opennlp.gpu.rocm;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.NonNull;

/**
 * Represents an AMD GPU device available through ROCm.
 * This class encapsulates information about an AMD GPU device.
 */
@Data
@Builder
@AllArgsConstructor
@NoArgsConstructor(force = true) // Keep this annotation
public class RocmDevice {
    
    // Add default values for all final fields
    private final int deviceId = 0;
    
    @NonNull
    private final String name = "Unknown"; // Default non-null value
    
    @NonNull
    private final String architecture = "Unknown"; // Default non-null value
    
    private final long memoryBytes = 0L;
    
    private final int computeUnits = 0;
    
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
