package org.apache.opennlp.gpu.common;

/**
 * Interface for managing GPU memory.
 */
public interface MemoryManager {
    
    /**
     * Allocate memory on the device.
     * 
     * @param size the size in bytes
     * @return a handle to the allocated memory
     */
    int allocate(long size);
    
    /**
     * Free memory on the device.
     * 
     * @param ptr the memory handle
     */
    void free(long ptr);
    
    /**
     * Copy data from host to device.
     * 
     * @param devicePtr the device memory handle
     * @param hostData the host data
     * @param size the size in bytes
     */
    void copyHostToDevice(long devicePtr, byte[] hostData, long size);
    
    /**
     * Copy data from device to host.
     * 
     * @param devicePtr the device memory handle
     * @param hostData the host data buffer
     * @param size the size in bytes
     */
    void copyDeviceToHost(long devicePtr, byte[] hostData, long size);
    
    /**
     * Release all memory.
     */
    void releaseAll();
}
