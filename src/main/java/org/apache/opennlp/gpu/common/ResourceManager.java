package org.apache.opennlp.gpu.common;

import org.jocl.cl_kernel;
import org.jocl.cl_mem;

/**
 * Interface for managing GPU resources.
 */
public interface ResourceManager {
    
    /**
     * Initialize the resource manager.
     * 
     * @return true if initialization succeeded
     */
    boolean initialize();
    
    /**
     * Release resources.
     */
    void release();
    
    /**
     * Get the memory manager.
     * 
     * @return the memory manager
     */
    MemoryManager getMemoryManager();
    
    /**
     * Release all resources.
     */
    void releaseAll();
    
    /**
     * Get or create a kernel with the specified name and source.
     * 
     * @param name the kernel name
     * @param source the kernel source code
     * @return the kernel
     */
    cl_kernel getOrCreateKernel(String name, String source);
    
    /**
     * Allocate a buffer with the specified size.
     * 
     * @param size the buffer size in bytes
     * @param readOnly true if the buffer is read-only
     * @return the buffer
     */
    cl_mem allocateBuffer(int size, boolean readOnly);
    
    /**
     * Allocate a named buffer with the specified size.
     * 
     * @param size the buffer size in bytes
     * @param name the buffer name
     * @return the buffer
     */
    cl_mem allocateBuffer(int size, String name);
    
    /**
     * Get cached data with the specified name.
     * 
     * @param name the data name
     * @return the cached data
     */
    Object getCachedData(String name);
    
    /**
     * Release the specified buffer.
     * 
     * @param buffer the buffer to release
     */
    void releaseBuffer(cl_mem buffer);
}
