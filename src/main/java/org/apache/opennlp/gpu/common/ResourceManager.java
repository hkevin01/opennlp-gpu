package org.apache.opennlp.gpu.common;

import org.jocl.cl_mem;
import org.jocl.cl_kernel;
import org.jocl.Pointer;

/**
 * Interface for resource managers used by compute providers.
 */
public interface ResourceManager {
    
    /**
     * Initialize the resource manager.
     * 
     * @return true if initialization was successful
     */
    boolean initialize();
    
    /**
     * Release all resources.
     */
    void release();
    
    /**
     * Release all cached resources.
     */
    void releaseAll();
    
    /**
     * Get the memory manager.
     * 
     * @return the memory manager
     */
    MemoryManager getMemoryManager();
    
    /**
     * Get or create an OpenCL kernel.
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
     * @param readOnly whether the buffer is read-only
     * @return the allocated buffer
     */
    cl_mem allocateBuffer(int size, boolean readOnly);
    
    /**
     * Allocate a buffer with the specified size and name.
     * 
     * @param size the buffer size in bytes
     * @param name the buffer name for caching
     * @return the allocated buffer
     */
    cl_mem allocateBuffer(int size, String name);
    
    /**
     * Get cached data by name.
     * 
     * @param name the data name
     * @return the cached data
     */
    Object getCachedData(String name);
    
    /**
     * Release a buffer.
     * 
     * @param buffer the buffer to release
     */
    void releaseBuffer(cl_mem buffer);
}
