package org.apache.opennlp.gpu.common;

import org.jocl.cl_kernel; // If these types are part of the interface
import org.jocl.cl_mem;    // If these types are part of the interface

/**
 * Interface for managing GPU resources.
 * This includes memory, kernels, and other platform-specific resources.
 */
public interface ResourceManager {

    /**
     * Initializes the resource manager.
     * @return true if initialization was successful, false otherwise.
     */
    boolean initialize();

    /**
     * Releases all resources managed by this manager.
     */
    void release(); // This is a common method.

    /**
     * Gets the memory manager associated with this resource manager.
     * @return The memory manager.
     */
    MemoryManager getMemoryManager(); // This is a common method.

    /**
     * Releases all allocated resources.
     * This might be redundant if 'release()' already does this,
     * but ensure its definition is clear.
     */
    void releaseAll(); // This is a common method.

    // The following methods are causing @Override errors in CudaResourceManager.
    // If they ARE part of this interface, their signatures must match exactly.
    // If they ARE NOT part of this interface, remove @Override in CudaResourceManager.

    /**
     * Gets or creates a compute kernel. (Primarily for OpenCL)
     * @param name The name of the kernel.
     * @param source The source code of the kernel.
     * @return The compiled kernel, or null if not applicable/error.
     */
    cl_kernel getOrCreateKernel(String name, String source);

    /**
     * Allocates a memory buffer on the device. (Primarily for OpenCL)
     * @param size The size of the buffer in bytes.
     * @param readOnly True if the buffer is read-only, false otherwise.
     * @return A reference to the allocated memory buffer, or null if not applicable/error.
     */
    cl_mem allocateBuffer(int size, boolean readOnly);

    /**
     * Allocates a named memory buffer on the device. (Primarily for OpenCL)
     * @param size The size of the buffer in bytes.
     * @param name The name to associate with the buffer.
     * @return A reference to the allocated memory buffer, or null if not applicable/error.
     */
    cl_mem allocateBuffer(int size, String name);

    /**
     * Retrieves cached data by name.
     * @param name The name of the cached data.
     * @return The cached data object, or null if not found.
     */
    Object getCachedData(String name);

    /**
     * Releases a specific memory buffer. (Primarily for OpenCL)
     * @param buffer The memory buffer to release.
     */
    void releaseBuffer(cl_mem buffer);
}
