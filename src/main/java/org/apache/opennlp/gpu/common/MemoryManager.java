package org.apache.opennlp.gpu.common;

import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * Manages memory buffers on the GPU device and handles data transfer
 * between host (CPU) and device (GPU) memory.
 */
public class MemoryManager {
    
    private static final Logger logger = LoggerFactory.getLogger(MemoryManager.class);
    
    private final cl_context context;
    private final cl_command_queue commandQueue;
    
    // Cache for buffers to avoid repeated allocations
    private final Map<String, cl_mem> bufferCache = new HashMap<>();
    
    /**
     * Creates a new memory manager for the specified OpenCL context and command queue.
     *
     * @param context the OpenCL context
     * @param commandQueue the OpenCL command queue
     */
    public MemoryManager(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        logger.debug("Memory manager initialized");
    }
    
    /**
     * Allocates a buffer on the GPU device.
     *
     * @param size the size of the buffer in bytes
     * @param readOnly whether the buffer is read-only for the kernel
     * @return the allocated buffer
     */
    public cl_mem allocateBuffer(long size, boolean readOnly) {
        int flags = readOnly ? CL.CL_MEM_READ_ONLY : CL.CL_MEM_READ_WRITE;
        
        try {
            int[] errorCode = new int[1];
            cl_mem buffer = CL.clCreateBuffer(context, flags, size, null, errorCode);
            
            if (errorCode[0] != CL.CL_SUCCESS) {
                throw new RuntimeException("Failed to allocate device memory: " + errorCode[0]);
            }
            
            logger.debug("Allocated buffer of size {} bytes", size);
            return buffer;
        } catch (Exception e) {
            logger.error("Error allocating device memory", e);
            throw new RuntimeException("Error allocating device memory", e);
        }
    }
    
    /**
     * Allocates or retrieves a cached buffer with the given key.
     *
     * @param key the cache key for the buffer
     * @param size the size of the buffer in bytes
     * @param readOnly whether the buffer is read-only for the kernel
     * @return the allocated or cached buffer
     */
    public cl_mem getOrAllocateBuffer(String key, long size, boolean readOnly) {
        if (bufferCache.containsKey(key)) {
            logger.debug("Using cached buffer for key: {}", key);
            return bufferCache.get(key);
        }
        
        cl_mem buffer = allocateBuffer(size, readOnly);
        bufferCache.put(key, buffer);
        return buffer;
    }
    
    /**
     * Copies data from host to device memory.
     *
     * @param hostArray the source array on the host
     * @param deviceBuffer the destination buffer on the device
     * @param size the number of bytes to copy
     */
    public void copyToDevice(Pointer hostArray, cl_mem deviceBuffer, long size) {
        try {
            CL.clEnqueueWriteBuffer(
                commandQueue, deviceBuffer, CL.CL_TRUE, 0,
                size, hostArray, 0, null, null);
            
            logger.debug("Copied {} bytes to device", size);
        } catch (Exception e) {
            logger.error("Error copying data to device", e);
            throw new RuntimeException("Error copying data to device", e);
        }
    }
    
    /**
     * Copies data from device to host memory.
     *
     * @param deviceBuffer the source buffer on the device
     * @param hostArray the destination array on the host
     * @param size the number of bytes to copy
     */
    public void copyFromDevice(cl_mem deviceBuffer, Pointer hostArray, long size) {
        try {
            CL.clEnqueueReadBuffer(
                commandQueue, deviceBuffer, CL.CL_TRUE, 0,
                size, hostArray, 0, null, null);
            
            logger.debug("Copied {} bytes from device", size);
        } catch (Exception e) {
            logger.error("Error copying data from device", e);
            throw new RuntimeException("Error copying data from device", e);
        }
    }
    
    /**
     * Releases a device buffer.
     *
     * @param buffer the buffer to release
     */
    public void releaseBuffer(cl_mem buffer) {
        try {
            CL.clReleaseMemObject(buffer);
            logger.debug("Released buffer");
        } catch (Exception e) {
            logger.error("Error releasing device buffer", e);
        }
    }
    
    /**
     * Clears the buffer cache and releases all buffers.
     */
    public void clearCache() {
        for (cl_mem buffer : bufferCache.values()) {
            releaseBuffer(buffer);
        }
        bufferCache.clear();
        logger.debug("Cleared buffer cache");
    }
    
    /**
     * Releases all resources associated with this memory manager.
     */
    public void release() {
        clearCache();
    }
}
