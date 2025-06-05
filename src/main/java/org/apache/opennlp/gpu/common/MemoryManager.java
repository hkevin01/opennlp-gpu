package org.apache.opennlp.gpu.common;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_mem;

import static org.jocl.CL.*;

import lombok.extern.slf4j.Slf4j;

/**
 * Manages memory allocations for GPU operations.
 */
@Slf4j
public class MemoryManager {
    
    private static final int DEFAULT_ALIGNMENT = 64;
    private cl_context context;
    private cl_command_queue commandQueue;
    
    /**
     * Default constructor.
     */
    public MemoryManager() {
        // Default constructor
    }
    
    /**
     * Creates a memory manager with OpenCL context and command queue.
     *
     * @param context the OpenCL context
     * @param commandQueue the OpenCL command queue
     */
    public MemoryManager(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        log.info("Created OpenCL memory manager");
    }
    
    /**
     * Allocates a direct byte buffer aligned to the specified boundary.
     *
     * @param size the buffer size in bytes
     * @param alignment the alignment boundary
     * @return the allocated buffer
     */
    public static ByteBuffer allocateAligned(long size, int alignment) {
        if (size <= 0) {
            throw new IllegalArgumentException("Size must be positive");
        }
        
        if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
            throw new IllegalArgumentException("Alignment must be a positive power of 2");
        }
        
        // Allocate with extra space for alignment
        long totalSize = size + alignment - 1;
        if (totalSize > Integer.MAX_VALUE) {
            throw new OutOfMemoryError("Requested allocation exceeds maximum buffer size");
        }
        
        ByteBuffer buffer = ByteBuffer.allocateDirect((int)totalSize);
        buffer.order(ByteOrder.nativeOrder());
        
        // Calculate alignment offset
        long address = getBufferAddress(buffer);
        long alignedAddress = (address + alignment - 1) & ~(alignment - 1);
        int offset = (int)(alignedAddress - address);
        
        // Create aligned slice
        buffer.position(offset);
        buffer.limit((int)size + offset);
        ByteBuffer alignedBuffer = buffer.slice();
        
        log.debug("Allocated aligned buffer: size={}, alignment={}, offset={}", size, alignment, offset);
        return alignedBuffer;
    }
    
    /**
     * Allocates a direct byte buffer aligned to the default boundary.
     *
     * @param size the buffer size in bytes
     * @return the allocated buffer
     */
    public static ByteBuffer allocateAligned(long size) {
        return allocateAligned(size, DEFAULT_ALIGNMENT);
    }
    
    /**
     * Gets the memory address of a direct byte buffer.
     *
     * @param buffer the buffer
     * @return the memory address
     */
    private static long getBufferAddress(ByteBuffer buffer) {
        if (!buffer.isDirect()) {
            throw new IllegalArgumentException("Buffer must be direct");
        }
        
        // This is a placeholder - actual implementation would use Unsafe or JNI
        // to get the real memory address
        return buffer.hashCode() & 0xFFFFFFFFL;
    }
    
    /**
     * Allocates an OpenCL buffer.
     *
     * @param size the buffer size in bytes
     * @param readOnly whether the buffer is read-only
     * @return the allocated buffer
     */
    public cl_mem allocateBuffer(int size, boolean readOnly) {
        if (context == null) {
            throw new IllegalStateException("OpenCL context not initialized");
        }
        
        long flags = readOnly ? CL_MEM_READ_ONLY : CL_MEM_READ_WRITE;
        int[] errCode = new int[1];
        cl_mem buffer = clCreateBuffer(context, flags, size, null, errCode);
        
        if (errCode[0] != CL_SUCCESS) {
            throw new RuntimeException("Failed to allocate OpenCL buffer: " + errCode[0]);
        }
        
        log.debug("Allocated OpenCL buffer: size={}, readOnly={}", size, readOnly);
        return buffer;
    }
    
    /**
     * Copies data from host to device.
     *
     * @param hostPtr the host pointer
     * @param deviceBuffer the device buffer
     * @param size the size in bytes
     */
    public void copyToDevice(Pointer hostPtr, cl_mem deviceBuffer, int size) {
        if (commandQueue == null) {
            throw new IllegalStateException("OpenCL command queue not initialized");
        }
        
        int errCode = clEnqueueWriteBuffer(commandQueue, deviceBuffer, CL_TRUE, 0,
                size, hostPtr, 0, null, null);
        
        if (errCode != CL_SUCCESS) {
            throw new RuntimeException("Failed to copy data to device: " + errCode);
        }
        
        log.debug("Copied data to device: size={}", size);
    }
    
    /**
     * Copies data from device to host.
     *
     * @param deviceBuffer the device buffer
     * @param hostPtr the host pointer
     * @param size the size in bytes
     */
    public void copyFromDevice(cl_mem deviceBuffer, Pointer hostPtr, int size) {
        if (commandQueue == null) {
            throw new IllegalStateException("OpenCL command queue not initialized");
        }
        
        int errCode = clEnqueueReadBuffer(commandQueue, deviceBuffer, CL_TRUE, 0,
                size, hostPtr, 0, null, null);
        
        if (errCode != CL_SUCCESS) {
            throw new RuntimeException("Failed to copy data from device: " + errCode);
        }
        
        log.debug("Copied data from device: size={}", size);
    }
    
    /**
     * Releases an OpenCL buffer.
     *
     * @param buffer the buffer to release
     */
    public void releaseBuffer(cl_mem buffer) {
        if (buffer != null) {
            int errCode = clReleaseMemObject(buffer);
            if (errCode != CL_SUCCESS) {
                log.warn("Failed to release OpenCL buffer: {}", errCode);
            } else {
                log.debug("Released OpenCL buffer");
            }
        }
    }
    
    /**
     * Releases all resources.
     */
    public void release() {
        log.info("Releasing memory manager resources");
        // Release any cached resources here
    }
}
