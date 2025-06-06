package org.apache.opennlp.gpu.compute;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.cuda.CudaUtil;

/**
 * CUDA implementation of matrix operations.
 * This class uses NVIDIA's CUDA platform for GPU-accelerated matrix operations.
 */
public class CudaMatrixOperation implements MatrixOperation {
    // Add explicit logger declaration
    private static final Logger log = LoggerFactory.getLogger(CudaMatrixOperation.class);
    
    private final ComputeProvider provider;
    private boolean initialized = false;
    private int deviceId = 0;
    
    // Implement the required method from the MatrixOperation interface
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
    
    // JNI method declarations for CUDA matrix operations
    private native long allocateDeviceMemory(long size);
    private native void freeDeviceMemory(long devicePtr);
    private native void copyHostToDevice(float[] hostArray, long devicePtr, int size);
    private native void copyDeviceToHost(long devicePtr, float[] hostArray, int size);
    private native void cudaMatrixMultiply(long aPtr, long bPtr, long cPtr, int rowsA, int colsB, int sharedDim);
    private native void cudaMatrixAdd(long aPtr, long bPtr, long cPtr, int elements);
    private native void cudaMatrixSubtract(long aPtr, long bPtr, long cPtr, int elements);
    private native void cudaMatrixScalarMultiply(long aPtr, long bPtr, float scalar, int elements);
    private native void cudaMatrixTranspose(long aPtr, long bPtr, int rows, int cols);
    
    /**
     * Creates a new CUDA matrix operation with the specified provider.
     *
     * @param provider the compute provider to use
     */
    public CudaMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
        log.info("Initializing CUDA matrix operations with provider: {}", provider.getName());
        
        // Initialize CUDA
        if (!CudaUtil.isAvailable()) {
            throw new RuntimeException("CUDA is not available");
        }
        
        try {
            // Load the native library for CUDA matrix operations
            System.loadLibrary("opennlp_cuda_matrix");
            initialized = true;
            log.info("CUDA matrix operations initialized successfully");
        } catch (UnsatisfiedLinkError e) {
            log.error("Failed to load CUDA matrix operations library", e);
            throw new RuntimeException("Failed to initialize CUDA matrix operations", e);
        }
    }
    
    @Override
    public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) {
        if (!initialized) {
            throw new IllegalStateException("CUDA matrix operations not initialized");
        }
        
        log.debug("CUDA matrix multiply: {}x{} * {}x{}", rowsA, sharedDim, sharedDim, colsB);
        
        // Allocate device memory
        long aPtr = allocateDeviceMemory(a.length * Float.BYTES);
        long bPtr = allocateDeviceMemory(b.length * Float.BYTES);
        long cPtr = allocateDeviceMemory(c.length * Float.BYTES);
        
        try {
            // Copy input data to device
            copyHostToDevice(a, aPtr, a.length);
            copyHostToDevice(b, bPtr, b.length);
            
            // Perform matrix multiplication
            cudaMatrixMultiply(aPtr, bPtr, cPtr, rowsA, colsB, sharedDim);
            
            // Copy result back to host
            copyDeviceToHost(cPtr, c, c.length);
        } finally {
            // Free device memory
            freeDeviceMemory(aPtr);
            freeDeviceMemory(bPtr);
            freeDeviceMemory(cPtr);
        }
    }
    
    @Override
    public void add(float[] a, float[] b, float[] c, int elements) {
        if (!initialized) {
            throw new IllegalStateException("CUDA matrix operations not initialized");
        }
        
        log.debug("CUDA matrix add: {} elements", elements);
        
        // Allocate device memory
        long aPtr = allocateDeviceMemory(elements * Float.BYTES);
        long bPtr = allocateDeviceMemory(elements * Float.BYTES);
        long cPtr = allocateDeviceMemory(elements * Float.BYTES);
        
        try {
            // Copy input data to device
            copyHostToDevice(a, aPtr, elements);
            copyHostToDevice(b, bPtr, elements);
            
            // Perform matrix addition
            cudaMatrixAdd(aPtr, bPtr, cPtr, elements);
            
            // Copy result back to host
            copyDeviceToHost(cPtr, c, elements);
        } finally {
            // Free device memory
            freeDeviceMemory(aPtr);
            freeDeviceMemory(bPtr);
            freeDeviceMemory(cPtr);
        }
    }
    
    @Override
    public void subtract(float[] a, float[] b, float[] c, int elements) {
        if (!initialized) {
            throw new IllegalStateException("CUDA matrix operations not initialized");
        }
        
        log.debug("CUDA matrix subtract: {} elements", elements);
        
        // Allocate device memory
        long aPtr = allocateDeviceMemory(elements * Float.BYTES);
        long bPtr = allocateDeviceMemory(elements * Float.BYTES);
        long cPtr = allocateDeviceMemory(elements * Float.BYTES);
        
        try {
            // Copy input data to device
            copyHostToDevice(a, aPtr, elements);
            copyHostToDevice(b, bPtr, elements);
            
            // Perform matrix subtraction
            cudaMatrixSubtract(aPtr, bPtr, cPtr, elements);
            
            // Copy result back to host
            copyDeviceToHost(cPtr, c, elements);
        } finally {
            // Free device memory
            freeDeviceMemory(aPtr);
            freeDeviceMemory(bPtr);
            freeDeviceMemory(cPtr);
        }
    }
    
    @Override
    public void scalarMultiply(float[] a, float[] b, float scalar, int elements) {
        if (!initialized) {
            throw new IllegalStateException("CUDA matrix operations not initialized");
        }
        
        log.debug("CUDA scalar multiply: {} elements by {}", elements, scalar);
        
        // Allocate device memory
        long aPtr = allocateDeviceMemory(elements * Float.BYTES);
        long bPtr = allocateDeviceMemory(elements * Float.BYTES);
        
        try {
            // Copy input data to device
            copyHostToDevice(a, aPtr, elements);
            
            // Perform scalar multiplication
            cudaMatrixScalarMultiply(aPtr, bPtr, scalar, elements);
            
            // Copy result back to host
            copyDeviceToHost(bPtr, b, elements);
        } finally {
            // Free device memory
            freeDeviceMemory(aPtr);
            freeDeviceMemory(bPtr);
        }
    }
    
    @Override
    public void transpose(float[] a, float[] b, int rows, int cols) {
        if (!initialized) {
            throw new IllegalStateException("CUDA matrix operations not initialized");
        }
        
        log.debug("CUDA matrix transpose: {}x{}", rows, cols);
        
        // Allocate device memory
        long aPtr = allocateDeviceMemory(rows * cols * Float.BYTES);
        long bPtr = allocateDeviceMemory(rows * cols * Float.BYTES);
        
        try {
            // Copy input data to device
            copyHostToDevice(a, aPtr, rows * cols);
            
            // Perform matrix transpose
            cudaMatrixTranspose(aPtr, bPtr, rows, cols);
            
            // Copy result back to host
            copyDeviceToHost(bPtr, b, rows * cols);
        } finally {
            // Free device memory
            freeDeviceMemory(aPtr);
            freeDeviceMemory(bPtr);
        }
    }
    
    @Override
    public void release() {
        log.info("Releasing CUDA matrix operation resources");
        // No resources to release at this level
        // Native resources are managed per-operation
    }
}
