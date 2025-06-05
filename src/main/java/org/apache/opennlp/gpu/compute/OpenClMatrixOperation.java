package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ResourceManager;
import org.jocl.*;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * OpenCL implementation of matrix operations.
 * This class uses OpenCL for GPU-accelerated matrix operations.
 */
@Slf4j
@RequiredArgsConstructor
public class OpenClMatrixOperation implements MatrixOperation {
    
    @Getter
    private final ComputeProvider provider;
    private final ResourceManager resourceManager;
    
    // Kernel source code
    private static final String MATRIX_MULTIPLY_KERNEL = 
            "__kernel void matrixMultiply(__global const float* a, __global const float* b, " +
            "__global float* c, const int m, const int n, const int k) { " +
            "    const int row = get_global_id(0); " +
            "    const int col = get_global_id(1); " +
            "    if (row < m && col < n) { " +
            "        float sum = 0.0f; " +
            "        for (int i = 0; i < k; i++) { " +
            "            sum += a[row * k + i] * b[i * n + col]; " +
            "        } " +
            "        c[row * n + col] = sum; " +
            "    } " +
            "}";
    
    private static final String MATRIX_ADD_KERNEL = 
            "__kernel void matrixAdd(__global const float* a, __global const float* b, " +
            "__global float* c, const int size) { " +
            "    const int i = get_global_id(0); " +
            "    if (i < size) { " +
            "        c[i] = a[i] + b[i]; " +
            "    } " +
            "}";
    
    private static final String MATRIX_SUBTRACT_KERNEL = 
            "__kernel void matrixSubtract(__global const float* a, __global const float* b, " +
            "__global float* c, const int size) { " +
            "    const int i = get_global_id(0); " +
            "    if (i < size) { " +
            "        c[i] = a[i] - b[i]; " +
            "    } " +
            "}";
    
    private static final String SCALAR_MULTIPLY_KERNEL = 
            "__kernel void scalarMultiply(__global const float* a, __global float* b, " +
            "const float scalar, const int size) { " +
            "    const int i = get_global_id(0); " +
            "    if (i < size) { " +
            "        b[i] = a[i] * scalar; " +
            "    } " +
            "}";
    
    private static final String MATRIX_TRANSPOSE_KERNEL = 
            "__kernel void matrixTranspose(__global const float* a, __global float* b, " +
            "const int rows, const int cols) { " +
            "    const int i = get_global_id(0); " +
            "    const int j = get_global_id(1); " +
            "    if (i < rows && j < cols) { " +
            "        b[j * rows + i] = a[i * cols + j]; " +
            "    } " +
            "}";
    
    /**
     * Creates a new OpenCL matrix operation with the specified provider.
     *
     * @param provider the compute provider to use
     */
    public OpenClMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
        this.resourceManager = provider.getResourceManager();
        logger.info("Initializing OpenCL matrix operations with provider: {}", provider.getName());
    }
    
    @Override
    public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) {
        logger.debug("OpenCL matrix multiply: {}x{} * {}x{}", rowsA, sharedDim, sharedDim, colsB);
        
        try {
            // Get kernel
            cl_kernel kernel = (cl_kernel) resourceManager.getOrCreateKernel(
                    "matrixMultiply", MATRIX_MULTIPLY_KERNEL);
            
            // Allocate buffers
            cl_mem aBuffer = (cl_mem) resourceManager.allocateBuffer(rowsA * sharedDim * Sizeof.cl_float, "float");
            cl_mem bBuffer = (cl_mem) resourceManager.allocateBuffer(sharedDim * colsB * Sizeof.cl_float, "float");
            cl_mem cBuffer = (cl_mem) resourceManager.allocateBuffer(rowsA * colsB * Sizeof.cl_float, "float");
            
            // Copy data to device
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    aBuffer, CL.CL_TRUE, 0, rowsA * sharedDim * Sizeof.cl_float,
                    Pointer.to(a), 0, null, null);
            
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    bBuffer, CL.CL_TRUE, 0, sharedDim * colsB * Sizeof.cl_float,
                    Pointer.to(b), 0, null, null);
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cBuffer));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{rowsA}));
            CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{colsB}));
            CL.clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{sharedDim}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{rowsA, colsB};
            CL.clEnqueueNDRangeKernel(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    kernel, 2, null, globalWorkSize, null, 0, null, null);
            
            // Read results
            CL.clEnqueueReadBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    cBuffer, CL.CL_TRUE, 0, rowsA * colsB * Sizeof.cl_float,
                    Pointer.to(c), 0, null, null);
            
            // Release buffers
            resourceManager.releaseBuffer(aBuffer);
            resourceManager.releaseBuffer(bBuffer);
            resourceManager.releaseBuffer(cBuffer);
            
        } catch (Exception e) {
            logger.error("Error in OpenCL matrix multiplication", e);
            // Fall back to CPU implementation
            new CpuMatrixOperation(provider).multiply(a, b, c, rowsA, colsB, sharedDim);
        }
    }
    
    @Override
    public void add(float[] a, float[] b, float[] c, int elements) {
        logger.debug("OpenCL matrix add: {} elements", elements);
        
        try {
            // Get kernel
            cl_kernel kernel = (cl_kernel) resourceManager.getOrCreateKernel(
                    "matrixAdd", MATRIX_ADD_KERNEL);
            
            // Allocate buffers
            cl_mem aBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            cl_mem bBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            cl_mem cBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            
            // Copy data to device
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    aBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(a), 0, null, null);
            
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    bBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(b), 0, null, null);
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cBuffer));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{elements}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{elements};
            CL.clEnqueueNDRangeKernel(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    kernel, 1, null, globalWorkSize, null, 0, null, null);
            
            // Read results
            CL.clEnqueueReadBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    cBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(c), 0, null, null);
            
            // Release buffers
            resourceManager.releaseBuffer(aBuffer);
            resourceManager.releaseBuffer(bBuffer);
            resourceManager.releaseBuffer(cBuffer);
            
        } catch (Exception e) {
            logger.error("Error in OpenCL matrix addition", e);
            // Fall back to CPU implementation
            new CpuMatrixOperation(provider).add(a, b, c, elements);
        }
    }
    
    @Override
    public void subtract(float[] a, float[] b, float[] c, int elements) {
        logger.debug("OpenCL matrix subtract: {} elements", elements);
        
        try {
            // Get kernel
            cl_kernel kernel = (cl_kernel) resourceManager.getOrCreateKernel(
                    "matrixSubtract", MATRIX_SUBTRACT_KERNEL);
            
            // Allocate buffers
            cl_mem aBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            cl_mem bBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            cl_mem cBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            
            // Copy data to device
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    aBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(a), 0, null, null);
            
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    bBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(b), 0, null, null);
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cBuffer));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{elements}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{elements};
            CL.clEnqueueNDRangeKernel(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    kernel, 1, null, globalWorkSize, null, 0, null, null);
            
            // Read results
            CL.clEnqueueReadBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    cBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(c), 0, null, null);
            
            // Release buffers
            resourceManager.releaseBuffer(aBuffer);
            resourceManager.releaseBuffer(bBuffer);
            resourceManager.releaseBuffer(cBuffer);
            
        } catch (Exception e) {
            logger.error("Error in OpenCL matrix subtraction", e);
            // Fall back to CPU implementation
            new CpuMatrixOperation(provider).subtract(a, b, c, elements);
        }
    }
    
    @Override
    public void scalarMultiply(float[] a, float[] b, float scalar, int elements) {
        logger.debug("OpenCL scalar multiply: {} elements by {}", elements, scalar);
        
        try {
            // Get kernel
            cl_kernel kernel = (cl_kernel) resourceManager.getOrCreateKernel(
                    "scalarMultiply", SCALAR_MULTIPLY_KERNEL);
            
            // Allocate buffers
            cl_mem aBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            cl_mem bBuffer = (cl_mem) resourceManager.allocateBuffer(elements * Sizeof.cl_float, "float");
            
            // Copy data to device
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    aBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(a), 0, null, null);
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_float, Pointer.to(new float[]{scalar}));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{elements}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{elements};
            CL.clEnqueueNDRangeKernel(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    kernel, 1, null, globalWorkSize, null, 0, null, null);
            
            // Read results
            CL.clEnqueueReadBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    bBuffer, CL.CL_TRUE, 0, elements * Sizeof.cl_float,
                    Pointer.to(b), 0, null, null);
            
            // Release buffers
            resourceManager.releaseBuffer(aBuffer);
            resourceManager.releaseBuffer(bBuffer);
            
        } catch (Exception e) {
            logger.error("Error in OpenCL scalar multiplication", e);
            // Fall back to CPU implementation
            new CpuMatrixOperation(provider).scalarMultiply(a, b, scalar, elements);
        }
    }
    
    @Override
    public void transpose(float[] a, float[] b, int rows, int cols) {
        logger.debug("OpenCL matrix transpose: {}x{}", rows, cols);
        
        try {
            // Get kernel
            cl_kernel kernel = (cl_kernel) resourceManager.getOrCreateKernel(
                    "matrixTranspose", MATRIX_TRANSPOSE_KERNEL);
            
            // Allocate buffers
            cl_mem aBuffer = (cl_mem) resourceManager.allocateBuffer(rows * cols * Sizeof.cl_float, "float");
            cl_mem bBuffer = (cl_mem) resourceManager.allocateBuffer(rows * cols * Sizeof.cl_float, "float");
            
            // Copy data to device
            CL.clEnqueueWriteBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    aBuffer, CL.CL_TRUE, 0, rows * cols * Sizeof.cl_float,
                    Pointer.to(a), 0, null, null);
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_int, Pointer.to(new int[]{rows}));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{cols}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{rows, cols};
            CL.clEnqueueNDRangeKernel(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    kernel, 2, null, globalWorkSize, null, 0, null, null);
            
            // Read results
            CL.clEnqueueReadBuffer(
                    (cl_command_queue) resourceManager.getCachedData("commandQueue"),
                    bBuffer, CL.CL_TRUE, 0, rows * cols * Sizeof.cl_float,
                    Pointer.to(b), 0, null, null);
            
            // Release buffers
            resourceManager.releaseBuffer(aBuffer);
            resourceManager.releaseBuffer(bBuffer);
            
        } catch (Exception e) {
            logger.error("Error in OpenCL matrix transpose", e);
            // Fall back to CPU implementation
            new CpuMatrixOperation(provider).transpose(a, b, rows, cols);
        }
    }
    
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
    
    @Override
    public void release() {
        logger.info("Releasing OpenCL matrix operation resources");
        // No specific resources to release, as ResourceManager handles cleanup
    }
}
