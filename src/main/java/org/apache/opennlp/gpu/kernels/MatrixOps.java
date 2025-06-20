package org.apache.opennlp.gpu.kernels;

import static org.jocl.CL.CL_MEM_COPY_HOST_PTR;
import static org.jocl.CL.CL_MEM_READ_ONLY;
import static org.jocl.CL.CL_MEM_WRITE_ONLY;
import static org.jocl.CL.CL_SUCCESS;
import static org.jocl.CL.CL_TRUE;
import static org.jocl.CL.clBuildProgram;
import static org.jocl.CL.clCreateBuffer;
import static org.jocl.CL.clCreateKernel;
import static org.jocl.CL.clCreateProgramWithSource;
import static org.jocl.CL.clEnqueueNDRangeKernel;
import static org.jocl.CL.clEnqueueReadBuffer;
import static org.jocl.CL.clReleaseMemObject;
import static org.jocl.CL.clSetKernelArg;

import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_program;

/**
 * GPU-accelerated matrix operations using OpenCL kernels
 */
public class MatrixOps {
    
    private final cl_context context;
    private final cl_command_queue commandQueue;
    private final cl_device_id device;
    
    // Pre-compiled kernels
    private cl_program program;
    private cl_kernel matrixMultiplyKernel;
    private cl_kernel matrixAddKernel;
    private cl_kernel vectorNormalizeKernel;
    
    // OpenCL kernel source code
    private static final String MATRIX_KERNELS = 
        "__kernel void matrix_multiply(__global float* A, __global float* B, __global float* C, " +
        "                              int M, int N, int K) {" +
        "    int row = get_global_id(0);" +
        "    int col = get_global_id(1);" +
        "    if (row < M && col < N) {" +
        "        float sum = 0.0f;" +
        "        for (int k = 0; k < K; k++) {" +
        "            sum += A[row * K + k] * B[k * N + col];" +
        "        }" +
        "        C[row * N + col] = sum;" +
        "    }" +
        "}" +
        
        "__kernel void matrix_add(__global float* A, __global float* B, __global float* C, int size) {" +
        "    int idx = get_global_id(0);" +
        "    if (idx < size) {" +
        "        C[idx] = A[idx] + B[idx];" +
        "    }" +
        "}" +
        
        "__kernel void vector_normalize(__global float* vector, int size) {" +
        "    int idx = get_global_id(0);" +
        "    if (idx < size) {" +
        "        // Simple L2 normalization (simplified version)" +
        "        float norm = sqrt(vector[idx] * vector[idx]);" +
        "        if (norm > 0.0f) vector[idx] /= norm;" +
        "    }" +
        "}";
    
    public MatrixOps(cl_context context, cl_command_queue commandQueue, cl_device_id device) {
        this.context = context;
        this.commandQueue = commandQueue;
        this.device = device;
        
        initializeKernels();
    }
    
    private void initializeKernels() {
        try {
            // Create program from source
            program = clCreateProgramWithSource(context, 1, new String[]{MATRIX_KERNELS}, null, null);
            
            // Build program
            int buildResult = clBuildProgram(program, 0, null, null, null, null);
            if (buildResult != CL_SUCCESS) {
                // Fallback to CPU if GPU compilation fails
                System.err.println("Warning: GPU kernel compilation failed, using CPU fallback");
                return;
            }
            
            // Create kernels
            matrixMultiplyKernel = clCreateKernel(program, "matrix_multiply", null);
            matrixAddKernel = clCreateKernel(program, "matrix_add", null);
            vectorNormalizeKernel = clCreateKernel(program, "vector_normalize", null);
            
        } catch (Exception e) {
            System.err.println("Warning: GPU kernel initialization failed: " + e.getMessage());
            // Continue with CPU fallback
        }
    }
    
    /**
     * Perform matrix multiplication: C = A * B using GPU kernels
     */
    public void matrixMultiply(float[] a, float[] b, float[] c, int m, int n, int k) {
        if (matrixMultiplyKernel == null) {
            // Fallback to CPU implementation
            matrixMultiplyCpu(a, b, c, m, n, k);
            return;
        }
        
        try {
            // Create buffers
            cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * a.length, Pointer.to(a), null);
            cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * b.length, Pointer.to(b), null);
            cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_float * c.length, null, null);
            
            // Set kernel arguments
            clSetKernelArg(matrixMultiplyKernel, 0, Sizeof.cl_mem, Pointer.to(bufferA));
            clSetKernelArg(matrixMultiplyKernel, 1, Sizeof.cl_mem, Pointer.to(bufferB));
            clSetKernelArg(matrixMultiplyKernel, 2, Sizeof.cl_mem, Pointer.to(bufferC));
            clSetKernelArg(matrixMultiplyKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            clSetKernelArg(matrixMultiplyKernel, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
            clSetKernelArg(matrixMultiplyKernel, 5, Sizeof.cl_int, Pointer.to(new int[]{k}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{m, n};
            clEnqueueNDRangeKernel(commandQueue, matrixMultiplyKernel, 2, null,
                    globalWorkSize, null, 0, null, null);
            
            // Read result
            clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0,
                    c.length * Sizeof.cl_float, Pointer.to(c), 0, null, null);
            
            // Cleanup
            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);
            
        } catch (Exception e) {
            System.err.println("GPU matrix multiply failed, falling back to CPU: " + e.getMessage());
            matrixMultiplyCpu(a, b, c, m, n, k);
        }
    }
    
    /**
     * CPU fallback for matrix multiplication
     */
    private void matrixMultiplyCpu(float[] a, float[] b, float[] c, int m, int n, int k) {
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
    }
    
    /**
     * Perform matrix addition: C = A + B using GPU kernels
     */
    public void matrixAdd(float[] a, float[] b, float[] c, int size) {
        if (matrixAddKernel == null) {
            // Fallback to CPU implementation
            matrixAddCpu(a, b, c, size);
            return;
        }
        
        try {
            // Create buffers
            cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * size, Pointer.to(a), null);
            cl_mem bufferB = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                    Sizeof.cl_float * size, Pointer.to(b), null);
            cl_mem bufferC = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
                    Sizeof.cl_float * size, null, null);
            
            // Set kernel arguments
            clSetKernelArg(matrixAddKernel, 0, Sizeof.cl_mem, Pointer.to(bufferA));
            clSetKernelArg(matrixAddKernel, 1, Sizeof.cl_mem, Pointer.to(bufferB));
            clSetKernelArg(matrixAddKernel, 2, Sizeof.cl_mem, Pointer.to(bufferC));
            clSetKernelArg(matrixAddKernel, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));
            
            // Execute kernel
            long[] globalWorkSize = new long[]{size};
            clEnqueueNDRangeKernel(commandQueue, matrixAddKernel, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            // Read result
            clEnqueueReadBuffer(commandQueue, bufferC, CL_TRUE, 0,
                    size * Sizeof.cl_float, Pointer.to(c), 0, null, null);
            
            // Cleanup
            clReleaseMemObject(bufferA);
            clReleaseMemObject(bufferB);
            clReleaseMemObject(bufferC);
            
        } catch (Exception e) {
            System.err.println("GPU matrix add failed, falling back to CPU: " + e.getMessage());
            matrixAddCpu(a, b, c, size);
        }
    }
    
    /**
     * CPU fallback for matrix addition
     */
    private void matrixAddCpu(float[] a, float[] b, float[] c, int size) {
        for (int i = 0; i < size; i++) {
            c[i] = a[i] + b[i];
        }
    }
    
    /**
     * Release OpenCL resources
     */
    public void release() {
        // TODO: Release OpenCL kernels and buffers
        // For now, this is a no-op
    }
}
