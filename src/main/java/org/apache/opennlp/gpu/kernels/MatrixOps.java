package org.apache.opennlp.gpu.kernels;

import java.io.IOException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.io.InputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.nio.charset.StandardCharsets;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.HashMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import java.util.Map;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.opennlp.gpu.common.MemoryManager;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.CL;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.Pointer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.Sizeof;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_command_queue;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_context;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_device_id;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_kernel;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_mem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.cl_program;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Provides GPU-accelerated matrix operations using OpenCL.
 * These operations are commonly used in machine learning algorithms.
 */
public class MatrixOps {
    private static final Logger logger = LoggerFactory.getLogger(MatrixOps.class);
    
    private final cl_context context;
    private final cl_command_queue commandQueue;
    private final MemoryManager memoryManager;
    
    // Cache of compiled OpenCL programs
    private final Map<String, cl_program> programCache = new HashMap<>();
    private final Map<String, cl_kernel> kernelCache = new HashMap<>();
    
    /**
     * Creates a new MatrixOps instance for the specified OpenCL context and device.
     *
     * @param context the OpenCL context
     * @param commandQueue the OpenCL command queue
     */
    public MatrixOps(cl_context context, cl_command_queue commandQueue) {
        this.context = context;
        this.commandQueue = commandQueue;
        this.memoryManager = new MemoryManager(context, commandQueue);
        
        // Initialize OpenCL programs
        try {
            initializePrograms();
            logger.info("Matrix operations initialized");
        } catch (Exception e) {
            logger.error("Failed to initialize matrix operations", e);
            throw new RuntimeException("Failed to initialize matrix operations", e);
        }
    }
    
    /**
     * Performs matrix multiplication: C = A * B.
     *
     * @param a matrix A (m x k)
     * @param b matrix B (k x n)
     * @param c output matrix C (m x n)
     * @param m number of rows in A and C
     * @param n number of columns in B and C
     * @param k number of columns in A and rows in B
     */
    public void matrixMultiply(float[] a, float[] b, float[] c, int m, int n, int k) {
        logger.debug("Matrix multiply: A({} x {}), B({} x {}), C({} x {})",
                m, k, k, n, m, n);
        
        // Allocate device memory
        cl_mem aBuffer = memoryManager.allocateBuffer(m * k * Sizeof.cl_float, true);
        cl_mem bBuffer = memoryManager.allocateBuffer(k * n * Sizeof.cl_float, true);
        cl_mem cBuffer = memoryManager.allocateBuffer(m * n * Sizeof.cl_float, false);
        
        try {
            // Copy input data to device
            memoryManager.copyToDevice(Pointer.to(a), aBuffer, m * k * Sizeof.cl_float);
            memoryManager.copyToDevice(Pointer.to(b), bBuffer, k * n * Sizeof.cl_float);
            
            // Get the kernel
            cl_kernel kernel = getKernel("matrixMultiply");
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cBuffer));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{m}));
            CL.clSetKernelArg(kernel, 4, Sizeof.cl_int, Pointer.to(new int[]{n}));
            CL.clSetKernelArg(kernel, 5, Sizeof.cl_int, Pointer.to(new int[]{k}));
            
            // Execute the kernel
            long[] globalWorkSize = new long[]{m, n};
            CL.clEnqueueNDRangeKernel(commandQueue, kernel, 2, null,
                    globalWorkSize, null, 0, null, null);
            
            // Read the result back
            memoryManager.copyFromDevice(cBuffer, Pointer.to(c), m * n * Sizeof.cl_float);
            
        } finally {
            // Clean up
            memoryManager.releaseBuffer(aBuffer);
            memoryManager.releaseBuffer(bBuffer);
            memoryManager.releaseBuffer(cBuffer);
        }
    }
    
    /**
     * Performs element-wise addition: C = A + B.
     *
     * @param a matrix A
     * @param b matrix B
     * @param c output matrix C
     * @param size number of elements
     */
    public void matrixAdd(float[] a, float[] b, float[] c, int size) {
        logger.debug("Matrix add: size={}", size);
        
        // Allocate device memory
        cl_mem aBuffer = memoryManager.allocateBuffer(size * Sizeof.cl_float, true);
        cl_mem bBuffer = memoryManager.allocateBuffer(size * Sizeof.cl_float, true);
        cl_mem cBuffer = memoryManager.allocateBuffer(size * Sizeof.cl_float, false);
        
        try {
            // Copy input data to device
            memoryManager.copyToDevice(Pointer.to(a), aBuffer, size * Sizeof.cl_float);
            memoryManager.copyToDevice(Pointer.to(b), bBuffer, size * Sizeof.cl_float);
            
            // Get the kernel
            cl_kernel kernel = getKernel("matrixAdd");
            
            // Set kernel arguments
            CL.clSetKernelArg(kernel, 0, Sizeof.cl_mem, Pointer.to(aBuffer));
            CL.clSetKernelArg(kernel, 1, Sizeof.cl_mem, Pointer.to(bBuffer));
            CL.clSetKernelArg(kernel, 2, Sizeof.cl_mem, Pointer.to(cBuffer));
            CL.clSetKernelArg(kernel, 3, Sizeof.cl_int, Pointer.to(new int[]{size}));
            
            // Execute the kernel
            long[] globalWorkSize = new long[]{size};
            CL.clEnqueueNDRangeKernel(commandQueue, kernel, 1, null,
                    globalWorkSize, null, 0, null, null);
            
            // Read the result back
            memoryManager.copyFromDevice(cBuffer, Pointer.to(c), size * Sizeof.cl_float);
            
        } finally {
            // Clean up
            memoryManager.releaseBuffer(aBuffer);
            memoryManager.releaseBuffer(bBuffer);
            memoryManager.releaseBuffer(cBuffer);
        }
    }
    
    /**
     * Initialize OpenCL programs from kernel source files.
     */
    private void initializePrograms() throws IOException {
        // Load and compile the matrix operations program
        String source = loadKernelSource("/opencl/matrix_ops.cl");
        createProgram("matrix_ops", source);
        
        // Create kernels from the program
        createKernel("matrix_ops", "matrixMultiply");
        createKernel("matrix_ops", "matrixAdd");
    }
    
    /**
     * Loads a kernel source file from the classpath.
     *
     * @param resourcePath the path to the kernel source file
     * @return the kernel source code as a string
     */
    private String loadKernelSource(String resourcePath) throws IOException {
        try (InputStream is = getClass().getResourceAsStream(resourcePath)) {
            if (is == null) {
                throw new IOException("Kernel source not found: " + resourcePath);
            }
            
            byte[] bytes = is.readAllBytes();
            return new String(bytes, StandardCharsets.UTF_8);
        }
    }
    
    /**
     * Creates and compiles an OpenCL program from source code.
     *
     * @param name the name of the program
     * @param source the OpenCL source code
     */
    private void createProgram(String name, String source) {
        try {
            // Create the program
            cl_program program = CL.clCreateProgramWithSource(context, 1,
                    new String[]{source}, null, null);
            
            // Build the program
            int err = CL.clBuildProgram(program, 0, null, null, null, null);
            if (err != CL.CL_SUCCESS) {
                // If there was an error, get the build log
                cl_device_id[] devices = new cl_device_id[1];
                CL.clGetContextInfo(context, CL.CL_CONTEXT_DEVICES,
                        Sizeof.cl_device_id, Pointer.to(devices), null);
                
                long[] logSize = new long[1];
                CL.clGetProgramBuildInfo(program, devices[0], CL.CL_PROGRAM_BUILD_LOG,
                        0, null, logSize);
                
                byte[] logData = new byte[(int)logSize[0]];
                CL.clGetProgramBuildInfo(program, devices[0], CL.CL_PROGRAM_BUILD_LOG,
                        logData.length, Pointer.to(logData), null);
                
                String log = new String(logData, 0, logData.length, StandardCharsets.UTF_8);
                logger.error("OpenCL program build error: {}", log);
                throw new RuntimeException("Error building OpenCL program: " + log);
            }
            
            programCache.put(name, program);
            logger.debug("Created OpenCL program: {}", name);
        } catch (Exception e) {
            logger.error("Error creating OpenCL program", e);
            throw new RuntimeException("Error creating OpenCL program", e);
        }
    }
    
    /**
     * Creates an OpenCL kernel from a compiled program.
     *
     * @param programName the name of the program
     * @param kernelName the name of the kernel function
     */
    private void createKernel(String programName, String kernelName) {
        try {
            cl_program program = programCache.get(programName);
            if (program == null) {
                throw new IllegalArgumentException("Program not found: " + programName);
            }
            
            // Create the kernel
            cl_kernel kernel = CL.clCreateKernel(program, kernelName, null);
            kernelCache.put(kernelName, kernel);
            logger.debug("Created OpenCL kernel: {}", kernelName);
        } catch (Exception e) {
            logger.error("Error creating OpenCL kernel", e);
            throw new RuntimeException("Error creating OpenCL kernel", e);
        }
    }
    
    /**
     * Gets a cached OpenCL kernel.
     *
     * @param kernelName the name of the kernel
     * @return the OpenCL kernel
     */
    private cl_kernel getKernel(String kernelName) {
        cl_kernel kernel = kernelCache.get(kernelName);
        if (kernel == null) {
            throw new IllegalArgumentException("Kernel not found: " + kernelName);
        }
        return kernel;
    }
    
    /**
     * Releases all resources associated with this instance.
     */
    public void release() {
        // Release kernels
        for (cl_kernel kernel : kernelCache.values()) {
            CL.clReleaseKernel(kernel);
        }
        kernelCache.clear();
        
        // Release programs
        for (cl_program program : programCache.values()) {
            CL.clReleaseProgram(program);
        }
        programCache.clear();
        
        // Release memory manager
        memoryManager.release();
        
        logger.info("Matrix operations released");
    }
}
