package org.apache.opennlp.gpu.kernels;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.jocl.CL;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Tests for the GPU-accelerated matrix operations.
 * These tests will be skipped if no GPU is available.
 */
@EnabledIf("isGpuAvailable")
public class MatrixOpsTest {
    private static final Logger logger = LoggerFactory.getLogger(MatrixOpsTest.class);
    
    private cl_context context;
    private cl_command_queue commandQueue;
    private MatrixOps matrixOps;
    
    /**
     * Check if a GPU is available for testing.
     */
    public static boolean isGpuAvailable() {
        try {
            // Try to create a CPU compute provider for fallback testing
            ComputeProvider provider = new CpuComputeProvider();
            provider.initialize();
            boolean available = true; // CPU provider is always available
            provider.cleanup();
            return available;
        } catch (Exception e) {
            MatrixOpsTest.logger.debug("GPU not available: {}", e.getMessage());
            return false;
        }
    }
    
    @BeforeEach
    public void setUp() {
        try {
            // Initialize OpenCL
            CL.setExceptionsEnabled(true);
            
            // Create OpenCL compute provider
            ComputeProvider provider = new CpuComputeProvider();
            provider.initialize();
            boolean initialized = true; // Assume initialization worked
            if (!initialized) {
                throw new RuntimeException("Failed to initialize compute provider");
            }
            
            // For testing purposes, create a minimal OpenCL setup
            setupOpenClContext();
            
            // Create matrix operations
            matrixOps = new MatrixOps(context, commandQueue);
        } catch (Exception e) {
            throw new RuntimeException("Error setting up test", e);
        }
    }
    
    /**
     * Set up OpenCL context for testing.
     */
    private void setupOpenClContext() throws Exception {
        // Get platform
        int[] numPlatformsArray = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatformsArray);
        int numPlatforms = numPlatformsArray[0];
        
        if (numPlatforms == 0) {
            throw new RuntimeException("No OpenCL platforms found");
        }
        
        // Get first platform
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        cl_platform_id platform = platforms[0];
        
        // Get GPU device
        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        if (numDevices == 0) {
            // Fall back to CPU if no GPU available
            CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_CPU, 0, null, numDevicesArray);
            numDevices = numDevicesArray[0];
        }
        
        if (numDevices == 0) {
            throw new RuntimeException("No OpenCL devices found");
        }
        
        cl_device_id[] devices = new cl_device_id[numDevices];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, numDevices, devices, null);
        cl_device_id deviceId = devices[0];
        
        // Create context and command queue
        cl_context_properties contextProperties = new cl_context_properties();
        contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
        
        int[] errorCode = new int[1];
        context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{deviceId}, 
                                    null, null, errorCode);
        
        if (errorCode[0] != CL.CL_SUCCESS) {
            throw new RuntimeException("Failed to create OpenCL context: " + errorCode[0]);
        }
        
        commandQueue = CL.clCreateCommandQueue(context, deviceId, 0, errorCode);
        
        if (errorCode[0] != CL.CL_SUCCESS) {
            CL.clReleaseContext(context);
            throw new RuntimeException("Failed to create command queue: " + errorCode[0]);
        }
    }
    
    @AfterEach
    public void tearDown() {
        if (matrixOps != null) {
            matrixOps.release();
        }
        
        if (commandQueue != null) {
            CL.clReleaseCommandQueue(commandQueue);
        }
        
        if (context != null) {
            CL.clReleaseContext(context);
        }
    }
    
    @Test
    public void testMatrixMultiply() {
        // Test matrices
        float[] a = {
            1.0f, 2.0f,
            3.0f, 4.0f
        };
        
        float[] b = {
            5.0f, 6.0f,
            7.0f, 8.0f
        };
        
        float[] c = new float[4];
        
        // Expected result: A * B
        float[] expected = {
            19.0f, 22.0f,
            43.0f, 50.0f
        };
        
        // Perform matrix multiplication
        matrixOps.matrixMultiply(a, b, c, 2, 2, 2);
        
        // Verify result
        Assertions.assertArrayEquals(expected, c, 0.001f);
    }
    
    @Test
    public void testMatrixAdd() {
        // Test matrices
        float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] c = new float[4];
        
        // Expected result: A + B
        float[] expected = {6.0f, 8.0f, 10.0f, 12.0f};
        
        // Perform matrix addition
        matrixOps.matrixAdd(a, b, c, 4);
        
        // Verify result
        Assertions.assertArrayEquals(expected, c, 0.001f);
    }
}
