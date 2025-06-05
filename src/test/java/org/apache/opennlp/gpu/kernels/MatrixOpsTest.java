package org.apache.opennlp.gpu.kernels;

import org.apache.opennlp.gpu.common.GpuDevice;
import org.jocl.*;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIf;

import java.util.List;

import static org.junit.jupiter.api.Assertions.*;

/**
 * Tests for the GPU-accelerated matrix operations.
 * These tests will be skipped if no GPU is available.
 */
@EnabledIf("isGpuAvailable")
public class MatrixOpsTest {
    
    private cl_context context;
    private cl_command_queue commandQueue;
    private MatrixOps matrixOps;
    
    /**
     * Check if a GPU is available for testing.
     */
    public static boolean isGpuAvailable() {
        try {
            List<GpuDevice> devices = GpuDevice.getAvailableDevices();
            return !devices.isEmpty();
        } catch (Exception e) {
            return false;
        }
    }
    
    @BeforeEach
    public void setUp() {
        try {
            // Initialize OpenCL
            CL.setExceptionsEnabled(true);
            
            // Get GPU device
            List<GpuDevice> devices = GpuDevice.getAvailableDevices();
            if (devices.isEmpty()) {
                throw new RuntimeException("No GPU devices available");
            }
            
            GpuDevice device = devices.get(0);
            
            // Create context and command queue
            cl_context_properties contextProperties = new cl_context_properties();
            contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, device.getDeviceId().getInfo());
            
            int[] errorCode = new int[1];
            context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device.getDeviceId()}, 
                                        null, null, errorCode);
            
            commandQueue = CL.clCreateCommandQueue(context, device.getDeviceId(), 0, errorCode);
            
            // Create matrix operations
            matrixOps = new MatrixOps(context, commandQueue);
        } catch (Exception e) {
            throw new RuntimeException("Error setting up test", e);
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
        assertArrayEquals(expected, c, 0.001f);
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
        assertArrayEquals(expected, c, 0.001f);
    }
}
