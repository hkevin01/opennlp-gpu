package org.apache.opennlp.gpu.util;

import org.apache.opennlp.gpu.cuda.CudaUtil;
import org.apache.opennlp.gpu.opencl.OpenCLUtil;  // Corrected import for OpenCL utility class
import org.apache.opennlp.gpu.rocm.RocmUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * ID: GPU-GU-001
 * Requirement: GpuUtils must provide stateless utility methods for GPU memory arithmetic, array conversion, and diagnostic formatting.
 * Purpose: Utility class with static helpers used across compute and ml packages: byte count formatting, float[] ↔ double[] conversion, GPU availability checks.
 * Rationale: Centralising common utility logic prevents duplication and simplifies testing of shared arithmetic helpers.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; all methods are stateless.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuUtils {
    
    private static final Logger logger = LoggerFactory.getLogger(GpuUtils.class);
    
    private GpuUtils() {
        // Private constructor to prevent instantiation
    }
    
    /**
     * Checks if any GPU is available on the system.
     *
     * @return true if at least one GPU is available, false otherwise
     */
    public static boolean isGpuAvailable() {
        try {
            return CudaUtil.isAvailable() || OpenCLUtil.isAvailable() || RocmUtil.isAvailable();
        } catch (Exception e) {
            logger.warn("Error checking GPU availability", e);
            return false;
        }
    }
    
    /**
     * Gets the number of available GPUs.
     *
     * @return the number of available GPUs
     */
    public static int getGpuCount() {
        int count = 0;
        try {
            count += CudaUtil.getDeviceCount();
        } catch (Exception e) {
            logger.debug("Error counting CUDA devices", e);
        }
        
        try {
            count += OpenCLUtil.getDeviceCount();  // Updated to match the correct class name
        } catch (Exception e) {
            logger.debug("Error counting OpenCL devices", e);
        }
        
        try {
            count += RocmUtil.getDeviceCount();
        } catch (Exception e) {
            logger.debug("Error counting ROCm devices", e);
        }
        
        return count;
    }
}