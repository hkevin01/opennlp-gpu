package org.apache.opennlp.gpu.util;

import org.apache.opennlp.gpu.cuda.CudaUtil;
import org.apache.opennlp.gpu.opencl.OpenCLUtil;  // Corrected import for OpenCL utility class
import org.apache.opennlp.gpu.rocm.RocmUtil;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for GPU-related operations.
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