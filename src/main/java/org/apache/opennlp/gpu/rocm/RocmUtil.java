package org.apache.opennlp.gpu.rocm;

import org.apache.opennlp.gpu.util.NativeLibraryLoader;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Utility class for ROCm operations, providing initialization and basic operations.
 * This enables support for AMD GPUs using the ROCm platform.
 */
public class RocmUtil {
    private static final Logger logger = LoggerFactory.getLogger(RocmUtil.class);
    private static boolean initialized = false;
    private static boolean available = false;
    
    // Static initializer to load native library
    static {
        try {
            if (NativeLibraryLoader.loadLibrary("opennlp_rocm")) {
                logger.info("ROCm native library loaded successfully");
            } else {
                logger.warn("Failed to load ROCm native library");
            }
        } catch (Exception e) {
            logger.error("Error loading ROCm native library", e);
        }
    }
    
    // JNI method declarations for ROCm operations
    private static native boolean initializeRocm();
    private static native int getDeviceCount();
    private static native String getDeviceName(int deviceId);
    private static native long getDeviceMemory(int deviceId);
    private static native int getComputeCapability(int deviceId);
    
    /**
     * Initialize ROCm if available.
     * @return true if ROCm is available and initialized successfully
     */
    public static synchronized boolean initialize() {
        if (initialized) {
            return available;
        }
        
        try {
            // Initialize ROCm
            available = initializeRocm();
            
            if (available) {
                int deviceCount = getDeviceCount();
                logger.info("ROCm initialized successfully. Found {} device(s)", deviceCount);
                
                // Log information about each device
                for (int i = 0; i < deviceCount; i++) {
                    logger.info("ROCm Device {}: {}", i, getDeviceName(i));
                    logger.info("  Memory: {} MB", getDeviceMemory(i) / (1024 * 1024));
                    logger.info("  Compute Capability: {}", getComputeCapability(i));
                }
            } else {
                logger.warn("ROCm initialization failed");
            }
        } catch (UnsatisfiedLinkError e) {
            logger.warn("ROCm native library not found: {}", e.getMessage());
            available = false;
        } catch (Exception e) {
            logger.error("Error initializing ROCm", e);
            available = false;
        }
        
        initialized = true;
        return available;
    }
    
    /**
     * Check if ROCm is available.
     * @return true if ROCm is available
     */
    public static boolean isAvailable() {
        if (!initialized) {
            initialize();
        }
        return available;
    }
    
    /**
     * Get the number of ROCm devices.
     * @return the number of devices, or 0 if ROCm is not available
     */
    public static int getDeviceCount() {
        if (!isAvailable()) {
            return 0;
        }
        return getDeviceCount();
    }
}
