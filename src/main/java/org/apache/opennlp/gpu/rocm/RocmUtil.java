package org.apache.opennlp.gpu.rocm;

import org.apache.opennlp.gpu.util.NativeLibraryLoader;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Utility class for ROCm operations, providing initialization and basic operations
 * for AMD GPU acceleration.
 */
@Slf4j
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class RocmUtil {
    private static boolean initialized = false;
    private static boolean available = false;
    
    // Static initializer to load native library
    static {
        try {
            if (NativeLibraryLoader.loadLibrary("opennlp_rocm")) {
                log.info("ROCm native library loaded successfully");
            } else {
                log.warn("Failed to load ROCm native library");
            }
        } catch (Exception e) {
            log.error("Error loading ROCm native library", e);
        }
    }
    
    // JNI method declarations for ROCm operations
    private static native boolean initializeRocm();
    private static native int getDeviceCount();
    private static native String getDeviceName(int deviceId);
    private static native long getDeviceMemory(int deviceId);
    private static native String getDeviceArchitecture(int deviceId);
    
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
                log.info("ROCm initialized successfully. Found {} device(s)", deviceCount);
                
                // Log information about each device
                for (int i = 0; i < deviceCount; i++) {
                    log.info("ROCm Device {}: {}", i, getDeviceName(i));
                    log.info("  Memory: {} MB", getDeviceMemory(i) / (1024 * 1024));
                    log.info("  Architecture: {}", getDeviceArchitecture(i));
                }
            } else {
                log.warn("ROCm initialization failed - no compatible AMD GPUs found");
            }
        } catch (Exception e) {
            log.error("Error initializing ROCm", e);
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
    
    /**
     * Verify ROCm installation and requirements.
     * @return a string describing the verification results
     */
    public static String verifyRocmInstallation() {
        StringBuilder result = new StringBuilder();
        
        // Check environment variables
        String rocmPath = System.getenv("ROCM_PATH");
        if (rocmPath == null) {
            rocmPath = "/opt/rocm"; // Default path
        }
        result.append("ROCm path: ").append(rocmPath).append("\n");
        
        // Check for HIP_PLATFORM
        String hipPlatform = System.getenv("HIP_PLATFORM");
        result.append("HIP platform: ").append(hipPlatform != null ? hipPlatform : "not set (using default)").append("\n");
        
        // Check if native library can be loaded
        if (isAvailable()) {
            result.append("ROCm status: Available\n");
            result.append("Device count: ").append(getDeviceCount()).append("\n");
        } else {
            result.append("ROCm status: Not available\n");
            result.append("Possible issues:\n");
            result.append("- ROCm not installed or not in PATH\n");
            result.append("- No compatible AMD GPUs found\n");
            result.append("- Missing required libraries\n");
        }
        
        return result.toString();
    }
}
