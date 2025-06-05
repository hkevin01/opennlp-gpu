package org.apache.opennlp.gpu.cuda;

import org.apache.opennlp.gpu.util.NativeLibraryLoader;
import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * Utility class for CUDA operations, providing initialization and basic operations.
 */
@Slf4j
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public class CudaUtil {
    private static boolean initialized = false;
    private static boolean available = false;
    
    // Static initializer to load native library
    static {
        try {
            if (NativeLibraryLoader.loadLibrary("opennlp_cuda")) {
                log.info("CUDA native library loaded successfully");
            } else {
                log.warn("Failed to load CUDA native library");
            }
        } catch (Exception e) {
            log.error("Error loading CUDA native library", e);
        }
    }
    
    // JNI method declarations for CUDA operations
    private static native boolean initializeCuda();
    private static native int getDeviceCount();
    private static native String getDeviceName(int deviceId);
    private static native long getDeviceMemory(int deviceId);
    private static native int getComputeCapability(int deviceId);
    
    /**
     * Initialize CUDA if available.
     * @return true if CUDA is available and initialized successfully
     */
    public static synchronized boolean initialize() {
        if (initialized) {
            return available;
        }
        
        try {
            // Initialize CUDA
            available = initializeCuda();
            
            if (available) {
                int deviceCount = getDeviceCount();
                log.info("CUDA initialized successfully. Found {} device(s)", deviceCount);
                
                // Log information about each device
                for (int i = 0; i < deviceCount; i++) {
                    log.info("CUDA Device {}: {}", i, getDeviceName(i));
                    log.info("  Memory: {} MB", getDeviceMemory(i) / (1024 * 1024));
                    log.info("  Compute Capability: {}", getComputeCapability(i));
                }
            } else {
                log.warn("CUDA initialization failed");
            }
        } catch (UnsatisfiedLinkError e) {
            log.warn("CUDA native library not found: {}", e.getMessage());
            available = false;
        } catch (Exception e) {
            log.error("Error initializing CUDA", e);
            available = false;
        }
        
        initialized = true;
        return available;
    }
    
    /**
     * Check if CUDA is available.
     * @return true if CUDA is available
     */
    public static boolean isAvailable() {
        if (!initialized) {
            initialize();
        }
        return available;
    }
    
    /**
     * Get the number of CUDA devices.
     * @return the number of devices, or 0 if CUDA is not available
     */
    public static int getDeviceCount() {
        if (!isAvailable()) {
            return 0;
        }
        return getDeviceCount();
    }
}
