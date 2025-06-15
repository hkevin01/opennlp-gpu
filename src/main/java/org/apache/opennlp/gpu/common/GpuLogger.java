package org.apache.opennlp.gpu.common;

/**
 * Simple logging wrapper for GPU operations
 * Provides consistent logging interface across the GPU acceleration framework
 */
public class GpuLogger {
    
    private final String className;
    
    private GpuLogger(String className) {
        this.className = className;
    }
    
    public static GpuLogger getLogger(Class<?> clazz) {
        return new GpuLogger(clazz.getSimpleName());
    }
    
    public void info(String message) {
        System.out.println("[INFO] " + className + ": " + message);
    }
    
    public void warn(String message) {
        System.out.println("[WARN] " + className + ": " + message);
    }
    
    public void error(String message) {
        System.err.println("[ERROR] " + className + ": " + message);
    }
    
    public void error(String message, Exception e) {
        System.err.println("[ERROR] " + className + ": " + message);
        if (e != null) {
            e.printStackTrace();
        }
    }
    
    public void debug(String message) {
        // Only print debug messages if debug mode is enabled
        String debugMode = System.getProperty("gpu.debug", "false");
        if ("true".equalsIgnoreCase(debugMode)) {
            System.out.println("[DEBUG] " + className + ": " + message);
        }
    }
}
