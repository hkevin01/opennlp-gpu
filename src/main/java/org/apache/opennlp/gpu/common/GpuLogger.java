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
    
    public void info(String message, Object... params) {
        System.out.println("[INFO] " + className + ": " + formatMessage(message, params));
    }
    
    public void warn(String message) {
        System.out.println("[WARN] " + className + ": " + message);
    }
    
    public void warn(String message, Object... params) {
        System.out.println("[WARN] " + className + ": " + formatMessage(message, params));
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
    
    public void error(String message, Object... params) {
        System.err.println("[ERROR] " + className + ": " + formatMessage(message, params));
    }
    
    public void debug(String message) {
        // Only print debug messages if debug mode is enabled
        String debugMode = System.getProperty("gpu.debug", "false");
        if ("true".equalsIgnoreCase(debugMode)) {
            System.out.println("[DEBUG] " + className + ": " + message);
        }
    }
    
    public void debug(String message, Object... params) {
        String debugMode = System.getProperty("gpu.debug", "false");
        if ("true".equalsIgnoreCase(debugMode)) {
            System.out.println("[DEBUG] " + className + ": " + formatMessage(message, params));
        }
    }
    
    private String formatMessage(String message, Object... params) {
        if (params == null || params.length == 0) {
            return message;
        }
        
        // Simple parameter substitution - replace {} with parameters
        String result = message;
        for (Object param : params) {
            if (result.contains("{}")) {
                result = result.replaceFirst("\\{\\}", String.valueOf(param));
            }
        }
        return result;
    }
}
