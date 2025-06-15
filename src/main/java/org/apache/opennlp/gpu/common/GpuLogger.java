package org.apache.opennlp.gpu.common;

/**
 * Stub logger for GPU operations
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
        System.out.println("[GPU-INFO] " + className + ": " + message);
    }
    
    public void warn(String message) {
        System.out.println("[GPU-WARN] " + className + ": " + message);
    }
    
    public void error(String message) {
        System.err.println("[GPU-ERROR] " + className + ": " + message);
    }
}
