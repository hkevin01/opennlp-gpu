package org.apache.opennlp.gpu.common;

/**
 * ID: GL-001
 * Requirement: GpuLogger must provide a consistent, lightweight logging wrapper for all GPU subsystem classes.
 * Purpose: Drop-in logger that routes info/warn/error/debug messages to stdout/stderr, respecting the gpu.debug system property.
 * Rationale: Using a custom wrapper avoids pulling in a full SLF4J backend at runtime when the extension is embedded in a host application.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Writes to stdout (info, debug) and stderr (error); no file I/O.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
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
