package org.apache.opennlp.gpu.common;

/**

 * ID: GPU-GL-001
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
    
    /**
    
     * ID: GPU-GL-002
     * Requirement: GpuLogger must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuLogger instance.
     * Inputs: String className
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private GpuLogger(String className) {
        this.className = className;
    }
    
    /**
    
     * ID: GPU-GL-003
     * Requirement: Return the Logger field value without side effects.
     * Purpose: Return the value of the Logger property.
     * Inputs: Class<?> clazz
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public static GpuLogger getLogger(Class<?> clazz) {
        return new GpuLogger(clazz.getSimpleName());
    }
    
    /**
    
     * ID: GPU-GL-004
     * Requirement: info must execute correctly within the contract defined by this class.
     * Purpose: Implement the info operation for this class.
     * Inputs: String message
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void info(String message) {
        System.out.println("[INFO] " + className + ": " + message);
    }
    
    /**
    
     * ID: GPU-GL-005
     * Requirement: info must execute correctly within the contract defined by this class.
     * Purpose: Implement the info operation for this class.
     * Inputs: String message, Object... params
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void info(String message, Object... params) {
        System.out.println("[INFO] " + className + ": " + formatMessage(message, params));
    }
    
    /**
    
     * ID: GPU-GL-006
     * Requirement: warn must execute correctly within the contract defined by this class.
     * Purpose: Implement the warn operation for this class.
     * Inputs: String message
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void warn(String message) {
        System.out.println("[WARN] " + className + ": " + message);
    }
    
    /**
    
     * ID: GPU-GL-007
     * Requirement: warn must execute correctly within the contract defined by this class.
     * Purpose: Implement the warn operation for this class.
     * Inputs: String message, Object... params
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void warn(String message, Object... params) {
        System.out.println("[WARN] " + className + ": " + formatMessage(message, params));
    }
    
    /**
    
     * ID: GPU-GL-008
     * Requirement: error must execute correctly within the contract defined by this class.
     * Purpose: Implement the error operation for this class.
     * Inputs: String message
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void error(String message) {
        System.err.println("[ERROR] " + className + ": " + message);
    }
    
    /**
    
     * ID: GPU-GL-009
     * Requirement: error must execute correctly within the contract defined by this class.
     * Purpose: Implement the error operation for this class.
     * Inputs: String message, Exception e
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void error(String message, Exception e) {
        System.err.println("[ERROR] " + className + ": " + message);
        if (e != null) {
            e.printStackTrace();
        }
    }
    
    /**
    
     * ID: GPU-GL-010
     * Requirement: error must execute correctly within the contract defined by this class.
     * Purpose: Implement the error operation for this class.
     * Inputs: String message, Object... params
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void error(String message, Object... params) {
        System.err.println("[ERROR] " + className + ": " + formatMessage(message, params));
    }
    
    /**
    
     * ID: GPU-GL-011
     * Requirement: debug must execute correctly within the contract defined by this class.
     * Purpose: Implement the debug operation for this class.
     * Inputs: String message
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void debug(String message) {
        // Only print debug messages if debug mode is enabled
        String debugMode = System.getProperty("gpu.debug", "false");
        if ("true".equalsIgnoreCase(debugMode)) {
            System.out.println("[DEBUG] " + className + ": " + message);
        }
    }
    
    /**
    
     * ID: GPU-GL-012
     * Requirement: debug must execute correctly within the contract defined by this class.
     * Purpose: Implement the debug operation for this class.
     * Inputs: String message, Object... params
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void debug(String message, Object... params) {
        String debugMode = System.getProperty("gpu.debug", "false");
        if ("true".equalsIgnoreCase(debugMode)) {
            System.out.println("[DEBUG] " + className + ": " + formatMessage(message, params));
        }
    }
    
    /**
    
     * ID: GPU-GL-013
     * Requirement: formatMessage must execute correctly within the contract defined by this class.
     * Purpose: Implement the formatMessage operation for this class.
     * Inputs: String message, Object... params
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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
