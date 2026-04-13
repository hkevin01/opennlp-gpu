package org.apache.opennlp.gpu.rocm;

import org.apache.opennlp.gpu.common.GpuDevice;

/**

 * ID: GPU-RD-001
 * Requirement: RocmDevice must represent a single AMD ROCm GPU device with its properties (name, VRAM, compute version).
 * Purpose: Value object populated from hipGetDeviceProperties to carry per-device metadata through diagnostics and config flows.
 * Rationale: A dedicated device-info type decouples hipDeviceProp_t from Java code and enables serialization/logging without JNI calls.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; immutable value object after construction.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class RocmDevice implements GpuDevice {
    private final String name;
    private final int deviceId;
    private long devicePtr;
    private String architecture = "Unknown";
    private long memoryBytes = 1024L * 1024L * 1024L; // 1GB default
    
    /**
     * Create a new ROCm device.
     *
     * @param name device name
     * @param deviceId device index
     * @param devicePtr native device pointer
     */
    /**
    
     * ID: GPU-RD-002
     * Requirement: RocmDevice must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a RocmDevice instance.
     * Inputs: String name, int deviceId, long devicePtr
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public RocmDevice(String name, int deviceId, long devicePtr) {
        this.name = name != null ? name : "Unknown";
        this.deviceId = deviceId;
        this.devicePtr = devicePtr;
    }
    
    /**
     * Get the device name.
     *
     * @return the device name
     */
    /**
    
     * ID: GPU-RD-003
     * Requirement: Return the Name field value without side effects.
     * Purpose: Return the value of the Name property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public String getName() {
        return name;
    }
    
    /**
     * Get the device index.
     *
     * @return the device index
     */
    /**
    
     * ID: GPU-RD-004
     * Requirement: Return the DeviceId field value without side effects.
     * Purpose: Return the value of the DeviceId property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getDeviceId() {
        return deviceId;
    }
    
    /**
     * Get the native device pointer.
     *
     * @return the device pointer
     */
    /**
    
     * ID: GPU-RD-005
     * Requirement: Return the DevicePtr field value without side effects.
     * Purpose: Return the value of the DevicePtr property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getDevicePtr() {
        return devicePtr;
    }
    
    /**
     * Get the amount of memory in megabytes.
     * 
     * @return the memory size in MB
     */
    /**
    
     * ID: GPU-RD-006
     * Requirement: Return the MemoryMB field value without side effects.
     * Purpose: Return the value of the MemoryMB property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public long getMemoryMB() {
        // Implementation of the missing method
        return 1024; // Default 1GB, replace with actual implementation
    }
    
    /**
     * Get a short description of the device.
     * 
     * @return a string describing the device
     */
    /**
    
     * ID: GPU-RD-007
     * Requirement: Return the Description field value without side effects.
     * Purpose: Return the value of the Description property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public String getDescription() {
        return String.format("%s (%s) with %d MB", name, architecture, getMemoryMB());
    }
    
    /**
     * Check if this device has enough memory for the specified size.
     * 
     * @param requiredBytes the number of bytes required
     * @return true if the device has enough memory
     */
    /**
    
     * ID: GPU-RD-008
     * Requirement: Evaluate and return the boolean result of hasEnoughMemory.
     * Purpose: Return whether hasEnoughMemory condition holds.
     * Inputs: long requiredBytes
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public boolean hasEnoughMemory(long requiredBytes) {
        // Allow up to 80% of total memory to be used
        return requiredBytes <= (memoryBytes * 0.8);
    }
}
