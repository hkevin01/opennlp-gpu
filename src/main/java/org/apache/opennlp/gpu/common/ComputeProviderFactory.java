package org.apache.opennlp.gpu.common;

/**

 * ID: GPU-CPF-001
 * Requirement: ComputeProviderFactory must construct and return the correct ComputeProvider implementation for a given Type.
 * Purpose: Factory that decouples caller code from concrete provider classes (CpuComputeProvider, CudaComputeProvider, OpenClComputeProvider, RocmComputeProvider).
 * Rationale: Factory pattern eliminates switch/case duplication across callers and makes adding new vendors a single-point change.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; all construction is side-effect-free.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class ComputeProviderFactory {
    
    /**
    
     * ID: GPU-CPF-002
     * Requirement: Return the DefaultProvider field value without side effects.
     * Purpose: Return the value of the DefaultProvider property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public static ComputeProvider getDefaultProvider() {
        return new CpuComputeProvider();
    }
    
    /**
    
     * ID: GPU-CPF-003
     * Requirement: createProvider must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new Provider.
     * Inputs: ComputeProvider.Type type
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public static ComputeProvider createProvider(ComputeProvider.Type type) {
        switch (type) {
            case CPU:
                return new CpuComputeProvider();
            case OPENCL:
                return new OpenClComputeProvider();
            case CUDA:
                return new CudaComputeProvider();
            case ROCM:
                return new RocmComputeProvider();
            default:
                return new CpuComputeProvider();
        }
    }
    
    /**
    
     * ID: GPU-CPF-004
     * Requirement: Return the Provider field value without side effects.
     * Purpose: Return the value of the Provider property.
     * Inputs: ComputeProvider.Type type
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public static ComputeProvider getProvider(ComputeProvider.Type type) {
        return ComputeProviderFactory.createProvider(type);
    }
}
