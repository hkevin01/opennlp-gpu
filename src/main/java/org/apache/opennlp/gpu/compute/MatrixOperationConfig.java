package org.apache.opennlp.gpu.compute;

/**

 * ID: GPU-MOC-001
 * Requirement: MatrixOperationConfig must hold per-operation configuration parameters (work-group sizes, precision, tiling factors) for GPU kernel dispatch.
 * Purpose: Value object that tunes GPU kernel launch parameters for specific operation types and device characteristics.
 * Rationale: Optimal GPU kernel parameters are device-dependent; externalising them to a config object enables tuning without recompilation.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None beyond holding configuration state.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class MatrixOperationConfig {
    // Example properties - replace with actual config needed
    private int preferredWorkGroupSizeMultiple;
    private boolean usePinnedMemory;

    public MatrixOperationConfig() {
        // Default constructor
    }

    // Example getter/setter - add actual methods
    public int getPreferredWorkGroupSizeMultiple() {
        return preferredWorkGroupSizeMultiple;
    }

    public void setPreferredWorkGroupSizeMultiple(int preferredWorkGroupSizeMultiple) {
        this.preferredWorkGroupSizeMultiple = preferredWorkGroupSizeMultiple;
    }

    public boolean shouldUsePinnedMemory() {
        return usePinnedMemory;
    }

    public void setUsePinnedMemory(boolean usePinnedMemory) {
        this.usePinnedMemory = usePinnedMemory;
    }
}
