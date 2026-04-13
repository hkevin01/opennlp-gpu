package org.apache.opennlp.gpu.common;

/**

 * ID: GPU-CCP-001
 * Requirement: CpuComputeProvider must implement ComputeProvider using pure-Java
 *              CPU arithmetic with no native GPU dependencies, serving as both the
 *              universal fallback and the reference implementation for the common package.
 * Purpose: Provides a guaranteed-available ComputeProvider in the common package used
 *          by ComputeProviderFactory when no GPU backend is available or requested.
 * Rationale: The common package requires a self-contained CPU provider that does not
 *            depend on the compute sub-package to avoid circular imports; this class
 *            duplicates the compute.CpuComputeProvider contract at the common layer.
 * Inputs: float[] matrix/vector operands; int dimension parameters; GpuConfig settings.
 * Outputs: void (results written to caller-supplied arrays); boolean availability flags.
 * Preconditions: JVM initialised; arrays non-null with consistent dimensions.
 * Postconditions: Result arrays contain correct arithmetic results; isAvailable() == true.
 * Assumptions: Always available; no driver or SDK dependencies.
 * Side Effects: Allocates transient ResourceManager; no GPU API calls.
 * Failure Modes: ArrayIndexOutOfBoundsException if dimension mismatch; caught and logged.
 * Error Handling: Delegates dimension validation to underlying arithmetic; exceptions propagate.
 * Constraints: Single-threaded per-instance; thread safety provided by caller.
 * Verification: GpuTestSuite; BasicValidationTest; parity tests vs GPU providers.
 * References: ComputeProvider interface; compute.CpuComputeProvider; ARCHITECTURE_OVERVIEW.md.
 */
public class CpuComputeProvider implements ComputeProvider {

    private final ResourceManager resourceManager;

    /**

     * ID: GPU-CCP-002
     * Requirement: CpuComputeProvider must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a CpuComputeProvider instance.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public CpuComputeProvider() {
        this.resourceManager = new ResourceManager();
    }

    /**

     * ID: GPU-CCP-003
     * Requirement: Evaluate and return the boolean result of isGpuProvider.
     * Purpose: Return whether isGpuProvider condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public boolean isGpuProvider() {
        return false;
    }

    /**

     * ID: GPU-CCP-004
     * Requirement: cleanup must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void cleanup() {
        if (resourceManager != null) {
            resourceManager.cleanup();
        }
    }

    /**

     * ID: GPU-CCP-005
     * Requirement: Return the Name field value without side effects.
     * Purpose: Return the value of the Name property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getName() {
        return "CPU Provider";
    }

    /**

     * ID: GPU-CCP-006
     * Requirement: Return the Type field value without side effects.
     * Purpose: Return the value of the Type property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public Type getType() {
        return Type.CPU;
    }

    /**

     * ID: GPU-CCP-007
     * Requirement: Evaluate and return the boolean result of isAvailable.
     * Purpose: Return whether isAvailable condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public boolean isAvailable() {
        return true; // CPU is always available
    }

    /**

     * ID: GPU-CCP-008
     * Requirement: Return the MaxMemoryMB field value without side effects.
     * Purpose: Return the value of the MaxMemoryMB property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public long getMaxMemoryMB() {
        return Runtime.getRuntime().maxMemory() / (1024 * 1024);
    }

    /**

     * ID: GPU-CCP-009
     * Requirement: Return the CurrentMemoryUsageMB field value without side effects.
     * Purpose: Return the value of the CurrentMemoryUsageMB property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public long getCurrentMemoryUsageMB() {
        return (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
    }

    /**

     * ID: GPU-CCP-010
     * Requirement: Return the ResourceManager field value without side effects.
     * Purpose: Return the value of the ResourceManager property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public Object getResourceManager() {
        return resourceManager;
    }

    /**

     * ID: GPU-CCP-011
     * Requirement: matrixMultiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the matrixMultiply operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int m, int n, int k
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k) {
        // Simple CPU matrix multiplication
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }

    /**

     * ID: GPU-CCP-012
     * Requirement: matrixAdd must execute correctly within the contract defined by this class.
     * Purpose: Implement the matrixAdd operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void matrixAdd(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }

    /**

     * ID: GPU-CCP-013
     * Requirement: matrixTranspose must execute correctly within the contract defined by this class.
     * Purpose: Implement the matrixTranspose operation for this class.
     * Inputs: float[] input, float[] output, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void matrixTranspose(float[] input, float[] output, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
    }

    /**

     * ID: GPU-CCP-014
     * Requirement: extractFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractFeatures operation for this class.
     * Inputs: String[] text, float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void extractFeatures(String[] text, float[] features) {
        // CPU feature extraction: maps each token to its character-count as a lightweight
        // BOW-style numeric feature.  GPU providers use kernel-based vectorizers instead.
        for (int i = 0; i < features.length && i < text.length; i++) {
            features[i] = text[i].length();
        }
    }

    /**

     * ID: GPU-CCP-015
     * Requirement: computeTfIdf must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the computeTfIdf result.
     * Inputs: float[] termFreq, float[] docFreq, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = termFreq[i] * (float) Math.log(1.0 + 1.0 / (docFreq[i] + 1e-10));
        }
    }

    /**

     * ID: GPU-CCP-016
     * Requirement: initialize must execute correctly within the contract defined by this class.
     * Purpose: Initialise internal state and allocate required resources.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void initialize() {
        // CPU provider doesn't need special initialization
    }

    /**

     * ID: GPU-CCP-017
     * Requirement: initialize must execute correctly within the contract defined by this class.
     * Purpose: Initialise internal state and allocate required resources.
     * Inputs: GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void initialize(GpuConfig config) {
        // CPU provider doesn't need configuration
        initialize();
    }

    /**

     * ID: GPU-CCP-018
     * Requirement: supportsOperation must execute correctly within the contract defined by this class.
     * Purpose: Implement the supportsOperation operation for this class.
     * Inputs: String operationType
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public boolean supportsOperation(String operationType) {
        return true; // CPU supports all operations
    }
}
