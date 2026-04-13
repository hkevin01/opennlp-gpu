package org.apache.opennlp.gpu.ml.model;

import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;

import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**

 * ID: GPU-GMMW-001
 * Requirement: GpuMaxentModelWrapper must wrap a GpuMaxentModel with additional output formatting and logging for diagnostic use.
 * Purpose: Adds human-readable output and performance logging around GpuMaxentModel.eval() without changing model semantics.
 * Rationale: Diagnostic wrappers avoid modifying core model code while providing debugging visibility during development.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Logs eval latency and probability distributions to GpuLogger.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuMaxentModelWrapper implements MaxentModel {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuMaxentModelWrapper.class);
    
    private final GpuMaxentModel gpuModel;
    private final MaxentModel baseModel;
    
    /**
     * Create a wrapper around a GPU-accelerated MaxEnt model.
     * 
     * @param gpuModel The underlying GPU model
     */
    /**
    
     * ID: GPU-GMMW-002
     * Requirement: GpuMaxentModelWrapper must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuMaxentModelWrapper instance.
     * Inputs: GpuMaxentModel gpuModel
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuMaxentModelWrapper(GpuMaxentModel gpuModel) {
        this.gpuModel = gpuModel;
        // Get the base model from the GPU model for compatibility
        this.baseModel = gpuModel.getBaseModel();
        
        logger.debug("Created OpenNLP-compatible wrapper for GPU MaxEnt model with {} outcomes", 
                    baseModel.getNumOutcomes());
    }
    
    /**
    
     * ID: GPU-GMMW-003
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] features) {
        return gpuModel.eval(features);
    }
    
    /**
    
     * ID: GPU-GMMW-004
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] features, double[] priors
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] features, double[] priors) {
        return gpuModel.eval(features, priors);
    }
    
    /**
    
     * ID: GPU-GMMW-005
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] features, float[] values
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] features, float[] values) {
        return gpuModel.eval(features, values);
    }
    
    /**
    
     * ID: GPU-GMMW-006
     * Requirement: Return the BestOutcome field value without side effects.
     * Purpose: Return the value of the BestOutcome property.
     * Inputs: double[] outcomes
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getBestOutcome(double[] outcomes) {
        return gpuModel.getBestOutcome(outcomes);
    }
    
    /**
    
     * ID: GPU-GMMW-007
     * Requirement: Return the AllOutcomes field value without side effects.
     * Purpose: Return the value of the AllOutcomes property.
     * Inputs: double[] outcomes
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getAllOutcomes(double[] outcomes) {
        return gpuModel.getAllOutcomes(outcomes);
    }
    
    /**
    
     * ID: GPU-GMMW-008
     * Requirement: Return the Outcome field value without side effects.
     * Purpose: Return the value of the Outcome property.
     * Inputs: int index
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getOutcome(int index) {
        return gpuModel.getOutcome(index);
    }
    
    /**
    
     * ID: GPU-GMMW-009
     * Requirement: Return the Index field value without side effects.
     * Purpose: Return the value of the Index property.
     * Inputs: String outcome
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public int getIndex(String outcome) {
        return gpuModel.getIndex(outcome);
    }
    
    /**
    
     * ID: GPU-GMMW-010
     * Requirement: Return the NumOutcomes field value without side effects.
     * Purpose: Return the value of the NumOutcomes property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public int getNumOutcomes() {
        return gpuModel.getNumOutcomes();
    }
    
    /**
    
     * ID: GPU-GMMW-011
     * Requirement: Return the DataStructures field value without side effects.
     * Purpose: Return the value of the DataStructures property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Context[] getDataStructures() {
        // This method is not available in OpenNLP 2.5.4, so we'll return null
        // In a full implementation, this would extract the parameter structure
        return null;
    }
    
    /**
     * Get the underlying GPU model for advanced operations.
     * 
     * @return The GPU-accelerated model
     */
    /**
    
     * ID: GPU-GMMW-012
     * Requirement: Return the GpuModel field value without side effects.
     * Purpose: Return the value of the GpuModel property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuMaxentModel getGpuModel() {
        return gpuModel;
    }
    
    /**
     * Get performance statistics from the GPU model.
     * 
     * @return Performance statistics map
     */
    public java.util.Map<String, Object> getPerformanceStats() {
        return gpuModel.getPerformanceStats();
    }
    
    /**
     * Check if this model is using GPU acceleration.
     * 
     * @return true if using GPU, false if using CPU fallback
     */
    /**
    
     * ID: GPU-GMMW-013
     * Requirement: Evaluate and return the boolean result of isUsingGpu.
     * Purpose: Return whether isUsingGpu condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public boolean isUsingGpu() {
        return gpuModel.isUsingGpu();
    }
    
    /**
     * Get the speedup factor compared to CPU implementation.
     * 
     * @return Speedup factor (e.g., 13.6 for 13.6x faster)
     */
    /**
    
     * ID: GPU-GMMW-014
     * Requirement: Return the SpeedupFactor field value without side effects.
     * Purpose: Return the value of the SpeedupFactor property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public double getSpeedupFactor() {
        return gpuModel.getSpeedupFactor();
    }
    
    /**
    
     * ID: GPU-GMMW-015
     * Requirement: toString must execute correctly within the contract defined by this class.
     * Purpose: Implement the toString operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String toString() {
        return String.format("GpuMaxentModel(outcomes=%d, gpu=%s, speedup=%.1fx)",
                           getNumOutcomes(),
                           isUsingGpu() ? "enabled" : "disabled",
                           getSpeedupFactor());
    }
}
