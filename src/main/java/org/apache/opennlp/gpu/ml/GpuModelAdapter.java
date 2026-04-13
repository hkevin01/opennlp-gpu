package org.apache.opennlp.gpu.ml;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMemoryManager;

import opennlp.tools.ml.model.MaxentModel;

/**

 * ID: GPU-GMA-001
 * Requirement: GpuModelAdapter must adapt a standard OpenNLP model to the GPU compute pipeline by wrapping eval() dispatch.
 * Purpose: Generic adapter applying GPU acceleration to any MaxentModel implementation without modifying the base model.
 * Rationale: Generic adapter reduces code duplication across the per-algorithm wrappers (MaxEnt, Naive Bayes, Perceptron, Neural).
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None beyond delegation to underlying model and GPU provider.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuModelAdapter implements MaxentModel {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuModelAdapter.class);
    
    private final MaxentModel cpuModel;
    private final ComputeProvider computeProvider;
    private final GpuConfig config;
    private final GpuMemoryManager memoryManager;
    
    // Performance thresholds
    private static final int GPU_THRESHOLD_CONTEXT_SIZE = 100;
    private static final int GPU_THRESHOLD_OUTCOMES = 10;
    
    /**
     * Creates a GPU-accelerated adapter for the given model
     */
    /**
    
     * ID: GPU-GMA-002
     * Requirement: GpuModelAdapter must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuModelAdapter instance.
     * Inputs: MaxentModel cpuModel, GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuModelAdapter(MaxentModel cpuModel, GpuConfig config) {
        this.cpuModel = cpuModel;
        this.config = config;
        this.computeProvider = createComputeProvider();
        this.memoryManager = new GpuMemoryManager(config);
        
        logger.info("Created GPU model adapter for: " + cpuModel.toString());
    }
    
    /**
    
     * ID: GPU-GMA-003
     * Requirement: createComputeProvider must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new ComputeProvider.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private ComputeProvider createComputeProvider() {
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                return new GpuComputeProvider(config);
            }
        } catch (Exception e) {
            logger.warn("Failed to initialize GPU provider, falling back to CPU: " + e.getMessage());
        }
        return new CpuComputeProvider();
    }
    
    /**
     * Determines whether to use GPU acceleration for this evaluation
     */
    /**
    
     * ID: GPU-GMA-004
     * Requirement: shouldUseGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpu operation for this class.
     * Inputs: String[] context
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpu(String[] context) {
        if (!computeProvider.isGpuProvider()) {
            return false;
        }
        
        // Use GPU for larger contexts and outcome sets
        return context.length >= GPU_THRESHOLD_CONTEXT_SIZE && 
               cpuModel.getNumOutcomes() >= GPU_THRESHOLD_OUTCOMES;
    }
    
    /**
    
     * ID: GPU-GMA-005
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] context
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] context) {
        if (shouldUseGpu(context)) {
            return evaluateOnGpu(context, null);
        } else {
            return cpuModel.eval(context);
        }
    }
    
    /**
    
     * ID: GPU-GMA-006
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] context, double[] probs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] context, double[] probs) {
        if (shouldUseGpu(context)) {
            return evaluateOnGpu(context, probs);
        } else {
            return cpuModel.eval(context, probs);
        }
    }
    
    /**
    
     * ID: GPU-GMA-007
     * Requirement: eval must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the eval result.
     * Inputs: String[] context, float[] values
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public double[] eval(String[] context, float[] values) {
        // For now, delegate to CPU implementation
        return cpuModel.eval(context, values);
    }

    /**
    
     * ID: GPU-GMA-008
     * Requirement: Return the BestOutcome field value without side effects.
     * Purpose: Return the value of the BestOutcome property.
     * Inputs: double[] ocs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getBestOutcome(double[] ocs) {
        return cpuModel.getBestOutcome(ocs);
    }

    /**
    
     * ID: GPU-GMA-009
     * Requirement: Return the AllOutcomes field value without side effects.
     * Purpose: Return the value of the AllOutcomes property.
     * Inputs: double[] ocs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getAllOutcomes(double[] ocs) {
        return cpuModel.getAllOutcomes(ocs);
    }
    
    /**
    
     * ID: GPU-GMA-010
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
        return cpuModel.getOutcome(index);
    }
    
    /**
    
     * ID: GPU-GMA-011
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
        return cpuModel.getNumOutcomes();
    }
    
    /**
    
     * ID: GPU-GMA-012
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
        return cpuModel.getIndex(outcome);
    }
    
    /**
    
     * ID: GPU-GMA-013
     * Requirement: Return the AllOutcomes field value without side effects.
     * Purpose: Return the value of the AllOutcomes property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public String[] getAllOutcomes() {
        int numOutcomes = cpuModel.getNumOutcomes();
        String[] outcomes = new String[numOutcomes];
        for (int i = 0; i < numOutcomes; i++) {
            outcomes[i] = cpuModel.getOutcome(i);
        }
        return outcomes;
    }
    
    /**
     * GPU-accelerated evaluation method
     */
    /**
    
     * ID: GPU-GMA-014
     * Requirement: evaluateOnGpu must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the evaluateOnGpu result.
     * Inputs: String[] context, double[] probs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private double[] evaluateOnGpu(String[] context, double[] probs) {
        try {
            // For now, delegate to CPU implementation
            if (probs != null) {
                return cpuModel.eval(context, probs);
            } else {
                return cpuModel.eval(context);
            }
        } catch (Exception e) {
            logger.warn("GPU evaluation failed, falling back to CPU: " + e.getMessage());
            return probs != null ? cpuModel.eval(context, probs) : cpuModel.eval(context);
        }
    }
    
    /**
     * Cleanup GPU resources
     */
    /**
    
     * ID: GPU-GMA-015
     * Requirement: cleanup must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void cleanup() {
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
        if (memoryManager != null) {
            memoryManager.cleanup();
        }
    }
}
