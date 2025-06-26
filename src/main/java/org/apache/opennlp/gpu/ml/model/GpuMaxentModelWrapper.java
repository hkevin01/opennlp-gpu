package org.apache.opennlp.gpu.ml.model;

import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;

import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * Wrapper that makes GPU-accelerated MaxEnt model compatible with OpenNLP's MaxentModel interface.
 * 
 * This wrapper provides seamless integration with existing OpenNLP code while providing
 * GPU acceleration under the hood. All OpenNLP MaxentModel methods are supported.
 * 
 * Usage:
 * <pre>
 * {@code
 * // Existing OpenNLP code works unchanged
 * MaxentModel model = GpuMaxentTrainer.train(language, events, params);
 * 
 * // Use standard OpenNLP prediction methods
 * double[] outcomes = model.eval(features);
 * String bestOutcome = model.getBestOutcome(outcomes);
 * }
 * </pre>
 * 
 * @author OpenNLP GPU Extension Team
 * @since 1.0.0
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
    public GpuMaxentModelWrapper(GpuMaxentModel gpuModel) {
        this.gpuModel = gpuModel;
        // Get the base model from the GPU model for compatibility
        this.baseModel = gpuModel.getBaseModel();
        
        logger.debug("Created OpenNLP-compatible wrapper for GPU MaxEnt model with {} outcomes", 
                    baseModel.getNumOutcomes());
    }
    
    @Override
    public double[] eval(String[] features) {
        return gpuModel.eval(features);
    }
    
    @Override
    public double[] eval(String[] features, double[] priors) {
        return gpuModel.eval(features, priors);
    }
    
    @Override
    public double[] eval(String[] features, float[] values) {
        return gpuModel.eval(features, values);
    }
    
    @Override
    public String getBestOutcome(double[] outcomes) {
        return gpuModel.getBestOutcome(outcomes);
    }
    
    @Override
    public String getAllOutcomes(double[] outcomes) {
        return gpuModel.getAllOutcomes(outcomes);
    }
    
    @Override
    public String getOutcome(int index) {
        return gpuModel.getOutcome(index);
    }
    
    @Override
    public int getIndex(String outcome) {
        return gpuModel.getIndex(outcome);
    }
    
    @Override
    public int getNumOutcomes() {
        return gpuModel.getNumOutcomes();
    }
    
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
    public boolean isUsingGpu() {
        return gpuModel.isUsingGpu();
    }
    
    /**
     * Get the speedup factor compared to CPU implementation.
     * 
     * @return Speedup factor (e.g., 13.6 for 13.6x faster)
     */
    public double getSpeedupFactor() {
        return gpuModel.getSpeedupFactor();
    }
    
    @Override
    public String toString() {
        return String.format("GpuMaxentModel(outcomes=%d, gpu=%s, speedup=%.1fx)",
                           getNumOutcomes(),
                           isUsingGpu() ? "enabled" : "disabled",
                           getSpeedupFactor());
    }
}
