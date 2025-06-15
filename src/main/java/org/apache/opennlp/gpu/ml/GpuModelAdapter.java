package org.apache.opennlp.gpu.ml;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMemoryManager;
import org.apache.opennlp.maxent.MaxentModel;
import org.apache.opennlp.model.Context;



/**
 * Adapter that wraps any MaxEnt model to provide GPU acceleration
 * when beneficial, while maintaining full compatibility with the 
 * standard MaxEnt interface.
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
    public GpuModelAdapter(MaxentModel cpuModel, GpuConfig config) {
        this.cpuModel = cpuModel;
        this.config = config;
        this.computeProvider = createComputeProvider();
        this.memoryManager = new GpuMemoryManager(config);
        
        logger.info("Created GPU model adapter for: " + cpuModel.toString());
    }
    
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
    private boolean shouldUseGpu(String[] context) {
        if (!computeProvider.isGpuProvider()) {
            return false;
        }
        
        // Use GPU for larger contexts and outcome sets
        return context.length >= GPU_THRESHOLD_CONTEXT_SIZE && 
               cpuModel.getNumOutcomes() >= GPU_THRESHOLD_OUTCOMES;
    }
    
    public double[] eval(String[] context) {
        if (shouldUseGpu(context)) {
            return evaluateOnGpu(context, null);
        } else {
            return cpuModel.eval(context);
        }
    }
    
    public double[] eval(String[] context, double[] probs) {
        if (shouldUseGpu(context)) {
            return evaluateOnGpu(context, probs);
        } else {
            return cpuModel.eval(context, probs);
        }
    }
    
    public double[] eval(String[] context, float[] probs) {
        // Convert float array to double array for compatibility
        double[] doubleProbs = null;
        if (probs != null) {
            doubleProbs = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                doubleProbs[i] = probs[i];
            }
        }
        return eval(context, doubleProbs);
    }
    
    public String getOutcome(int index) {
        return cpuModel.getOutcome(index);
    }
    
    public int getNumOutcomes() {
        return cpuModel.getNumOutcomes();
    }
    
    public int getIndex(String outcome) {
        return cpuModel.getIndex(outcome);
    }
    
    public String[] getAllOutcomes() {
        return cpuModel.getAllOutcomes();
    }
    
    public Object[] getDataStructures() {
        return cpuModel.getDataStructures();
    }
    
    /**
     * GPU-accelerated evaluation method
     */
    private double[] evaluateOnGpu(String[] context, double[] probs) {
        try {
            // For now, delegate to CPU implementation
            // TODO: Implement actual GPU acceleration
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
    public void cleanup() {
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
        if (memoryManager != null) {
            memoryManager.cleanup();
        }
    }
}
