package org.apache.opennlp.gpu.ml;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.maxent.MaxentModel;
import org.apache.opennlp.model.Context;



/**
 * Factory for creating GPU-accelerated ML models
 * Provides a unified interface for creating GPU-enhanced versions
 * of OpenNLP machine learning models.
 */
public class GpuModelFactory {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuModelFactory.class);
    
    private final GpuConfig config;
    
    /**
     * Creates a new GPU model factory with the given configuration
     */
    public GpuModelFactory(GpuConfig config) {
        this.config = config;
        logger.info("Created GPU model factory with GPU enabled: " + config.isGpuEnabled());
    }
    
    /**
     * Creates a GPU-accelerated MaxEnt model
     */
    public MaxentModel createGpuMaxentModel(MaxentModel cpuModel) {
        try {
            if (config.isGpuEnabled()) {
                return new GpuMaxentModel(cpuModel, config);
            } else {
                logger.info("GPU disabled, returning CPU model");
                return cpuModel;
            }
        } catch (Exception e) {
            logger.warn("Failed to create GPU MaxEnt model, falling back to CPU: " + e.getMessage());
            return cpuModel;
        }
    }
    
    /**
     * Creates a GPU-accelerated model adapter
     */
    public MaxentModel createGpuModelAdapter(MaxentModel cpuModel) {
        try {
            return new GpuModelAdapter(cpuModel, config);
        } catch (Exception e) {
            logger.warn("Failed to create GPU model adapter, returning CPU model: " + e.getMessage());
            return cpuModel;
        }
    }
    
    /**
     * Determines if GPU acceleration should be used for the given model
     */
    public boolean shouldUseGpu(MaxentModel model) {
        return config.isGpuEnabled() && 
               model.getNumOutcomes() > 10;
    }
    
    /**
     * Gets the current GPU configuration
     */
    public GpuConfig getConfig() {
        return config;
    }
}
