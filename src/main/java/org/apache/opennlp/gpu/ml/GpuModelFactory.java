package org.apache.opennlp.gpu.ml;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;

import opennlp.tools.ml.model.MaxentModel;

/**

 * ID: GPU-GMF-001
 * Requirement: GpuModelFactory must create GPU-accelerated wrappers for OpenNLP ML models (MaxEnt, NaiveBayes, Perceptron, Neural).
 * Purpose: Factory providing convenience constructors for all supported GPU model types with shared GpuConfig.
 * Rationale: Centralises GPU wrapper creation for the ml package, mirroring the integration.GpuModelFactory for ML-level callers.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; stateless factory.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
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
