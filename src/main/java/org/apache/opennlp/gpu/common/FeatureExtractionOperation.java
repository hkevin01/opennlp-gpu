package org.apache.opennlp.gpu.common;

/**
 * Interface for feature extraction operations.
 */
public interface FeatureExtractionOperation {
    
    /**
     * Gets the compute provider.
     *
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Extracts features from the input data.
     * 
     * @param inputData the input data
     * @return the extracted features
     */
    default Object extract(Object inputData) {
        // Default implementation does nothing
        return null;
    }
    
    /**
     * Checks if this operation is supported on the current device.
     * 
     * @return true if the operation is supported
     */
    default boolean isSupported() {
        return true;
    }
    
    // Add any other methods that are required by implementations
    // For example:
    // float[] extractFeatures(String[] tokens);
    // float[] computeTfIdf(String[] documents);
    // float computeCosineSimilarity(float[] vector1, float[] vector2);
    // void release();
}
