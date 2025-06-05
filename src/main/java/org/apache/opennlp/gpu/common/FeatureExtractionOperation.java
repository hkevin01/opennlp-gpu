package org.apache.opennlp.gpu.common;

/**
 * Interface for feature extraction operations that can be executed on different compute devices.
 */
public interface FeatureExtractionOperation {
    
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
}
