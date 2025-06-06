package org.apache.opennlp.gpu.common;

import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Adapter for CPU-based feature extraction operations.
 */
public class FeatureExtractionAdapter implements FeatureExtractionOperation {
    
    private static final Logger logger = LoggerFactory.getLogger(FeatureExtractionAdapter.class);
    
    private final CpuFeatureExtractionOperation delegate;
    private final ComputeProvider provider;
    
    // Add default constructor
    public FeatureExtractionAdapter() {
        this.delegate = null;
        this.provider = null;
        logger.warn("Default constructor called - adapter may not function properly");
    }
    
    // Add constructor that takes both parameters 
    public FeatureExtractionAdapter(CpuFeatureExtractionOperation delegate, ComputeProvider provider) {
        this.delegate = delegate;
        this.provider = provider;
    }
    
    // Implement all methods from the FeatureExtractionOperation interface
    public int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap) {
        // Use delegate if available, otherwise provide fallback implementation
        if (delegate != null) {
            return delegate.extractNGrams(tokens, numTokens, maxNGramLength, featureMap);
        }
        return 0; // Fallback
    }
    
    public float[] computeTfIdf(String[] documents) {
        logger.debug("Delegating computeTfIdf to CpuFeatureExtractionOperation");
        // Direct call with matching parameter types
        return delegate.computeTfIdf(documents);
    }
    
    public float computeCosineSimilarity(float[] vector1, float[] vector2) {
        logger.debug("Delegating computeCosineSimilarity to CpuFeatureExtractionOperation");
        // Direct call with matching parameter types
        return delegate.computeCosineSimilarity(vector1, vector2);
    }
    
    public void release() {
        // Use delegate if available, otherwise do nothing
        if (delegate != null) {
            delegate.release();
        }
    }
    
    // Fix the extractFeatures method to safely use delegate when available
    public float[] extractFeatures(String[] tokens) {
        logger.debug("Delegating extractFeatures to CpuFeatureExtractionOperation");
        if (delegate != null && delegate instanceof CpuFeatureExtractionOperation) {
            return ((CpuFeatureExtractionOperation) delegate).extractFeatures(tokens);
        }
        // Fallback implementation
        return new float[tokens.length];
    }
    
    public ComputeProvider getProvider() {
        return provider;
    }
}
