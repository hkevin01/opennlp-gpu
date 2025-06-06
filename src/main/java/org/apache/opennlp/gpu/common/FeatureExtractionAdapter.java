package org.apache.opennlp.gpu.common;

import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;

/**
 * Adapter class to bridge between different FeatureExtractionOperation interfaces.
 */
public class FeatureExtractionAdapter implements FeatureExtractionOperation {
    
    private final CpuFeatureExtractionOperation delegate;
    
    /**
     * Creates a new feature extraction adapter.
     *
     * @param delegate the CPU implementation to delegate to
     */
    public FeatureExtractionAdapter(CpuFeatureExtractionOperation delegate) {
        this.delegate = delegate;
    }
    
    public ComputeProvider getProvider() {
        return delegate.getProvider();
    }
    
    public float[] extractFeatures(String[] tokens) {
        return delegate.extractFeatures(tokens);
    }
    
    public float[] computeTfIdf(String[] documents) {
        return delegate.computeTfIdf(documents);
    }
    
    public float computeCosineSimilarity(float[] vector1, float[] vector2) {
        return delegate.computeCosineSimilarity(vector1, vector2);
    }
    
    public void release() {
        delegate.release();
    }
}
