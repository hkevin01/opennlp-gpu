package org.apache.opennlp.gpu.common;

/**
 * Interface for feature extraction operations.
 */
public interface FeatureExtractionOperation {
    /**
     * Gets the compute provider used by this operation.
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Extract features from tokens.
     * @param tokens the tokens to extract features from
     * @return the extracted features
     */
    float[] extractFeatures(String[] tokens);
    
    /**
     * Compute TF-IDF for documents.
     * @param documents the documents to compute TF-IDF for
     * @return the TF-IDF values
     */
    float[] computeTfIdf(String[] documents);
    
    /**
     * Compute cosine similarity between vectors.
     * @param vector1 the first vector
     * @param vector2 the second vector
     * @return the cosine similarity
     */
    float computeCosineSimilarity(float[] vector1, float[] vector2);
    
    /**
     * Releases resources used by this operation.
     */
    void release();
}
