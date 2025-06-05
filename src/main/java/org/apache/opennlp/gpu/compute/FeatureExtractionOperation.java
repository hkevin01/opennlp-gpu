package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Interface for feature extraction operations that can be accelerated on different hardware.
 */
public interface FeatureExtractionOperation {
    
    /**
     * Extract n-gram features from text.
     *
     * @param tokens array of token indices
     * @param numTokens number of tokens
     * @param maxNGramLength maximum n-gram length
     * @param featureMap output feature map (index -> count)
     * @return number of features extracted
     */
    int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap);
    
    /**
     * Compute TF-IDF features.
     *
     * @param termFreq term frequency array
     * @param docFreq document frequency array
     * @param numDocs total number of documents
     * @param tfidf output TF-IDF array
     * @param numTerms number of terms
     */
    void computeTfIdf(float[] termFreq, float[] docFreq, int numDocs, float[] tfidf, int numTerms);
    
    /**
     * Compute cosine similarity between document vectors.
     *
     * @param docVectors document vectors (each row is a document vector)
     * @param numDocs number of documents
     * @param vectorSize size of each document vector
     * @param similarities output similarity matrix
     */
    void computeCosineSimilarity(float[] docVectors, int numDocs, int vectorSize, float[] similarities);
    
    /**
     * Get the compute provider used by this operation.
     *
     * @return the compute provider
     */
    ComputeProvider getProvider();
    
    /**
     * Release resources associated with this operation.
     */
    void release();
}
