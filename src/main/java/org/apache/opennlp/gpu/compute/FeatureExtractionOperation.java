package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**

 * Requirement: FeatureExtractionOperation must define the interface for GPU-dispatched NLP feature extraction from token contexts.
 * Purpose: Interface specifying feature extraction from raw String context arrays to normalised float feature vectors.
 * Rationale: Abstracting extraction behind an interface allows the same NLP pipeline to use CPU, CUDA, OpenCL, or ROCm extraction at runtime.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; implementors define side effects.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
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
