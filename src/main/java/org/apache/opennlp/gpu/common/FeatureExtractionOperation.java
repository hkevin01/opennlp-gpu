package org.apache.opennlp.gpu.common;

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
