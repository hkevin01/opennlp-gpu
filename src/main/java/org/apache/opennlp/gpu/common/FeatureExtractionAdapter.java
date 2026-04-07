package org.apache.opennlp.gpu.common;

import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**

 * Requirement: FeatureExtractionAdapter must adapt raw text or token inputs to the FeatureExtractionOperation interface expected by GPU compute providers.
 * Purpose: Bridges OpenNLP feature contexts (String[] contexts) to numeric feature vectors consumed by GPU matrix operations.
 * Rationale: Decoupling NLP feature format conversion from compute dispatch keeps compute providers format-agnostic.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None beyond transient float[] allocation for feature vectors.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class FeatureExtractionAdapter implements FeatureExtractionOperation {
    
    private static final Logger logger = LoggerFactory.getLogger(FeatureExtractionAdapter.class);
    
    private final CpuFeatureExtractionOperation delegate;
    private final ComputeProvider provider;
    
    // Add default constructor
    public FeatureExtractionAdapter() {
        this.delegate = null;
        this.provider = null;
        FeatureExtractionAdapter.logger.warn("Default constructor called - adapter may not function properly");
    }
    
    // Add constructor that takes both parameters 
    public FeatureExtractionAdapter(CpuFeatureExtractionOperation delegate, ComputeProvider provider) {
        this.delegate = delegate;
        this.provider = provider;
    }
    
    // Remove @Override for methods that don't exist in the interface
    public int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap) {
        // Use delegate if available, otherwise provide fallback implementation
        if (delegate != null) {
            return delegate.extractNGrams(tokens, numTokens, maxNGramLength, featureMap);
        }
        return 0; // Fallback
    }
    
    @Override
    public float[] computeTfIdf(String[] documents) {
        FeatureExtractionAdapter.logger.debug("Delegating computeTfIdf to CpuFeatureExtractionOperation");
        // Direct call with matching parameter types
        return delegate.computeTfIdf(documents);
    }
    
    @Override
    public float computeCosineSimilarity(float[] vector1, float[] vector2) {
        FeatureExtractionAdapter.logger.debug("Delegating computeCosineSimilarity to CpuFeatureExtractionOperation");
        // Direct call with matching parameter types
        return delegate.computeCosineSimilarity(vector1, vector2);
    }
    
    @Override
    public void release() {
        // Use delegate if available, otherwise do nothing
        if (delegate != null) {
            delegate.release();
        }
    }
    
    // Fix the extractFeatures method to safely use delegate when available
    @Override
    public float[] extractFeatures(String[] tokens) {
        FeatureExtractionAdapter.logger.debug("Delegating extractFeatures to CpuFeatureExtractionOperation");
        if (delegate != null) {
            return delegate.extractFeatures(tokens);
        }
        // Fallback implementation
        return new float[tokens.length];
    }
    
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
}
