package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.FeatureExtractionOperation;

/**

 * ID: GPU-CFEO-001
 * Requirement: CpuFeatureExtractionOperation must implement FeatureExtractionOperation using pure-Java CPU arithmetic with no native dependencies.
 * Purpose: Returns feature vectors for NLP contexts using Java-only computation, serving as the reference and fallback implementation.
 * Rationale: A correct CPU implementation validates all GPU kernels through parity tests and ensures correctness on systems without GPU hardware.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Allocates transient float[] arrays; no external I/O.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class CpuFeatureExtractionOperation implements FeatureExtractionOperation {

    private final ComputeProvider provider;

    /**
     * Creates a new CPU feature extraction operation.
     *
     * @param provider the compute provider
     */
    /**

     * ID: GPU-CFEO-002
     * Requirement: CpuFeatureExtractionOperation must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a CpuFeatureExtractionOperation instance.
     * Inputs: ComputeProvider provider
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public CpuFeatureExtractionOperation(ComputeProvider provider) {
        this.provider = provider;
    }

    // Add required getProvider method
    /**

     * ID: GPU-CFEO-003
     * Requirement: Return the Provider field value without side effects.
     * Purpose: Return the value of the Provider property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public ComputeProvider getProvider() {
        return provider;
    }

    // Implement other required methods of FeatureExtractionOperation
    /**

     * ID: GPU-CFEO-004
     * Requirement: extractNGrams must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractNGrams operation for this class.
     * Inputs: int[] tokens, int numTokens, int maxNGramLength, int[] featureMap
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap) {
        // Count n-grams of lengths 1..maxNGramLength from tokens[0..numTokens-1] and
        // accumulate them into featureMap using a polynomial rolling hash index.
        if (tokens == null || featureMap == null || numTokens <= 0) return 0;
        int mapSize = featureMap.length;
        int count = 0;
        for (int n = 1; n <= maxNGramLength; n++) {
            for (int i = 0; i <= numTokens - n; i++) {
                int hash = 1;
                for (int j = i; j < i + n; j++) {
                    hash = 31 * hash + tokens[j];
                }
                featureMap[Math.abs(hash) % mapSize]++;
                count++;
            }
        }
        return count;
    }

    /**

     * ID: GPU-CFEO-005
     * Requirement: computeTfIdf must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the computeTfIdf result.
     * Inputs: String[] documents
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float[] computeTfIdf(String[] documents) {
        // Compute a single normalised term-frequency score per document:
        // the frequency of the most common token divided by total token count.
        if (documents == null || documents.length == 0) return new float[0];
        float[] result = new float[documents.length];
        for (int di = 0; di < documents.length; di++) {
            String[] words = documents[di].split("\\s+");
            if (words.length == 0) { result[di] = 0f; continue; }
            java.util.Map<String, Integer> freq = new java.util.HashMap<>();
            for (String w : words) freq.merge(w.toLowerCase(), 1, Integer::sum);
            int maxFreq = freq.values().stream().mapToInt(Integer::intValue).max().orElse(1);
            result[di] = (float) maxFreq / words.length;
        }
        return result;
    }

    /**

     * ID: GPU-CFEO-006
     * Requirement: computeCosineSimilarity must execute correctly within the contract defined by this class.
     * Purpose: Compute and return the computeCosineSimilarity result.
     * Inputs: float[] vector1, float[] vector2
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float computeCosineSimilarity(float[] vector1, float[] vector2) {
        // Standard dot-product cosine similarity: dot(v1,v2) / (||v1|| * ||v2||).
        if (vector1 == null || vector2 == null || vector1.length == 0) return 0.0f;
        int len = Math.min(vector1.length, vector2.length);
        float dot = 0f, n1 = 0f, n2 = 0f;
        for (int i = 0; i < len; i++) {
            dot += vector1[i] * vector2[i];
            n1  += vector1[i] * vector1[i];
            n2  += vector2[i] * vector2[i];
        }
        return (n1 == 0f || n2 == 0f) ? 0f : dot / (float) Math.sqrt(n1 * n2);
    }

    /**

     * ID: GPU-CFEO-007
     * Requirement: release must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void release() {
        // Release resources
    }

    // Add method for adapter to use
    /**

     * ID: GPU-CFEO-008
     * Requirement: extractFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractFeatures operation for this class.
     * Inputs: String[] tokens
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float[] extractFeatures(String[] tokens) {
        // Hash-based bag-of-words feature vector: each token is mapped to a normalised
        // float in [0,1] via its absolute hashCode divided by Integer.MAX_VALUE.
        if (tokens == null || tokens.length == 0) return new float[0];
        float[] features = new float[tokens.length];
        for (int i = 0; i < tokens.length; i++) {
            features[i] = (float)(tokens[i].hashCode() & 0x7FFFFFFF) / Integer.MAX_VALUE;
        }
        return features;
    }
}
