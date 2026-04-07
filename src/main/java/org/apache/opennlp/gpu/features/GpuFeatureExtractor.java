package org.apache.opennlp.gpu.features;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.stream.IntStream;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.MatrixOperation;

/**

 * ID: GPU-GFE-001
 * Requirement: GpuFeatureExtractor must extract numeric feature vectors from NLP token contexts using the active GPU compute backend.
 * Purpose: Higher-level wrapper around FeatureExtractionOperation that integrates token normalisation, vocabulary lookup, and TF-IDF weighting.
 * Rationale: Consolidating feature extraction logic here decouples NLP feature engineering from GPU dispatch plumbing.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Allocates transient feature arrays; may cache vocabulary index for repeat calls.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuFeatureExtractor {

    private static final GpuLogger logger = GpuLogger.getLogger(GpuFeatureExtractor.class);

    private final ComputeProvider provider;
    private final GpuConfig config;
    private final MatrixOperation matrixOp;

    // Feature extraction parameters
    private final Map<String, Integer> vocabulary = new HashMap<String, Integer>();
    private final Map<String, Float> idfScores = new HashMap<String, Float>();
    private int vocabularySize = 0;

    // Performance thresholds
    private static final int MIN_DOCS_FOR_GPU = 100;
    private static final int MIN_FEATURES_FOR_GPU = 1000;

    public GpuFeatureExtractor(ComputeProvider provider, GpuConfig config, MatrixOperation matrixOp) {
        this.provider  = Objects.requireNonNull(provider,  "provider must not be null");
        this.config    = Objects.requireNonNull(config,    "config must not be null");
        this.matrixOp  = Objects.requireNonNull(matrixOp,  "matrixOp must not be null");
        logger.info("Initialized GPU feature extractor with " + provider.getName());
    }

    /**
     * Extract n-gram features from text documents
     */
    public float[][] extractNGramFeatures(String[] documents, int ngramSize, int maxFeatures) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (ngramSize < 1) throw new IllegalArgumentException("ngramSize must be >= 1, got: " + ngramSize);
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);
        if (documents.length == 0) return new float[0][0];
        logger.debug("Extracting " + ngramSize + "-gram features from " + documents.length + " documents");

        // Build vocabulary
        buildVocabulary(documents, ngramSize, maxFeatures);

        // Extract features
        float[][] features = new float[documents.length][vocabularySize];

        if (shouldUseGpu(documents.length, vocabularySize)) {
            extractNGramFeaturesGpu(documents, features, ngramSize);
        } else {
            extractNGramFeaturesCpu(documents, features, ngramSize);
        }

        return features;
    }

    /**
     * Extract TF-IDF features from text documents
     */
    public float[][] extractTfIdfFeatures(String[] documents, int ngramSize, int maxFeatures) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (ngramSize < 1) throw new IllegalArgumentException("ngramSize must be >= 1, got: " + ngramSize);
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);
        if (documents.length == 0) return new float[0][0];
        logger.debug("Extracting TF-IDF features from " + documents.length + " documents");

        // First extract n-gram features
        float[][] tfFeatures = extractNGramFeatures(documents, ngramSize, maxFeatures);

        // Calculate IDF scores
        calculateIdfScores(documents, ngramSize);

        // Apply TF-IDF transformation
        float[][] tfidfFeatures = new float[documents.length][vocabularySize];

        if (shouldUseGpu(documents.length, vocabularySize)) {
            applyTfIdfTransformationGpu(tfFeatures, tfidfFeatures);
        } else {
            applyTfIdfTransformationCpu(tfFeatures, tfidfFeatures);
        }

        return tfidfFeatures;
    }

    /**
     * Extract context window features around target words
     */
    public float[][] extractContextFeatures(String[] documents, String[] targetWords, int windowSize) {
        logger.debug("Extracting context features with window size " + windowSize);

        List<float[]> allFeatures = new ArrayList<float[]>();

        for (String document : documents) {
            String[] tokens = tokenize(document);

            for (int i = 0; i < tokens.length; i++) {
                for (String target : targetWords) {
                    if (tokens[i].equals(target)) {
                        float[] contextFeature = extractContextWindow(tokens, i, windowSize);
                        allFeatures.add(contextFeature);
                    }
                }
            }
        }

        return allFeatures.toArray(new float[allFeatures.size()][]);
    }

    /**
     * Apply feature normalization
     */
    public void normalizeFeatures(float[][] features) {
        Objects.requireNonNull(features, "features must not be null");
        if (features.length == 0) return;
        logger.debug("Normalizing features for " + features.length + " documents");

        int numDocs = features.length;
        int numFeatures = features[0].length;

        if (shouldUseGpu(numDocs, numFeatures)) {
            normalizeFeaturesGpu(features, numDocs, numFeatures);
        } else {
            normalizeFeaturesCpu(features, numDocs, numFeatures);
        }
    }

    // Private helper methods

    private void buildVocabulary(String[] documents, int ngramSize, int maxFeatures) {
        Map<String, Integer> ngramCounts = new HashMap<String, Integer>();

        // Count n-grams
        for (String document : documents) {
            String[] tokens = tokenize(document);
            List<String> ngrams = generateNGrams(tokens, ngramSize);

            for (String ngram : ngrams) {
                ngramCounts.put(ngram, ngramCounts.getOrDefault(ngram, 0) + 1);
            }
        }

        // Select top features
        List<Map.Entry<String, Integer>> sortedNgrams = new ArrayList<Map.Entry<String, Integer>>(ngramCounts.entrySet());
        sortedNgrams.sort((a, b) -> b.getValue().compareTo(a.getValue()));

        vocabulary.clear();
        vocabularySize = Math.min(maxFeatures, sortedNgrams.size());

        for (int i = 0; i < vocabularySize; i++) {
            vocabulary.put(sortedNgrams.get(i).getKey(), i);
        }

        logger.debug("Built vocabulary with " + vocabularySize + " features");
    }

    private void calculateIdfScores(String[] documents, int ngramSize) {
        Map<String, Integer> documentFrequency = new HashMap<String, Integer>();

        // Count document frequency for each n-gram
        for (String document : documents) {
            String[] tokens = tokenize(document);
            List<String> ngrams = generateNGrams(tokens, ngramSize);

            Map<String, Boolean> seenInDoc = new HashMap<String, Boolean>();
            for (String ngram : ngrams) {
                if (!seenInDoc.containsKey(ngram) && vocabulary.containsKey(ngram)) {
                    documentFrequency.put(ngram, documentFrequency.getOrDefault(ngram, 0) + 1);
                    seenInDoc.put(ngram, true);
                }
            }
        }

        // Calculate IDF scores
        idfScores.clear();
        int totalDocs = documents.length;

        for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
            String ngram = entry.getKey();
            int df = documentFrequency.getOrDefault(ngram, 1);
            float idf = (float) Math.log((double) totalDocs / df);
            idfScores.put(ngram, idf);
        }

        logger.debug("Calculated IDF scores for " + idfScores.size() + " features");
    }

    private String[] tokenize(String text) {
        // Simple whitespace tokenization (can be enhanced)
        return text.toLowerCase().split("\\s+");
    }

    private List<String> generateNGrams(String[] tokens, int n) {
        List<String> ngrams = new ArrayList<String>();

        for (int i = 0; i <= tokens.length - n; i++) {
            StringBuilder ngram = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) ngram.append("_");
                ngram.append(tokens[i + j]);
            }
            ngrams.add(ngram.toString());
        }

        return ngrams;
    }

    private float[] extractContextWindow(String[] tokens, int targetIndex, int windowSize) {
        float[] context = new float[windowSize * 2];

        int start = Math.max(0, targetIndex - windowSize);
        int end = Math.min(tokens.length, targetIndex + windowSize + 1);

        int contextIndex = 0;
        for (int i = start; i < end; i++) {
            if (i != targetIndex && contextIndex < context.length) {
                // Simple hash-based feature (can be enhanced with embeddings)
                context[contextIndex] = Math.abs(tokens[i].hashCode()) % 1000;
                contextIndex++;
            }
        }

        return context;
    }

    // CPU implementations

    private void extractNGramFeaturesCpu(String[] documents, float[][] features, int ngramSize) {
        for (int docIndex = 0; docIndex < documents.length; docIndex++) {
            String[] tokens = tokenize(documents[docIndex]);
            List<String> ngrams = generateNGrams(tokens, ngramSize);

            // Count n-gram frequencies
            for (String ngram : ngrams) {
                Integer featureIndex = vocabulary.get(ngram);
                if (featureIndex != null) {
                    features[docIndex][featureIndex]++;
                }
            }
        }
    }

    private void applyTfIdfTransformationCpu(float[][] tfFeatures, float[][] tfidfFeatures) {
        for (int docIndex = 0; docIndex < tfFeatures.length; docIndex++) {
            for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
                String ngram = entry.getKey();
                int featureIndex = entry.getValue();

                float tf = tfFeatures[docIndex][featureIndex];
                float idf = idfScores.getOrDefault(ngram, 0.0f);
                tfidfFeatures[docIndex][featureIndex] = tf * idf;
            }
        }
    }

    private void normalizeFeaturesCpu(float[][] features, int numDocs, int numFeatures) {
        for (int docIndex = 0; docIndex < numDocs; docIndex++) {
            float[] docFeatures = features[docIndex];

            // Calculate L2 norm
            float norm = 0.0f;
            for (int i = 0; i < numFeatures; i++) {
                norm += docFeatures[i] * docFeatures[i];
            }
            norm = (float) Math.sqrt(norm);

            // Normalize
            if (norm > 0.0f) {
                for (int i = 0; i < numFeatures; i++) {
                    docFeatures[i] /= norm;
                }
            }
        }
    }

    // GPU implementations (stubs for now)

    // GPU implementations using parallel streams
    // (provides real multi-core speedup without requiring GPU hardware)

    private void extractNGramFeaturesGpu(String[] documents, float[][] features, int ngramSize) {
        logger.debug("Parallel n-gram extraction for " + documents.length + " documents");
        IntStream.range(0, documents.length).parallel().forEach(docIndex -> {
            String[] tokens = tokenize(documents[docIndex]);
            List<String> ngrams = generateNGrams(tokens, ngramSize);
            for (String ngram : ngrams) {
                Integer featureIndex = vocabulary.get(ngram);
                if (featureIndex != null) {
                    // Atomic increment not needed: each docIndex writes to its own row
                    features[docIndex][featureIndex]++;
                }
            }
            // Normalise by document length (term frequency)
            int tokenCount = tokens.length;
            if (tokenCount > 0) {
                for (int i = 0; i < features[docIndex].length; i++) {
                    features[docIndex][i] /= tokenCount;
                }
            }
        });
    }

    private void applyTfIdfTransformationGpu(float[][] tfFeatures, float[][] tfidfFeatures) {
        logger.debug("Parallel TF-IDF transformation for " + tfFeatures.length + " documents");
        // Each document row is independent — safe to parallelise
        IntStream.range(0, tfFeatures.length).parallel().forEach(docIndex -> {
            for (Map.Entry<String, Integer> entry : vocabulary.entrySet()) {
                int featureIndex = entry.getValue();
                float idf = idfScores.getOrDefault(entry.getKey(), 0.0f);
                tfidfFeatures[docIndex][featureIndex] = tfFeatures[docIndex][featureIndex] * idf;
            }
        });
    }

    private void normalizeFeaturesGpu(float[][] features, int numDocs, int numFeatures) {
        logger.debug("Parallel L2 normalisation for " + numDocs + " document vectors");
        // Row-level normalisation: each document vector is independent
        IntStream.range(0, numDocs).parallel().forEach(docIndex -> {
            float norm = 0.0f;
            for (int i = 0; i < numFeatures; i++) {
                norm += features[docIndex][i] * features[docIndex][i];
            }
            norm = (float) Math.sqrt(norm);
            if (norm > 0.0f) {
                float invNorm = 1.0f / norm;
                for (int i = 0; i < numFeatures; i++) {
                    features[docIndex][i] *= invNorm;
                }
            }
        });
    }

    // Helper methods

    private boolean shouldUseGpu(int numDocuments, int numFeatures) {
        return provider.isGpuProvider() &&
               config.isGpuEnabled() &&
               numDocuments >= MIN_DOCS_FOR_GPU &&
               numFeatures >= MIN_FEATURES_FOR_GPU;
    }

    public int getVocabularySize() {
        return vocabularySize;
    }

    public Map<String, Integer> getVocabulary() {
        return new HashMap<String, Integer>(vocabulary);
    }

    public void release() {
        vocabulary.clear();
        idfScores.clear();
        vocabularySize = 0;
        logger.debug("Released feature extractor resources");
    }
}
