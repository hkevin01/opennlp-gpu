package org.apache.opennlp.gpu.features;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.nio.file.Path;
import java.io.IOException;
import java.util.Arrays;
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
    private TfIdfAlgorithms.NormalizationOptions normalizationOptions = TfIdfAlgorithms.NormalizationOptions.defaultOptions();
    private TfIdfAlgorithms.NGramBlendOptions nGramBlendOptions = TfIdfAlgorithms.NGramBlendOptions.unigramOnly();
    private TfIdfAlgorithms.FeatureSelectionMethod featureSelectionMethod = TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY;
    private TfIdfAlgorithms.ClassBalanceOptions classBalanceOptions = TfIdfAlgorithms.ClassBalanceOptions.defaultOptions();
    private TfIdfAlgorithms.IDFSmoothingStrategy idfSmoothingStrategy = TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH;
    private double bm25PlusDelta = 1.0;
    private double bm25LowerTfBound = 0.0;
    private int minDocumentFrequency = 1;
    private int maxDocumentFrequency = Integer.MAX_VALUE;
    private TfIdfAlgorithms.VocabularyState vocabularyState;
    private TfIdfAlgorithms.CalibrationMetadata calibrationMetadata = TfIdfAlgorithms.CalibrationMetadata.none();
    private int vocabularySize = 0;

    // Performance thresholds
    private static final int MIN_DOCS_FOR_GPU = 100;
    private static final int MIN_FEATURES_FOR_GPU = 1000;

    public static final class CalibratedVectorizationResult {
        private final TfIdfAlgorithms.VectorizationResult rawResult;
        private final float[][] calibratedDenseVectors;
        private final TfIdfAlgorithms.SparseVector[] calibratedSparseVectors;
        private final float[] rawScores;
        private final float[] calibratedScores;
        private final TfIdfAlgorithms.CalibrationMetadata calibrationMetadata;

        public CalibratedVectorizationResult(TfIdfAlgorithms.VectorizationResult rawResult,
                                             float[][] calibratedDenseVectors,
                                             TfIdfAlgorithms.SparseVector[] calibratedSparseVectors,
                                             float[] rawScores,
                                             float[] calibratedScores,
                                             TfIdfAlgorithms.CalibrationMetadata calibrationMetadata) {
            this.rawResult = rawResult;
            this.calibratedDenseVectors = calibratedDenseVectors;
            this.calibratedSparseVectors = calibratedSparseVectors;
            this.rawScores = rawScores;
            this.calibratedScores = calibratedScores;
            this.calibrationMetadata = calibrationMetadata;
        }

        public TfIdfAlgorithms.VectorizationResult getRawResult() {
            return rawResult;
        }

        public float[][] getCalibratedDenseVectors() {
            return calibratedDenseVectors;
        }

        public TfIdfAlgorithms.SparseVector[] getCalibratedSparseVectors() {
            return calibratedSparseVectors;
        }

        public float[] getRawScores() {
            return rawScores;
        }

        public float[] getCalibratedScores() {
            return calibratedScores;
        }

        public TfIdfAlgorithms.CalibrationMetadata getCalibrationMetadata() {
            return calibrationMetadata;
        }
    }

    public static final class TrainingBundleBuildResult {
        private final CalibratedVectorizationResult calibratedVectorizationResult;
        private final TfIdfAlgorithms.TrainingArtifactBundle trainingArtifactBundle;

        public TrainingBundleBuildResult(CalibratedVectorizationResult calibratedVectorizationResult,
                                         TfIdfAlgorithms.TrainingArtifactBundle trainingArtifactBundle) {
            this.calibratedVectorizationResult = calibratedVectorizationResult;
            this.trainingArtifactBundle = trainingArtifactBundle;
        }

        public CalibratedVectorizationResult getCalibratedVectorizationResult() {
            return calibratedVectorizationResult;
        }

        public TfIdfAlgorithms.TrainingArtifactBundle getTrainingArtifactBundle() {
            return trainingArtifactBundle;
        }
    }

    /**

     * ID: GPU-GFE-002
     * Requirement: GpuFeatureExtractor must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuFeatureExtractor instance.
     * Inputs: ComputeProvider provider, GpuConfig config, MatrixOperation matrixOp
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuFeatureExtractor(ComputeProvider provider, GpuConfig config, MatrixOperation matrixOp) {
        this.provider  = Objects.requireNonNull(provider,  "provider must not be null");
        this.config    = Objects.requireNonNull(config,    "config must not be null");
        this.matrixOp  = Objects.requireNonNull(matrixOp,  "matrixOp must not be null");
        logger.info("Initialized GPU feature extractor with " + provider.getName());
    }

    /**
     * Extract n-gram features from text documents
     */
    /**

     * ID: GPU-GFE-003
     * Requirement: extractNGramFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractNGramFeatures operation for this class.
     * Inputs: String[] documents, int ngramSize, int maxFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**

     * ID: GPU-GFE-004
     * Requirement: extractTfIdfFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractTfIdfFeatures operation for this class.
     * Inputs: String[] documents, int ngramSize, int maxFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float[][] extractTfIdfFeatures(String[] documents, int ngramSize, int maxFeatures) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (ngramSize < 1) throw new IllegalArgumentException("ngramSize must be >= 1, got: " + ngramSize);
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);
        if (documents.length == 0) return new float[0][0];
        logger.debug("Extracting TF-IDF features from " + documents.length + " documents");
        return extractTfIdfVectors(documents, ngramSize, maxFeatures,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF).getDenseVectors();
    }

    /**
     * Extract N-gram-aware TF-IDF vectors aligned with vocabulary indices.
     * The result includes both dense and sparse representations.
     */
    public TfIdfAlgorithms.VectorizationResult extractTfIdfVectors(String[] documents,
                                                                   int ngramSize,
                                                                   int maxFeatures) {
        return extractTfIdfVectors(documents, ngramSize, maxFeatures,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF);
    }

    /**
     * Extract N-gram-aware TF-IDF vectors using the selected weighting scheme.
     */
    public TfIdfAlgorithms.VectorizationResult extractTfIdfVectors(String[] documents,
                                                                   int ngramSize,
                                                                   int maxFeatures,
                                                                   TfIdfAlgorithms.WeightingScheme scheme) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (ngramSize < 1) throw new IllegalArgumentException("ngramSize must be >= 1, got: " + ngramSize);
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);
        if (documents.length == 0) {
            return new TfIdfAlgorithms.VectorizationResult(new HashMap<String, Integer>(), new float[0][0],
                    new TfIdfAlgorithms.SparseVector[0], scheme,
                    new TfIdfAlgorithms.VocabularyState(new HashMap<String, Integer>(), new HashMap<String, Integer>(),
                            0, 0.0, TfIdfAlgorithms.NGramBlendOptions.singleN(ngramSize)));
        }

        TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                maxFeatures,
                scheme,
                normalizationOptions,
                TfIdfAlgorithms.NGramBlendOptions.singleN(ngramSize),
                featureSelectionMethod,
            classBalanceOptions,
            idfSmoothingStrategy,
            minDocumentFrequency,
            maxDocumentFrequency,
            bm25PlusDelta,
            bm25LowerTfBound,
                null,
                true
        );

        TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(documents, options);

        vocabulary.clear();
        vocabulary.putAll(result.getVocabulary());
        vocabularySize = vocabulary.size();
        vocabularyState = result.getVocabularyState();
        return result;
    }

    /**
     * Extract TF-IDF vectors with blend/selection configuration and optional labels.
     */
    public TfIdfAlgorithms.VectorizationResult extractTfIdfVectors(String[] documents,
                                                                   int maxFeatures,
                                                                   TfIdfAlgorithms.WeightingScheme scheme,
                                                                   String[] labels) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);
        if (documents.length == 0) {
            return new TfIdfAlgorithms.VectorizationResult(new HashMap<String, Integer>(), new float[0][0],
                    new TfIdfAlgorithms.SparseVector[0], scheme,
                    new TfIdfAlgorithms.VocabularyState(new HashMap<String, Integer>(), new HashMap<String, Integer>(),
                            0, 0.0, nGramBlendOptions));
        }

        TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                maxFeatures,
                scheme,
                normalizationOptions,
                nGramBlendOptions,
                featureSelectionMethod,
            classBalanceOptions,
            idfSmoothingStrategy,
            minDocumentFrequency,
            maxDocumentFrequency,
            bm25PlusDelta,
            bm25LowerTfBound,
                labels,
                true
        );

        TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(documents, options);

        // Keep existing extractor state aligned with the vectorizer output.
        vocabulary.clear();
        vocabulary.putAll(result.getVocabulary());
        vocabularySize = vocabulary.size();
        vocabularyState = result.getVocabularyState();

        return result;
    }

    /**
     * Configure centralized token normalization for all feature extraction paths.
     */
    public void setNormalizationOptions(TfIdfAlgorithms.NormalizationOptions normalizationOptions) {
        this.normalizationOptions = Objects.requireNonNull(normalizationOptions, "normalizationOptions must not be null");
    }

    /**
     * Configure linear blending for unigram/bigram/trigram features.
     */
    public void setNGramBlendOptions(TfIdfAlgorithms.NGramBlendOptions nGramBlendOptions) {
        this.nGramBlendOptions = Objects.requireNonNull(nGramBlendOptions, "nGramBlendOptions must not be null");
    }

    /**
     * Configure vocabulary feature pruning strategy.
     */
    public void setFeatureSelectionMethod(TfIdfAlgorithms.FeatureSelectionMethod featureSelectionMethod) {
        this.featureSelectionMethod = Objects.requireNonNull(featureSelectionMethod, "featureSelectionMethod must not be null");
    }

    /**
     * Configure class-balance behavior for discriminative feature scoring.
     */
    public void setClassBalanceOptions(TfIdfAlgorithms.ClassBalanceOptions classBalanceOptions) {
        this.classBalanceOptions = Objects.requireNonNull(classBalanceOptions, "classBalanceOptions must not be null");
    }

    /**
     * Configure IDF smoothing strategy.
     */
    public void setIdfSmoothingStrategy(TfIdfAlgorithms.IDFSmoothingStrategy idfSmoothingStrategy) {
        this.idfSmoothingStrategy = Objects.requireNonNull(idfSmoothingStrategy, "idfSmoothingStrategy must not be null");
    }

    /**
     * Configure BM25+ parameters used when weighting scheme is BM25_PLUS.
     */
    public void setBm25PlusOptions(double bm25PlusDelta, double bm25LowerTfBound) {
        if (bm25PlusDelta < 0.0) {
            throw new IllegalArgumentException("bm25PlusDelta must be >= 0.0, got: " + bm25PlusDelta);
        }
        if (bm25LowerTfBound < 0.0) {
            throw new IllegalArgumentException("bm25LowerTfBound must be >= 0.0, got: " + bm25LowerTfBound);
        }
        this.bm25PlusDelta = bm25PlusDelta;
        this.bm25LowerTfBound = bm25LowerTfBound;
    }

    /**
     * Configure min/max document-frequency cutoffs used during vocabulary selection.
     */
    public void setDocumentFrequencyCutoffs(int minDocumentFrequency, int maxDocumentFrequency) {
        if (minDocumentFrequency < 1) {
            throw new IllegalArgumentException("minDocumentFrequency must be >= 1, got: " + minDocumentFrequency);
        }
        if (maxDocumentFrequency < minDocumentFrequency) {
            throw new IllegalArgumentException("maxDocumentFrequency must be >= minDocumentFrequency");
        }
        this.minDocumentFrequency = minDocumentFrequency;
        this.maxDocumentFrequency = maxDocumentFrequency;
    }

    /**
     * Persist the latest vocabulary/df state for reproducible inference.
     */
    public void saveVocabularyState(Path path) throws IOException {
        if (vocabularyState == null) {
            throw new IllegalStateException("No vocabulary state available. Run extractTfIdfVectors first.");
        }
        TfIdfAlgorithms.saveVocabularyState(vocabularyState, path);
    }

    /**
     * Load a persisted vocabulary state and update in-memory extractor state.
     */
    public void loadVocabularyState(Path path) throws IOException {
        vocabularyState = TfIdfAlgorithms.loadVocabularyState(path);
        vocabulary.clear();
        vocabulary.putAll(vocabularyState.getVocabulary());
        vocabularySize = vocabulary.size();
        nGramBlendOptions = vocabularyState.getNgramBlendOptions();
        idfSmoothingStrategy = vocabularyState.getIdfSmoothingStrategy();
        minDocumentFrequency = vocabularyState.getMinDocumentFrequency();
        maxDocumentFrequency = vocabularyState.getMaxDocumentFrequency();
        calibrationMetadata = vocabularyState.getCalibrationMetadata();
    }

    /**
     * Vectorize using a previously loaded/persisted vocabulary state.
     */
    public TfIdfAlgorithms.VectorizationResult extractTfIdfVectorsWithLoadedVocabulary(String[] documents,
                                                                                        TfIdfAlgorithms.WeightingScheme scheme) {
        if (vocabularyState == null) {
            throw new IllegalStateException("No vocabulary state loaded. Call loadVocabularyState first.");
        }
        return TfIdfAlgorithms.vectorizeDocumentsWithVocabulary(
                documents,
                vocabularyState,
                scheme,
                normalizationOptions,
                true
        );
    }

        /**
         * Single-call vectorization plus optional calibration fit/apply pipeline.
         */
        public CalibratedVectorizationResult extractTfIdfVectorsWithCalibration(String[] documents,
                                             int maxFeatures,
                                             TfIdfAlgorithms.WeightingScheme scheme,
                                             String[] labels,
                                             TfIdfAlgorithms.VectorCalibrationMethod vectorCalibrationMethod,
                                             TfIdfAlgorithms.ScoreCalibrationMethod scoreCalibrationMethod,
                                             boolean persistCalibrationMetadata) {
        TfIdfAlgorithms.VectorizationResult raw = extractTfIdfVectors(documents, maxFeatures, scheme, labels);
        float[] rawScores = computeDenseVectorNormScores(raw.getDenseVectors());
        TfIdfAlgorithms.CalibrationMetadata fitted = TfIdfAlgorithms.fitCalibrationMetadata(
            raw.getDenseVectors(),
            rawScores,
            vectorCalibrationMethod,
            scoreCalibrationMethod
        );
        TfIdfAlgorithms.CalibrationApplicationResult calibrated = TfIdfAlgorithms.applyCalibration(
            raw.getDenseVectors(),
            rawScores,
            fitted
        );

        if (persistCalibrationMetadata && vocabularyState != null) {
            vocabularyState = vocabularyState.withCalibrationMetadata(fitted);
            calibrationMetadata = fitted;
        }

        return new CalibratedVectorizationResult(
            raw,
            calibrated.getCalibratedDenseVectors(),
            toSparse(calibrated.getCalibratedDenseVectors()),
            rawScores,
            calibrated.getCalibratedScores(),
            fitted
        );
        }

        /**
         * Vectorize with loaded vocabulary, then apply persisted calibration metadata.
         */
        public CalibratedVectorizationResult extractTfIdfVectorsWithLoadedCalibration(String[] documents,
                                               TfIdfAlgorithms.WeightingScheme scheme) {
        TfIdfAlgorithms.VectorizationResult raw = extractTfIdfVectorsWithLoadedVocabulary(documents, scheme);
        float[] rawScores = computeDenseVectorNormScores(raw.getDenseVectors());

        TfIdfAlgorithms.CalibrationMetadata metadata = (vocabularyState == null)
            ? TfIdfAlgorithms.CalibrationMetadata.none()
            : vocabularyState.getCalibrationMetadata();
        TfIdfAlgorithms.CalibrationApplicationResult calibrated = TfIdfAlgorithms.applyCalibration(
            raw.getDenseVectors(),
            rawScores,
            metadata
        );

        return new CalibratedVectorizationResult(
            raw,
            calibrated.getCalibratedDenseVectors(),
            toSparse(calibrated.getCalibratedDenseVectors()),
            rawScores,
            calibrated.getCalibratedScores(),
            metadata
        );
        }

        /**
         * Build a compact strongly-typed training artifact bundle in one call.
         */
        public TrainingBundleBuildResult buildTrainingArtifactBundle(String[] documents,
                                     int maxFeatures,
                                     TfIdfAlgorithms.WeightingScheme scheme,
                                     String[] labels,
                                     TfIdfAlgorithms.VectorCalibrationMethod vectorCalibrationMethod,
                                     TfIdfAlgorithms.ScoreCalibrationMethod scoreCalibrationMethod,
                                     int diagnosticsTopK,
                                     boolean imbalanceAdjusted) {
        CalibratedVectorizationResult calibrated = extractTfIdfVectorsWithCalibration(
            documents,
            maxFeatures,
            scheme,
            labels,
            vectorCalibrationMethod,
            scoreCalibrationMethod,
            true
        );

        TfIdfAlgorithms.VocabularyDiagnostics diagnostics = computeVocabularyDiagnostics(documents, maxFeatures, labels);
        TfIdfAlgorithms.PerClassDiagnosticsReport diagnosticsReport = aggregatePerClassDiagnostics(
            diagnostics,
            labels,
            diagnosticsTopK,
            imbalanceAdjusted
        );

        int cols = calibrated.getCalibratedDenseVectors().length == 0
            ? calibrated.getRawResult().getVocabulary().size()
            : calibrated.getCalibratedDenseVectors()[0].length;
        TfIdfAlgorithms.SparseCsrMatrix csr = toSparseCsr(calibrated.getCalibratedSparseVectors(), cols);

        TfIdfAlgorithms.VocabularyState bundleState = calibrated.getRawResult()
            .getVocabularyState()
            .withCalibrationMetadata(calibrated.getCalibrationMetadata());
        TfIdfAlgorithms.TrainingArtifactBundle bundle = TfIdfAlgorithms.buildTrainingArtifactBundle(
            bundleState,
            calibrated.getCalibrationMetadata(),
            diagnosticsReport,
            csr
        );
        return new TrainingBundleBuildResult(calibrated, bundle);
        }

        /**
         * Build a compact artifact bundle from already-computed components.
         */
        public TfIdfAlgorithms.TrainingArtifactBundle buildTrainingArtifactBundle(TfIdfAlgorithms.VocabularyState vocabularyState,
                                               TfIdfAlgorithms.CalibrationMetadata calibrationMetadata,
                                               TfIdfAlgorithms.PerClassDiagnosticsReport diagnosticsSummary,
                                               TfIdfAlgorithms.SparseCsrMatrix csrMatrix) {
        return TfIdfAlgorithms.buildTrainingArtifactBundle(
            vocabularyState,
            calibrationMetadata,
            diagnosticsSummary,
            csrMatrix
        );
        }

    /**
     * Compute per-term vocabulary diagnostics using current extractor configuration.
     */
    public TfIdfAlgorithms.VocabularyDiagnostics computeVocabularyDiagnostics(String[] documents,
                                                                              int maxFeatures) {
        return computeVocabularyDiagnostics(documents, maxFeatures, null);
    }

    /**
     * Compute per-term vocabulary diagnostics with optional labels for discriminative scoring.
     */
    public TfIdfAlgorithms.VocabularyDiagnostics computeVocabularyDiagnostics(String[] documents,
                                                                              int maxFeatures,
                                                                              String[] labels) {
        Objects.requireNonNull(documents, "documents must not be null");
        if (maxFeatures < 1) throw new IllegalArgumentException("maxFeatures must be >= 1, got: " + maxFeatures);

        TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                maxFeatures,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                normalizationOptions,
                nGramBlendOptions,
                featureSelectionMethod,
                classBalanceOptions,
                idfSmoothingStrategy,
                minDocumentFrequency,
                maxDocumentFrequency,
            bm25PlusDelta,
            bm25LowerTfBound,
                labels,
                true
        );
        return TfIdfAlgorithms.computeVocabularyDiagnostics(documents, options);
    }

    /**
     * Aggregate diagnostics into per-class top-k summaries.
     */
    public TfIdfAlgorithms.PerClassDiagnosticsReport aggregatePerClassDiagnostics(
            TfIdfAlgorithms.VocabularyDiagnostics diagnostics,
            String[] labels,
            int topK,
            boolean imbalanceAdjusted) {
        return TfIdfAlgorithms.aggregatePerClassDiagnostics(diagnostics, labels, topK, imbalanceAdjusted);
    }

    /**
     * Convert sparse vectors to CSR using explicit column count.
     */
    public TfIdfAlgorithms.SparseCsrMatrix toSparseCsr(TfIdfAlgorithms.SparseVector[] sparseVectors,
                                                        int cols) {
        return TfIdfAlgorithms.toSparseCsr(sparseVectors, cols);
    }

    /**
     * Convert sparse vectors to CSR using current extractor vocabulary size.
     */
    public TfIdfAlgorithms.SparseCsrMatrix toSparseCsr(TfIdfAlgorithms.SparseVector[] sparseVectors) {
        return TfIdfAlgorithms.toSparseCsr(sparseVectors, vocabularySize);
    }

    /**
     * Persist sparse CSR matrix for ANN/retrieval pipelines.
     */
    public void saveSparseCsr(TfIdfAlgorithms.SparseCsrMatrix matrix, Path path) throws IOException {
        TfIdfAlgorithms.saveSparseCsr(matrix, path);
    }

    /**
     * Load sparse CSR matrix from disk.
     */
    public TfIdfAlgorithms.SparseCsrMatrix loadSparseCsr(Path path) throws IOException {
        return TfIdfAlgorithms.loadSparseCsr(path);
    }

    /**
     * Compute inner product similarity between two CSR rows.
     */
    public float dotProduct(TfIdfAlgorithms.SparseCsrMatrix matrix, int rowA, int rowB) {
        return TfIdfAlgorithms.dotProduct(matrix, rowA, rowB);
    }

    /**
     * Compute cosine similarity between two CSR rows.
     */
    public float cosineSimilarity(TfIdfAlgorithms.SparseCsrMatrix matrix, int rowA, int rowB) {
        return TfIdfAlgorithms.cosineSimilarity(matrix, rowA, rowB);
    }

    /**
     * Search nearest rows by cosine similarity.
     */
    public List<TfIdfAlgorithms.CsrSearchResult> searchTopKByCosine(TfIdfAlgorithms.SparseCsrMatrix matrix,
                                                                     int queryRow,
                                                                     int topK) {
        return TfIdfAlgorithms.searchTopKByCosine(matrix, queryRow, topK);
    }

    /**
     * Search nearest rows by inner product.
     */
    public List<TfIdfAlgorithms.CsrSearchResult> searchTopKByInnerProduct(TfIdfAlgorithms.SparseCsrMatrix matrix,
                                                                           int queryRow,
                                                                           int topK) {
        return TfIdfAlgorithms.searchTopKByInnerProduct(matrix, queryRow, topK);
    }

    /**
     * Calibrate dense vectors for cross-corpus comparability.
     */
    public float[][] calibrateDenseVectors(float[][] denseVectors,
                                           TfIdfAlgorithms.VectorCalibrationMethod method) {
        return TfIdfAlgorithms.calibrateDenseVectors(denseVectors, method);
    }

    /**
     * Calibrate scalar scores for cross-corpus comparability.
     */
    public float[] calibrateScores(float[] scores,
                                   TfIdfAlgorithms.ScoreCalibrationMethod method) {
        return TfIdfAlgorithms.calibrateScores(scores, method);
    }

    /**
     * Evaluate score comparability metrics across corpora.
     */
    public TfIdfAlgorithms.ComparabilityMetrics evaluateComparability(float[] baselineScores, float[] comparedScores) {
        return TfIdfAlgorithms.evaluateComparability(baselineScores, comparedScores);
    }

    /**
     * Extract context window features around target words
     */
    /**

     * ID: GPU-GFE-005
     * Requirement: extractContextFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractContextFeatures operation for this class.
     * Inputs: String[] documents, String[] targetWords, int windowSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**

     * ID: GPU-GFE-006
     * Requirement: normalizeFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the normalizeFeatures operation for this class.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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

    /**

     * ID: GPU-GFE-007
     * Requirement: buildVocabulary must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new ocabulary.
     * Inputs: String[] documents, int ngramSize, int maxFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-009
     * Requirement: tokenize must execute correctly within the contract defined by this class.
     * Purpose: Implement the tokenize operation for this class.
     * Inputs: String text
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private String[] tokenize(String text) {
        return TfIdfAlgorithms.tokenizeNormalized(text, normalizationOptions);
    }

    private static float[] computeDenseVectorNormScores(float[][] denseVectors) {
        if (denseVectors == null) {
            return new float[0];
        }
        float[] scores = new float[denseVectors.length];
        for (int i = 0; i < denseVectors.length; i++) {
            double sum = 0.0;
            for (float v : denseVectors[i]) sum += v * v;
            scores[i] = (float) Math.sqrt(sum);
        }
        return scores;
    }

    private static TfIdfAlgorithms.SparseVector[] toSparse(float[][] denseVectors) {
        if (denseVectors == null) {
            return new TfIdfAlgorithms.SparseVector[0];
        }
        TfIdfAlgorithms.SparseVector[] sparse = new TfIdfAlgorithms.SparseVector[denseVectors.length];
        for (int r = 0; r < denseVectors.length; r++) {
            float[] row = denseVectors[r];
            int nnz = 0;
            for (float v : row) if (v != 0.0f) nnz++;

            int[] idx = new int[nnz];
            float[] vals = new float[nnz];
            int p = 0;
            for (int c = 0; c < row.length; c++) {
                if (row[c] != 0.0f) {
                    idx[p] = c;
                    vals[p] = row[c];
                    p++;
                }
            }
            sparse[r] = new TfIdfAlgorithms.SparseVector(idx, vals);
        }
        return sparse;
    }

    /**

     * ID: GPU-GFE-010
     * Requirement: generateNGrams must execute correctly within the contract defined by this class.
     * Purpose: Implement the generateNGrams operation for this class.
     * Inputs: String[] tokens, int n
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-011
     * Requirement: extractContextWindow must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractContextWindow operation for this class.
     * Inputs: String[] tokens, int targetIndex, int windowSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-012
     * Requirement: extractNGramFeaturesCpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractNGramFeaturesCpu operation for this class.
     * Inputs: String[] documents, float[][] features, int ngramSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-014
     * Requirement: normalizeFeaturesCpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the normalizeFeaturesCpu operation for this class.
     * Inputs: float[][] features, int numDocs, int numFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-015
     * Requirement: extractNGramFeaturesGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractNGramFeaturesGpu operation for this class.
     * Inputs: String[] documents, float[][] features, int ngramSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-017
     * Requirement: normalizeFeaturesGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the normalizeFeaturesGpu operation for this class.
     * Inputs: float[][] features, int numDocs, int numFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**

     * ID: GPU-GFE-018
     * Requirement: shouldUseGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpu operation for this class.
     * Inputs: int numDocuments, int numFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpu(int numDocuments, int numFeatures) {
        return provider.isGpuProvider() &&
               config.isGpuEnabled() &&
               numDocuments >= MIN_DOCS_FOR_GPU &&
               numFeatures >= MIN_FEATURES_FOR_GPU;
    }

    /**

     * ID: GPU-GFE-019
     * Requirement: Return the VocabularySize field value without side effects.
     * Purpose: Return the value of the VocabularySize property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getVocabularySize() {
        return vocabularySize;
    }

    /**

     * ID: GPU-GFE-020
     * Requirement: Return the Vocabulary field value without side effects.
     * Purpose: Return the value of the Vocabulary property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Map<String, Integer> getVocabulary() {
        return new HashMap<String, Integer>(vocabulary);
    }

    /**

     * ID: GPU-GFE-021
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
        vocabulary.clear();
        vocabularyState = null;
        calibrationMetadata = TfIdfAlgorithms.CalibrationMetadata.none();
        vocabularySize = 0;
        logger.debug("Released feature extractor resources");
    }
}
