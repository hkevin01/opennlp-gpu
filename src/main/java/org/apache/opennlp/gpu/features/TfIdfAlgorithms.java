package org.apache.opennlp.gpu.features;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * Shared TF-IDF algorithms used by multiple compute backends.
 */
public final class TfIdfAlgorithms {

    public enum WeightingScheme {
        RAW_TF_IDF,
        SUBLINEAR_TF_IDF,
        BM25
    }

    public enum FeatureSelectionMethod {
        FREQUENCY,
        INFORMATION_GAIN,
        CHI_SQUARE
    }

    public enum IDFSmoothingStrategy {
        STANDARD_SMOOTH,
        PROBABILISTIC_IDF,
        BM25_IDF
    }

    public enum DenseVectorQuantization {
        FLOAT32,
        FLOAT16,
        INT8
    }

    public static final class ClassBalanceOptions {
        public final boolean macroAverage;
        public final boolean useClassPriorWeighting;

        public ClassBalanceOptions(boolean macroAverage, boolean useClassPriorWeighting) {
            this.macroAverage = macroAverage;
            this.useClassPriorWeighting = useClassPriorWeighting;
        }

        public static ClassBalanceOptions defaultOptions() {
            return new ClassBalanceOptions(true, false);
        }
    }

    public static final class NormalizationOptions {
        public final boolean normalizeUnicode;
        public final boolean stripPunctuation;
        public final boolean removeStopwords;
        public final Set<String> stopwords;

        public NormalizationOptions(boolean normalizeUnicode,
                                    boolean stripPunctuation,
                                    boolean removeStopwords,
                                    Set<String> stopwords) {
            this.normalizeUnicode = normalizeUnicode;
            this.stripPunctuation = stripPunctuation;
            this.removeStopwords = removeStopwords;
            this.stopwords = stopwords == null ? Collections.emptySet() : new HashSet<>(stopwords);
        }

        public static NormalizationOptions defaultOptions() {
            return new NormalizationOptions(true, true, true, defaultEnglishStopwords());
        }
    }

    public static final class NGramBlendOptions {
        private final Map<Integer, Double> ngramWeights;

        public NGramBlendOptions(Map<Integer, Double> ngramWeights) {
            this.ngramWeights = new HashMap<>();
            if (ngramWeights != null) {
                for (Map.Entry<Integer, Double> e : ngramWeights.entrySet()) {
                    if (e.getKey() != null && e.getKey() >= 1 && e.getValue() != null && e.getValue() > 0.0) {
                        this.ngramWeights.put(e.getKey(), e.getValue());
                    }
                }
            }
            if (this.ngramWeights.isEmpty()) {
                this.ngramWeights.put(1, 1.0);
            }
        }

        public static NGramBlendOptions unigramOnly() {
            return new NGramBlendOptions(Collections.singletonMap(1, 1.0));
        }

        public static NGramBlendOptions singleN(int n) {
            return new NGramBlendOptions(Collections.singletonMap(Math.max(1, n), 1.0));
        }

        public static NGramBlendOptions linearMix(double unigramWeight, double bigramWeight, double trigramWeight) {
            Map<Integer, Double> weights = new HashMap<>();
            if (unigramWeight > 0.0) weights.put(1, unigramWeight);
            if (bigramWeight > 0.0) weights.put(2, bigramWeight);
            if (trigramWeight > 0.0) weights.put(3, trigramWeight);
            return new NGramBlendOptions(weights);
        }

        public Set<Integer> enabledNValues() {
            return new HashSet<>(ngramWeights.keySet());
        }

        public double getWeight(int n) {
            return ngramWeights.getOrDefault(n, 0.0);
        }

        public Map<Integer, Double> getWeights() {
            return new HashMap<>(ngramWeights);
        }
    }

    public static final class VectorizationOptions {
        public final int maxFeatures;
        public final WeightingScheme weightingScheme;
        public final NormalizationOptions normalizationOptions;
        public final NGramBlendOptions ngramBlendOptions;
        public final FeatureSelectionMethod featureSelectionMethod;
        public final ClassBalanceOptions classBalanceOptions;
        public final IDFSmoothingStrategy idfSmoothingStrategy;
        public final int minDocumentFrequency;
        public final int maxDocumentFrequency;
        public final String[] labels;
        public final boolean useCache;

        public VectorizationOptions(int maxFeatures,
                                    WeightingScheme weightingScheme,
                                    NormalizationOptions normalizationOptions,
                                    NGramBlendOptions ngramBlendOptions,
                                    FeatureSelectionMethod featureSelectionMethod,
                                    ClassBalanceOptions classBalanceOptions,
                                    IDFSmoothingStrategy idfSmoothingStrategy,
                                    int minDocumentFrequency,
                                    int maxDocumentFrequency,
                                    String[] labels,
                                    boolean useCache) {
            this.maxFeatures = maxFeatures;
            this.weightingScheme = weightingScheme == null ? WeightingScheme.RAW_TF_IDF : weightingScheme;
            this.normalizationOptions = normalizationOptions == null ? NormalizationOptions.defaultOptions() : normalizationOptions;
            this.ngramBlendOptions = ngramBlendOptions == null ? NGramBlendOptions.unigramOnly() : ngramBlendOptions;
            this.featureSelectionMethod = featureSelectionMethod == null ? FeatureSelectionMethod.FREQUENCY : featureSelectionMethod;
            this.classBalanceOptions = classBalanceOptions == null ? ClassBalanceOptions.defaultOptions() : classBalanceOptions;
            this.idfSmoothingStrategy = idfSmoothingStrategy == null ? IDFSmoothingStrategy.STANDARD_SMOOTH : idfSmoothingStrategy;
            this.minDocumentFrequency = Math.max(1, minDocumentFrequency);
            this.maxDocumentFrequency = maxDocumentFrequency <= 0 ? Integer.MAX_VALUE : maxDocumentFrequency;
            this.labels = labels;
            this.useCache = useCache;
        }

        public static VectorizationOptions defaultForMaxFeatures(int maxFeatures) {
            return new VectorizationOptions(
                    maxFeatures,
                    WeightingScheme.RAW_TF_IDF,
                    NormalizationOptions.defaultOptions(),
                    NGramBlendOptions.unigramOnly(),
                    FeatureSelectionMethod.FREQUENCY,
                    ClassBalanceOptions.defaultOptions(),
                    IDFSmoothingStrategy.STANDARD_SMOOTH,
                    1,
                    Integer.MAX_VALUE,
                    null,
                    true
            );
        }
    }

    public static final class SparseVector {
        private final int[] indices;
        private final float[] values;

        public SparseVector(int[] indices, float[] values) {
            this.indices = indices;
            this.values = values;
        }

        public int[] getIndices() {
            return indices;
        }

        public float[] getValues() {
            return values;
        }
    }

    public static final class VocabularyState {
        private final Map<String, Integer> vocabulary;
        private final Map<String, Integer> documentFrequency;
        private final int totalDocuments;
        private final double averageDocumentLength;
        private final NGramBlendOptions ngramBlendOptions;
        private final IDFSmoothingStrategy idfSmoothingStrategy;
        private final int minDocumentFrequency;
        private final int maxDocumentFrequency;

        public VocabularyState(Map<String, Integer> vocabulary,
                               Map<String, Integer> documentFrequency,
                               int totalDocuments,
                               double averageDocumentLength,
                               NGramBlendOptions ngramBlendOptions,
                               IDFSmoothingStrategy idfSmoothingStrategy,
                               int minDocumentFrequency,
                               int maxDocumentFrequency) {
            this.vocabulary = new HashMap<>(vocabulary);
            this.documentFrequency = new HashMap<>(documentFrequency);
            this.totalDocuments = totalDocuments;
            this.averageDocumentLength = averageDocumentLength;
            this.ngramBlendOptions = ngramBlendOptions == null ? NGramBlendOptions.unigramOnly() : ngramBlendOptions;
            this.idfSmoothingStrategy = idfSmoothingStrategy == null ? IDFSmoothingStrategy.STANDARD_SMOOTH : idfSmoothingStrategy;
            this.minDocumentFrequency = Math.max(1, minDocumentFrequency);
            this.maxDocumentFrequency = maxDocumentFrequency <= 0 ? Integer.MAX_VALUE : maxDocumentFrequency;
        }

        /**
         * Backward-compatible constructor for older callers.
         */
        public VocabularyState(Map<String, Integer> vocabulary,
                               Map<String, Integer> documentFrequency,
                               int totalDocuments,
                               double averageDocumentLength,
                               NGramBlendOptions ngramBlendOptions) {
            this(vocabulary, documentFrequency, totalDocuments, averageDocumentLength,
                    ngramBlendOptions, IDFSmoothingStrategy.STANDARD_SMOOTH, 1, Integer.MAX_VALUE);
        }

        public Map<String, Integer> getVocabulary() {
            return new HashMap<>(vocabulary);
        }

        public Map<String, Integer> getDocumentFrequency() {
            return new HashMap<>(documentFrequency);
        }

        public int getTotalDocuments() {
            return totalDocuments;
        }

        public double getAverageDocumentLength() {
            return averageDocumentLength;
        }

        public NGramBlendOptions getNgramBlendOptions() {
            return ngramBlendOptions;
        }

        public IDFSmoothingStrategy getIdfSmoothingStrategy() {
            return idfSmoothingStrategy;
        }

        public int getMinDocumentFrequency() {
            return minDocumentFrequency;
        }

        public int getMaxDocumentFrequency() {
            return maxDocumentFrequency;
        }
    }

    public static final class LoadedDenseVectors {
        private final float[][] denseVectors;
        private final DenseVectorQuantization quantization;

        public LoadedDenseVectors(float[][] denseVectors, DenseVectorQuantization quantization) {
            this.denseVectors = denseVectors;
            this.quantization = quantization;
        }

        public float[][] getDenseVectors() {
            return denseVectors;
        }

        public DenseVectorQuantization getQuantization() {
            return quantization;
        }
    }

    public static final class VectorizationResult {
        private final Map<String, Integer> vocabulary;
        private final float[][] denseVectors;
        private final SparseVector[] sparseVectors;
        private final WeightingScheme scheme;
        private final VocabularyState vocabularyState;

        public VectorizationResult(Map<String, Integer> vocabulary,
                                   float[][] denseVectors,
                                   SparseVector[] sparseVectors,
                                   WeightingScheme scheme,
                                   VocabularyState vocabularyState) {
            this.vocabulary = vocabulary;
            this.denseVectors = denseVectors;
            this.sparseVectors = sparseVectors;
            this.scheme = scheme;
            this.vocabularyState = vocabularyState;
        }

        public Map<String, Integer> getVocabulary() {
            return vocabulary;
        }

        public float[][] getDenseVectors() {
            return denseVectors;
        }

        public SparseVector[] getSparseVectors() {
            return sparseVectors;
        }

        public WeightingScheme getScheme() {
            return scheme;
        }

        public VocabularyState getVocabularyState() {
            return vocabularyState;
        }
    }

    private static final class CorpusCacheEntry {
        private final String corpusFingerprint;
        private final long createdAtNanos;
        private final List<Map<String, Double>> docTermWeights;
        private final int[] docLengths;
        private final double avgDocLength;
        private final Map<String, Integer> corpusDocumentFrequency;
        private final Map<String, Double> corpusFrequency;

        private CorpusCacheEntry(String corpusFingerprint,
                                 long createdAtNanos,
                                 List<Map<String, Double>> docTermWeights,
                                 int[] docLengths,
                                 double avgDocLength,
                                 Map<String, Integer> corpusDocumentFrequency,
                                 Map<String, Double> corpusFrequency) {
            this.corpusFingerprint = corpusFingerprint;
            this.createdAtNanos = createdAtNanos;
            this.docTermWeights = docTermWeights;
            this.docLengths = docLengths;
            this.avgDocLength = avgDocLength;
            this.corpusDocumentFrequency = corpusDocumentFrequency;
            this.corpusFrequency = corpusFrequency;
        }
    }

    private static final int CACHE_FORMAT_V1 = 1;
    private static final int CACHE_FORMAT_V2 = 2;
    private static final int DENSE_VECTOR_FORMAT_VERSION = 1;
    private static final int MAX_CACHE_ENTRIES = 32;
    private static final double BM25_K1 = 1.2;
    private static final double BM25_B = 0.75;
    private static final double EPS = 1e-12;

    private static final Map<String, CorpusCacheEntry> CORPUS_CACHE = Collections.synchronizedMap(
            new LinkedHashMap<String, CorpusCacheEntry>(64, 0.75f, true) {
                @Override
                protected boolean removeEldestEntry(Map.Entry<String, CorpusCacheEntry> eldest) {
                    return size() > MAX_CACHE_ENTRIES;
                }
            }
    );

    private TfIdfAlgorithms() {
        // utility class
    }

    /**
     * Compute a scalar TF-IDF score per document from dense vectors.
     */
    public static float[] computeDocumentScores(String[] documents) {
        VectorizationResult result = vectorizeDocuments(
                documents,
                1,
                Integer.MAX_VALUE,
                WeightingScheme.RAW_TF_IDF,
                NormalizationOptions.defaultOptions(),
                true
        );

        float[][] dense = result.getDenseVectors();
        float[] scores = new float[dense.length];
        for (int i = 0; i < dense.length; i++) {
            double sumSquares = 0.0;
            for (float v : dense[i]) {
                sumSquares += v * v;
            }
            scores[i] = (float) Math.sqrt(sumSquares);
        }
        return scores;
    }

    /**
     * Backward-compatible vectorization API for a single n-gram size.
     */
    public static VectorizationResult vectorizeDocuments(String[] documents,
                                                         int ngramSize,
                                                         int maxFeatures,
                                                         WeightingScheme scheme,
                                                         NormalizationOptions options,
                                                         boolean useCache) {
        VectorizationOptions vectorizationOptions = new VectorizationOptions(
                maxFeatures,
                scheme,
                options,
                NGramBlendOptions.singleN(ngramSize),
                FeatureSelectionMethod.FREQUENCY,
                ClassBalanceOptions.defaultOptions(),
                IDFSmoothingStrategy.STANDARD_SMOOTH,
                1,
                Integer.MAX_VALUE,
                null,
                useCache
        );
        return vectorizeDocuments(documents, vectorizationOptions);
    }

    /**
     * Extended vectorization API with weighted n-gram blending and discriminative feature selection.
     */
    public static VectorizationResult vectorizeDocuments(String[] documents,
                                                         VectorizationOptions options) {
        Objects.requireNonNull(options, "options must not be null");
        if (documents == null || documents.length == 0) {
            return new VectorizationResult(
                    Collections.emptyMap(),
                    new float[0][0],
                    new SparseVector[0],
                    options.weightingScheme,
                    new VocabularyState(Collections.emptyMap(), Collections.emptyMap(), 0, 0.0,
                            options.ngramBlendOptions, options.idfSmoothingStrategy,
                            options.minDocumentFrequency, options.maxDocumentFrequency)
            );
        }
        if (options.maxFeatures < 1) {
            throw new IllegalArgumentException("maxFeatures must be >= 1, got " + options.maxFeatures);
        }

        CorpusCacheEntry cacheEntry = getOrBuildCorpusEntry(documents, options);

        Map<String, Integer> vocabulary = selectVocabulary(
                cacheEntry,
                options.maxFeatures,
                options.featureSelectionMethod,
                options.classBalanceOptions,
                options.labels,
                options.minDocumentFrequency,
                options.maxDocumentFrequency
        );

        Map<String, Integer> selectedDf = new HashMap<>();
        for (String term : vocabulary.keySet()) {
            selectedDf.put(term, cacheEntry.corpusDocumentFrequency.getOrDefault(term, 1));
        }

        float[][] dense = buildDenseVectors(
                cacheEntry,
                vocabulary,
                selectedDf,
                documents.length,
                options.weightingScheme,
                options.idfSmoothingStrategy
        );

        SparseVector[] sparse = new SparseVector[dense.length];
        for (int i = 0; i < dense.length; i++) {
            sparse[i] = toSparse(dense[i]);
        }

        VocabularyState state = new VocabularyState(
                vocabulary,
                selectedDf,
                documents.length,
                cacheEntry.avgDocLength,
                options.ngramBlendOptions,
                options.idfSmoothingStrategy,
                options.minDocumentFrequency,
                options.maxDocumentFrequency
        );

        return new VectorizationResult(
                Collections.unmodifiableMap(new HashMap<>(vocabulary)),
                dense,
                sparse,
                options.weightingScheme,
                state
        );
    }

    /**
     * Vectorize documents with an externally persisted vocabulary/df state.
     */
    public static VectorizationResult vectorizeDocumentsWithVocabulary(String[] documents,
                                                                       VocabularyState state,
                                                                       WeightingScheme scheme,
                                                                       NormalizationOptions normalizationOptions,
                                                                       boolean useCache) {
        Objects.requireNonNull(state, "state must not be null");
        if (documents == null || documents.length == 0) {
            return new VectorizationResult(
                    Collections.unmodifiableMap(state.getVocabulary()),
                    new float[0][state.getVocabulary().size()],
                    new SparseVector[0],
                    scheme,
                    state
            );
        }

        VectorizationOptions options = new VectorizationOptions(
                Math.max(1, state.getVocabulary().size()),
                scheme,
                normalizationOptions,
                state.getNgramBlendOptions(),
                FeatureSelectionMethod.FREQUENCY,
                ClassBalanceOptions.defaultOptions(),
                state.getIdfSmoothingStrategy(),
                state.getMinDocumentFrequency(),
                state.getMaxDocumentFrequency(),
                null,
                useCache
        );

        CorpusCacheEntry cacheEntry = getOrBuildCorpusEntry(documents, options);
        Map<String, Integer> vocabulary = state.getVocabulary();
        Map<String, Integer> df = state.getDocumentFrequency();

        float[][] dense = buildDenseVectors(cacheEntry, vocabulary, df, Math.max(1, state.getTotalDocuments()),
                scheme, state.getIdfSmoothingStrategy());
        SparseVector[] sparse = new SparseVector[dense.length];
        for (int i = 0; i < dense.length; i++) {
            sparse[i] = toSparse(dense[i]);
        }

        return new VectorizationResult(
                Collections.unmodifiableMap(vocabulary),
                dense,
                sparse,
                scheme,
                state
        );
    }

    public static String[] tokenizeNormalized(String document, NormalizationOptions options) {
        if (document == null) {
            return new String[0];
        }

        NormalizationOptions effectiveOptions = options == null ? NormalizationOptions.defaultOptions() : options;
        String normalized = document;

        if (effectiveOptions.normalizeUnicode) {
            normalized = Normalizer.normalize(normalized, Normalizer.Form.NFKC);
        }

        normalized = normalized.toLowerCase();

        if (effectiveOptions.stripPunctuation) {
            normalized = normalized.replaceAll("[\\p{Punct}\\p{S}]", " ");
        }

        normalized = normalized.replaceAll("\\s+", " ").trim();
        if (normalized.isEmpty()) {
            return new String[0];
        }

        String[] rawTokens = normalized.split(" ");
        if (!effectiveOptions.removeStopwords || effectiveOptions.stopwords.isEmpty()) {
            return rawTokens;
        }

        List<String> filtered = new ArrayList<>(rawTokens.length);
        for (String token : rawTokens) {
            if (!effectiveOptions.stopwords.contains(token)) {
                filtered.add(token);
            }
        }
        return filtered.toArray(new String[0]);
    }

    /**
     * Persist vocabulary and document-frequency metadata.
     */
    public static void saveVocabularyState(VocabularyState state, Path path) throws IOException {
        Objects.requireNonNull(state, "state must not be null");
        Objects.requireNonNull(path, "path must not be null");
        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(CACHE_FORMAT_V2);
            out.writeInt(state.totalDocuments);
            out.writeDouble(state.averageDocumentLength);

            out.writeInt(state.vocabulary.size());
            for (Map.Entry<String, Integer> e : state.vocabulary.entrySet()) {
                out.writeUTF(e.getKey());
                out.writeInt(e.getValue());
            }

            out.writeInt(state.documentFrequency.size());
            for (Map.Entry<String, Integer> e : state.documentFrequency.entrySet()) {
                out.writeUTF(e.getKey());
                out.writeInt(e.getValue());
            }

            Map<Integer, Double> blendWeights = state.ngramBlendOptions.getWeights();
            out.writeInt(blendWeights.size());
            for (Map.Entry<Integer, Double> e : blendWeights.entrySet()) {
                out.writeInt(e.getKey());
                out.writeDouble(e.getValue());
            }

            out.writeUTF(state.idfSmoothingStrategy.name());
            out.writeInt(state.minDocumentFrequency);
            out.writeInt(state.maxDocumentFrequency);
        }
    }

    /**
     * Load vocabulary metadata. Supports migration from V1 -> V2.
     */
    public static VocabularyState loadVocabularyState(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int version = in.readInt();
            if (version == CACHE_FORMAT_V2) {
                return readVocabularyStateV2(in);
            } else if (version == CACHE_FORMAT_V1) {
                return migrateVocabularyStateFromV1(in);
            }
            throw new IOException("Unsupported vocabulary state version: " + version);
        }
    }

    /**
     * Persist dense vectors in FLOAT32/FLOAT16/INT8 form for memory-efficient storage.
     */
    public static void saveDenseVectors(float[][] denseVectors,
                                        Path path,
                                        DenseVectorQuantization quantization) throws IOException {
        Objects.requireNonNull(denseVectors, "denseVectors must not be null");
        Objects.requireNonNull(path, "path must not be null");
        DenseVectorQuantization q = quantization == null ? DenseVectorQuantization.FLOAT32 : quantization;

        try (DataOutputStream out = new DataOutputStream(new BufferedOutputStream(Files.newOutputStream(path)))) {
            out.writeInt(DENSE_VECTOR_FORMAT_VERSION);
            out.writeUTF(q.name());
            out.writeInt(denseVectors.length);
            out.writeInt(denseVectors.length == 0 ? 0 : denseVectors[0].length);

            switch (q) {
                case FLOAT32:
                    for (float[] row : denseVectors) {
                        for (float v : row) out.writeFloat(v);
                    }
                    break;
                case FLOAT16:
                    for (float[] row : denseVectors) {
                        for (float v : row) out.writeShort(floatToHalf(v));
                    }
                    break;
                case INT8:
                    for (float[] row : denseVectors) {
                        float maxAbs = 0f;
                        for (float v : row) maxAbs = Math.max(maxAbs, Math.abs(v));
                        float scale = maxAbs < EPS ? 1f : maxAbs / 127f;
                        out.writeFloat(scale);
                        for (float v : row) {
                            int qv = Math.round(v / scale);
                            qv = Math.max(-127, Math.min(127, qv));
                            out.writeByte(qv);
                        }
                    }
                    break;
                default:
                    throw new IOException("Unsupported quantization: " + q);
            }
        }
    }

    public static LoadedDenseVectors loadDenseVectors(Path path) throws IOException {
        Objects.requireNonNull(path, "path must not be null");
        try (DataInputStream in = new DataInputStream(new BufferedInputStream(Files.newInputStream(path)))) {
            int version = in.readInt();
            if (version != DENSE_VECTOR_FORMAT_VERSION) {
                throw new IOException("Unsupported dense vector format version: " + version);
            }
            DenseVectorQuantization q = DenseVectorQuantization.valueOf(in.readUTF());
            int rows = in.readInt();
            int cols = in.readInt();
            float[][] vectors = new float[rows][cols];

            switch (q) {
                case FLOAT32:
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) vectors[i][j] = in.readFloat();
                    }
                    break;
                case FLOAT16:
                    for (int i = 0; i < rows; i++) {
                        for (int j = 0; j < cols; j++) vectors[i][j] = halfToFloat(in.readUnsignedShort());
                    }
                    break;
                case INT8:
                    for (int i = 0; i < rows; i++) {
                        float scale = in.readFloat();
                        for (int j = 0; j < cols; j++) {
                            vectors[i][j] = in.readByte() * scale;
                        }
                    }
                    break;
                default:
                    throw new IOException("Unsupported quantization: " + q);
            }
            return new LoadedDenseVectors(vectors, q);
        }
    }

    public static void clearCache() {
        synchronized (CORPUS_CACHE) {
            CORPUS_CACHE.clear();
        }
    }

    public static int getCacheSize() {
        synchronized (CORPUS_CACHE) {
            return CORPUS_CACHE.size();
        }
    }

    public static List<String> getCachedCorpusFingerprints() {
        synchronized (CORPUS_CACHE) {
            List<String> fingerprints = new ArrayList<>(CORPUS_CACHE.size());
            for (CorpusCacheEntry entry : CORPUS_CACHE.values()) {
                fingerprints.add(entry.corpusFingerprint);
            }
            return fingerprints;
        }
    }

    private static VocabularyState readVocabularyStateV2(DataInputStream in) throws IOException {
        int totalDocuments = in.readInt();
        double averageDocumentLength = in.readDouble();

        int vocabSize = in.readInt();
        Map<String, Integer> vocabulary = new HashMap<>(vocabSize * 2 + 1);
        for (int i = 0; i < vocabSize; i++) {
            vocabulary.put(in.readUTF(), in.readInt());
        }

        int dfSize = in.readInt();
        Map<String, Integer> documentFrequency = new HashMap<>(dfSize * 2 + 1);
        for (int i = 0; i < dfSize; i++) {
            documentFrequency.put(in.readUTF(), in.readInt());
        }

        int blendSize = in.readInt();
        Map<Integer, Double> blendWeights = new HashMap<>();
        for (int i = 0; i < blendSize; i++) {
            blendWeights.put(in.readInt(), in.readDouble());
        }

        IDFSmoothingStrategy smoothing = IDFSmoothingStrategy.valueOf(in.readUTF());
        int minDf = in.readInt();
        int maxDf = in.readInt();

        return new VocabularyState(
                vocabulary,
                documentFrequency,
                totalDocuments,
                averageDocumentLength,
                new NGramBlendOptions(blendWeights),
                smoothing,
                minDf,
                maxDf
        );
    }

    private static VocabularyState migrateVocabularyStateFromV1(DataInputStream in) throws IOException {
        int totalDocuments = in.readInt();
        double averageDocumentLength = in.readDouble();

        int vocabSize = in.readInt();
        Map<String, Integer> vocabulary = new HashMap<>(vocabSize * 2 + 1);
        for (int i = 0; i < vocabSize; i++) {
            vocabulary.put(in.readUTF(), in.readInt());
        }

        int dfSize = in.readInt();
        Map<String, Integer> documentFrequency = new HashMap<>(dfSize * 2 + 1);
        for (int i = 0; i < dfSize; i++) {
            documentFrequency.put(in.readUTF(), in.readInt());
        }

        int blendSize = in.readInt();
        Map<Integer, Double> blendWeights = new HashMap<>();
        for (int i = 0; i < blendSize; i++) {
            blendWeights.put(in.readInt(), in.readDouble());
        }

        return new VocabularyState(
                vocabulary,
                documentFrequency,
                totalDocuments,
                averageDocumentLength,
                new NGramBlendOptions(blendWeights),
                IDFSmoothingStrategy.STANDARD_SMOOTH,
                1,
                Integer.MAX_VALUE
        );
    }

    private static CorpusCacheEntry getOrBuildCorpusEntry(String[] documents,
                                                          VectorizationOptions options) {
        String cacheKey = buildCorpusFingerprint(documents, options.ngramBlendOptions, options.normalizationOptions);
        if (options.useCache) {
            synchronized (CORPUS_CACHE) {
                CorpusCacheEntry cached = CORPUS_CACHE.get(cacheKey);
                if (cached != null) {
                    return cached;
                }
            }
        }

        CorpusCacheEntry built = buildCorpusEntry(documents, options.ngramBlendOptions, options.normalizationOptions);
        if (options.useCache) {
            synchronized (CORPUS_CACHE) {
                CorpusCacheEntry cached = CORPUS_CACHE.get(cacheKey);
                if (cached != null) {
                    return cached;
                }
                CORPUS_CACHE.put(cacheKey, built);
            }
        }
        return built;
    }

    private static CorpusCacheEntry buildCorpusEntry(String[] documents,
                                                     NGramBlendOptions blendOptions,
                                                     NormalizationOptions normalizationOptions) {
        int numDocs = documents.length;
        List<Map<String, Double>> docTermWeights = new ArrayList<>(numDocs);
        int[] docLengths = new int[numDocs];

        Map<String, Integer> documentFrequency = new HashMap<>();
        Map<String, Double> corpusFrequency = new HashMap<>();

        for (int i = 0; i < numDocs; i++) {
            String[] tokens = tokenizeNormalized(documents[i], normalizationOptions);
            docLengths[i] = Math.max(1, tokens.length);

            Map<String, Double> termWeights = new HashMap<>();
            for (Integer n : blendOptions.enabledNValues()) {
                double nWeight = blendOptions.getWeight(n);
                if (nWeight <= 0.0) continue;

                List<String> ngrams = generateNGrams(tokens, n);
                for (String term : ngrams) {
                    termWeights.merge(term, nWeight, Double::sum);
                    corpusFrequency.merge(term, nWeight, Double::sum);
                }
            }

            Set<String> seenInDoc = new HashSet<>(termWeights.keySet());
            for (String term : seenInDoc) {
                documentFrequency.merge(term, 1, Integer::sum);
            }

            docTermWeights.add(termWeights);
        }

        double avgDocLength = Arrays.stream(docLengths).average().orElse(1.0);
        String fingerprint = buildCorpusFingerprint(documents, blendOptions, normalizationOptions);

        return new CorpusCacheEntry(
                fingerprint,
                System.nanoTime(),
                docTermWeights,
                docLengths,
                avgDocLength,
                documentFrequency,
                corpusFrequency
        );
    }

    private static Map<String, Integer> selectVocabulary(CorpusCacheEntry entry,
                                                         int maxFeatures,
                                                         FeatureSelectionMethod selectionMethod,
                                                         ClassBalanceOptions balanceOptions,
                                                         String[] labels,
                                                         int minDf,
                                                         int maxDf) {
        Map<String, Double> termScores;
        if (selectionMethod == FeatureSelectionMethod.CHI_SQUARE && isLabelSetValid(labels, entry.docTermWeights.size())) {
            termScores = scoreTermsByChiSquare(entry, labels, balanceOptions);
        } else if (selectionMethod == FeatureSelectionMethod.INFORMATION_GAIN && isLabelSetValid(labels, entry.docTermWeights.size())) {
            termScores = scoreTermsByInformationGain(entry, labels, balanceOptions);
        } else {
            termScores = new HashMap<>(entry.corpusFrequency);
        }

        List<Map.Entry<String, Double>> sorted = new ArrayList<>();
        for (Map.Entry<String, Double> e : termScores.entrySet()) {
            int df = entry.corpusDocumentFrequency.getOrDefault(e.getKey(), 0);
            if (df >= minDf && df <= maxDf) {
                sorted.add(e);
            }
        }

        sorted.sort((a, b) -> {
            int cmp = Double.compare(b.getValue(), a.getValue());
            return (cmp != 0) ? cmp : a.getKey().compareTo(b.getKey());
        });

        int vocabSize = Math.min(maxFeatures, sorted.size());
        Map<String, Integer> vocabulary = new HashMap<>(vocabSize * 2 + 1);
        for (int i = 0; i < vocabSize; i++) {
            vocabulary.put(sorted.get(i).getKey(), i);
        }
        return vocabulary;
    }

    private static float[][] buildDenseVectors(CorpusCacheEntry cacheEntry,
                                               Map<String, Integer> vocabulary,
                                               Map<String, Integer> documentFrequency,
                                               int totalDocsForIdf,
                                               WeightingScheme scheme,
                                               IDFSmoothingStrategy smoothingStrategy) {
        int numDocs = cacheEntry.docTermWeights.size();
        int vocabSize = vocabulary.size();
        float[][] dense = new float[numDocs][vocabSize];

        for (int docIndex = 0; docIndex < numDocs; docIndex++) {
            Map<String, Double> termWeights = cacheEntry.docTermWeights.get(docIndex);
            int docLength = Math.max(1, cacheEntry.docLengths[docIndex]);

            for (Map.Entry<String, Double> termEntry : termWeights.entrySet()) {
                String term = termEntry.getKey();
                Integer featureIndex = vocabulary.get(term);
                if (featureIndex == null) continue;

                double tf = termEntry.getValue();
                int df = Math.max(1, documentFrequency.getOrDefault(term, 1));
                double value;
                if (scheme == WeightingScheme.BM25) {
                    value = computeBm25Weight(tf, df, totalDocsForIdf, docLength, cacheEntry.avgDocLength);
                } else {
                    value = computeTfIdfWeight(tf, df, totalDocsForIdf, scheme, smoothingStrategy);
                }
                dense[docIndex][featureIndex] = (float) value;
            }
        }

        return dense;
    }

    private static boolean isLabelSetValid(String[] labels, int expectedSize) {
        return labels != null && labels.length == expectedSize;
    }

    private static Map<String, Double> scoreTermsByChiSquare(CorpusCacheEntry entry,
                                                              String[] labels,
                                                              ClassBalanceOptions balanceOptions) {
        int numDocs = labels.length;
        Map<String, Integer> classCounts = new HashMap<>();
        for (String label : labels) classCounts.merge(label, 1, Integer::sum);

        Map<String, Double> scores = new HashMap<>();
        for (String term : entry.corpusDocumentFrequency.keySet()) {
            double agg = balanceOptions.macroAverage ? 0.0 : Double.NEGATIVE_INFINITY;
            int classes = 0;

            for (Map.Entry<String, Integer> cls : classCounts.entrySet()) {
                String label = cls.getKey();
                int classSize = cls.getValue();

                int a = 0;
                int b = 0;
                int c = 0;
                int d = 0;

                for (int i = 0; i < numDocs; i++) {
                    boolean present = entry.docTermWeights.get(i).containsKey(term);
                    boolean inClass = label.equals(labels[i]);
                    if (present && inClass) a++;
                    else if (present) b++;
                    else if (inClass) c++;
                    else d++;
                }

                double numerator = (double) numDocs * Math.pow((a * d - b * c), 2);
                double denominator = (double) (a + b) * (c + d) * (a + c) * (b + d) + EPS;
                double score = numerator / denominator;

                if (balanceOptions.useClassPriorWeighting) {
                    double prior = (double) classSize / numDocs;
                    score *= (1.0 / Math.max(prior, EPS));
                }

                if (balanceOptions.macroAverage) {
                    agg += score;
                    classes++;
                } else {
                    agg = Math.max(agg, score);
                }
            }
            if (balanceOptions.macroAverage && classes > 0) {
                agg /= classes;
            }
            scores.put(term, Math.max(0.0, agg));
        }
        return scores;
    }

    private static Map<String, Double> scoreTermsByInformationGain(CorpusCacheEntry entry,
                                                                    String[] labels,
                                                                    ClassBalanceOptions balanceOptions) {
        int numDocs = labels.length;
        Map<String, Integer> classCounts = new HashMap<>();
        for (String label : labels) classCounts.merge(label, 1, Integer::sum);

        Map<String, Double> classWeights = new HashMap<>();
        if (balanceOptions.useClassPriorWeighting) {
            for (Map.Entry<String, Integer> e : classCounts.entrySet()) {
                double prior = (double) e.getValue() / numDocs;
                classWeights.put(e.getKey(), 1.0 / Math.max(prior, EPS));
            }
        } else {
            for (String label : classCounts.keySet()) {
                classWeights.put(label, 1.0);
            }
        }

        double entropyClass = weightedEntropy(classCounts, classWeights, numDocs);
        Map<String, Double> scores = new HashMap<>();

        for (String term : entry.corpusDocumentFrequency.keySet()) {
            Map<String, Integer> presentClassCounts = new HashMap<>();
            Map<String, Integer> absentClassCounts = new HashMap<>();
            int present = 0;
            int absent = 0;

            for (int i = 0; i < numDocs; i++) {
                boolean hasTerm = entry.docTermWeights.get(i).containsKey(term);
                if (hasTerm) {
                    present++;
                    presentClassCounts.merge(labels[i], 1, Integer::sum);
                } else {
                    absent++;
                    absentClassCounts.merge(labels[i], 1, Integer::sum);
                }
            }

            double conditionalEntropy =
                    ((double) present / numDocs) * weightedEntropy(presentClassCounts, classWeights, present)
                            + ((double) absent / numDocs) * weightedEntropy(absentClassCounts, classWeights, absent);
            double ig = Math.max(0.0, entropyClass - conditionalEntropy);

            if (balanceOptions.macroAverage) {
                scores.put(term, ig);
            } else {
                // Non-macro mode: retain compatibility by slight frequency emphasis
                scores.put(term, ig * Math.max(1.0, entry.corpusDocumentFrequency.getOrDefault(term, 1)));
            }
        }

        return scores;
    }

    private static double weightedEntropy(Map<String, Integer> counts,
                                          Map<String, Double> classWeights,
                                          int total) {
        if (total <= 0) return 0.0;
        double h = 0.0;
        for (Map.Entry<String, Integer> e : counts.entrySet()) {
            int count = e.getValue() == null ? 0 : e.getValue();
            if (count <= 0) continue;
            double p = (double) count / total;
            double weight = classWeights.getOrDefault(e.getKey(), 1.0);
            h -= weight * p * (Math.log(p + EPS) / Math.log(2.0));
        }
        return h;
    }

    private static SparseVector toSparse(float[] denseRow) {
        int nnz = 0;
        for (float v : denseRow) {
            if (v != 0.0f) nnz++;
        }

        int[] indices = new int[nnz];
        float[] values = new float[nnz];
        int p = 0;
        for (int i = 0; i < denseRow.length; i++) {
            if (denseRow[i] != 0.0f) {
                indices[p] = i;
                values[p] = denseRow[i];
                p++;
            }
        }
        return new SparseVector(indices, values);
    }

    private static String buildCorpusFingerprint(String[] documents,
                                                 NGramBlendOptions blendOptions,
                                                 NormalizationOptions normalizationOptions) {
        int docsHash = Arrays.hashCode(documents);
        int blendHash = blendOptions.getWeights().hashCode();
        int normHash = Objects.hash(
                normalizationOptions.normalizeUnicode,
                normalizationOptions.stripPunctuation,
                normalizationOptions.removeStopwords,
                normalizationOptions.stopwords.hashCode()
        );
        return Integer.toHexString(Objects.hash(docsHash, blendHash, normHash));
    }

    private static List<String> generateNGrams(String[] tokens, int n) {
        List<String> ngrams = new ArrayList<>();
        if (tokens.length < n) return ngrams;
        for (int i = 0; i <= tokens.length - n; i++) {
            StringBuilder sb = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) sb.append('_');
                sb.append(tokens[i + j]);
            }
            ngrams.add(sb.toString());
        }
        return ngrams;
    }

    private static double computeTfIdfWeight(double tf,
                                             int df,
                                             int totalDocs,
                                             WeightingScheme scheme,
                                             IDFSmoothingStrategy smoothing) {
        double effectiveTf = (scheme == WeightingScheme.SUBLINEAR_TF_IDF) ? Math.log1p(tf) : tf;
        double idf;
        switch (smoothing) {
            case PROBABILISTIC_IDF:
                idf = Math.log((totalDocs - df + 0.5) / (df + 0.5) + 1.0);
                break;
            case BM25_IDF:
                idf = Math.log(1.0 + (totalDocs - df + 0.5) / (df + 0.5));
                break;
            case STANDARD_SMOOTH:
            default:
                idf = Math.log((totalDocs + 1.0) / (df + 1.0)) + 1.0;
                break;
        }
        return effectiveTf * idf;
    }

    private static double computeBm25Weight(double tf,
                                            int df,
                                            int totalDocs,
                                            int docLength,
                                            double avgDocLength) {
        double idf = Math.log(1.0 + (totalDocs - df + 0.5) / (df + 0.5));
        double numerator = tf * (BM25_K1 + 1.0);
        double denominator = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * (docLength / Math.max(avgDocLength, EPS)));
        return idf * (numerator / Math.max(denominator, EPS));
    }

    private static short floatToHalf(float value) {
        int fbits = Float.floatToIntBits(value);
        int sign = (fbits >>> 16) & 0x8000;
        int val = (fbits & 0x7fffffff) + 0x1000;

        if (val >= 0x47800000) {
            if ((fbits & 0x7fffffff) >= 0x47800000) {
                if (val < 0x7f800000) {
                    return (short) (sign | 0x7c00);
                }
                return (short) (sign | 0x7c00 | ((fbits & 0x007fffff) >>> 13));
            }
            return (short) (sign | 0x7bff);
        }
        if (val >= 0x38800000) {
            return (short) (sign | ((val - 0x38000000) >>> 13));
        }
        if (val < 0x33000000) {
            return (short) sign;
        }
        val = (fbits & 0x7fffffff) >>> 23;
        return (short) (sign | ((((fbits & 0x7fffff) | 0x800000) + (0x800000 >>> (val - 102))) >>> (126 - val)));
    }

    private static float halfToFloat(int half) {
        int mant = half & 0x03ff;
        int exp = half & 0x7c00;
        int f;
        if (exp == 0x7c00) {
            exp = 0x3fc00;
        } else if (exp != 0) {
            exp += 0x1c000;
        } else if (mant != 0) {
            exp = 0x1c400;
            do {
                mant <<= 1;
                exp -= 0x400;
            } while ((mant & 0x400) == 0);
            mant &= 0x3ff;
        }
        f = ((half & 0x8000) << 16) | ((exp | mant) << 13);
        return Float.intBitsToFloat(f);
    }

    private static Set<String> defaultEnglishStopwords() {
        return new HashSet<>(Arrays.asList(
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
                "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"
        ));
    }
}
