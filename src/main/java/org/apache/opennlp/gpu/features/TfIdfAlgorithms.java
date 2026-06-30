package org.apache.opennlp.gpu.features;

import java.text.Normalizer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Shared TF-IDF algorithms used by multiple compute backends.
 *
 * <p>This helper intentionally computes a single scalar score per document
 * because {@code FeatureExtractionOperation#computeTfIdf(String[])} returns
 * {@code float[]} with one element per input document.</p>
 */
public final class TfIdfAlgorithms {

    public enum WeightingScheme {
        RAW_TF_IDF,
        SUBLINEAR_TF_IDF,
        BM25
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

    public static final class VectorizationResult {
        private final Map<String, Integer> vocabulary;
        private final float[][] denseVectors;
        private final SparseVector[] sparseVectors;
        private final WeightingScheme scheme;

        public VectorizationResult(Map<String, Integer> vocabulary,
                                   float[][] denseVectors,
                                   SparseVector[] sparseVectors,
                                   WeightingScheme scheme) {
            this.vocabulary = vocabulary;
            this.denseVectors = denseVectors;
            this.sparseVectors = sparseVectors;
            this.scheme = scheme;
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
    }

    private static final class CorpusCacheEntry {
        private final List<List<String>> docNgrams;
        private final int[] docLengths;
        private final double avgDocLength;
        private final Map<String, Integer> vocabulary;
        private final Map<String, Integer> documentFrequency;

        private CorpusCacheEntry(List<List<String>> docNgrams,
                                 int[] docLengths,
                                 double avgDocLength,
                                 Map<String, Integer> vocabulary,
                                 Map<String, Integer> documentFrequency) {
            this.docNgrams = docNgrams;
            this.docLengths = docLengths;
            this.avgDocLength = avgDocLength;
            this.vocabulary = vocabulary;
            this.documentFrequency = documentFrequency;
        }
    }

    private static final int MAX_CACHE_ENTRIES = 32;
    private static final double BM25_K1 = 1.2;
    private static final double BM25_B = 0.75;
    private static final ConcurrentHashMap<String, CorpusCacheEntry> CORPUS_CACHE = new ConcurrentHashMap<>();

    private TfIdfAlgorithms() {
        // utility class
    }

    /**
     * Compute a TF-IDF score per document using smoothed IDF:
     * idf(t) = ln((N + 1) / (df(t) + 1)) + 1
     *
     * <p>The per-document scalar is the L2 norm of the document TF-IDF vector.</p>
     *
     * @param documents input documents
     * @return one TF-IDF score per document
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

    public static VectorizationResult vectorizeDocuments(String[] documents,
                                                         int ngramSize,
                                                         int maxFeatures,
                                                         WeightingScheme scheme,
                                                         NormalizationOptions options,
                                                         boolean useCache) {
        if (documents == null || documents.length == 0) {
            return new VectorizationResult(Collections.emptyMap(), new float[0][0], new SparseVector[0], scheme);
        }
        if (ngramSize < 1) {
            throw new IllegalArgumentException("ngramSize must be >= 1, got " + ngramSize);
        }
        if (maxFeatures < 1) {
            throw new IllegalArgumentException("maxFeatures must be >= 1, got " + maxFeatures);
        }

        NormalizationOptions effectiveOptions = options == null ? NormalizationOptions.defaultOptions() : options;
        CorpusCacheEntry entry = getOrBuildCorpusEntry(documents, ngramSize, maxFeatures, effectiveOptions, useCache);

        int numDocs = documents.length;
        int vocabSize = entry.vocabulary.size();
        float[][] dense = new float[numDocs][vocabSize];
        SparseVector[] sparse = new SparseVector[numDocs];

        for (int docIndex = 0; docIndex < numDocs; docIndex++) {
            List<String> ngrams = entry.docNgrams.get(docIndex);
            int docLength = Math.max(1, entry.docLengths[docIndex]);

            Map<String, Integer> tfCounts = new HashMap<>();
            for (String gram : ngrams) {
                if (entry.vocabulary.containsKey(gram)) {
                    tfCounts.merge(gram, 1, Integer::sum);
                }
            }

            for (Map.Entry<String, Integer> termEntry : tfCounts.entrySet()) {
                String term = termEntry.getKey();
                int tfCount = termEntry.getValue();
                Integer featureIndex = entry.vocabulary.get(term);
                if (featureIndex == null) continue;

                int df = Math.max(1, entry.documentFrequency.getOrDefault(term, 1));
                double value;
                if (scheme == WeightingScheme.BM25) {
                    value = computeBm25Weight(tfCount, df, numDocs, docLength, entry.avgDocLength);
                } else {
                    value = computeTfIdfWeight(tfCount, df, numDocs, scheme);
                }

                dense[docIndex][featureIndex] = (float) value;
            }

            sparse[docIndex] = toSparse(dense[docIndex]);
        }

        return new VectorizationResult(Collections.unmodifiableMap(new HashMap<>(entry.vocabulary)), dense, sparse, scheme);
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
            normalized = normalized.replaceAll("[\\p{Punct}\\p{IsSymbol}]", " ");
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

    public static void clearCache() {
        CORPUS_CACHE.clear();
    }

    public static int getCacheSize() {
        return CORPUS_CACHE.size();
    }

    private static CorpusCacheEntry getOrBuildCorpusEntry(String[] documents,
                                                          int ngramSize,
                                                          int maxFeatures,
                                                          NormalizationOptions options,
                                                          boolean useCache) {
        String cacheKey = buildCacheKey(documents, ngramSize, maxFeatures, options);
        if (useCache) {
            CorpusCacheEntry cached = CORPUS_CACHE.get(cacheKey);
            if (cached != null) return cached;
        }

        int numDocs = documents.length;
        List<List<String>> docNgrams = new ArrayList<>(numDocs);
        int[] docLengths = new int[numDocs];

        Map<String, Integer> corpusCounts = new HashMap<>();
        for (int i = 0; i < numDocs; i++) {
            String[] tokens = tokenizeNormalized(documents[i], options);
            docLengths[i] = Math.max(tokens.length, 1);
            List<String> grams = generateNGrams(tokens, ngramSize);
            docNgrams.add(grams);
            for (String gram : grams) {
                corpusCounts.merge(gram, 1, Integer::sum);
            }
        }

        List<Map.Entry<String, Integer>> sorted = new ArrayList<>(corpusCounts.entrySet());
        sorted.sort((a, b) -> {
            int cmp = Integer.compare(b.getValue(), a.getValue());
            return (cmp != 0) ? cmp : a.getKey().compareTo(b.getKey());
        });

        int vocabSize = Math.min(maxFeatures, sorted.size());
        Map<String, Integer> vocabulary = new HashMap<>(vocabSize * 2 + 1);
        for (int i = 0; i < vocabSize; i++) {
            vocabulary.put(sorted.get(i).getKey(), i);
        }

        Map<String, Integer> df = new HashMap<>(vocabSize * 2 + 1);
        for (List<String> grams : docNgrams) {
            Set<String> seen = new HashSet<>();
            for (String gram : grams) {
                if (vocabulary.containsKey(gram) && seen.add(gram)) {
                    df.merge(gram, 1, Integer::sum);
                }
            }
        }

        double avgDocLength = Arrays.stream(docLengths).average().orElse(1.0);
        CorpusCacheEntry built = new CorpusCacheEntry(docNgrams, docLengths, avgDocLength, vocabulary, df);

        if (useCache) {
            if (CORPUS_CACHE.size() >= MAX_CACHE_ENTRIES) {
                String firstKey = CORPUS_CACHE.keys().nextElement();
                CORPUS_CACHE.remove(firstKey);
            }
            CORPUS_CACHE.putIfAbsent(cacheKey, built);
            return CORPUS_CACHE.get(cacheKey);
        }

        return built;
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
            float v = denseRow[i];
            if (v != 0.0f) {
                indices[p] = i;
                values[p] = v;
                p++;
            }
        }
        return new SparseVector(indices, values);
    }

    private static String buildCacheKey(String[] documents,
                                        int ngramSize,
                                        int maxFeatures,
                                        NormalizationOptions options) {
        return Integer.toHexString(Arrays.hashCode(documents))
                + "|n=" + ngramSize
                + "|m=" + maxFeatures
                + "|u=" + options.normalizeUnicode
                + "|p=" + options.stripPunctuation
                + "|s=" + options.removeStopwords
                + "|sw=" + options.stopwords.hashCode();
    }

    private static List<String> generateNGrams(String[] tokens, int n) {
        List<String> ngrams = new ArrayList<>();
        if (tokens.length < n) return ngrams;
        for (int i = 0; i <= tokens.length - n; i++) {
            StringBuilder ngram = new StringBuilder();
            for (int j = 0; j < n; j++) {
                if (j > 0) ngram.append('_');
                ngram.append(tokens[i + j]);
            }
            ngrams.add(ngram.toString());
        }
        return ngrams;
    }

    private static double computeTfIdfWeight(int tfCount, int df, int totalDocs, WeightingScheme scheme) {
        double tf;
        if (scheme == WeightingScheme.SUBLINEAR_TF_IDF) {
            tf = Math.log1p(tfCount);
        } else {
            tf = tfCount;
        }
        double idf = Math.log((totalDocs + 1.0) / (df + 1.0)) + 1.0;
        return tf * idf;
    }

    private static double computeBm25Weight(int tfCount, int df, int totalDocs, int docLength, double avgDocLength) {
        double idf = Math.log(1.0 + (totalDocs - df + 0.5) / (df + 0.5));
        double numerator = tfCount * (BM25_K1 + 1.0);
        double denominator = tfCount + BM25_K1 * (1.0 - BM25_B + BM25_B * (docLength / Math.max(avgDocLength, 1e-9)));
        return idf * (numerator / denominator);
    }

    private static Set<String> defaultEnglishStopwords() {
        return new HashSet<>(Arrays.asList(
                "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "he",
                "in", "is", "it", "its", "of", "on", "that", "the", "to", "was", "were", "will", "with"
        ));
    }
}
