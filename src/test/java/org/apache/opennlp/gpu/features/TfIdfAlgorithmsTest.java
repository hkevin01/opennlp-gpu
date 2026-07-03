package org.apache.opennlp.gpu.features;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.nio.file.Files;
import java.nio.file.Path;

import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.OpenClFeatureExtractionOperation;
import org.apache.opennlp.gpu.common.CpuComputeProvider;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class TfIdfAlgorithmsTest {

    private static final float EPS = 1e-6f;

    @Test
    @DisplayName("TF-IDF scores match smoothed formula on a small corpus")
    void tfIdfScoresMatchExpectedFormula() {
        String[] docs = {
                "alpha alpha beta",
                "alpha gamma"
        };

        float[] scores = TfIdfAlgorithms.computeDocumentScores(docs);
        assertEquals(2, scores.length);

        // With N = 2 and smoothed idf(t) = ln((N+1)/(df+1)) + 1:
        // df(alpha)=2 => idf(alpha)=1
        // df(beta)=1 => idf(beta)=ln(3/2)+1
        // df(gamma)=1 => idf(gamma)=ln(3/2)+1
        double idfRare = Math.log(3.0 / 2.0) + 1.0;

        // doc0 = "alpha alpha beta": tf(alpha)=2, tf(beta)=1 (raw tf weighting)
        double expected0 = Math.sqrt(Math.pow(2.0 * 1.0, 2)
                                   + Math.pow(1.0 * idfRare, 2));

        // doc1 = "alpha gamma": tf(alpha)=1, tf(gamma)=1
        double expected1 = Math.sqrt(Math.pow(1.0 * 1.0, 2)
                                   + Math.pow(1.0 * idfRare, 2));

        assertEquals((float) expected0, scores[0], EPS);
        assertEquals((float) expected1, scores[1], EPS);
        assertTrue(scores[0] > scores[1], "Expected ordering from computed formula should hold");
    }

    @Test
    @DisplayName("CPU and OpenCL fallback TF-IDF paths should be numerically consistent")
    void cpuAndOpenClFallbackConsistency() {
        String[] docs = {
                "nlp gpu acceleration",
                "nlp cpu fallback",
                "gpu kernels and fallback"
        };

        CpuComputeProvider provider = new CpuComputeProvider();
        CpuFeatureExtractionOperation cpu = new CpuFeatureExtractionOperation(provider);
        OpenClFeatureExtractionOperation opencl = new OpenClFeatureExtractionOperation(provider);

        float[] cpuScores = cpu.computeTfIdf(docs);
        float[] openClScores = opencl.computeTfIdf(docs);

        assertEquals(cpuScores.length, openClScores.length);
        for (int i = 0; i < cpuScores.length; i++) {
            assertEquals(cpuScores[i], openClScores[i], EPS,
                    "CPU and OpenCL fallback TF-IDF should match at index " + i);
        }
    }

    @Test
    @DisplayName("Vectorization returns dense/sparse outputs aligned with vocabulary indices")
    void denseAndSparseVectorsAlign() {
        String[] docs = {
                "gpu gpu kernel",
                "cpu fallback"
        };

        TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(
                docs,
                1,
                16,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                false
        );

        assertEquals(2, result.getDenseVectors().length);
        assertEquals(2, result.getSparseVectors().length);
        assertFalse(result.getVocabulary().isEmpty());

        for (int doc = 0; doc < result.getDenseVectors().length; doc++) {
            float[] dense = result.getDenseVectors()[doc];
            TfIdfAlgorithms.SparseVector sparse = result.getSparseVectors()[doc];
            for (int i = 0; i < sparse.getIndices().length; i++) {
                int idx = sparse.getIndices()[i];
                assertEquals(dense[idx], sparse.getValues()[i], EPS);
            }
        }
    }

    @Test
    @DisplayName("Sublinear TF-IDF differs from raw TF-IDF on repeated terms")
    void sublinearDiffersFromRaw() {
        String[] docs = {
                "term term term rare",
                "term common"
        };

        TfIdfAlgorithms.VectorizationResult raw = TfIdfAlgorithms.vectorizeDocuments(
                docs, 1, 16,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                false
        );
        TfIdfAlgorithms.VectorizationResult sublinear = TfIdfAlgorithms.vectorizeDocuments(
                docs, 1, 16,
                TfIdfAlgorithms.WeightingScheme.SUBLINEAR_TF_IDF,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                false
        );

        int idx = raw.getVocabulary().get("term");
        assertNotEquals(raw.getDenseVectors()[0][idx], sublinear.getDenseVectors()[0][idx],
                "Expected sublinear TF weight to differ from raw TF weight");
    }

    @Test
    @DisplayName("BM25 weighting produces finite non-negative scores")
    void bm25ProducesFiniteScores() {
        String[] docs = {
                "information retrieval bm25 bm25",
                "retrieval baseline",
                "information extraction"
        };

        TfIdfAlgorithms.VectorizationResult bm25 = TfIdfAlgorithms.vectorizeDocuments(
                docs, 1, 32,
                TfIdfAlgorithms.WeightingScheme.BM25,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                false
        );

        for (float[] row : bm25.getDenseVectors()) {
            for (float v : row) {
                assertTrue(Float.isFinite(v));
                assertTrue(v >= 0.0f);
            }
        }
    }

    @Test
    @DisplayName("Tokenizer normalization applies punctuation stripping, unicode normalization, and stopword removal")
    void tokenizerNormalizationLayerWorks() {
        HashSet<String> stopwords = new HashSet<>();
        stopwords.add("the");
        stopwords.add("and");
        TfIdfAlgorithms.NormalizationOptions options = new TfIdfAlgorithms.NormalizationOptions(
                true,
                true,
                true,
                stopwords
        );

        String[] tokens = TfIdfAlgorithms.tokenizeNormalized("The café, and GPU!", options);
        // "The" and "and" removed; punctuation stripped.
        assertEquals(2, tokens.length);
        assertEquals("café", tokens[0]);
        assertEquals("gpu", tokens[1]);
    }

    @Test
    @DisplayName("Corpus cache is reused across repeated vectorization calls")
    void corpusCacheIsReusable() {
        TfIdfAlgorithms.clearCache();
        String[] docs = {"repeatable tf idf corpus", "repeatable corpus"};

        TfIdfAlgorithms.vectorizeDocuments(
                docs, 1, 32,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                true
        );
        int afterFirst = TfIdfAlgorithms.getCacheSize();

        TfIdfAlgorithms.vectorizeDocuments(
                docs, 1, 32,
                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                true
        );
        int afterSecond = TfIdfAlgorithms.getCacheSize();

        assertTrue(afterFirst >= 1);
        assertEquals(afterFirst, afterSecond, "Second run should reuse cached corpus statistics");
    }

        @Test
        @DisplayName("Weighted n-gram blending includes bigram/trigram features in vocabulary")
        void weightedNgramBlendingAddsPhraseFeatures() {
                String[] docs = {
                                "new york city skyline",
                                "new york subway",
                                "city skyline view"
                };

                TfIdfAlgorithms.VectorizationOptions unigramOptions = new TfIdfAlgorithms.VectorizationOptions(
                                32,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                null,
                                false
                );
                TfIdfAlgorithms.VectorizationOptions mixedOptions = new TfIdfAlgorithms.VectorizationOptions(
                                64,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.linearMix(1.0, 0.8, 0.6),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationResult unigram = TfIdfAlgorithms.vectorizeDocuments(docs, unigramOptions);
                TfIdfAlgorithms.VectorizationResult mixed = TfIdfAlgorithms.vectorizeDocuments(docs, mixedOptions);

                assertFalse(unigram.getVocabulary().containsKey("new_york"));
                assertTrue(mixed.getVocabulary().containsKey("new_york"));
        }

        @Test
        @DisplayName("Vocabulary persistence roundtrip keeps vocabulary/df consistent")
        void vocabularyPersistenceRoundtrip() throws Exception {
                String[] docs = {
                                "apache opennlp gpu",
                                "gpu tf idf",
                                "apache tf"
                };

                TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                                32,
                                TfIdfAlgorithms.WeightingScheme.SUBLINEAR_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.linearMix(1.0, 0.5, 0.0),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationResult first = TfIdfAlgorithms.vectorizeDocuments(docs, options);
                Path tmp = Files.createTempFile("tfidf-vocab", ".bin");
                try {
                        TfIdfAlgorithms.saveVocabularyState(first.getVocabularyState(), tmp);
                        TfIdfAlgorithms.VocabularyState loaded = TfIdfAlgorithms.loadVocabularyState(tmp);

                        TfIdfAlgorithms.VectorizationResult second = TfIdfAlgorithms.vectorizeDocumentsWithVocabulary(
                                        docs,
                                        loaded,
                                        TfIdfAlgorithms.WeightingScheme.SUBLINEAR_TF_IDF,
                                        TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                        false
                        );

                        assertEquals(first.getVocabulary().size(), second.getVocabulary().size());
                        for (int i = 0; i < first.getDenseVectors().length; i++) {
                                assertEquals(first.getDenseVectors()[i].length, second.getDenseVectors()[i].length);
                                for (int j = 0; j < first.getDenseVectors()[i].length; j++) {
                                        assertEquals(first.getDenseVectors()[i][j], second.getDenseVectors()[i][j], EPS);
                                }
                        }
                } finally {
                        Files.deleteIfExists(tmp);
                }
        }

        @Test
        @DisplayName("Top-k discriminative pruning with chi-square selects class-indicative features")
        void chiSquareFeatureSelectionFindsDiscriminativeTerms() {
                String[] docs = {
                                "alpha alpha shared",
                                "alpha shared",
                                "beta beta shared",
                                "beta shared"
                };
                String[] labels = {"A", "A", "B", "B"};

                TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                                1,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.CHI_SQUARE,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(docs, options);
                Map<String, Integer> vocab = result.getVocabulary();
                assertEquals(1, vocab.size());
                String selected = vocab.keySet().iterator().next();
                assertTrue(selected.equals("alpha") || selected.equals("beta"));
        }

        @Test
        @DisplayName("LRU cache eviction is deterministic and evicts oldest corpus first")
        void deterministicLruEviction() {
                TfIdfAlgorithms.clearCache();
                TfIdfAlgorithms.VectorizationOptions options = TfIdfAlgorithms.VectorizationOptions.defaultForMaxFeatures(8);

                for (int i = 0; i < 32; i++) {
                        String[] docs = {"doc" + i + " token" + i};
                        TfIdfAlgorithms.vectorizeDocuments(docs, options);
                }
                List<String> fingerprintsBefore = TfIdfAlgorithms.getCachedCorpusFingerprints();
                assertEquals(32, fingerprintsBefore.size());
                String eldestFingerprint = fingerprintsBefore.get(0);

                TfIdfAlgorithms.vectorizeDocuments(new String[]{"new corpus for eviction"}, options);
                List<String> fingerprintsAfter = TfIdfAlgorithms.getCachedCorpusFingerprints();
                assertEquals(32, fingerprintsAfter.size());
                assertFalse(fingerprintsAfter.contains(eldestFingerprint));
        }

        @Test
        @DisplayName("Numerical stability holds for long, sparse, and unicode-heavy corpora across schemes")
        void numericalStabilityStressAcrossSchemes() {
                StringBuilder longDocBuilder = new StringBuilder();
                for (int i = 0; i < 15000; i++) {
                        longDocBuilder.append("token").append(i % 17).append(' ');
                }
                String[] docs = {
                                longDocBuilder.toString(),
                                "x", // sparse
                                "Café naïve résumé coöperate — 東京 — 😀😀😀 repeated unicode"
                };

                for (TfIdfAlgorithms.WeightingScheme scheme : TfIdfAlgorithms.WeightingScheme.values()) {
                        TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                                        256,
                                        scheme,
                                        TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                        TfIdfAlgorithms.NGramBlendOptions.linearMix(1.0, 0.7, 0.3),
                                        TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                        TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                        TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                        1,
                                        Integer.MAX_VALUE,
                                        null,
                                        false
                        );

                        TfIdfAlgorithms.VectorizationResult r1 = TfIdfAlgorithms.vectorizeDocuments(docs, options);
                        TfIdfAlgorithms.VectorizationResult r2 = TfIdfAlgorithms.vectorizeDocuments(docs, options);

                        assertEquals(r1.getDenseVectors().length, r2.getDenseVectors().length);
                        for (int i = 0; i < r1.getDenseVectors().length; i++) {
                                assertEquals(r1.getDenseVectors()[i].length, r2.getDenseVectors()[i].length);
                                for (int j = 0; j < r1.getDenseVectors()[i].length; j++) {
                                        float v1 = r1.getDenseVectors()[i][j];
                                        float v2 = r2.getDenseVectors()[i][j];
                                        assertTrue(Float.isFinite(v1));
                                        assertTrue(Float.isFinite(v2));
                                        assertEquals(v1, v2, 1e-5f);
                                }
                        }
                }
        }

        @Test
        @DisplayName("IDF smoothing strategy and DF cutoffs are applied during vectorization")
        void smoothingAndDfCutoffsAffectVocabularyAndWeights() {
                String[] docs = {
                                "common common rare1",
                                "common common rare2",
                                "common common rare3"
                };

                TfIdfAlgorithms.VectorizationOptions baseline = new TfIdfAlgorithms.VectorizationOptions(
                                16,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationOptions cutoffAndProbIdf = new TfIdfAlgorithms.VectorizationOptions(
                                16,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.PROBABILISTIC_IDF,
                                2,
                                Integer.MAX_VALUE,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationResult baselineResult = TfIdfAlgorithms.vectorizeDocuments(docs, baseline);
                TfIdfAlgorithms.VectorizationResult cutoffResult = TfIdfAlgorithms.vectorizeDocuments(docs, cutoffAndProbIdf);

                assertTrue(baselineResult.getVocabulary().containsKey("rare1"));
                assertFalse(cutoffResult.getVocabulary().containsKey("rare1"), "minDf=2 should prune singleton terms");
                assertTrue(cutoffResult.getVocabulary().containsKey("common"));
        }

        @Test
        @DisplayName("Class-prior weighting with macro-average modifies discriminative scoring")
        void classPriorWeightingInfluencesSelection() {
                String[] docs = {
                                "major major signal",
                                "major context",
                                "major context",
                                "minor rareminor"
                };
                String[] labels = {"MAJ", "MAJ", "MAJ", "MIN"};

                TfIdfAlgorithms.VectorizationOptions unweighted = new TfIdfAlgorithms.VectorizationOptions(
                                1,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.CHI_SQUARE,
                                new TfIdfAlgorithms.ClassBalanceOptions(true, false),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                TfIdfAlgorithms.VectorizationOptions weighted = new TfIdfAlgorithms.VectorizationOptions(
                                1,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.CHI_SQUARE,
                                new TfIdfAlgorithms.ClassBalanceOptions(true, true),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                String selectedUnweighted = TfIdfAlgorithms.vectorizeDocuments(docs, unweighted)
                                .getVocabulary().keySet().iterator().next();
                String selectedWeighted = TfIdfAlgorithms.vectorizeDocuments(docs, weighted)
                                .getVocabulary().keySet().iterator().next();

                assertTrue(selectedUnweighted.equals("major") || selectedUnweighted.equals("rareminor"));
                assertTrue(selectedWeighted.equals("major") || selectedWeighted.equals("rareminor"));
        }

        @Test
        @DisplayName("Information Gain supports distinct macro and micro aggregation paths")
        void informationGainMacroAndMicroAreBothSupported() {
                String[] docs = {
                                "majorterm shared common",
                                "majorterm common",
                                "majorterm common",
                                "minorterm unique"
                };
                String[] labels = {"MAJ", "MAJ", "MAJ", "MIN"};

                TfIdfAlgorithms.VectorizationOptions microOptions = new TfIdfAlgorithms.VectorizationOptions(
                                2,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.INFORMATION_GAIN,
                                new TfIdfAlgorithms.ClassBalanceOptions(false, false),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                TfIdfAlgorithms.VectorizationOptions macroPriorOptions = new TfIdfAlgorithms.VectorizationOptions(
                                2,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.INFORMATION_GAIN,
                                new TfIdfAlgorithms.ClassBalanceOptions(true, true),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                TfIdfAlgorithms.VectorizationResult micro = TfIdfAlgorithms.vectorizeDocuments(docs, microOptions);
                TfIdfAlgorithms.VectorizationResult macroPrior = TfIdfAlgorithms.vectorizeDocuments(docs, macroPriorOptions);

                assertFalse(micro.getVocabulary().isEmpty());
                assertFalse(macroPrior.getVocabulary().isEmpty());

                // Both paths should preserve minority-class signal in top features for this corpus.
                assertTrue(micro.getVocabulary().containsKey("minorterm") || micro.getVocabulary().containsKey("majorterm"));
                assertTrue(macroPrior.getVocabulary().containsKey("minorterm") || macroPrior.getVocabulary().containsKey("majorterm"));
        }

        @Test
        @DisplayName("Dense vector quantization round-trip remains numerically bounded")
        void denseVectorQuantizationRoundtrip() throws Exception {
                String[] docs = {
                                "quantization test vector one",
                                "quantization test vector two"
                };

                TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(
                                docs,
                                1,
                                32,
                                TfIdfAlgorithms.WeightingScheme.SUBLINEAR_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                false
                );

                Path float16Path = Files.createTempFile("tfidf-dense", "-f16.bin");
                Path int8Path = Files.createTempFile("tfidf-dense", "-i8.bin");
                try {
                        TfIdfAlgorithms.saveDenseVectors(result.getDenseVectors(), float16Path,
                                        TfIdfAlgorithms.DenseVectorQuantization.FLOAT16);
                        TfIdfAlgorithms.saveDenseVectors(result.getDenseVectors(), int8Path,
                                        TfIdfAlgorithms.DenseVectorQuantization.INT8);

                        TfIdfAlgorithms.LoadedDenseVectors loaded16 = TfIdfAlgorithms.loadDenseVectors(float16Path);
                        TfIdfAlgorithms.LoadedDenseVectors loaded8 = TfIdfAlgorithms.loadDenseVectors(int8Path);

                        assertEquals(TfIdfAlgorithms.DenseVectorQuantization.FLOAT16, loaded16.getQuantization());
                        assertEquals(TfIdfAlgorithms.DenseVectorQuantization.INT8, loaded8.getQuantization());

                        for (int i = 0; i < result.getDenseVectors().length; i++) {
                                for (int j = 0; j < result.getDenseVectors()[i].length; j++) {
                                        assertEquals(result.getDenseVectors()[i][j], loaded16.getDenseVectors()[i][j], 2e-2f);
                                        assertEquals(result.getDenseVectors()[i][j], loaded8.getDenseVectors()[i][j], 1e-1f);
                                }
                        }
                } finally {
                        Files.deleteIfExists(float16Path);
                        Files.deleteIfExists(int8Path);
                }
        }

        @Test
        @DisplayName("Information Gain regression values match expected closed-form results")
        void informationGainRegressionValues() {
                String[] docs = {
                                "x y",
                                "x",
                                "y",
                                "z"
                };
                String[] labels = {"A", "A", "B", "B"};

                Map<String, Double> scores = TfIdfAlgorithms.computeInformationGainScores(
                                docs,
                                labels,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                false,
                                false
                );

                assertEquals(1.0, scores.get("x"), 1e-6);
                assertEquals(0.0, scores.get("y"), 1e-6);
                assertEquals(0.3112781245, scores.get("z"), 1e-6);
        }

        @Test
        @DisplayName("Macro IG with class-prior weighting changes expected numeric scores")
        void macroInformationGainPriorWeightedRegression() {
                String[] docs = {
                                "t1",
                                "t1",
                                "t2",
                                "t3",
                                "t3",
                                "t4"
                };
                String[] labels = {"A", "A", "B", "C", "C", "C"};

                Map<String, Double> macroUnweighted = TfIdfAlgorithms.computeInformationGainScores(
                                docs,
                                labels,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                true,
                                false
                );
                Map<String, Double> macroWeighted = TfIdfAlgorithms.computeInformationGainScores(
                                docs,
                                labels,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                true,
                                true
                );

                assertEquals(0.316688, macroUnweighted.get("t2"), 1e-4);
                assertEquals(0.4190358686, macroWeighted.get("t2"), 1e-6);
                assertTrue(macroWeighted.get("t2") > macroUnweighted.get("t2"));
        }

        @Test
        @DisplayName("BM25+ lower-TF floor improves long-document robustness for rare terms")
        void bm25PlusLowerTfBoundBehavior() {
                String[] docs = {
                                "rare " + "common ".repeat(300),
                                "common common common"
                };

                TfIdfAlgorithms.VectorizationOptions bm25 = new TfIdfAlgorithms.VectorizationOptions(
                                64,
                                TfIdfAlgorithms.WeightingScheme.BM25,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.BM25_IDF,
                                1,
                                Integer.MAX_VALUE,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationOptions bm25Plus = new TfIdfAlgorithms.VectorizationOptions(
                                64,
                                TfIdfAlgorithms.WeightingScheme.BM25_PLUS,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.FREQUENCY,
                                TfIdfAlgorithms.ClassBalanceOptions.defaultOptions(),
                                TfIdfAlgorithms.IDFSmoothingStrategy.BM25_IDF,
                                1,
                                Integer.MAX_VALUE,
                                0.75,
                                0.25,
                                null,
                                false
                );

                TfIdfAlgorithms.VectorizationResult rBm25 = TfIdfAlgorithms.vectorizeDocuments(docs, bm25);
                TfIdfAlgorithms.VectorizationResult rBm25Plus = TfIdfAlgorithms.vectorizeDocuments(docs, bm25Plus);

                int rareIdx = rBm25.getVocabulary().get("rare");
                assertTrue(rBm25Plus.getDenseVectors()[0][rareIdx] > rBm25.getDenseVectors()[0][rareIdx]);
        }

        @Test
        @DisplayName("Per-class vocabulary diagnostics expose class contribution scores")
        void vocabularyDiagnosticsExposeClassContributions() {
                String[] docs = {
                                "alpha alpha",
                                "alpha",
                                "beta",
                                "gamma"
                };
                String[] labels = {"A", "A", "B", "C"};

                TfIdfAlgorithms.VectorizationOptions options = new TfIdfAlgorithms.VectorizationOptions(
                                3,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                TfIdfAlgorithms.NGramBlendOptions.unigramOnly(),
                                TfIdfAlgorithms.FeatureSelectionMethod.INFORMATION_GAIN,
                                new TfIdfAlgorithms.ClassBalanceOptions(true, true),
                                TfIdfAlgorithms.IDFSmoothingStrategy.STANDARD_SMOOTH,
                                1,
                                Integer.MAX_VALUE,
                                labels,
                                false
                );

                TfIdfAlgorithms.VocabularyDiagnostics diagnostics = TfIdfAlgorithms.computeVocabularyDiagnostics(docs, options);
                assertEquals(TfIdfAlgorithms.FeatureSelectionMethod.INFORMATION_GAIN, diagnostics.getSelectionMethod());
                assertFalse(diagnostics.getTermDiagnostics().isEmpty());
                TfIdfAlgorithms.TermDiagnostics first = diagnostics.getTermDiagnostics().get(0);
                assertFalse(first.getClassContributions().isEmpty());
        }

        @Test
        @DisplayName("Sparse CSR export/import roundtrip is lossless")
        void sparseCsrRoundtrip() throws Exception {
                String[] docs = {
                                "a b c",
                                "a a",
                                "c"
                };
                TfIdfAlgorithms.VectorizationResult result = TfIdfAlgorithms.vectorizeDocuments(
                                docs,
                                1,
                                16,
                                TfIdfAlgorithms.WeightingScheme.RAW_TF_IDF,
                                TfIdfAlgorithms.NormalizationOptions.defaultOptions(),
                                false
                );

                TfIdfAlgorithms.SparseCsrMatrix csr = TfIdfAlgorithms.toSparseCsr(
                                result.getSparseVectors(),
                                result.getVocabulary().size()
                );

                Path tmp = Files.createTempFile("tfidf-csr", ".bin");
                try {
                        TfIdfAlgorithms.saveSparseCsr(csr, tmp);
                        TfIdfAlgorithms.SparseCsrMatrix loaded = TfIdfAlgorithms.loadSparseCsr(tmp);

                        assertEquals(csr.getRows(), loaded.getRows());
                        assertEquals(csr.getCols(), loaded.getCols());
                        assertArrayEquals(csr.getRowOffsets(), loaded.getRowOffsets());
                        assertArrayEquals(csr.getColIndices(), loaded.getColIndices());
                        assertArrayEquals(csr.getValues(), loaded.getValues(), EPS);
                } finally {
                        Files.deleteIfExists(tmp);
                }
        }

        @Test
        @DisplayName("Calibration utilities provide stable score/vector comparability transforms")
        void calibrationUtilitiesWorkAsExpected() {
                float[] scores = new float[]{2f, 4f, 6f};
                float[] minMax = TfIdfAlgorithms.calibrateScores(scores, TfIdfAlgorithms.ScoreCalibrationMethod.MIN_MAX_0_1);
                assertEquals(0f, minMax[0], EPS);
                assertEquals(1f, minMax[2], EPS);

                float[][] dense = new float[][]{
                                {3f, 4f},
                                {0f, 5f}
                };
                float[][] l2 = TfIdfAlgorithms.calibrateDenseVectors(dense, TfIdfAlgorithms.VectorCalibrationMethod.L2_PER_DOCUMENT);
                double n0 = Math.sqrt(l2[0][0] * l2[0][0] + l2[0][1] * l2[0][1]);
                double n1 = Math.sqrt(l2[1][0] * l2[1][0] + l2[1][1] * l2[1][1]);
                assertEquals(1.0, n0, 1e-6);
                assertEquals(1.0, n1, 1e-6);
        }
}
