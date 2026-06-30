package org.apache.opennlp.gpu.features;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.HashSet;

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
                "a a b",
                "a c"
        };

        float[] scores = TfIdfAlgorithms.computeDocumentScores(docs);
        assertEquals(2, scores.length);

        // With N = 2 and smoothed idf(t) = ln((N+1)/(df+1)) + 1:
        // df(a)=2 => idf(a)=1
        // df(b)=1 => idf(b)=ln(3/2)+1
        // df(c)=1 => idf(c)=ln(3/2)+1
        double idfRare = Math.log(3.0 / 2.0) + 1.0;

        // doc0 = "a a b": tf(a)=2/3, tf(b)=1/3
        double expected0 = Math.sqrt(Math.pow((2.0 / 3.0) * 1.0, 2)
                                   + Math.pow((1.0 / 3.0) * idfRare, 2));

        // doc1 = "a c": tf(a)=1/2, tf(c)=1/2
        double expected1 = Math.sqrt(Math.pow(0.5 * 1.0, 2)
                                   + Math.pow(0.5 * idfRare, 2));

        assertEquals((float) expected0, scores[0], EPS);
        assertEquals((float) expected1, scores[1], EPS);
        assertTrue(scores[1] > scores[0], "Expected ordering from computed formula should hold");
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
}
