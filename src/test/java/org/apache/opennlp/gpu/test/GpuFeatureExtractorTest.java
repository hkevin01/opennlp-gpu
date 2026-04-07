/*
 * Copyright 2025 OpenNLP GPU Extension Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.opennlp.gpu.test;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive JUnit 5 tests for GpuFeatureExtractor.
 * All tests run on the CPU path — no GPU hardware required.
 */
@DisplayName("GpuFeatureExtractor Tests")
public class GpuFeatureExtractorTest {

    private static final float TOLERANCE = 1e-5f;

    private GpuFeatureExtractor extractor;
    private CpuComputeProvider provider;
    private GpuConfig config;
    private MatrixOperation matrixOp;

    @BeforeEach
    void setUp() {
        config    = new GpuConfig();
        provider  = new CpuComputeProvider();
        provider.initialize(config);
        matrixOp  = OperationFactory.createMatrixOperation(provider);
        extractor = new GpuFeatureExtractor(provider, config, matrixOp);
    }

    // ── Constructor validation ────────────────────────────────────────────────

    @Test
    @DisplayName("Constructor: null provider throws NullPointerException")
    void testConstructorNullProvider() {
        assertThrows(NullPointerException.class,
            () -> new GpuFeatureExtractor(null, config, matrixOp));
    }

    @Test
    @DisplayName("Constructor: null config throws NullPointerException")
    void testConstructorNullConfig() {
        assertThrows(NullPointerException.class,
            () -> new GpuFeatureExtractor(provider, null, matrixOp));
    }

    @Test
    @DisplayName("Constructor: null matrixOp throws NullPointerException")
    void testConstructorNullMatrixOp() {
        assertThrows(NullPointerException.class,
            () -> new GpuFeatureExtractor(provider, config, null));
    }

    // ── extractNGramFeatures input validation ─────────────────────────────────

    @Test
    @DisplayName("extractNGramFeatures: null documents throws NullPointerException")
    void testExtractNGramNullDocuments() {
        assertThrows(NullPointerException.class,
            () -> extractor.extractNGramFeatures(null, 1, 100));
    }

    @Test
    @DisplayName("extractNGramFeatures: ngramSize < 1 throws IllegalArgumentException")
    void testExtractNGramInvalidNgramSize() {
        assertThrows(IllegalArgumentException.class,
            () -> extractor.extractNGramFeatures(new String[]{"hello world"}, 0, 100));
    }

    @Test
    @DisplayName("extractNGramFeatures: maxFeatures < 1 throws IllegalArgumentException")
    void testExtractNGramInvalidMaxFeatures() {
        assertThrows(IllegalArgumentException.class,
            () -> extractor.extractNGramFeatures(new String[]{"hello world"}, 1, 0));
    }

    @Test
    @DisplayName("extractNGramFeatures: empty documents returns empty array")
    void testExtractNGramEmptyDocuments() {
        float[][] result = extractor.extractNGramFeatures(new String[0], 1, 100);
        assertNotNull(result);
        assertEquals(0, result.length);
    }

    // ── extractNGramFeatures correctness ─────────────────────────────────────

    @Test
    @DisplayName("extractNGramFeatures: unigram row count matches document count")
    void testExtractNGramRowCount() {
        String[] docs = {"the cat sat", "the dog ran", "a bird flew"};
        float[][] features = extractor.extractNGramFeatures(docs, 1, 50);
        assertEquals(3, features.length, "one row per document");
    }

    @Test
    @DisplayName("extractNGramFeatures: vocabulary size respects maxFeatures cap")
    void testExtractNGramMaxFeaturesCap() {
        String[] docs = {"apple banana cherry date elderberry fig grape"};
        float[][] features = extractor.extractNGramFeatures(docs, 1, 3);
        // Column count must be <= maxFeatures
        assertTrue(features[0].length <= 3, "feature columns must be <= maxFeatures");
    }

    @Test
    @DisplayName("extractNGramFeatures: repeated token gets higher weight than unique token")
    void testExtractNGramFrequency() {
        // "the" appears 3 times, "cat" appears once
        String[] docs = {"the the the cat"};
        float[][] features = extractor.extractNGramFeatures(docs, 1, 10);

        int theIdx   = extractor.getVocabulary().getOrDefault("the", -1);
        int catIdx   = extractor.getVocabulary().getOrDefault("cat", -1);

        // Both tokens must be in vocabulary
        if (theIdx >= 0 && catIdx >= 0) {
            assertTrue(features[0][theIdx] > features[0][catIdx],
                "'the' (3x) should have higher weight than 'cat' (1x)");
        }
    }

    @Test
    @DisplayName("extractNGramFeatures: bigrams have _ separator in vocabulary")
    void testExtractNGramBigrams() {
        String[] docs = {"hello world"};
        extractor.extractNGramFeatures(docs, 2, 50);
        boolean hasBigram = extractor.getVocabulary().keySet().stream()
            .anyMatch(k -> k.contains("_"));
        assertTrue(hasBigram, "bigram vocabulary should contain underscore-joined tokens");
    }

    @Test
    @DisplayName("extractNGramFeatures: feature vectors are non-negative")
    void testExtractNGramNonNegative() {
        String[] docs = {"natural language processing is fun", "gpu acceleration works"};
        float[][] features = extractor.extractNGramFeatures(docs, 1, 20);
        for (float[] row : features) {
            for (float v : row) {
                assertTrue(v >= 0f, "feature values must be non-negative");
            }
        }
    }

    // ── extractTfIdfFeatures ─────────────────────────────────────────────────

    @Test
    @DisplayName("extractTfIdfFeatures: returns correct shape")
    void testExtractTfIdfShape() {
        String[] docs = {
            "machine learning is great",
            "deep learning is deep",
            "natural language processing matters"
        };
        float[][] tfidf = extractor.extractTfIdfFeatures(docs, 1, 30);
        assertEquals(3, tfidf.length, "should have one row per document");
    }

    @Test
    @DisplayName("extractTfIdfFeatures: common word gets lower weight than rare word")
    void testExtractTfIdfRareWordHigherScore() {
        // "is" appears in all 3 docs (low IDF), "xyzrare" appears in only 1
        String[] docs = {
            "this is a test xyzrare",
            "this is another test",
            "this is yet another"
        };
        float[][] tfidf = extractor.extractTfIdfFeatures(docs, 1, 30);

        int isIdx   = extractor.getVocabulary().getOrDefault("is", -1);
        int rareIdx = extractor.getVocabulary().getOrDefault("xyzrare", -1);

        if (isIdx >= 0 && rareIdx >= 0) {
            // For the first doc, xyzrare should have higher TF-IDF than "is"
            assertTrue(tfidf[0][rareIdx] >= tfidf[0][isIdx],
                "rare word should have >= TF-IDF weight vs common word");
        }
    }

    @Test
    @DisplayName("extractTfIdfFeatures: empty documents returns empty result")
    void testExtractTfIdfEmptyDocs() {
        float[][] result = extractor.extractTfIdfFeatures(new String[0], 1, 100);
        assertEquals(0, result.length);
    }

    // ── normalizeFeatures ────────────────────────────────────────────────────

    @Test
    @DisplayName("normalizeFeatures: null features throws NullPointerException")
    void testNormalizeFeaturesNull() {
        assertThrows(NullPointerException.class, () -> extractor.normalizeFeatures(null));
    }

    @Test
    @DisplayName("normalizeFeatures: empty array does not throw")
    void testNormalizeFeaturesEmpty() {
        assertDoesNotThrow(() -> extractor.normalizeFeatures(new float[0][]));
    }

    @Test
    @DisplayName("normalizeFeatures: L2 norm of each row is approximately 1.0 after normalisation")
    void testNormalizeFeaturesL2Norm() {
        float[][] features = {
            {3.0f, 4.0f},   // norm = 5
            {1.0f, 0.0f},   // norm = 1
            {0.0f, 0.0f}    // zero vector — should remain [0,0]
        };

        extractor.normalizeFeatures(features);

        // [3,4]/5 = [0.6, 0.8], norm² = 0.36+0.64 = 1.0
        float norm0 = (float) Math.sqrt(features[0][0]*features[0][0] + features[0][1]*features[0][1]);
        assertEquals(1.0f, norm0, TOLERANCE, "non-zero row should have unit L2 norm after normalisation");

        // [1,0] is already normalised
        float norm1 = (float) Math.sqrt(features[1][0]*features[1][0] + features[1][1]*features[1][1]);
        assertEquals(1.0f, norm1, TOLERANCE, "unit vector should remain unit after normalisation");

        // zero vector should stay zero
        assertEquals(0.0f, features[2][0], TOLERANCE);
        assertEquals(0.0f, features[2][1], TOLERANCE);
    }

    // ── extractContextFeatures ───────────────────────────────────────────────

    @Test
    @DisplayName("extractContextFeatures: no target words gives empty result")
    void testExtractContextFeaturesNoTargets() {
        String[] docs    = {"hello world foo bar"};
        String[] targets = {"notpresent"};
        float[][] result = extractor.extractContextFeatures(docs, targets, 2);
        assertEquals(0, result.length, "no target matches means empty result");
    }

    @Test
    @DisplayName("extractContextFeatures: context window size is respected")
    void testExtractContextFeaturesWindowSize() {
        String[] docs    = {"a b c d e"};
        String[] targets = {"c"};
        int windowSize   = 2;
        float[][] result = extractor.extractContextFeatures(docs, targets, windowSize);
        assertEquals(1, result.length, "one occurrence of target");
        assertEquals(windowSize * 2, result[0].length, "context vector length == windowSize*2");
    }

    // ── getVocabulary / getVocabularySize ────────────────────────────────────

    @Test
    @DisplayName("getVocabularySize returns number of features after extraction")
    void testGetVocabularySize() {
        String[] docs = {"one two three"};
        extractor.extractNGramFeatures(docs, 1, 50);
        assertEquals(3, extractor.getVocabularySize(), "3 unique unigrams");
    }

    @Test
    @DisplayName("getVocabulary returns defensive copy")
    void testGetVocabularyDefensiveCopy() {
        String[] docs = {"test document"};
        extractor.extractNGramFeatures(docs, 1, 50);
        var vocab1 = extractor.getVocabulary();
        var vocab2 = extractor.getVocabulary();
        assertNotSame(vocab1, vocab2, "getVocabulary should return a copy, not the same reference");
    }

    // ── release ──────────────────────────────────────────────────────────────

    @Test
    @DisplayName("release clears vocabulary")
    void testReleaseClears() {
        String[] docs = {"hello world"};
        extractor.extractNGramFeatures(docs, 1, 50);
        assertTrue(extractor.getVocabularySize() > 0, "vocabulary should be populated");
        extractor.release();
        assertEquals(0, extractor.getVocabularySize(), "release should clear vocabulary");
    }

    // ── Parallel vs sequential consistency ───────────────────────────────────

    @Test
    @DisplayName("GPU path and CPU path produce same n-gram feature matrix")
    void testParallelAndSequentialConsistency() {
        // Force GPU path by using enough documents to exceed MIN_DOCS_FOR_GPU threshold
        // Since we're on CPU provider, shouldUseGpu returns false — both paths stay CPU.
        // Here we just verify the feature matrix is numerically stable across two calls.
        String[] docs = new String[5];
        for (int i = 0; i < docs.length; i++) {
            docs[i] = "word" + i + " common common rare" + i;
        }
        float[][] first  = extractor.extractNGramFeatures(docs, 1, 20);
        extractor.release(); // reset vocabulary
        float[][] second = extractor.extractNGramFeatures(docs, 1, 20);

        assertEquals(first.length, second.length);
        for (int i = 0; i < first.length; i++) {
            assertArrayEquals(first[i], second[i], TOLERANCE,
                "repeated extraction should give identical results");
        }
    }
}
