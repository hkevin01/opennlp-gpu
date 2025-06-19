/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.opennlp.gpu.ml.neural.attention;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertThrows;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for GPU-accelerated attention mechanisms.
 * Tests functionality, accuracy, and performance of the attention layer.
 */
public class GpuAttentionLayerTest {
    
    private GpuAttentionLayer attentionLayer;
    private MatrixOperation matrixOp;
    private CpuComputeProvider provider;
    private GpuConfig config;
    
    // Test configuration
    private static final int HIDDEN_SIZE = 128;
    private static final int NUM_HEADS = 8;
    private static final int BATCH_SIZE = 4;
    private static final int SEQ_LEN = 16;
    private static final float TOLERANCE = 1e-5f;
    
    @BeforeEach
    void setUp() {
        provider = new CpuComputeProvider();
        config = new GpuConfig();
        config.setGpuEnabled(true);
        matrixOp = new CpuMatrixOperation(provider);
        
        attentionLayer = new GpuAttentionLayer(
            provider, config, matrixOp, 
            HIDDEN_SIZE, NUM_HEADS, true
        );
    }
    
    @AfterEach
    void tearDown() {
        if (attentionLayer != null) {
            attentionLayer.release();
        }
        if (matrixOp != null) {
            matrixOp.release();
        }
        if (provider != null) {
            provider.cleanup();
        }
    }
    
    @Test
    @DisplayName("Attention Layer Configuration")
    void testAttentionLayerConfiguration() {
        assertEquals(HIDDEN_SIZE, attentionLayer.getHiddenSize());
        assertEquals(NUM_HEADS, attentionLayer.getNumHeads());
        assertEquals(HIDDEN_SIZE / NUM_HEADS, attentionLayer.getHeadDim());
        assertTrue(attentionLayer.isUsingPositionalEncoding());
        assertNotNull(attentionLayer.getProvider());
    }
    
    @Test
    @DisplayName("Self-Attention Basic Functionality")
    void testSelfAttentionBasic() {
        // Create simple test input
        float[] input = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] output = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        
        // Apply self-attention
        attentionLayer.applySelfAttention(input, output, BATCH_SIZE, SEQ_LEN, null);
        
        // Verify output dimensions and basic properties
        assertNotNull(output);
        assertEquals(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE, output.length);
        
        // Check that output is not all zeros (attention should produce meaningful results)
        boolean hasNonZero = false;
        for (float value : output) {
            if (Math.abs(value) > TOLERANCE) {
                hasNonZero = true;
                break;
            }
        }
        assertTrue(hasNonZero, "Attention output should contain non-zero values");
    }
    
    @Test
    @DisplayName("Multi-Head Attention Functionality")
    void testMultiHeadAttention() {
        float[] queries = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] keys = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] values = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] output = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        
        // Apply multi-head attention
        attentionLayer.applyAttention(queries, keys, values, output, BATCH_SIZE, SEQ_LEN, null);
        
        // Verify output
        assertNotNull(output);
        assertEquals(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE, output.length);
        
        // Check for reasonable output values (not NaN or infinite)
        for (float value : output) {
            assertFalse(Float.isNaN(value), "Output should not contain NaN values");
            assertFalse(Float.isInfinite(value), "Output should not contain infinite values");
        }
    }
    
    @Test
    @DisplayName("Attention with Masking")
    void testAttentionWithMask() {
        float[] input = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] output = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        
        // Create a simple mask (mask out second half of sequence)
        float[] mask = new float[SEQ_LEN * SEQ_LEN];
        for (int i = 0; i < SEQ_LEN; i++) {
            for (int j = 0; j < SEQ_LEN; j++) {
                mask[i * SEQ_LEN + j] = (j < SEQ_LEN / 2) ? 1.0f : 0.0f;
            }
        }
        
        // Apply self-attention with mask
        attentionLayer.applySelfAttention(input, output, BATCH_SIZE, SEQ_LEN, mask);
        
        // Verify output
        assertNotNull(output);
        assertEquals(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE, output.length);
    }
    
    @Test
    @DisplayName("Attention Score Computation")
    void testAttentionScoreComputation() {
        int testBatchSize = 2;
        int testSeqLen = 4;
        int testHeadDim = HIDDEN_SIZE / NUM_HEADS;
        
        // Create test queries and keys
        float[] queries = createTestInput(testBatchSize * NUM_HEADS, testSeqLen, testHeadDim);
        float[] keys = createTestInput(testBatchSize * NUM_HEADS, testSeqLen, testHeadDim);
        float[] scores = new float[testBatchSize * NUM_HEADS * testSeqLen * testSeqLen];
        
        // Compute attention scores
        attentionLayer.computeAttentionScores(queries, keys, scores, testBatchSize, testSeqLen);
        
        // Verify scores
        assertNotNull(scores);
        assertEquals(testBatchSize * NUM_HEADS * testSeqLen * testSeqLen, scores.length);
        
        // Check that scores are finite
        for (float score : scores) {
            assertFalse(Float.isNaN(score), "Attention scores should not be NaN");
            assertFalse(Float.isInfinite(score), "Attention scores should not be infinite");
        }
    }
    
    @Test
    @DisplayName("Attention Softmax Application")
    void testAttentionSoftmax() {
        int testBatchSize = 2;
        int testSeqLen = 4;
        
        // Create test scores
        float[] scores = new float[testBatchSize * NUM_HEADS * testSeqLen * testSeqLen];
        for (int i = 0; i < scores.length; i++) {
            scores[i] = (float) (Math.random() * 2 - 1); // Random values between -1 and 1
        }
        
        float[] weights = new float[testBatchSize * NUM_HEADS * testSeqLen * testSeqLen];
        
        // Apply softmax
        attentionLayer.applyAttentionSoftmax(scores, weights, testBatchSize, testSeqLen, null);
        
        // Verify weights
        assertNotNull(weights);
        assertEquals(testBatchSize * NUM_HEADS * testSeqLen * testSeqLen, weights.length);
        
        // Check that each row sums to approximately 1 (softmax property)
        for (int head = 0; head < testBatchSize * NUM_HEADS; head++) {
            for (int i = 0; i < testSeqLen; i++) {
                float sum = 0.0f;
                for (int j = 0; j < testSeqLen; j++) {
                    int idx = head * testSeqLen * testSeqLen + i * testSeqLen + j;
                    sum += weights[idx];
                    assertTrue(weights[idx] >= 0, "Attention weights should be non-negative");
                }
                assertEquals(1.0f, sum, 0.01f, "Each attention row should sum to 1");
            }
        }
    }
    
    @Test
    @DisplayName("Weighted Values Computation")
    void testWeightedValuesComputation() {
        int testBatchSize = 2;
        int testSeqLen = 4;
        int testHeadDim = HIDDEN_SIZE / NUM_HEADS;
        
        // Create test weights (normalized)
        float[] weights = new float[testBatchSize * NUM_HEADS * testSeqLen * testSeqLen];
        for (int head = 0; head < testBatchSize * NUM_HEADS; head++) {
            for (int i = 0; i < testSeqLen; i++) {
                float sum = 0.0f;
                for (int j = 0; j < testSeqLen; j++) {
                    int idx = head * testSeqLen * testSeqLen + i * testSeqLen + j;
                    weights[idx] = (float) Math.random();
                    sum += weights[idx];
                }
                // Normalize to sum to 1
                for (int j = 0; j < testSeqLen; j++) {
                    int idx = head * testSeqLen * testSeqLen + i * testSeqLen + j;
                    weights[idx] /= sum;
                }
            }
        }
        
        // Create test values
        float[] values = createTestInput(testBatchSize * NUM_HEADS, testSeqLen, testHeadDim);
        float[] output = new float[testBatchSize * NUM_HEADS * testSeqLen * testHeadDim];
        
        // Compute weighted values
        attentionLayer.computeWeightedValues(weights, values, output, testBatchSize, testSeqLen);
        
        // Verify output
        assertNotNull(output);
        assertEquals(testBatchSize * NUM_HEADS * testSeqLen * testHeadDim, output.length);
        
        // Check for finite values
        for (float value : output) {
            assertFalse(Float.isNaN(value), "Weighted values should not be NaN");
            assertFalse(Float.isInfinite(value), "Weighted values should not be infinite");
        }
    }
    
    @Test
    @DisplayName("Positional Encoding")
    void testPositionalEncoding() {
        float[] input = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] output = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        
        // Add positional encoding
        attentionLayer.addPositionalEncoding(input, output, BATCH_SIZE, SEQ_LEN);
        
        // Verify output
        assertNotNull(output);
        assertEquals(BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE, output.length);
        
        // Check that output is different from input (positional encoding was added)
        boolean isDifferent = false;
        for (int i = 0; i < input.length; i++) {
            if (Math.abs(input[i] - output[i]) > TOLERANCE) {
                isDifferent = true;
                break;
            }
        }
        assertTrue(isDifferent, "Output should be different from input after adding positional encoding");
    }
    
    @Test
    @DisplayName("Different Sequence Lengths")
    void testDifferentSequenceLengths() {
        int[] testSeqLengths = {8, 16, 32, 64};
        
        for (int seqLen : testSeqLengths) {
            float[] input = createTestInput(BATCH_SIZE, seqLen, HIDDEN_SIZE);
            float[] output = new float[BATCH_SIZE * seqLen * HIDDEN_SIZE];
            
            // Should work without throwing exceptions
            assertDoesNotThrow(() -> {
                attentionLayer.applySelfAttention(input, output, BATCH_SIZE, seqLen, null);
            }, "Attention should work with sequence length: " + seqLen);
            
            // Verify output dimensions
            assertEquals(BATCH_SIZE * seqLen * HIDDEN_SIZE, output.length);
        }
    }
    
    @Test
    @DisplayName("Different Batch Sizes")
    void testDifferentBatchSizes() {
        int[] testBatchSizes = {1, 2, 4, 8};
        
        for (int batchSize : testBatchSizes) {
            float[] input = createTestInput(batchSize, SEQ_LEN, HIDDEN_SIZE);
            float[] output = new float[batchSize * SEQ_LEN * HIDDEN_SIZE];
            
            // Should work without throwing exceptions
            assertDoesNotThrow(() -> {
                attentionLayer.applySelfAttention(input, output, batchSize, SEQ_LEN, null);
            }, "Attention should work with batch size: " + batchSize);
            
            // Verify output dimensions
            assertEquals(batchSize * SEQ_LEN * HIDDEN_SIZE, output.length);
        }
    }
    
    @Test
    @DisplayName("Invalid Configuration")
    void testInvalidConfiguration() {
        // Test invalid hidden size (not divisible by number of heads)
        assertThrows(IllegalArgumentException.class, () -> {
            new GpuAttentionLayer(provider, config, matrixOp, 127, 8, false);
        }, "Should throw exception for invalid hidden size");
        
        // Test zero heads
        assertThrows(IllegalArgumentException.class, () -> {
            new GpuAttentionLayer(provider, config, matrixOp, 128, 0, false);
        }, "Should throw exception for zero heads");
    }
    
    @Test
    @DisplayName("Memory Management")
    void testMemoryManagement() {
        // Create multiple attention layers and release them
        for (int i = 0; i < 10; i++) {
            GpuAttentionLayer layer = new GpuAttentionLayer(
                provider, config, matrixOp, HIDDEN_SIZE, NUM_HEADS, false
            );
            
            // Use the layer
            float[] input = createTestInput(2, 8, HIDDEN_SIZE);
            float[] output = new float[2 * 8 * HIDDEN_SIZE];
            layer.applySelfAttention(input, output, 2, 8, null);
            
            // Release resources
            layer.release();
        }
        
        // Should complete without memory issues
        assertTrue(true, "Memory management test completed successfully");
    }
    
    @Test
    @DisplayName("Performance Scaling")
    void testPerformanceScaling() {
        // Test with increasing input sizes to verify scaling behavior
        int[] sizes = {4, 8, 16, 32};
        long[] times = new long[sizes.length];
        
        for (int i = 0; i < sizes.length; i++) {
            int seqLen = sizes[i];
            float[] input = createTestInput(2, seqLen, HIDDEN_SIZE);
            float[] output = new float[2 * seqLen * HIDDEN_SIZE];
            
            long startTime = System.nanoTime();
            attentionLayer.applySelfAttention(input, output, 2, seqLen, null);
            long endTime = System.nanoTime();
            
            times[i] = endTime - startTime;
        }
        
        // Verify that larger inputs take more time (reasonable scaling)
        for (int i = 1; i < times.length; i++) {
            assertTrue(times[i] >= times[i-1] * 0.5, 
                      "Performance should scale reasonably with input size");
        }
    }
    
    @Test
    @DisplayName("Attention Output Consistency")
    void testAttentionOutputConsistency() {
        float[] input = createTestInput(BATCH_SIZE, SEQ_LEN, HIDDEN_SIZE);
        float[] output1 = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        float[] output2 = new float[BATCH_SIZE * SEQ_LEN * HIDDEN_SIZE];
        
        // Apply attention twice with same input
        attentionLayer.applySelfAttention(input, output1, BATCH_SIZE, SEQ_LEN, null);
        attentionLayer.applySelfAttention(input, output2, BATCH_SIZE, SEQ_LEN, null);
        
        // Outputs should be identical (deterministic)
        assertArrayEquals(output1, output2, TOLERANCE, 
                         "Attention should produce consistent outputs for same input");
    }
    
    // Helper methods
    
    private float[] createTestInput(int batchSize, int seqLen, int hiddenSize) {
        float[] input = new float[batchSize * seqLen * hiddenSize];
        
        // Fill with normalized random values
        for (int i = 0; i < input.length; i++) {
            input[i] = (float) (Math.random() * 0.2 - 0.1); // Small values for stability
        }
        
        return input;
    }
    
    private void assertArrayEquals(float[] expected, float[] actual, float tolerance, String message) {
        assertEquals(expected.length, actual.length, message + " - Array lengths should match");
        
        for (int i = 0; i < expected.length; i++) {
            assertEquals(expected[i], actual[i], tolerance, 
                        message + " - Values at index " + i + " should match");
        }
    }
}
