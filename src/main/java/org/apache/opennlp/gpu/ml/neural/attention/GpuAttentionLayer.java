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

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.MatrixOperation;

/**
 * GPU-accelerated attention mechanism implementation for neural networks.
 * Implements scaled dot-product attention and multi-head attention patterns
 * commonly used in transformer architectures.
 * 
 * Features:
 * - Multi-head attention with configurable heads
 * - Scaled dot-product attention mechanism
 * - Positional encoding support
 * - Dynamic memory management
 * - CPU fallback for smaller sequences
 * 
 * @author OpenNLP GPU Team
 * @since 2.0.0
 */
public class GpuAttentionLayer {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuAttentionLayer.class);
    
    private final ComputeProvider provider;
    private final MatrixOperation matrixOp;
    private final GpuConfig config;
    
    // Attention configuration
    private final int hiddenSize;
    private final int numHeads;
    private final int headDim;
    private final float scaleFactor;
    private final boolean usePositionalEncoding;
    
    // Model parameters
    private float[] queryWeights;
    private float[] keyWeights;
    private float[] valueWeights;
    private float[] outputWeights;
    private float[] positionalEncodings;
    
    // Working memory buffers
    private float[] queryBuffer;
    private float[] keyBuffer;
    private float[] valueBuffer;
    private float[] attentionScores;
    private float[] attentionWeights;
    private float[] contextBuffer;
    
    // Performance thresholds
    private static final int MIN_SEQUENCE_LENGTH_FOR_GPU = 32;
    private static final int MIN_HIDDEN_SIZE_FOR_GPU = 128;
    private static final int MIN_BATCH_SIZE_FOR_GPU = 4;
    
    /**
     * Creates a new GPU attention layer.
     * 
     * @param provider the compute provider
     * @param config the GPU configuration
     * @param matrixOp the matrix operations implementation
     * @param hiddenSize the hidden dimension size
     * @param numHeads the number of attention heads
     * @param usePositionalEncoding whether to use positional encoding
     */
    public GpuAttentionLayer(ComputeProvider provider, GpuConfig config, MatrixOperation matrixOp,
                            int hiddenSize, int numHeads, boolean usePositionalEncoding) {
        this.provider = provider;
        this.config = config;
        this.matrixOp = matrixOp;
        this.hiddenSize = hiddenSize;
        this.numHeads = numHeads;
        this.usePositionalEncoding = usePositionalEncoding;
        
        // Validate configuration
        if (hiddenSize % numHeads != 0) {
            throw new IllegalArgumentException("Hidden size must be divisible by number of heads");
        }
        
        this.headDim = hiddenSize / numHeads;
        this.scaleFactor = 1.0f / (float) Math.sqrt(headDim);
        
        logger.info("Initialized GPU attention layer: hiddenSize={}, numHeads={}, headDim={}", 
                   hiddenSize, numHeads, headDim);
        
        initializeParameters();
        allocateBuffers();
    }
    
    /**
     * Apply multi-head attention to input sequences.
     * 
     * @param queries the query sequences [batchSize, seqLen, hiddenSize]
     * @param keys the key sequences [batchSize, seqLen, hiddenSize]
     * @param values the value sequences [batchSize, seqLen, hiddenSize]
     * @param output the output buffer [batchSize, seqLen, hiddenSize]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     * @param mask optional attention mask (null if not used)
     */
    public void applyAttention(float[] queries, float[] keys, float[] values, float[] output,
                              int batchSize, int seqLen, float[] mask) {
        
        logger.debug("Applying attention: batchSize={}, seqLen={}, hiddenSize={}", 
                    batchSize, seqLen, hiddenSize);
        
        if (shouldUseGpu(batchSize, seqLen)) {
            applyAttentionGpu(queries, keys, values, output, batchSize, seqLen, mask);
        } else {
            applyAttentionCpu(queries, keys, values, output, batchSize, seqLen, mask);
        }
    }
    
    /**
     * Apply self-attention where queries, keys, and values are the same.
     * 
     * @param input the input sequences [batchSize, seqLen, hiddenSize]
     * @param output the output buffer [batchSize, seqLen, hiddenSize]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     * @param mask optional attention mask (null if not used)
     */
    public void applySelfAttention(float[] input, float[] output, int batchSize, int seqLen, float[] mask) {
        logger.debug("Applying self-attention: batchSize={}, seqLen={}", batchSize, seqLen);
        applyAttention(input, input, input, output, batchSize, seqLen, mask);
    }
    
    /**
     * Compute attention scores between queries and keys.
     * 
     * @param queries the query matrix [batchSize * numHeads, seqLen, headDim]
     * @param keys the key matrix [batchSize * numHeads, seqLen, headDim]
     * @param scores the output scores [batchSize * numHeads, seqLen, seqLen]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     */
    public void computeAttentionScores(float[] queries, float[] keys, float[] scores,
                                      int batchSize, int seqLen) {
        
        int totalHeads = batchSize * numHeads;
        
        // Compute Q * K^T for each head
        for (int head = 0; head < totalHeads; head++) {
            int queryOffset = head * seqLen * headDim;
            int keyOffset = head * seqLen * headDim;
            int scoreOffset = head * seqLen * seqLen;
            
            // Matrix multiplication: queries * keys^T
            computeScaledDotProduct(
                queries, queryOffset, 
                keys, keyOffset,
                scores, scoreOffset,
                seqLen, headDim
            );
        }
    }
    
    /**
     * Apply softmax to attention scores to get attention weights.
     * 
     * @param scores the attention scores [batchSize * numHeads, seqLen, seqLen]
     * @param weights the output weights [batchSize * numHeads, seqLen, seqLen]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     * @param mask optional attention mask
     */
    public void applyAttentionSoftmax(float[] scores, float[] weights, int batchSize, int seqLen, float[] mask) {
        int totalHeads = batchSize * numHeads;
        
        for (int head = 0; head < totalHeads; head++) {
            for (int i = 0; i < seqLen; i++) {
                int rowOffset = head * seqLen * seqLen + i * seqLen;
                
                // Apply mask if provided
                if (mask != null) {
                    applyMask(scores, rowOffset, mask, i * seqLen, seqLen);
                }
                
                // Apply softmax to each row
                float[] rowInput = new float[seqLen];
                float[] rowOutput = new float[seqLen];
                System.arraycopy(scores, rowOffset, rowInput, 0, seqLen);
                matrixOp.softmax(rowInput, rowOutput, seqLen);
                System.arraycopy(rowOutput, 0, weights, rowOffset, seqLen);
            }
        }
    }
    
    /**
     * Compute weighted sum of values using attention weights.
     * 
     * @param weights the attention weights [batchSize * numHeads, seqLen, seqLen]
     * @param values the value matrix [batchSize * numHeads, seqLen, headDim]
     * @param output the output matrix [batchSize * numHeads, seqLen, headDim]
     * @param batchSize the batch size
     * @param seqLen the sequence length
     */
    public void computeWeightedValues(float[] weights, float[] values, float[] output,
                                     int batchSize, int seqLen) {
        
        int totalHeads = batchSize * numHeads;
        
        for (int head = 0; head < totalHeads; head++) {
            int weightOffset = head * seqLen * seqLen;
            int valueOffset = head * seqLen * headDim;
            int outputOffset = head * seqLen * headDim;
            
            // Matrix multiplication: weights * values
            float[] weightsMatrix = new float[seqLen * seqLen];
            float[] valuesMatrix = new float[seqLen * headDim];
            float[] outputMatrix = new float[seqLen * headDim];
            
            System.arraycopy(weights, weightOffset, weightsMatrix, 0, seqLen * seqLen);
            System.arraycopy(values, valueOffset, valuesMatrix, 0, seqLen * headDim);
            
            matrixOp.multiply(weightsMatrix, valuesMatrix, outputMatrix, seqLen, headDim, seqLen);
            
            System.arraycopy(outputMatrix, 0, output, outputOffset, seqLen * headDim);
        }
    }
    
    /**
     * Add positional encodings to input embeddings.
     * 
     * @param input the input embeddings [batchSize, seqLen, hiddenSize]
     * @param output the output with positional encodings
     * @param batchSize the batch size
     * @param seqLen the sequence length
     */
    public void addPositionalEncoding(float[] input, float[] output, int batchSize, int seqLen) {
        if (!usePositionalEncoding) {
            System.arraycopy(input, 0, output, 0, input.length);
            return;
        }
        
        logger.debug("Adding positional encoding: seqLen={}", seqLen);
        
        for (int batch = 0; batch < batchSize; batch++) {
            for (int pos = 0; pos < seqLen; pos++) {
                for (int dim = 0; dim < hiddenSize; dim++) {
                    int inputIdx = batch * seqLen * hiddenSize + pos * hiddenSize + dim;
                    int posIdx = pos * hiddenSize + dim;
                    output[inputIdx] = input[inputIdx] + positionalEncodings[posIdx];
                }
            }
        }
    }
    
    // Private implementation methods
    
    private void initializeParameters() {
        logger.debug("Initializing attention parameters");
        
        // Initialize weight matrices with Xavier initialization
        int weightsSize = hiddenSize * hiddenSize;
        queryWeights = new float[weightsSize];
        keyWeights = new float[weightsSize];
        valueWeights = new float[weightsSize];
        outputWeights = new float[weightsSize];
        
        initializeWeights(queryWeights, hiddenSize, hiddenSize);
        initializeWeights(keyWeights, hiddenSize, hiddenSize);
        initializeWeights(valueWeights, hiddenSize, hiddenSize);
        initializeWeights(outputWeights, hiddenSize, hiddenSize);
        
        // Initialize positional encodings if enabled
        if (usePositionalEncoding) {
            initializePositionalEncodings();
        }
    }
    
    private void allocateBuffers() {
        logger.debug("Allocating attention buffers");
        
        // Allocate working memory (assume maximum reasonable sizes)
        int maxSeqLen = 512;
        int maxBatchSize = 32;
        
        queryBuffer = new float[maxBatchSize * maxSeqLen * hiddenSize];
        keyBuffer = new float[maxBatchSize * maxSeqLen * hiddenSize];
        valueBuffer = new float[maxBatchSize * maxSeqLen * hiddenSize];
        attentionScores = new float[maxBatchSize * numHeads * maxSeqLen * maxSeqLen];
        attentionWeights = new float[maxBatchSize * numHeads * maxSeqLen * maxSeqLen];
        contextBuffer = new float[maxBatchSize * numHeads * maxSeqLen * headDim];
    }
    
    private void initializeWeights(float[] weights, int rows, int cols) {
        float scale = (float) Math.sqrt(2.0 / (rows + cols)); // Xavier initialization
        
        for (int i = 0; i < weights.length; i++) {
            weights[i] = (float) (Math.random() * 2 * scale - scale);
        }
    }
    
    private void initializePositionalEncodings() {
        int maxSeqLen = 512; // Maximum sequence length
        positionalEncodings = new float[maxSeqLen * hiddenSize];
        
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int dim = 0; dim < hiddenSize; dim++) {
                int idx = pos * hiddenSize + dim;
                
                if (dim % 2 == 0) {
                    // Even dimensions: sin
                    float angle = (float) (pos / Math.pow(10000.0, 2.0 * dim / hiddenSize));
                    positionalEncodings[idx] = (float) Math.sin(angle);
                } else {
                    // Odd dimensions: cos
                    float angle = (float) (pos / Math.pow(10000.0, 2.0 * (dim - 1) / hiddenSize));
                    positionalEncodings[idx] = (float) Math.cos(angle);
                }
            }
        }
    }
    
    private boolean shouldUseGpu(int batchSize, int seqLen) {
        return provider.isGpuProvider() && 
               config.isGpuEnabled() &&
               batchSize >= MIN_BATCH_SIZE_FOR_GPU &&
               seqLen >= MIN_SEQUENCE_LENGTH_FOR_GPU &&
               hiddenSize >= MIN_HIDDEN_SIZE_FOR_GPU;
    }
    
    private void applyAttentionGpu(float[] queries, float[] keys, float[] values, float[] output,
                                  int batchSize, int seqLen, float[] mask) {
        logger.debug("Using GPU for attention computation");
        
        try {
            // Transform inputs to query, key, value matrices
            transformInputs(queries, keys, values, batchSize, seqLen);
            
            // Compute attention scores
            computeAttentionScores(queryBuffer, keyBuffer, attentionScores, batchSize, seqLen);
            
            // Apply softmax to get attention weights
            applyAttentionSoftmax(attentionScores, attentionWeights, batchSize, seqLen, mask);
            
            // Compute weighted values
            computeWeightedValues(attentionWeights, valueBuffer, contextBuffer, batchSize, seqLen);
            
            // Apply output projection
            projectOutput(contextBuffer, output, batchSize, seqLen);
            
        } catch (Exception e) {
            logger.warn("GPU attention failed, falling back to CPU: " + e.getMessage());
            applyAttentionCpu(queries, keys, values, output, batchSize, seqLen, mask);
        }
    }
    
    private void applyAttentionCpu(float[] queries, float[] keys, float[] values, float[] output,
                                  int batchSize, int seqLen, float[] mask) {
        logger.debug("Using CPU for attention computation");
        
        // Simple CPU implementation - single-head for simplicity
        for (int batch = 0; batch < batchSize; batch++) {
            for (int i = 0; i < seqLen; i++) {
                for (int dim = 0; dim < hiddenSize; dim++) {
                    float sum = 0.0f;
                    float weightSum = 0.0f;
                    
                    // Compute attention for this position
                    for (int j = 0; j < seqLen; j++) {
                        float score = computeSimpleAttentionScore(queries, keys, batch, i, j, seqLen);
                        
                        if (mask == null || mask[i * seqLen + j] > 0) {
                            float weight = (float) Math.exp(score);
                            sum += weight * getValue(values, batch, j, dim, seqLen);
                            weightSum += weight;
                        }
                    }
                    
                    // Normalize and store result
                    int outputIdx = batch * seqLen * hiddenSize + i * hiddenSize + dim;
                    output[outputIdx] = weightSum > 0 ? sum / weightSum : 0.0f;
                }
            }
        }
    }
    
    private void transformInputs(float[] queries, float[] keys, float[] values, int batchSize, int seqLen) {
        // Apply linear transformations: Q = XW_Q, K = XW_K, V = XW_V
        
        // Transform queries
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] inputRow = new float[hiddenSize];
            float[] outputRow = new float[hiddenSize];
            System.arraycopy(queries, i * hiddenSize, inputRow, 0, hiddenSize);
            
            // Simple matrix-vector multiplication
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    sum += inputRow[k] * queryWeights[k * hiddenSize + j];
                }
                outputRow[j] = sum;
            }
            
            System.arraycopy(outputRow, 0, queryBuffer, i * hiddenSize, hiddenSize);
        }
        
        // Transform keys
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] inputRow = new float[hiddenSize];
            float[] outputRow = new float[hiddenSize];
            System.arraycopy(keys, i * hiddenSize, inputRow, 0, hiddenSize);
            
            // Simple matrix-vector multiplication
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    sum += inputRow[k] * keyWeights[k * hiddenSize + j];
                }
                outputRow[j] = sum;
            }
            
            System.arraycopy(outputRow, 0, keyBuffer, i * hiddenSize, hiddenSize);
        }
        
        // Transform values
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] inputRow = new float[hiddenSize];
            float[] outputRow = new float[hiddenSize];
            System.arraycopy(values, i * hiddenSize, inputRow, 0, hiddenSize);
            
            // Simple matrix-vector multiplication
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    sum += inputRow[k] * valueWeights[k * hiddenSize + j];
                }
                outputRow[j] = sum;
            }
            
            System.arraycopy(outputRow, 0, valueBuffer, i * hiddenSize, hiddenSize);
        }
        
        // Reshape for multi-head attention
        reshapeForMultiHead(queryBuffer, batchSize, seqLen);
        reshapeForMultiHead(keyBuffer, batchSize, seqLen);
        reshapeForMultiHead(valueBuffer, batchSize, seqLen);
    }
    
    private void reshapeForMultiHead(float[] matrix, int batchSize, int seqLen) {
        // Reshape from [batchSize, seqLen, hiddenSize] to [batchSize * numHeads, seqLen, headDim]
        // This is done in-place for efficiency
        
        float[] temp = new float[matrix.length];
        System.arraycopy(matrix, 0, temp, 0, matrix.length);
        
        for (int batch = 0; batch < batchSize; batch++) {
            for (int head = 0; head < numHeads; head++) {
                for (int seq = 0; seq < seqLen; seq++) {
                    for (int dim = 0; dim < headDim; dim++) {
                        int srcIdx = batch * seqLen * hiddenSize + seq * hiddenSize + head * headDim + dim;
                        int dstIdx = (batch * numHeads + head) * seqLen * headDim + seq * headDim + dim;
                        matrix[dstIdx] = temp[srcIdx];
                    }
                }
            }
        }
    }
    
    private void computeScaledDotProduct(float[] queries, int queryOffset, 
                                       float[] keys, int keyOffset,
                                       float[] scores, int scoreOffset,
                                       int seqLen, int headDim) {
        
        // Compute Q * K^T and scale
        for (int i = 0; i < seqLen; i++) {
            for (int j = 0; j < seqLen; j++) {
                float dotProduct = 0.0f;
                
                for (int d = 0; d < headDim; d++) {
                    dotProduct += queries[queryOffset + i * headDim + d] * 
                                 keys[keyOffset + j * headDim + d];
                }
                
                scores[scoreOffset + i * seqLen + j] = dotProduct * scaleFactor;
            }
        }
    }
    
    private void applyMask(float[] scores, int scoreOffset, float[] mask, int maskOffset, int seqLen) {
        for (int i = 0; i < seqLen; i++) {
            if (mask[maskOffset + i] == 0) {
                scores[scoreOffset + i] = Float.NEGATIVE_INFINITY;
            }
        }
    }
    
    private void projectOutput(float[] context, float[] output, int batchSize, int seqLen) {
        // Reshape context back from multi-head format
        float[] reshaped = new float[batchSize * seqLen * hiddenSize];
        
        for (int batch = 0; batch < batchSize; batch++) {
            for (int head = 0; head < numHeads; head++) {
                for (int seq = 0; seq < seqLen; seq++) {
                    for (int dim = 0; dim < headDim; dim++) {
                        int srcIdx = (batch * numHeads + head) * seqLen * headDim + seq * headDim + dim;
                        int dstIdx = batch * seqLen * hiddenSize + seq * hiddenSize + head * headDim + dim;
                        reshaped[dstIdx] = context[srcIdx];
                    }
                }
            }
        }
        
        // Apply output projection
        for (int i = 0; i < batchSize * seqLen; i++) {
            float[] inputRow = new float[hiddenSize];
            float[] outputRow = new float[hiddenSize];
            System.arraycopy(reshaped, i * hiddenSize, inputRow, 0, hiddenSize);
            
            // Simple matrix-vector multiplication
            for (int j = 0; j < hiddenSize; j++) {
                float sum = 0.0f;
                for (int k = 0; k < hiddenSize; k++) {
                    sum += inputRow[k] * outputWeights[k * hiddenSize + j];
                }
                outputRow[j] = sum;
            }
            
            System.arraycopy(outputRow, 0, output, i * hiddenSize, hiddenSize);
        }
    }
    
    private float computeSimpleAttentionScore(float[] queries, float[] keys, 
                                            int batch, int i, int j, int seqLen) {
        float score = 0.0f;
        
        for (int dim = 0; dim < hiddenSize; dim++) {
            int queryIdx = batch * seqLen * hiddenSize + i * hiddenSize + dim;
            int keyIdx = batch * seqLen * hiddenSize + j * hiddenSize + dim;
            score += queries[queryIdx] * keys[keyIdx];
        }
        
        return score * scaleFactor;
    }
    
    private float getValue(float[] values, int batch, int pos, int dim, int seqLen) {
        int idx = batch * seqLen * hiddenSize + pos * hiddenSize + dim;
        return values[idx];
    }
    
    /**
     * Release resources used by the attention layer.
     */
    public void release() {
        logger.debug("Releasing attention layer resources");
        
        queryWeights = null;
        keyWeights = null;
        valueWeights = null;
        outputWeights = null;
        positionalEncodings = null;
        
        queryBuffer = null;
        keyBuffer = null;
        valueBuffer = null;
        attentionScores = null;
        attentionWeights = null;
        contextBuffer = null;
    }
    
    // Getters for testing and introspection
    
    public int getHiddenSize() { return hiddenSize; }
    public int getNumHeads() { return numHeads; }
    public int getHeadDim() { return headDim; }
    public boolean isUsingPositionalEncoding() { return usePositionalEncoding; }
    public ComputeProvider getProvider() { return provider; }
}
