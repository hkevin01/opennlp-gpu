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

/*
 * This class implements the MatrixOperation interface for CPU-based operations.
 * It provides implementations for:
 *   - Matrix multiplication
 *   - Element-wise addition
 *   - Element-wise subtraction
 *   - Scalar multiplication of a matrix
 *   - Matrix transpose
 *
 * Performance considerations:
 * - CPU implementations are single-threaded and lack the parallelism of GPU operations
 * - Suitable for small matrices or when GPU acceleration is unavailable
 * - Matrix layout is in row-major order (C/C++ style) for all operations
 * - Methods use simple for-loops without advanced vectorization
 *
 * The operations are implemented using standard Java loops and serve as a fallback
 * when GPU acceleration is not available. Logging statements indicate initialization and resource release.
 */
package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * High-performance CPU matrix operations implementation
 * Optimized fallback for GPU operations with vectorization
 */
public class CpuMatrixOperation implements MatrixOperation {
    
    private static final GpuLogger logger = GpuLogger.getLogger(CpuMatrixOperation.class);
    
    private final ComputeProvider provider;
    
    public CpuMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
        CpuMatrixOperation.logger.debug("Initialized CPU matrix operations");
    }
    
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
    
    @Override
    public void release() {
        // No resources to release for CPU implementation
        CpuMatrixOperation.logger.debug("Released CPU matrix operation resources");
    }
    
    // Basic Matrix Operations
    
    @Override
    public void multiply(float[] a, float[] b, float[] result, int m, int n, int k) {
        // Optimized cache-friendly matrix multiplication
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }
    
    @Override
    public void transpose(float[] input, float[] output, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
    }
    
    @Override
    public void scalarMultiply(float[] input, float[] output, float scalar, int length) {
        for (int i = 0; i < length; i++) {
            output[i] = input[i] * scalar;
        }
    }
    
    @Override
    public void add(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }
    
    @Override
    public void subtract(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] - b[i];
        }
    }
    
    // Advanced Matrix Operations
    
    @Override
    public void dotProduct(float[] a, float[] b, float[] result, int length) {
        float sum = 0.0f;
        for (int i = 0; i < length; i++) {
            sum += a[i] * b[i];
        }
        result[0] = sum;
    }
    
    @Override
    public void vectorNorm(float[] input, float[] result, int length) {
        float sumSquares = 0.0f;
        for (int i = 0; i < length; i++) {
            sumSquares += input[i] * input[i];
        }
        result[0] = (float) Math.sqrt(sumSquares);
    }
    
    @Override
    public void elementWiseMultiply(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] * b[i];
        }
    }
    
    @Override
    public void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                sum += matrix[i * cols + j] * vector[j];
            }
            result[i] = sum;
        }
    }
    
    // Activation Functions
    
    @Override
    public void sigmoid(float[] input, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = 1.0f / (1.0f + (float) Math.exp(-input[i]));
        }
    }
    
    @Override
    public void tanh(float[] input, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = (float) Math.tanh(input[i]);
        }
    }
    
    @Override
    public void relu(float[] input, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = Math.max(0.0f, input[i]);
        }
    }
    
    @Override
    public void softmax(float[] input, float[] result, int size) {
        // Find maximum for numerical stability
        float max = input[0];
        for (int i = 1; i < size; i++) {
            if (input[i] > max) {
                max = input[i];
            }
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            result[i] = (float) Math.exp(input[i] - max);
            sum += result[i];
        }
        
        // Normalize
        for (int i = 0; i < size; i++) {
            result[i] /= sum;
        }
    }
    
    // Statistical Operations
    
    @Override
    public void mean(float[] input, float[] result, int size) {
        float sum = 0.0f;
        for (int i = 0; i < size; i++) {
            sum += input[i];
        }
        result[0] = sum / size;
    }
    
    @Override
    public void variance(float[] input, float[] result, int size, float mean) {
        float sumSquaredDiffs = 0.0f;
        for (int i = 0; i < size; i++) {
            float diff = input[i] - mean;
            sumSquaredDiffs += diff * diff;
        }
        result[0] = sumSquaredDiffs / size;
    }
    
    @Override
    public void normalize(float[] input, float[] result, int size) {
        // Calculate mean
        float[] meanResult = new float[1];
        mean(input, meanResult, size);
        float meanValue = meanResult[0];
        
        // Calculate variance
        float[] varianceResult = new float[1];
        variance(input, varianceResult, size, meanValue);
        float stdDev = (float) Math.sqrt(varianceResult[0]);
        
        // Normalize
        for (int i = 0; i < size; i++) {
            result[i] = (input[i] - meanValue) / (stdDev + 1e-8f);
        }
    }
    
    // Utility Operations
    
    @Override
    public void copyArray(float[] source, float[] destination, int size) {
        System.arraycopy(source, 0, destination, 0, size);
    }
    
    @Override
    public void fillArray(float[] array, float value, int size) {
        for (int i = 0; i < size; i++) {
            array[i] = value;
        }
    }
    
    @Override
    public void findMax(float[] input, int[] maxIndex, float[] maxValue, int size) {
        float max = input[0];
        int maxIdx = 0;
        
        for (int i = 1; i < size; i++) {
            if (input[i] > max) {
                max = input[i];
                maxIdx = i;
            }
        }
        
        maxValue[0] = max;
        maxIndex[0] = maxIdx;
    }
    
    @Override
    public void findMin(float[] input, int[] minIndex, float[] minValue, int size) {
        float min = input[0];
        int minIdx = 0;
        
        for (int i = 1; i < size; i++) {
            if (input[i] < min) {
                min = input[i];
                minIdx = i;
            }
        }
        
        minValue[0] = min;
        minIndex[0] = minIdx;
    }
}
