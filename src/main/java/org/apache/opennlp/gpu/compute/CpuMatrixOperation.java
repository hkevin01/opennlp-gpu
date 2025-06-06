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
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class CpuMatrixOperation implements MatrixOperation {
    private static final Logger log = LoggerFactory.getLogger(CpuMatrixOperation.class);
    
    // The ComputeProvider associated with this matrix operation instance.
    private final ComputeProvider provider;
    
    /**
     * Creates a new CPU matrix operation.
     *
     * @param provider the compute provider used for this operation.
     */
    public CpuMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
        CpuMatrixOperation.log.info("Initializing CPU matrix operations");
    }
    
    /**
     * Returns the associated compute provider.
     *
     * @return the compute provider.
     */
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
    
    /**
     * Multiplies two matrices using the standard CPU algorithm.
     *
     * @param a the first input matrix with dimensions rowsA x sharedDim.
     * @param b the second input matrix with dimensions sharedDim x colsB.
     * @param c the output matrix with dimensions rowsA x colsB.
     * @param rowsA the number of rows in matrix a.
     * @param colsB the number of columns in matrix b.
     * @param sharedDim the shared dimension between matrices a and b.
     */
    @Override
    public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) {
        // For each row in matrix A
        for (int i = 0; i < rowsA; i++) {
            // For each column in matrix B
            for (int j = 0; j < colsB; j++) {
                c[i * colsB + j] = 0; // Initialize the result cell
                // Perform the dot product
                for (int k = 0; k < sharedDim; k++) {
                    c[i * colsB + j] += a[i * sharedDim + k] * b[k * colsB + j];
                }
            }
        }
    }
    
    /**
     * Performs element-wise addition of two matrices.
     *
     * @param a the first input matrix.
     * @param b the second input matrix.
     * @param c the output matrix where the sum is stored.
     * @param elements the number of elements in each matrix.
     */
    @Override
    public void add(float[] a, float[] b, float[] c, int elements) {
        // For each element, perform the addition
        for (int i = 0; i < elements; i++) {
            c[i] = a[i] + b[i];
        }
    }
    
    /**
     * Performs element-wise subtraction of two matrices.
     *
     * @param a the first input matrix.
     * @param b the second input matrix.
     * @param c the output matrix where the difference is stored.
     * @param elements the number of elements in each matrix.
     */
    @Override
    public void subtract(float[] a, float[] b, float[] c, int elements) {
        // For each element, perform the subtraction
        for (int i = 0; i < elements; i++) {
            c[i] = a[i] - b[i];
        }
    }
    
    /**
     * Multiplies each element of the matrix by a scalar value.
     *
     * @param a the input matrix.
     * @param b the output matrix after scalar multiplication.
     * @param scalar the scalar multiplier.
     * @param elements the number of elements in the matrix.
     */
    @Override
    public void scalarMultiply(float[] a, float[] b, float scalar, int elements) {
        // For each element, multiply by the scalar
        for (int i = 0; i < elements; i++) {
            b[i] = a[i] * scalar;
        }
    }
    
    /**
     * Transposes a matrix.
     *
     * @param a the input matrix with dimensions rows x cols.
     * @param b the output matrix which will contain the transpose of a.
     * @param rows the number of rows in the input matrix.
     * @param cols the number of columns in the input matrix.
     */
    @Override
    public void transpose(float[] a, float[] b, int rows, int cols) {
        // For a proper transpose
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                b[j * rows + i] = a[i * cols + j];
            }
        }
    }
    
    /**
     * Releases any resources held by this operation.
     * For CPU-based matrix operations, this typically performs no action.
     */
    @Override
    public void release() {
        CpuMatrixOperation.log.info("Releasing CPU matrix operation resources");
    }
}
