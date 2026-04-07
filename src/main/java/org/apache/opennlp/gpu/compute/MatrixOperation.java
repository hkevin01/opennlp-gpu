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
package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**

 * ID: GPU-MO-001
 * Requirement: Define a hardware-agnostic interface for all matrix and vector
 *   operations required by GPU-accelerated NLP pipelines.
 * Purpose: Provides a single contract fulfilled by CPU, CUDA, ROCm, and OpenCL
 *   backends, enabling transparent substitution without caller changes.
 * Rationale: Isolating compute operations behind an interface decouples NLP
 *   algorithms from GPU driver details, simplifying testing and multi-platform
 *   deployment.
 * Inputs: float[] arrays representing dense row-major matrices/vectors;
 *   integer dimension parameters (rows, cols, size).
 * Outputs: Results written to caller-allocated float[] result arrays.
 * Preconditions: All input arrays must be non-null and sized exactly as the
 *   dimension parameters require. No overlapping source/destination arrays.
 * Postconditions: result[] contains computed values. Input arrays are unchanged.
 * Assumptions: float32 precision is sufficient for OpenNLP ML workloads.
 * Side Effects: Implementations may enqueue GPU kernel launches or block until
 *   completion depending on backend.
 * Failure Modes: ArrayIndexOutOfBoundsException if dimensions are mismatched.
 *   GPU out-of-memory is propagated as RuntimeException by implementations.
 * Constraints: All arrays use row-major (C-order) layout. Size parameters must
 *   be positive integers.
 * Verification: Unit-tested via MatrixOpsTest; integration-tested with
 *   OpenNLPTestDataIntegration.
 * References: BLAS Level-2/3 nomenclature; OpenNLP ML model evaluation pipeline.
 */
public interface MatrixOperation {

    /**

     * ID: GPU-MO-002
     * Requirement: Return the backing compute provider for introspection and
     *   resource lifecycle management.
     * Outputs: Non-null ComputeProvider; never null after initialization.
     */
    ComputeProvider getProvider();

    /**

     * ID: GPU-MO-003
     * Requirement: Release all GPU/native resources held by this operation
     *   instance. Must be idempotent.
     * Side Effects: Frees GPU buffers, unregisters kernels, and nulls handles.
     * Postconditions: All associated GPU memory is reclaimed.
     */
    void release();

    // ---- Basic Matrix Operations ----

    /**

     * ID: GPU-MO-004
     * Requirement: Compute C = A * B (general matrix multiply, GEMM).
     * Inputs:
     *   a      - row-major matrix A, length = m * k
     *   b      - row-major matrix B, length = k * n
     *   result - pre-allocated output, length = m * n
     *   m      - rows of A
     *   n      - cols of B
     *   k      - shared dimension (cols of A == rows of B)
     * Constraints: m, n, k > 0. Result must be allocated with length >= m*n.
     */
    void multiply(float[] a, float[] b, float[] result, int m, int n, int k);

    /**

     * ID: GPU-MO-005
     * Requirement: Transpose matrix A(rows × cols) into output B(cols × rows).
     * Inputs:
     *   input  - row-major source, length = rows * cols
     *   output - row-major destination, length = cols * rows
     */
    void transpose(float[] input, float[] output, int rows, int cols);

    /**

     * ID: GPU-MO-006
     * Requirement: Multiply every element of input by scalar, write to output.
     * Inputs: scalar - finite float32 multiplier.
     */
    void scalarMultiply(float[] input, float[] output, float scalar, int length);

    /**

     * ID: GPU-MO-007
     * Requirement: Element-wise addition: result[i] = a[i] + b[i].
     */
    void add(float[] a, float[] b, float[] result, int size);

    /**

     * ID: GPU-MO-008
     * Requirement: Element-wise subtraction: result[i] = a[i] - b[i].
     */
    void subtract(float[] a, float[] b, float[] result, int size);

    // ---- Advanced ML Operations ----

    /**

     * ID: GPU-MO-009
     * Requirement: Compute dot product sum(a[i]*b[i]) into result[0].
     */
    void dotProduct(float[] a, float[] b, float[] result, int length);

    /**

     * ID: GPU-MO-010
     * Requirement: Compute L2 norm of input vector, write to result[0].
     */
    void vectorNorm(float[] input, float[] result, int length);

    /**

     * ID: GPU-MO-011
     * Requirement: Element-wise product: result[i] = a[i] * b[i] (Hadamard).
     */
    void elementWiseMultiply(float[] a, float[] b, float[] result, int size);

    /**

     * ID: GPU-MO-012
     * Requirement: Matrix-vector product: result(rows) = matrix(rows×cols) * vector(cols).
     */
    void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols);

    // ---- Activation Functions ----

    /**

     * ID: GPU-MO-013
     * Requirement: Apply sigmoid σ(x) = 1/(1+e^-x) element-wise.
     * Constraints: Numerically stable for |x| < 87 (float32 range).
     */
    void sigmoid(float[] input, float[] result, int size);

    /**

     * ID: GPU-MO-014
     * Requirement: Apply hyperbolic tangent tanh(x) element-wise.
     */
    void tanh(float[] input, float[] result, int size);

    /**

     * ID: GPU-MO-015
     * Requirement: Apply ReLU max(0, x) element-wise.
     */
    void relu(float[] input, float[] result, int size);

    /**

     * ID: GPU-MO-016
     * Requirement: Apply numerically-stable softmax over the full size vector.
     * Rationale: exp(x - max(x)) formulation prevents float overflow.
     */
    void softmax(float[] input, float[] result, int size);

    // ---- Statistical Operations ----

    /**

     * ID: GPU-MO-017
     * Requirement: Compute arithmetic mean, write to result[0].
     */
    void mean(float[] input, float[] result, int size);

    /**

     * ID: GPU-MO-018
     * Requirement: Compute population variance given a pre-computed mean.
     * Inputs: mean - pre-computed mean value from mean().
     */
    void variance(float[] input, float[] result, int size, float mean);

    /**

     * ID: GPU-MO-019
     * Requirement: Normalize input to zero-mean, unit-variance in-place style.
     * Postconditions: result has mean ≈ 0 and std ≈ 1 (epsilon-guarded).
     */
    void normalize(float[] input, float[] result, int size);

    // ---- Utility Operations ----

    /**

     * ID: GPU-MO-020
     * Requirement: Copy size elements from source to destination (memcpy semantics).
     * Preconditions: source and destination must not overlap.
     */
    void copyArray(float[] source, float[] destination, int size);

    /**

     * ID: GPU-MO-021
     * Requirement: Fill first size elements of array with the given constant value.
     */
    void fillArray(float[] array, float value, int size);

    /**

     * ID: GPU-MO-022
     * Requirement: Find the maximum value and its index over size elements.
     * Outputs:
     *   maxIndex[0]  - 0-based index of maximum element.
     *   maxValue[0]  - value of maximum element.
     */
    void findMax(float[] input, int[] maxIndex, float[] maxValue, int size);

    /**

     * ID: GPU-MO-023
     * Requirement: Find the minimum value and its index over size elements.
     * Outputs:
     *   minIndex[0]  - 0-based index of minimum element.
     *   minValue[0]  - value of minimum element.
     */
    void findMin(float[] input, int[] minIndex, float[] minValue, int size);
}
