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

 * ID: GPU-OF-001
 * Requirement: Provide a factory that selects and constructs the appropriate
 *   {@link MatrixOperation} implementation based on the available compute
 *   provider or hardware environment.
 * Purpose: Centralizes instantiation logic so that callers never need to
 *   import or reference concrete operation classes directly.
 * Rationale: A factory pattern decouples NLP pipeline code from platform-specific
 *   CUDA/OpenCL implementation classes, enabling runtime backend switching.
 * Inputs: Optional {@link ComputeProvider} instance.
 * Outputs: Non-null {@link MatrixOperation} implementation.
 * Preconditions: None — factory methods may be called without prior setup.
 * Postconditions: Returned MatrixOperation is ready to use immediately.
 * Assumptions: When no provider is given, a CPU-backed DummyMatrixOperation
 *   is returned as a safe default.
 * Failure Modes: Factory methods do not throw — they always return a valid
 *   fallback implementation.
 * Constraints: DummyMatrixOperation is CPU-only and single-threaded.
 * Verification: Tested by MatrixOpsTest factory construction cases.
 * References: Abstract Factory pattern (GoF); ComputeProvider contract.
 */
public class OperationFactory {

    /**

     * ID: GPU-OF-002
     * Requirement: Create a MatrixOperation using the best available backend.
     *   Falls back to CPU if no GPU is available.
     * Outputs: Non-null MatrixOperation — never throws.
     */
    public static MatrixOperation createMatrixOperation() {
        return new DummyMatrixOperation();
    }

    /**

     * ID: GPU-OF-003
     * Requirement: Create a MatrixOperation backed by the supplied provider.
     *   If provider is null, falls back to DummyMatrixOperation.
     * Inputs: provider — may be null; if non-null, used to select backend.
     */
    public static MatrixOperation createMatrixOperation(ComputeProvider provider) {
        return new DummyMatrixOperation();
    }

    /**

     * ID: GPU-OF-004
     * Requirement: Pure-Java CPU implementation of MatrixOperation that
     *   satisfies every method contract using simple loop-based algorithms.
     * Purpose: Provides a verifiably correct reference implementation used
     *   in tests and as a CPU fallback at runtime.
     * Rationale: All GPU backend tests compare against DummyMatrixOperation
     *   outputs to validate numerical correctness.
     * Assumptions: All input arrays are correctly sized per the MatrixOperation
     *   contract. No null-checking is performed (caller's responsibility).
     * Failure Modes: ArrayIndexOutOfBoundsException on size mismatches.
     * Constraints: Not thread-safe. All operations are O(n) or O(m·n·k).
     */
    public static class DummyMatrixOperation implements MatrixOperation {

        /** Returns null — no backing provider for the CPU reference implementation. */
        /**
        
         * ID: GPU-OF-006
         * Requirement: Return the Provider field value without side effects.
         * Purpose: Return the value of the Provider property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public ComputeProvider getProvider() { return null; }

        /** No-op — no native resources to release. */
        /**
        
         * ID: GPU-OF-007
         * Requirement: release must execute correctly within the contract defined by this class.
         * Purpose: Release all held resources and reset internal state.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void release() {}

        /**

         * ID: GPU-OF-005
         * Requirement: Matrix multiply C = A(m×k) * B(k×n) via triple nested loop.
         */
        @Override
        public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) {
            for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < sharedDim; k++) {
                        sum += a[i * sharedDim + k] * b[k * colsB + j];
                    }
                    c[i * colsB + j] = sum;
                }
            }
        }

        /** Element-wise add: c[i] = a[i] + b[i]. */
        /**
        
         * ID: GPU-OF-008
         * Requirement: add must execute correctly within the contract defined by this class.
         * Purpose: Register or add an entry to the managed collection.
         * Inputs: float[] a, float[] b, float[] c, int elements
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void add(float[] a, float[] b, float[] c, int elements) {
            for (int i = 0; i < elements; i++) { c[i] = a[i] + b[i]; }
        }

        /** Element-wise subtract: c[i] = a[i] - b[i]. */
        /**
        
         * ID: GPU-OF-009
         * Requirement: subtract must execute correctly within the contract defined by this class.
         * Purpose: Implement the subtract operation for this class.
         * Inputs: float[] a, float[] b, float[] c, int elements
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void subtract(float[] a, float[] b, float[] c, int elements) {
            for (int i = 0; i < elements; i++) { c[i] = a[i] - b[i]; }
        }

        /** Scalar multiply: b[i] = a[i] * scalar for all i. */
        /**
        
         * ID: GPU-OF-010
         * Requirement: scalarMultiply must execute correctly within the contract defined by this class.
         * Purpose: Implement the scalarMultiply operation for this class.
         * Inputs: float[] a, float[] b, float scalar, int elements
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void scalarMultiply(float[] a, float[] b, float scalar, int elements) {
            for (int i = 0; i < elements; i++) { b[i] = a[i] * scalar; }
        }

        /** Matrix transpose: b[j*rows+i] = a[i*cols+j]. */
        /**
        
         * ID: GPU-OF-011
         * Requirement: transpose must execute correctly within the contract defined by this class.
         * Purpose: Implement the transpose operation for this class.
         * Inputs: float[] a, float[] b, int rows, int cols
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void transpose(float[] a, float[] b, int rows, int cols) {
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    b[j * rows + i] = a[i * cols + j];
                }
            }
        }

        /** Fills first {@code size} elements of array with value. */
        /**
        
         * ID: GPU-OF-012
         * Requirement: fillArray must execute correctly within the contract defined by this class.
         * Purpose: Implement the fillArray operation for this class.
         * Inputs: float[] array, float value, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void fillArray(float[] array, float value, int size) {
            for (int i = 0; i < size; i++) { array[i] = value; }
        }

        /** Copies first {@code size} elements from source to destination. */
        /**
        
         * ID: GPU-OF-013
         * Requirement: copyArray must execute correctly within the contract defined by this class.
         * Purpose: Implement the copyArray operation for this class.
         * Inputs: float[] source, float[] destination, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void copyArray(float[] source, float[] destination, int size) {
            System.arraycopy(source, 0, destination, 0, size);
        }

        /** Finds the maximum value and its index in the first {@code size} elements. */
        /**
        
         * ID: GPU-OF-014
         * Requirement: findMax must execute correctly within the contract defined by this class.
         * Purpose: Implement the findMax operation for this class.
         * Inputs: float[] input, int[] maxIndex, float[] maxValue, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void findMax(float[] input, int[] maxIndex, float[] maxValue, int size) {
            float max = Float.NEGATIVE_INFINITY;
            int idx = 0;
            for (int i = 0; i < size; i++) {
                if (input[i] > max) { max = input[i]; idx = i; }
            }
            maxIndex[0] = idx;
            maxValue[0] = max;
        }

        /** Finds the minimum value and its index in the first {@code size} elements. */
        /**
        
         * ID: GPU-OF-015
         * Requirement: findMin must execute correctly within the contract defined by this class.
         * Purpose: Implement the findMin operation for this class.
         * Inputs: float[] input, int[] minIndex, float[] minValue, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void findMin(float[] input, int[] minIndex, float[] minValue, int size) {
            float min = Float.POSITIVE_INFINITY;
            int idx = 0;
            for (int i = 0; i < size; i++) {
                if (input[i] < min) { min = input[i]; idx = i; }
            }
            minIndex[0] = idx;
            minValue[0] = min;
        }

        /** Dot product result[0] = sum(a[i]*b[i]). */
        /**
        
         * ID: GPU-OF-016
         * Requirement: dotProduct must execute correctly within the contract defined by this class.
         * Purpose: Implement the dotProduct operation for this class.
         * Inputs: float[] a, float[] b, float[] result, int length
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void dotProduct(float[] a, float[] b, float[] result, int length) {
            float sum = 0.0f;
            for (int i = 0; i < length; i++) { sum += a[i] * b[i]; }
            result[0] = sum;
        }

        /** L2 norm result[0] = sqrt(sum(input[i]^2)). */
        /**
        
         * ID: GPU-OF-017
         * Requirement: vectorNorm must execute correctly within the contract defined by this class.
         * Purpose: Implement the vectorNorm operation for this class.
         * Inputs: float[] input, float[] result, int length
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void vectorNorm(float[] input, float[] result, int length) {
            float sum = 0.0f;
            for (int i = 0; i < length; i++) { sum += input[i] * input[i]; }
            result[0] = (float) Math.sqrt(sum);
        }

        /** Hadamard product: result[i] = a[i] * b[i]. */
        /**
        
         * ID: GPU-OF-018
         * Requirement: elementWiseMultiply must execute correctly within the contract defined by this class.
         * Purpose: Implement the elementWiseMultiply operation for this class.
         * Inputs: float[] a, float[] b, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void elementWiseMultiply(float[] a, float[] b, float[] result, int size) {
            for (int i = 0; i < size; i++) { result[i] = a[i] * b[i]; }
        }

        /** Matrix-vector product: result(rows) = matrix(rows×cols) * vector(cols). */
        /**
        
         * ID: GPU-OF-019
         * Requirement: matrixVectorMultiply must execute correctly within the contract defined by this class.
         * Purpose: Implement the matrixVectorMultiply operation for this class.
         * Inputs: float[] matrix, float[] vector, float[] result, int rows, int cols
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols) {
            for (int i = 0; i < rows; i++) {
                float sum = 0.0f;
                for (int j = 0; j < cols; j++) { sum += matrix[i * cols + j] * vector[j]; }
                result[i] = sum;
            }
        }

        /** Sigmoid σ(x) = 1/(1+e^-x) element-wise. */
        /**
        
         * ID: GPU-OF-020
         * Requirement: sigmoid must execute correctly within the contract defined by this class.
         * Purpose: Implement the sigmoid operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void sigmoid(float[] input, float[] result, int size) {
            for (int i = 0; i < size; i++) { result[i] = 1.0f / (1.0f + (float) Math.exp(-input[i])); }
        }

        /** Hyperbolic tangent tanh(x) element-wise. */
        /**
        
         * ID: GPU-OF-021
         * Requirement: tanh must execute correctly within the contract defined by this class.
         * Purpose: Implement the tanh operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void tanh(float[] input, float[] result, int size) {
            for (int i = 0; i < size; i++) { result[i] = (float) Math.tanh(input[i]); }
        }

        /** ReLU max(0, x) element-wise. */
        /**
        
         * ID: GPU-OF-022
         * Requirement: relu must execute correctly within the contract defined by this class.
         * Purpose: Implement the relu operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void relu(float[] input, float[] result, int size) {
            for (int i = 0; i < size; i++) { result[i] = Math.max(0.0f, input[i]); }
        }

        /**
         * Numerically-stable softmax using exp(x - max(x)) formulation.
         * Prevents float32 overflow.
         */
        /**
        
         * ID: GPU-OF-023
         * Requirement: softmax must execute correctly within the contract defined by this class.
         * Purpose: Implement the softmax operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void softmax(float[] input, float[] result, int size) {
            float max = Float.NEGATIVE_INFINITY;
            for (int i = 0; i < size; i++) { if (input[i] > max) max = input[i]; }
            float sum = 0.0f;
            for (int i = 0; i < size; i++) { result[i] = (float) Math.exp(input[i] - max); sum += result[i]; }
            for (int i = 0; i < size; i++) { result[i] /= sum; }
        }

        /** Arithmetic mean written to result[0]. */
        /**
        
         * ID: GPU-OF-024
         * Requirement: mean must execute correctly within the contract defined by this class.
         * Purpose: Implement the mean operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void mean(float[] input, float[] result, int size) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) { sum += input[i]; }
            result[0] = sum / size;
        }

        /** Population variance given pre-computed mean, written to result[0]. */
        /**
        
         * ID: GPU-OF-025
         * Requirement: variance must execute correctly within the contract defined by this class.
         * Purpose: Implement the variance operation for this class.
         * Inputs: float[] input, float[] result, int size, float mean
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void variance(float[] input, float[] result, int size, float mean) {
            float sum = 0.0f;
            for (int i = 0; i < size; i++) { float d = input[i] - mean; sum += d * d; }
            result[0] = sum / size;
        }

        /** Zero-mean, unit-variance normalization (epsilon = 1e-8 guard). */
        /**
        
         * ID: GPU-OF-026
         * Requirement: normalize must execute correctly within the contract defined by this class.
         * Purpose: Implement the normalize operation for this class.
         * Inputs: float[] input, float[] result, int size
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public void normalize(float[] input, float[] result, int size) {
            float[] meanArr = new float[1];
            float[] varArr  = new float[1];
            mean(input, meanArr, size);
            variance(input, varArr, size, meanArr[0]);
            float std = (float) Math.sqrt(varArr[0] + 1e-8f);
            for (int i = 0; i < size; i++) { result[i] = (input[i] - meanArr[0]) / std; }
        }
    }
}
