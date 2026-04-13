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
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;

/**

 * ID: GPU-CCP-001
 * Requirement: Provide a pure-Java CPU implementation of ComputeProvider that
 *   executes all matrix and NLP operations without native GPU dependencies,
 *   serving as both a production fallback and a test baseline.
 * Purpose: Guarantees that the OpenNLP GPU extension works correctly on any
 *   JVM — including CI environments without GPU hardware — by implementing
 *   every ComputeProvider method in portable Java.
 * Rationale: A CPU fallback eliminates hard dependencies on native drivers,
 *   simplifies Docker/cloud deployment, and provides reference outputs for
 *   validating GPU kernel correctness.
 * Inputs: float[] arrays (row-major) and String[] token arrays as defined by
 *   ComputeProvider; dimension parameters must be positive integers.
 * Outputs: Results written to caller-supplied arrays.
 * Preconditions: Arrays must be correctly sized per operation contract.
 * Postconditions: Results are numerically equivalent to reference formulae.
 * Assumptions: Single-threaded use per instance. All float values are finite.
 * Side Effects: None beyond writing to result arrays and emitting debug logs.
 * Failure Modes: ArrayIndexOutOfBoundsException if dimension parameters are
 *   inconsistent with array lengths — no bounds-coercion is performed.
 * Constraints: O(m·n·k) time complexity for matrixMultiply; no parallelism.
 * Verification: Tested by MatrixOpsTest; used as reference in GPU correctness
 *   tests inside OpenNlpIntegrationTest.
 * References: Standard BLAS DGEMM algorithm; OpenNLP model evaluation contract.
 */
public class CpuComputeProvider implements ComputeProvider {

    private static final GpuLogger logger = GpuLogger.getLogger(CpuComputeProvider.class);

    /**

     * ID: GPU-CCP-002
     * Requirement: Report that this provider does not use a GPU device.
     */
    @Override
    public boolean isGpuProvider() {
        return false;
    }

    /**

     * ID: GPU-CCP-003
     * Requirement: Release provider resources. For CPU this is a no-op logged
     *   for operational traceability.
     */
    @Override
    public void cleanup() {
        logger.debug("CPU compute provider cleaned up");
    }

    /** Returns the display name "CPU Provider". */
    /**
    
     * ID: GPU-CCP-014
     * Requirement: Return the Name field value without side effects.
     * Purpose: Return the value of the Name property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String getName() { return "CPU Provider"; }

    /** Returns {@link Type#CPU}. */
    /**
    
     * ID: GPU-CCP-015
     * Requirement: Return the Type field value without side effects.
     * Purpose: Return the value of the Type property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public Type getType() { return Type.CPU; }

    /**

     * ID: GPU-CCP-004
     * Requirement: Always return true — the CPU is unconditionally available.
     */
    @Override
    public boolean isAvailable() { return true; }

    /**

     * ID: GPU-CCP-005
     * Requirement: Report JVM maximum heap memory as the effective memory limit.
     * Outputs: Runtime.maxMemory() / (1024*1024), in MB.
     */
    @Override
    public long getMaxMemoryMB() {
        return Runtime.getRuntime().maxMemory() / (1024L * 1024L);
    }

    /**

     * ID: GPU-CCP-006
     * Requirement: Report current JVM heap consumption in MB.
     * Outputs: (totalMemory - freeMemory) / (1024*1024).
     */
    @Override
    public long getCurrentMemoryUsageMB() {
        return (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024L * 1024L);
    }

    /** Returns null — the CPU provider requires no native resource manager. */
    /**
    
     * ID: GPU-CCP-016
     * Requirement: Return the ResourceManager field value without side effects.
     * Purpose: Return the value of the ResourceManager property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public Object getResourceManager() { return null; }

    /**

     * ID: GPU-CCP-007
     * Requirement: Compute matrix product C = A(m×k) * B(k×n) using the
     *   standard O(m·n·k) triple-loop algorithm.
     * Rationale: Correctness over performance; GPU backends replace this.
     * Inputs:
     *   a[m*k], b[k*n] — row-major operands; result[m*n] — pre-allocated output.
     */
    @Override
    public void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k) {
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

    /**

     * ID: GPU-CCP-008
     * Requirement: Element-wise addition: result[i] = a[i] + b[i].
     */
    @Override
    public void matrixAdd(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }

    /**

     * ID: GPU-CCP-009
     * Requirement: Transpose matrix input(rows×cols) into output(cols×rows).
     * Algorithm: output[j*rows + i] = input[i*cols + j] for all i, j.
     */
    @Override
    public void matrixTranspose(float[] input, float[] output, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
    }

    /**

     * ID: GPU-CCP-010
     * Requirement: Produce a simple numeric feature vector from text tokens.
     * Algorithm: features[i] = (float) text[i].length() for i < min(text.length, features.length).
     * Rationale: Provides a deterministic, dependency-free baseline feature
     *   extractor for unit testing and CPU fallback mode.
     */
    @Override
    public void extractFeatures(String[] text, float[] features) {
        int limit = Math.min(text.length, features.length);
        for (int i = 0; i < limit; i++) {
            features[i] = text[i].length();
        }
    }

    /**

     * ID: GPU-CCP-011
     * Requirement: Compute TF-IDF score:
     *   result[i] = termFreq[i] * ln(1 + 1 / (docFreq[i] + ε))
     *   where ε = 1e-10 guards against division by zero.
     * References: Salton & Buckley (1988) TF-IDF formulation.
     */
    @Override
    public void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size) {
        final double epsilon = 1e-10;
        for (int i = 0; i < size; i++) {
            result[i] = termFreq[i] * (float) Math.log(1.0 + 1.0 / (docFreq[i] + epsilon));
        }
    }

    /**

     * ID: GPU-CCP-012
     * Requirement: Initialize the CPU provider. No-op beyond trace logging.
     */
    @Override
    public void initialize() {
        logger.debug("Initializing CPU compute provider");
    }

    /** Delegates to {@link #initialize()} — config is ignored for CPU. */
    /**
    
     * ID: GPU-CCP-017
     * Requirement: initialize must execute correctly within the contract defined by this class.
     * Purpose: Initialise internal state and allocate required resources.
     * Inputs: GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void initialize(GpuConfig config) {
        initialize();
    }

    /**

     * ID: GPU-CCP-013
     * Requirement: Report that all operation types are supported by the CPU backend.
     */
    @Override
    public boolean supportsOperation(String operationType) {
        return true;
    }
}
