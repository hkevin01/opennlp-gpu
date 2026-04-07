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
package org.apache.opennlp.gpu.common;

/**

 * Requirement: Define the top-level contract for all compute backends
 *   (CPU, CUDA, ROCm, OpenCL) used by the OpenNLP GPU acceleration layer.
 * Purpose: Enables transparent runtime substitution of compute backends based
 *   on hardware availability, supporting graceful CPU fallback.
 * Rationale: A single interface isolates NLP algorithm code from GPU driver
 *   specifics, enabling cross-platform operation and simplified unit testing
 *   via mock providers.
 * Inputs: GpuConfig for initialization; float[] arrays for compute methods.
 * Outputs: Computed results written to caller-owned arrays; status methods
 *   return scalars or booleans.
 * Preconditions: initialize() or initialize(GpuConfig) must be called before
 *   any compute operation.
 * Postconditions: After cleanup(), no further compute calls are valid.
 * Assumptions: A single ComputeProvider instance is used by at most one thread
 *   unless the implementation explicitly documents thread safety.
 * Side Effects: Initialization may allocate GPU device contexts, command queues,
 *   or thread pools. Cleanup releases all such resources.
 * Failure Modes: Calling compute methods before initialize() results in
 *   IllegalStateException from implementations.
 * Constraints: Memory measurements are in megabytes (MB). Array layouts are
 *   row-major (C-order), float32 precision.
 * Verification: Tested via CpuComputeProvider unit tests and GPU integration
 *   tests in OpenNLPTestDataIntegration.
 * References: OpenNLP AbstractModel evaluation contract; BLAS operation naming.
 */
public interface ComputeProvider {

    /**

     * Requirement: Enumerate the supported hardware compute backends.
     * Purpose: Drive provider selection and capability reporting.
     */
    enum Type {
        /** Java-only CPU backend — always available. */
        CPU,
        /** NVIDIA CUDA backend — requires CUDA Toolkit ≥ 11 and compatible GPU. */
        CUDA,
        /** Cross-vendor OpenCL backend — requires ICD loader and OpenCL driver. */
        OPENCL,
        /** AMD ROCm/HIP backend — requires ROCm ≥ 5.0 and compatible AMD GPU. */
        ROCM
    }

    /**

     * Requirement: Return a human-readable display name for this provider.
     * Outputs: Non-null, non-empty string (e.g., "CPU Provider", "CUDA 12.1").
     */
    String getName();

    /**

     * Requirement: Report whether this provider is operational and ready to
     *   accept compute requests without error.
     * Outputs: true if hardware is present, drivers are loaded, and memory is
     *   allocated; false otherwise.
     */
    boolean isAvailable();

    /**

     * Requirement: Initialize the provider with the supplied configuration.
     *   Must be idempotent — repeated calls with the same config are safe.
     * Side Effects: May allocate GPU context, command queues, or JNI buffers.
     * Error Handling: Throws RuntimeException if hardware initialization fails
     *   and no CPU fallback is configured.
     */
    void initialize(GpuConfig config);

    /**

     * Requirement: Initialize with default internal configuration.
     *   Equivalent to initialize(new GpuConfig()).
     */
    void initialize();

    /**

     * Requirement: Distinguish GPU-backed providers from the CPU provider.
     * Outputs: true for CUDA/OpenCL/ROCm providers; false for CPU.
     */
    default boolean isGpuProvider() {
        return false;
    }

    /**

     * Requirement: Return the provider type enum value.
     */
    Type getType();

    /**

     * Requirement: Report the maximum memory available on this device in MB.
     * Outputs: Positive long; 0 if measurement is unavailable.
     */
    long getMaxMemoryMB();

    /**

     * Requirement: Report current memory consumption by this provider in MB.
     * Outputs: Non-negative long; ≤ getMaxMemoryMB().
     */
    long getCurrentMemoryUsageMB();

    /**

     * Requirement: Report whether this provider can execute the named operation.
     * Inputs: operationType — operation identifier string (e.g., "matmul", "tfidf").
     * Outputs: true if the operation is implemented and ready.
     */
    boolean supportsOperation(String operationType);

    /**

     * Requirement: Return the native resource manager handle for advanced use.
     * Outputs: Provider-specific object (e.g., CudaResourceManager); null for CPU.
     */
    Object getResourceManager();

    // ---- Compute Operations ----

    /**

     * Requirement: General matrix multiplication C = A(m×k) * B(k×n).
     * Inputs:
     *   a      - row-major float[m*k]
     *   b      - row-major float[k*n]
     *   result - pre-allocated float[m*n]; must not alias a or b
     *   m, n, k - positive integer dimensions
     */
    void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k);

    /**

     * Requirement: Element-wise addition result[i] = a[i] + b[i].
     * Inputs: All arrays length ≥ size.
     */
    void matrixAdd(float[] a, float[] b, float[] result, int size);

    /**

     * Requirement: Matrix transpose output(cols×rows) = input(rows×cols).
     */
    void matrixTranspose(float[] input, float[] output, int rows, int cols);

    /**

     * Requirement: Extract numeric feature vectors from raw text tokens.
     * Inputs:
     *   text     - array of token strings; length determines feature inputs
     *   features - pre-allocated float[] for feature output
     * Postconditions: features[i] populated for 0 ≤ i < min(text.length, features.length).
     */
    void extractFeatures(String[] text, float[] features);

    /**

     * Requirement: Compute TF-IDF scores: result[i] = termFreq[i] * log(1 + 1/docFreq[i]).
     * Inputs: termFreq, docFreq — non-negative float arrays of length ≥ size.
     * Error Handling: docFreq[i] = 0 is guarded with epsilon to avoid division by zero.
     */
    void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size);

    // ---- Lifecycle ----

    /**

     * Requirement: Return structured capability metadata for this provider.
     * Outputs: Non-null ProviderCapabilities instance.
     */
    default ProviderCapabilities getCapabilities() {
        return new ProviderCapabilities();
    }

    /**

     * Requirement: Release all resources held by this provider. Idempotent.
     * Side Effects: Frees GPU memory, destroys CUDA context or OpenCL command queue.
     * Postconditions: No further compute operations are valid on this instance.
     */
    default void cleanup() {
        // Default no-op for CPU provider
    }

    /**

     * Requirement: Carry provider capability flags as a plain value object.
     * Purpose: Enables callers to query parallelism and GPU support without
     *   casting to concrete provider types.
     */
    class ProviderCapabilities {
        private boolean supportsParallelComputation = false;
        private boolean supportsGpuAcceleration = false;
        private int maxThreads = 1;

        /** Whether this provider supports concurrent operation execution. */
        public boolean supportsParallelComputation() { return supportsParallelComputation; }
        /** Whether hardware GPU acceleration is active. */
        public boolean supportsGpuAcceleration() { return supportsGpuAcceleration; }
        /** Maximum thread count used by this provider. */
        public int getMaxThreads() { return maxThreads; }

        public void setSupportsParallelComputation(boolean supports) { this.supportsParallelComputation = supports; }
        public void setSupportsGpuAcceleration(boolean supports) { this.supportsGpuAcceleration = supports; }
        public void setMaxThreads(int threads) { this.maxThreads = threads; }
    }
}
