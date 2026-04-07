package org.apache.opennlp.gpu.common;

/**

 * Requirement: MemoryManager must define the lifecycle contract for GPU memory buffers: allocate, copy, and release.
 * Purpose: Interface required by every GPU backend for native buffer management, ensuring no GPU memory leaks across JNI boundaries.
 * Rationale: Interface separation lets CPU and GPU providers implement memory lifecycle with platform-specific APIs (cudaMalloc, hipMalloc, etc.).
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; implementors handle all side effects.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public interface MemoryManager {
    int allocate(long size);
    void free(long ptr);
    void copyHostToDevice(long devicePtr, byte[] hostData, long size);
    void copyDeviceToHost(long devicePtr, byte[] hostData, long size);
    void releaseAll();
}
