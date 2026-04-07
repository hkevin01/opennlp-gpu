package org.apache.opennlp.gpu.common;

/**

 * Requirement: GpuDevice must expose device identity and memory capacity for a single GPU or compute device.
 * Purpose: Lightweight interface for querying per-device metadata (name, device ID, VRAM in MB) used by diagnostics and configuration.
 * Rationale: Abstracting device queries behind an interface allows swapping CUDA/ROCm/OpenCL device representations without caller changes.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; read-only query interface.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public interface GpuDevice {
    String getName();
    int getDeviceId();
    
    // Add missing method referenced in GpuDemoMain
    long getMemoryMB();
}
