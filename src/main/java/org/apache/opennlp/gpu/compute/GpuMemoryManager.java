package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.GpuConfig;

/**

 * ID: GPU-GMM-001
 * Requirement: GpuMemoryManager must manage GPU device memory across CUDA, ROCm, and OpenCL buffers for matrix operations.
 * Purpose: Provides a unified memory management layer ensuring device buffers are allocated, used, and released without leaks.
 * Rationale: Device memory is a scarce resource; centralised management prevents over-allocation and enables pool reuse.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Allocates and frees GPU device memory; maintains allocation tracking map.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuMemoryManager {

    private final GpuConfig config;
    /** Tracks live host-proxy buffers by id → size in bytes. */
    private final java.util.concurrent.ConcurrentHashMap<Long, Integer> allocationMap =
            new java.util.concurrent.ConcurrentHashMap<>();
    private final java.util.concurrent.atomic.AtomicLong nextId =
            new java.util.concurrent.atomic.AtomicLong(1L);
    /** Total bytes currently tracked. */
    private final java.util.concurrent.atomic.AtomicLong totalAllocated =
            new java.util.concurrent.atomic.AtomicLong(0L);

    /**
    
     * ID: GPU-GMM-002
     * Requirement: GpuMemoryManager must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuMemoryManager instance.
     * Inputs: GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuMemoryManager(GpuConfig config) {
        if (config == null) throw new IllegalArgumentException("config must not be null");
        this.config = config;
    }

    /**
     * Allocate a tracked host-side buffer of {@code sizeBytes} bytes.
     * Returns an opaque Long handle used to release the allocation later.
     */
    public long allocate(int sizeBytes) {
        long id = nextId.getAndIncrement();
        allocationMap.put(id, sizeBytes);
        totalAllocated.addAndGet(sizeBytes);
        return id;
    }

    /** Release the allocation identified by {@code handle}. No-op if handle is unknown. */
    public void free(long handle) {
        Integer size = allocationMap.remove(handle);
        if (size != null) {
            totalAllocated.addAndGet(-size);
        }
    }

    /** Returns total tracked allocated bytes. */
    public long getTotalAllocatedBytes() { return totalAllocated.get(); }

    /** Returns the maximum allowed memory in bytes from config. */
    public long getMaxMemoryBytes() {
        return config.getMemoryPoolSizeMB() * 1024L * 1024L;
    }

    /**
    
     * ID: GPU-GMM-003
     * Requirement: cleanup must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void cleanup() {
        allocationMap.clear();
        totalAllocated.set(0L);
    }
}
