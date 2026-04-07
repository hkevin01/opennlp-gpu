package org.apache.opennlp.gpu.common;

/**
 * ID: CPF-001
 * Requirement: ComputeProviderFactory must construct and return the correct ComputeProvider implementation for a given Type.
 * Purpose: Factory that decouples caller code from concrete provider classes (CpuComputeProvider, CudaComputeProvider, OpenClComputeProvider, RocmComputeProvider).
 * Rationale: Factory pattern eliminates switch/case duplication across callers and makes adding new vendors a single-point change.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; all construction is side-effect-free.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class ComputeProviderFactory {
    
    public static ComputeProvider getDefaultProvider() {
        return new CpuComputeProvider();
    }
    
    public static ComputeProvider createProvider(ComputeProvider.Type type) {
        switch (type) {
            case CPU:
                return new CpuComputeProvider();
            case OPENCL:
                return new OpenClComputeProvider();
            case CUDA:
                return new CudaComputeProvider();
            case ROCM:
                return new RocmComputeProvider();
            default:
                return new CpuComputeProvider();
        }
    }
    
    public static ComputeProvider getProvider(ComputeProvider.Type type) {
        return ComputeProviderFactory.createProvider(type);
    }
}
