/*
 * ID: G-001
 * Requirement: gpu_config.h must define compile-time GPU platform selection macros and C++ namespace utility functions.
 * Purpose: Central header for conditional compilation: selects CUDA, ROCm, or CPU-only paths; provides isGpuAvailable() and getPlatformName() for runtime queries.
 * Rationale: A single platform-selection header prevents ifdef duplication across all native source files.
 * Inputs: JNI call parameters or direct C++ function arguments as documented per function.
 * Outputs: Return values / JNI jboolean, jint, jstring, jfloat array results.
 * Preconditions: GPU runtime (CUDA/ROCm/OpenCL) initialised; JNI env pointer valid.
 * Postconditions: GPU resources allocated or reported unavailable; error codes returned.
 * Assumptions: Compiled with matching GPU SDK headers (CUDA 11+, ROCm 5+, or CPU fallback).
 * Side Effects: Allocates/frees GPU device memory; prints error messages to stderr on failure.
 * Failure Modes: GPU API errors return false/0/NULL; errors logged via fprintf(stderr).
 * Error Handling: CHECK_CUDA_ERROR / CHECK_HIP_ERROR macros abort on API failures.
 * Constraints: GPU memory bounded by device; operations are single-device by default.
 * Verification: Integration tests in src/test; GpuDiagnostics CLI probe.
 * References: CUDA Toolkit docs; ROCm/HIP API; OpenCL 1.2 spec; JNI Programmer's Guide.
 */

#ifndef GPU_CONFIG_H
#define GPU_CONFIG_H

// Auto-generated GPU configuration header
// Generated at build time based on available GPU platforms

#define GPU_PLATFORM "ROCM"

#define USE_CUDA
#define USE_ROCM
/* #undef USE_CPU_ONLY */

// Platform-specific includes
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <cublas_v2.h>
#define GPU_DEVICE_COUNT() ({ int count = 0; cudaGetDeviceCount(&count); count; })
#define GPU_DEVICE_NAME(id, name, size) \
    cudaGetDeviceProperties(&prop, id); \
    strncpy(name, prop.name, size);
#endif

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hipblas.h>
#define GPU_DEVICE_COUNT() ({ int count = 0; hipGetDeviceCount(&count); count; })
#define GPU_DEVICE_NAME(id, name, size) \
    hipGetDeviceProperties(&prop, id);  \
    strncpy(name, prop.name, size);
#endif

#ifdef USE_CPU_ONLY
#define GPU_DEVICE_COUNT() 0
#define GPU_DEVICE_NAME(id, name, size) strncpy(name, "CPU-Only", size);
#endif

// Common GPU operation abstractions
namespace opennlp
{
    namespace gpu
    {

        enum class Platform
        {
#ifdef USE_CUDA
            CUDA,
#endif
#ifdef USE_ROCM
            ROCM,
#endif
            CPU_ONLY
        };

        inline Platform getCurrentPlatform()
        {
#ifdef USE_CUDA
            return Platform::CUDA;
#elif defined(USE_ROCM)
            return Platform::ROCM;
#else
            return Platform::CPU_ONLY;
#endif
        }

        inline const char *getPlatformName()
        {
            return GPU_PLATFORM;
        }

        inline bool isGpuAvailable()
        {
#if defined(USE_CUDA) || defined(USE_ROCM)
            return GPU_DEVICE_COUNT() > 0;
#else
            return false;
#endif
        }

    } // namespace gpu
} // namespace opennlp

#endif // GPU_CONFIG_H
