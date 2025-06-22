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
#define GPU_DEVICE_NAME(id, name, size) cudaGetDeviceProperties(&prop, id); strncpy(name, prop.name, size);
#endif

#ifdef USE_ROCM
#include <hip/hip_runtime.h>
#include <hipblas.h>
#define GPU_DEVICE_COUNT() ({ int count = 0; hipGetDeviceCount(&count); count; })
#define GPU_DEVICE_NAME(id, name, size) hipGetDeviceProperties(&prop, id); strncpy(name, prop.name, size);
#endif

#ifdef USE_CPU_ONLY
#define GPU_DEVICE_COUNT() 0
#define GPU_DEVICE_NAME(id, name, size) strncpy(name, "CPU-Only", size);
#endif

// Common GPU operation abstractions
namespace opennlp {
namespace gpu {

enum class Platform {
#ifdef USE_CUDA
    CUDA,
#endif
#ifdef USE_ROCM
    ROCM,
#endif
    CPU_ONLY
};

inline Platform getCurrentPlatform() {
#ifdef USE_CUDA
    return Platform::CUDA;
#elif defined(USE_ROCM)
    return Platform::ROCM;
#else
    return Platform::CPU_ONLY;
#endif
}

inline const char* getPlatformName() {
    return GPU_PLATFORM;
}

inline bool isGpuAvailable() {
#if defined(USE_CUDA) || defined(USE_ROCM)
    return GPU_DEVICE_COUNT() > 0;
#else
    return false;
#endif
}

} // namespace gpu
} // namespace opennlp

#endif // GPU_CONFIG_H
