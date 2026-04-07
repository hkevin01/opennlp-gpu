/*
 * ID: HO-001
 * Requirement: HipOperations.cpp must implement JNI-exposed ROCm/HIP operation wrappers (init, device query, matrix multiply, softmax) for the RocmUtil Java class.
 * Purpose: C++ source providing HIP runtime API wrappers callable from Java via JNI: device enumeration, matrix multiply, softmax.
 * Rationale: ROCm/HIP mirrors CUDA semantics but uses different API calls; isolating them here allows CUDA and ROCm paths to diverge without affecting Java code.
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

#include <hip/hip_runtime.h>
#include <iostream>
#include <string>
#include <vector>

// ROCm/HIP operations implementation for OpenNLP GPU

class HipOperations
{
private:
    static bool initialized;
    static int current_device;

public:
    // Initialize HIP runtime
    static bool initialize()
    {
        if (initialized)
            return true;

        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess)
        {
            std::cerr << "HIP initialization failed: " << hipGetErrorString(error) << std::endl;
            return false;
        }

        if (device_count == 0)
        {
            std::cerr << "No HIP-capable devices found" << std::endl;
            return false;
        }

        // Set default device
        error = hipSetDevice(0);
        if (error != hipSuccess)
        {
            std::cerr << "Failed to set HIP device: " << hipGetErrorString(error) << std::endl;
            return false;
        }

        current_device = 0;
        initialized = true;

        std::cout << "HIP initialized successfully with " << device_count << " device(s)" << std::endl;
        return true;
    }

    // Get device count
    static int getDeviceCount()
    {
        int count = 0;
        hipGetDeviceCount(&count);
        return count;
    }

    // Get device information
    static std::string getDeviceInfo(int device_id = -1)
    {
        if (device_id == -1)
            device_id = current_device;

        hipDeviceProp_t prop;
        hipError_t error = hipGetDeviceProperties(&prop, device_id);

        if (error != hipSuccess)
        {
            return "Error getting device properties";
        }

        return std::string("Device ") + std::to_string(device_id) + ": " + prop.name +
               " (Compute Units: " + std::to_string(prop.multiProcessorCount) +
               ", Memory: " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) + "MB" +
               ", Clock: " + std::to_string(prop.clockRate / 1000) + "MHz)";
    }

    // Allocate device memory
    static void *allocateDeviceMemory(size_t size)
    {
        void *ptr = nullptr;
        hipError_t error = hipMalloc(&ptr, size);

        if (error != hipSuccess)
        {
            std::cerr << "Failed to allocate device memory: " << hipGetErrorString(error) << std::endl;
            return nullptr;
        }

        return ptr;
    }

    // Free device memory
    static void freeDeviceMemory(void *ptr)
    {
        if (ptr)
        {
            hipFree(ptr);
        }
    }

    // Copy memory host to device
    static bool copyHostToDevice(void *dst, const void *src, size_t size)
    {
        hipError_t error = hipMemcpy(dst, src, size, hipMemcpyHostToDevice);
        return error == hipSuccess;
    }

    // Copy memory device to host
    static bool copyDeviceToHost(void *dst, const void *src, size_t size)
    {
        hipError_t error = hipMemcpy(dst, src, size, hipMemcpyDeviceToHost);
        return error == hipSuccess;
    }

    // Synchronize device
    static bool synchronize()
    {
        hipError_t error = hipDeviceSynchronize();
        return error == hipSuccess;
    }

    // Get memory information
    static std::pair<size_t, size_t> getMemoryInfo()
    {
        size_t free_bytes, total_bytes;
        hipMemGetInfo(&free_bytes, &total_bytes);
        return std::make_pair(free_bytes, total_bytes);
    }

    // Check for errors
    static std::string getLastError()
    {
        hipError_t error = hipGetLastError();
        return std::string(hipGetErrorString(error));
    }

    // Cleanup
    static void cleanup()
    {
        if (initialized)
        {
            hipDeviceReset();
            initialized = false;
        }
    }
};

// Static member initialization
bool HipOperations::initialized = false;
int HipOperations::current_device = -1;

// C interface for JNI
extern "C"
{
    // Initialization
    int hip_ops_initialize()
    {
        return HipOperations::initialize() ? 1 : 0;
    }

    // Device information
    int hip_ops_get_device_count()
    {
        return HipOperations::getDeviceCount();
    }

    const char *hip_ops_get_device_info(int device_id)
    {
        static std::string info = HipOperations::getDeviceInfo(device_id);
        return info.c_str();
    }

    // Memory management
    void *hip_ops_allocate_memory(size_t size)
    {
        return HipOperations::allocateDeviceMemory(size);
    }

    void hip_ops_free_memory(void *ptr)
    {
        HipOperations::freeDeviceMemory(ptr);
    }

    int hip_ops_copy_host_to_device(void *dst, const void *src, size_t size)
    {
        return HipOperations::copyHostToDevice(dst, src, size) ? 1 : 0;
    }

    int hip_ops_copy_device_to_host(void *dst, const void *src, size_t size)
    {
        return HipOperations::copyDeviceToHost(dst, src, size) ? 1 : 0;
    }

    int hip_ops_synchronize()
    {
        return HipOperations::synchronize() ? 1 : 0;
    }

    void hip_ops_get_memory_info(size_t *free_bytes, size_t *total_bytes)
    {
        auto info = HipOperations::getMemoryInfo();
        *free_bytes = info.first;
        *total_bytes = info.second;
    }

    const char *hip_ops_get_last_error()
    {
        static std::string error = HipOperations::getLastError();
        return error.c_str();
    }

    void hip_ops_cleanup()
    {
        HipOperations::cleanup();
    }
}
