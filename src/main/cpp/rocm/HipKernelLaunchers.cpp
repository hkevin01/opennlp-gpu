#include <hip/hip_runtime.h>
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>
#include <memory>

// Forward declare kernels from kernels.cpp
extern "C"
{
    __global__ void hip_matrix_multiply(const float *A, const float *B, float *C,
                                        int m, int n, int k);
    __global__ void hip_softmax(float *data, int size, int num_classes);
    __global__ void hip_vector_add(const float *a, const float *b, float *result, int size);
    __global__ void hip_dot_product(const float *features, const float *weights,
                                    float *result, int feature_size, int batch_size);
}

// HIP kernel launcher implementations for OpenNLP GPU operations

class HipKernelLaunchers
{
public:
    // Initialize HIP device
    static bool initialize()
    {
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess || device_count == 0)
        {
            std::cerr << "No HIP devices found or HIP not available" << std::endl;
            return false;
        }

        // Set device 0 as default
        error = hipSetDevice(0);
        if (error != hipSuccess)
        {
            std::cerr << "Failed to set HIP device" << std::endl;
            return false;
        }

        return true;
    }

    // Launch matrix multiplication kernel
    static bool launchMatrixMultiply(const float *h_A, const float *h_B, float *h_C,
                                     int m, int n, int k)
    {
        // Allocate device memory
        float *d_A, *d_B, *d_C;
        size_t size_A = m * k * sizeof(float);
        size_t size_B = k * n * sizeof(float);
        size_t size_C = m * n * sizeof(float);

        hipMalloc(&d_A, size_A);
        hipMalloc(&d_B, size_B);
        hipMalloc(&d_C, size_C);

        // Copy data to device
        hipMemcpy(d_A, h_A, size_A, hipMemcpyHostToDevice);
        hipMemcpy(d_B, h_B, size_B, hipMemcpyHostToDevice);

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((n + blockSize.x - 1) / blockSize.x,
                      (m + blockSize.y - 1) / blockSize.y);

        hipLaunchKernelGGL(hip_matrix_multiply,
                           gridSize, blockSize, 0, 0,
                           d_A, d_B, d_C, m, n, k);

        // Copy result back
        hipMemcpy(h_C, d_C, size_C, hipMemcpyDeviceToHost);

        // Cleanup
        hipFree(d_A);
        hipFree(d_B);
        hipFree(d_C);

        return hipGetLastError() == hipSuccess;
    }

    // Launch softmax kernel
    static bool launchSoftmax(float *h_data, int batch_size, int num_classes)
    {
        float *d_data;
        size_t size = batch_size * num_classes * sizeof(float);

        hipMalloc(&d_data, size);
        hipMemcpy(d_data, h_data, size, hipMemcpyHostToDevice);

        int total_size = batch_size * num_classes;
        dim3 blockSize(256);
        dim3 gridSize((total_size + blockSize.x - 1) / blockSize.x);

        hipLaunchKernelGGL(hip_softmax,
                           gridSize, blockSize, 0, 0,
                           d_data, total_size, num_classes);

        hipMemcpy(h_data, d_data, size, hipMemcpyDeviceToHost);
        hipFree(d_data);

        return hipGetLastError() == hipSuccess;
    }

    // Launch vector addition kernel
    static bool launchVectorAdd(const float *h_a, const float *h_b, float *h_result, int size)
    {
        float *d_a, *d_b, *d_result;
        size_t mem_size = size * sizeof(float);

        hipMalloc(&d_a, mem_size);
        hipMalloc(&d_b, mem_size);
        hipMalloc(&d_result, mem_size);

        hipMemcpy(d_a, h_a, mem_size, hipMemcpyHostToDevice);
        hipMemcpy(d_b, h_b, mem_size, hipMemcpyHostToDevice);

        dim3 blockSize(256);
        dim3 gridSize((size + blockSize.x - 1) / blockSize.x);

        hipLaunchKernelGGL(hip_vector_add,
                           gridSize, blockSize, 0, 0,
                           d_a, d_b, d_result, size);

        hipMemcpy(h_result, d_result, mem_size, hipMemcpyDeviceToHost);

        hipFree(d_a);
        hipFree(d_b);
        hipFree(d_result);

        return hipGetLastError() == hipSuccess;
    }

    // Launch dot product kernel
    static bool launchDotProduct(const float *h_features, const float *h_weights,
                                 float *h_result, int feature_size, int batch_size)
    {
        float *d_features, *d_weights, *d_result;
        size_t features_size = batch_size * feature_size * sizeof(float);
        size_t weights_size = feature_size * sizeof(float);
        size_t result_size = batch_size * sizeof(float);

        hipMalloc(&d_features, features_size);
        hipMalloc(&d_weights, weights_size);
        hipMalloc(&d_result, result_size);

        hipMemcpy(d_features, h_features, features_size, hipMemcpyHostToDevice);
        hipMemcpy(d_weights, h_weights, weights_size, hipMemcpyHostToDevice);

        dim3 blockSize(256);
        dim3 gridSize((batch_size + blockSize.x - 1) / blockSize.x);

        hipLaunchKernelGGL(hip_dot_product,
                           gridSize, blockSize, 0, 0,
                           d_features, d_weights, d_result, feature_size, batch_size);

        hipMemcpy(h_result, d_result, result_size, hipMemcpyDeviceToHost);

        hipFree(d_features);
        hipFree(d_weights);
        hipFree(d_result);

        return hipGetLastError() == hipSuccess;
    }

    // Get device information
    static std::string getDeviceInfo()
    {
        int device;
        hipGetDevice(&device);

        hipDeviceProp_t prop;
        hipGetDeviceProperties(&prop, device);

        return std::string("Device: ") + prop.name +
               ", Compute Units: " + std::to_string(prop.multiProcessorCount) +
               ", Memory: " + std::to_string(prop.totalGlobalMem / (1024 * 1024)) + "MB";
    }
};

// C interface for JNI
extern "C"
{
    // C wrapper functions
    int hip_initialize()
    {
        return HipKernelLaunchers::initialize() ? 1 : 0;
    }

    int hip_matrix_multiply_wrapper(const float *A, const float *B, float *C,
                                    int m, int n, int k)
    {
        return HipKernelLaunchers::launchMatrixMultiply(A, B, C, m, n, k) ? 1 : 0;
    }

    int hip_softmax_wrapper(float *data, int batch_size, int num_classes)
    {
        return HipKernelLaunchers::launchSoftmax(data, batch_size, num_classes) ? 1 : 0;
    }

    int hip_vector_add_wrapper(const float *a, const float *b, float *result, int size)
    {
        return HipKernelLaunchers::launchVectorAdd(a, b, result, size) ? 1 : 0;
    }

    int hip_dot_product_wrapper(const float *features, const float *weights,
                                float *result, int feature_size, int batch_size)
    {
        return HipKernelLaunchers::launchDotProduct(features, weights, result,
                                                    feature_size, batch_size)
                   ? 1
                   : 0;
    }

    const char *hip_get_device_info()
    {
        static std::string info = HipKernelLaunchers::getDeviceInfo();
        return info.c_str();
    }
}
