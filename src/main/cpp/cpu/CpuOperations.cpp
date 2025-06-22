#include <iostream>
#include <cmath>
#include <vector>
#include <string>

// CPU-only operations implementation for OpenNLP GPU fallback

class CpuOperations
{
public:
    // Initialize (no-op for CPU)
    static bool initialize()
    {
        std::cout << "Using CPU-only computation" << std::endl;
        return true;
    }

    // Matrix multiplication
    static bool matrixMultiply(const float *A, const float *B, float *C,
                               int m, int n, int k)
    {
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++)
                {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = sum;
            }
        }
        return true;
    }

    // Softmax activation
    static bool softmax(float *data, int batch_size, int num_classes)
    {
        for (int i = 0; i < batch_size; i++)
        {
            float *sample = data + i * num_classes;

            // Find max for numerical stability
            float max_val = sample[0];
            for (int j = 1; j < num_classes; j++)
            {
                if (sample[j] > max_val)
                    max_val = sample[j];
            }

            // Compute exponentials and sum
            float sum = 0.0f;
            for (int j = 0; j < num_classes; j++)
            {
                sample[j] = std::exp(sample[j] - max_val);
                sum += sample[j];
            }

            // Normalize
            for (int j = 0; j < num_classes; j++)
            {
                sample[j] /= sum;
            }
        }
        return true;
    }

    // Vector addition
    static bool vectorAdd(const float *a, const float *b, float *result, int size)
    {
        for (int i = 0; i < size; i++)
        {
            result[i] = a[i] + b[i];
        }
        return true;
    }

    // Dot product
    static bool dotProduct(const float *features, const float *weights,
                           float *result, int feature_size, int batch_size)
    {
        for (int i = 0; i < batch_size; i++)
        {
            float sum = 0.0f;
            const float *sample_features = features + i * feature_size;

            for (int j = 0; j < feature_size; j++)
            {
                sum += sample_features[j] * weights[j];
            }

            result[i] = sum;
        }
        return true;
    }

    // Get device information
    static std::string getDeviceInfo()
    {
        return "CPU-only computation (no GPU acceleration)";
    }

    // Cleanup (no-op for CPU)
    static void cleanup()
    {
        // Nothing to clean up for CPU
    }
};

// C interface
extern "C"
{
    int cpu_initialize()
    {
        return CpuOperations::initialize() ? 1 : 0;
    }

    int cpu_matrix_multiply_wrapper(const float *A, const float *B, float *C,
                                    int m, int n, int k)
    {
        return CpuOperations::matrixMultiply(A, B, C, m, n, k) ? 1 : 0;
    }

    int cpu_softmax_wrapper(float *data, int batch_size, int num_classes)
    {
        return CpuOperations::softmax(data, batch_size, num_classes) ? 1 : 0;
    }

    int cpu_vector_add_wrapper(const float *a, const float *b, float *result, int size)
    {
        return CpuOperations::vectorAdd(a, b, result, size) ? 1 : 0;
    }

    int cpu_dot_product_wrapper(const float *features, const float *weights,
                                float *result, int feature_size, int batch_size)
    {
        return CpuOperations::dotProduct(features, weights, result,
                                         feature_size, batch_size)
                   ? 1
                   : 0;
    }

    const char *cpu_get_device_info()
    {
        static std::string info = CpuOperations::getDeviceInfo();
        return info.c_str();
    }

    void cpu_ops_cleanup()
    {
        CpuOperations::cleanup();
    }
}
