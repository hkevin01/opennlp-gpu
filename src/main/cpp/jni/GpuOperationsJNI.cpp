#include <jni.h>
#include <iostream>
#include <string>
#include <vector>

#ifdef USE_CUDA
extern "C"
{
    // CUDA function declarations
    int cuda_initialize();
    int cuda_matrix_multiply_wrapper(const float *A, const float *B, float *C, int m, int n, int k);
    int cuda_softmax_wrapper(float *data, int batch_size, int num_classes);
    const char *cuda_get_device_info();
    void cuda_ops_cleanup();
}
#endif

#ifdef USE_ROCM
extern "C"
{
    // ROCm/HIP function declarations
    int hip_initialize();
    int hip_matrix_multiply_wrapper(const float *A, const float *B, float *C, int m, int n, int k);
    int hip_softmax_wrapper(float *data, int batch_size, int num_classes);
    const char *hip_get_device_info();
    void hip_ops_cleanup();
    int hip_ops_get_device_count();
}
#endif

// JNI implementations
extern "C"
{

    JNIEXPORT jboolean JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_initializeGpu(JNIEnv *env, jobject obj)
    {
#ifdef USE_CUDA
        return cuda_initialize() == 1 ? JNI_TRUE : JNI_FALSE;
#elif defined(USE_ROCM)
        return hip_initialize() == 1 ? JNI_TRUE : JNI_FALSE;
#else
        return JNI_FALSE; // CPU-only build
#endif
    }

    JNIEXPORT jint JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_getDeviceCount(JNIEnv *env, jobject obj)
    {
#ifdef USE_ROCM
        return hip_ops_get_device_count();
#elif defined(USE_CUDA)
        // Add CUDA device count function if needed
        return 1; // Placeholder
#else
        return 0;
#endif
    }

    JNIEXPORT jstring JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_getDeviceInfo(JNIEnv *env, jobject obj, jint deviceId)
    {
        std::string info;

#ifdef USE_CUDA
        info = std::string(cuda_get_device_info());
#elif defined(USE_ROCM)
        info = std::string(hip_get_device_info());
#else
        info = "CPU-only build - no GPU support";
#endif

        return env->NewStringUTF(info.c_str());
    }

    JNIEXPORT jboolean JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_matrixMultiply(JNIEnv *env, jobject obj,
                                                                          jfloatArray matrixA, jfloatArray matrixB, jfloatArray matrixC,
                                                                          jint m, jint n, jint k)
    {

        // Get array elements
        jfloat *a = env->GetFloatArrayElements(matrixA, NULL);
        jfloat *b = env->GetFloatArrayElements(matrixB, NULL);
        jfloat *c = env->GetFloatArrayElements(matrixC, NULL);

        bool result = false;

#ifdef USE_CUDA
        result = cuda_matrix_multiply_wrapper(a, b, c, m, n, k) == 1;
#elif defined(USE_ROCM)
        result = hip_matrix_multiply_wrapper(a, b, c, m, n, k) == 1;
#else
        // CPU fallback - simple implementation
        for (int i = 0; i < m; i++)
        {
            for (int j = 0; j < n; j++)
            {
                float sum = 0.0f;
                for (int l = 0; l < k; l++)
                {
                    sum += a[i * k + l] * b[l * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        result = true;
#endif

        // Release array elements
        env->ReleaseFloatArrayElements(matrixA, a, 0);
        env->ReleaseFloatArrayElements(matrixB, b, 0);
        env->ReleaseFloatArrayElements(matrixC, c, 0);

        return result ? JNI_TRUE : JNI_FALSE;
    }

    JNIEXPORT jboolean JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_applySoftmax(JNIEnv *env, jobject obj,
                                                                        jfloatArray data, jint batchSize, jint numClasses)
    {

        jfloat *array = env->GetFloatArrayElements(data, NULL);
        bool result = false;

#ifdef USE_CUDA
        result = cuda_softmax_wrapper(array, batchSize, numClasses) == 1;
#elif defined(USE_ROCM)
        result = hip_softmax_wrapper(array, batchSize, numClasses) == 1;
#else
        // CPU fallback - softmax implementation
        for (int i = 0; i < batchSize; i++)
        {
            float *sample = array + i * numClasses;

            // Find max for numerical stability
            float max_val = sample[0];
            for (int j = 1; j < numClasses; j++)
            {
                if (sample[j] > max_val)
                    max_val = sample[j];
            }

            // Compute exponentials and sum
            float sum = 0.0f;
            for (int j = 0; j < numClasses; j++)
            {
                sample[j] = expf(sample[j] - max_val);
                sum += sample[j];
            }

            // Normalize
            for (int j = 0; j < numClasses; j++)
            {
                sample[j] /= sum;
            }
        }
        result = true;
#endif

        env->ReleaseFloatArrayElements(data, array, 0);
        return result ? JNI_TRUE : JNI_FALSE;
    }

    JNIEXPORT void JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_cleanup(JNIEnv *env, jobject obj)
    {
#ifdef USE_CUDA
        cuda_ops_cleanup();
#elif defined(USE_ROCM)
        hip_ops_cleanup();
#endif
    }

    JNIEXPORT jstring JNICALL
    Java_org_apache_opennlp_gpu_compute_GpuComputeProvider_getPlatformInfo(JNIEnv *env, jobject obj)
    {
        std::string platform_info;

#ifdef USE_CUDA
        platform_info = "NVIDIA CUDA";
#elif defined(USE_ROCM)
        platform_info = "AMD ROCm/HIP";
#else
        platform_info = "CPU-only";
#endif

        return env->NewStringUTF(platform_info.c_str());
    }

} // extern "C"
