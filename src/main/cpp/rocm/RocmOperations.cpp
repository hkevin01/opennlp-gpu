#include <jni.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <string>

// Forward declarations of HIP kernels
extern void launchMatrixMultiply(const float *A, const float *B, float *C, int m, int n, int k);
extern void launchMatrixAdd(const float *A, const float *B, float *C, int size);
extern void launchMatrixSubtract(const float *A, const float *B, float *C, int size);
extern void launchMatrixScalarMultiply(const float *A, float *B, float scalar, int size);
extern void launchMatrixTranspose(const float *A, float *B, int rows, int cols);
extern void launchTfIdf(const float *termFreq, const float *docFreq, float *tfidf, int numTerms, int numDocs);
extern void launchCosineSimilarity(const float *vectors, float *similarities, int numVectors, int vectorSize);
extern int launchExtractNGrams(const int *tokens, int *featureCounts, int numTokens, int maxNGramLength, int vocabularySize);

// Error checking macro
#define CHECK_HIP_ERROR(err)                                                                         \
    do                                                                                               \
    {                                                                                                \
        if (err != hipSuccess)                                                                       \
        {                                                                                            \
            fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            return;                                                                                  \
        }                                                                                            \
    } while (0)

// Error checking macro with return value
#define CHECK_HIP_ERROR_RET(err, retval)                                                             \
    do                                                                                               \
    {                                                                                                \
        if (err != hipSuccess)                                                                       \
        {                                                                                            \
            fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
            return retval;                                                                           \
        }                                                                                            \
    } while (0)

// JNI method implementations for RocmUtil class
extern "C"
{

    JNIEXPORT jboolean JNICALL Java_org_apache_opennlp_gpu_rocm_RocmUtil_initializeRocm(JNIEnv *env, jclass cls)
    {
        // Check if ROCm/HIP is available
        int deviceCount;
        hipError_t error = hipGetDeviceCount(&deviceCount);

        if (error != hipSuccess || deviceCount == 0)
        {
            return JNI_FALSE;
        }

        // ROCm is available
        return JNI_TRUE;
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_rocm_RocmUtil_getDeviceCount(JNIEnv *env, jclass cls)
    {
        int deviceCount;
        hipError_t error = hipGetDeviceCount(&deviceCount);

        if (error != hipSuccess)
        {
            return 0;
        }

        return deviceCount;
    }

    JNIEXPORT jstring JNICALL Java_org_apache_opennlp_gpu_rocm_RocmUtil_getDeviceName(JNIEnv *env, jclass cls, jint deviceId)
    {
        hipDeviceProp_t deviceProp;
        hipError_t error = hipGetDeviceProperties(&deviceProp, deviceId);

        if (error != hipSuccess)
        {
            return env->NewStringUTF("Unknown");
        }

        return env->NewStringUTF(deviceProp.name);
    }

    JNIEXPORT jlong JNICALL Java_org_apache_opennlp_gpu_rocm_RocmUtil_getDeviceMemory(JNIEnv *env, jclass cls, jint deviceId)
    {
        hipDeviceProp_t deviceProp;
        hipError_t error = hipGetDeviceProperties(&deviceProp, deviceId);

        if (error != hipSuccess)
        {
            return 0;
        }

        return deviceProp.totalGlobalMem;
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_rocm_RocmUtil_getComputeCapability(JNIEnv *env, jclass cls, jint deviceId)
    {
        hipDeviceProp_t deviceProp;
        hipError_t error = hipGetDeviceProperties(&deviceProp, deviceId);

        if (error != hipSuccess)
        {
            return 0;
        }

        // HIP doesn't have the exact same concept as CUDA compute capability
        // Return a combination of major and minor version
        return deviceProp.major * 10 + deviceProp.minor;
    }

    // Resource management
    JNIEXPORT jlong JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_allocateDeviceMemory(JNIEnv *env, jobject obj, jlong size)
    {
        void *devicePtr = nullptr;
        hipError_t error = hipMalloc(&devicePtr, size);
        if (error != hipSuccess)
        {
            return 0;
        }
        return (jlong)devicePtr;
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_freeDeviceMemory(JNIEnv *env, jobject obj, jlong devicePtr)
    {
        if (devicePtr != 0)
        {
            hipFree((void *)devicePtr);
        }
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_copyHostToDevice(JNIEnv *env, jobject obj, jfloatArray hostArray, jlong devicePtr, jint size)
    {
        jfloat *hostData = env->GetFloatArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy((void *)devicePtr, hostData, size * sizeof(float), hipMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_HIP_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_copyDeviceToHost(JNIEnv *env, jobject obj, jlong devicePtr, jfloatArray hostArray, jint size)
    {
        jfloat *hostData = env->GetFloatArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy(hostData, (void *)devicePtr, size * sizeof(float), hipMemcpyDeviceToHost);
        env->ReleaseFloatArrayElements(hostArray, hostData, 0);
        CHECK_HIP_ERROR(error);
    }

    // Matrix operations
    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_rocmMatrixMultiply(JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint rowsA, jint colsB, jint sharedDim)
    {
        launchMatrixMultiply((const float *)aPtr, (const float *)bPtr, (float *)cPtr, rowsA, colsB, sharedDim);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_rocmMatrixAdd(JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint size)
    {
        launchMatrixAdd((const float *)aPtr, (const float *)bPtr, (float *)cPtr, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_rocmMatrixSubtract(JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint size)
    {
        launchMatrixSubtract((const float *)aPtr, (const float *)bPtr, (float *)cPtr, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_rocmMatrixScalarMultiply(JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jfloat scalar, jint size)
    {
        launchMatrixScalarMultiply((const float *)aPtr, (float *)bPtr, scalar, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmMatrixOperation_rocmMatrixTranspose(JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jint rows, jint cols)
    {
        launchMatrixTranspose((const float *)aPtr, (float *)bPtr, rows, cols);
    }

    // Feature extraction operations
    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_copyIntHostToDevice(JNIEnv *env, jobject obj, jintArray hostArray, jlong devicePtr, jint size)
    {
        jint *hostData = env->GetIntArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy((void *)devicePtr, hostData, size * sizeof(int), hipMemcpyHostToDevice);
        env->ReleaseIntArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_HIP_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_copyIntDeviceToHost(JNIEnv *env, jobject obj, jlong devicePtr, jintArray hostArray, jint size)
    {
        jint *hostData = env->GetIntArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy(hostData, (void *)devicePtr, size * sizeof(int), hipMemcpyDeviceToHost);
        env->ReleaseIntArrayElements(hostArray, hostData, 0);
        CHECK_HIP_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_copyFloatHostToDevice(JNIEnv *env, jobject obj, jfloatArray hostArray, jlong devicePtr, jint size)
    {
        jfloat *hostData = env->GetFloatArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy((void *)devicePtr, hostData, size * sizeof(float), hipMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_HIP_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_copyFloatDeviceToHost(JNIEnv *env, jobject obj, jlong devicePtr, jfloatArray hostArray, jint size)
    {
        jfloat *hostData = env->GetFloatArrayElements(hostArray, nullptr);
        hipError_t error = hipMemcpy(hostData, (void *)devicePtr, size * sizeof(float), hipMemcpyDeviceToHost);
        env->ReleaseFloatArrayElements(hostArray, hostData, 0);
        CHECK_HIP_ERROR(error);
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_rocmExtractNGrams(JNIEnv *env, jobject obj, jlong tokensPtr, jint numTokens, jint maxNGramLength, jlong featureMapPtr, jint featureMapSize)
    {
        return launchExtractNGrams((const int *)tokensPtr, (int *)featureMapPtr, numTokens, maxNGramLength, featureMapSize);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_rocmComputeTfIdf(JNIEnv *env, jobject obj, jlong termFreqPtr, jlong docFreqPtr, jint numDocs, jlong tfidfPtr, jint numTerms)
    {
        launchTfIdf((const float *)termFreqPtr, (const float *)docFreqPtr, (float *)tfidfPtr, numTerms, numDocs);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_RocmFeatureExtractionOperation_rocmComputeCosineSimilarity(JNIEnv *env, jobject obj, jlong docVectorsPtr, jint numDocs, jint vectorSize, jlong similaritiesPtr)
    {
        launchCosineSimilarity((const float *)docVectorsPtr, (float *)similaritiesPtr, numDocs, vectorSize);
    }

} // extern "C"
