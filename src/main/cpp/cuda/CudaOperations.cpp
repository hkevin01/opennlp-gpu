#include <jni.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <string>

// Forward declarations of CUDA kernels
extern void launchMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k);
extern void launchMatrixAdd(const float* A, const float* B, float* C, int size);
extern void launchMatrixSubtract(const float* A, const float* B, float* C, int size);
extern void launchMatrixScalarMultiply(const float* A, float* B, float scalar, int size);
extern void launchMatrixTranspose(const float* A, float* B, int rows, int cols);
extern void launchTfIdf(const float* termFreq, const float* docFreq, float* tfidf, int numTerms, int numDocs);
extern void launchCosineSimilarity(const float* vectors, float* similarities, int numVectors, int vectorSize);
extern int launchExtractNGrams(const int* tokens, int* featureCounts, int numTokens, int maxNGramLength, int vocabularySize);

// Error checking macro
#define CHECK_CUDA_ERROR(err)                                                                          \
    do                                                                                                 \
    {                                                                                                  \
        if (err != cudaSuccess)                                                                        \
        {                                                                                              \
            fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
            return;                                                                                    \
        }                                                                                              \
    } while (0)

// Error checking macro with return value
#define CHECK_CUDA_ERROR_RET(err, retval) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
        return retval; \
    } \
} while(0)

// JNI method implementations for CudaUtil class
extern "C"
{

    JNIEXPORT jboolean JNICALL Java_org_apache_opennlp_gpu_cuda_CudaUtil_initializeCuda(JNIEnv *env, jclass cls)
    {
        // Check if CUDA is available
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);

        if (error != cudaSuccess || deviceCount == 0)
        {
            return JNI_FALSE;
        }

        // CUDA is available
        return JNI_TRUE;
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_cuda_CudaUtil_getDeviceCount(JNIEnv *env, jclass cls)
    {
        int deviceCount;
        cudaError_t error = cudaGetDeviceCount(&deviceCount);

        if (error != cudaSuccess)
        {
            return 0;
        }

        return deviceCount;
    }

    JNIEXPORT jstring JNICALL Java_org_apache_opennlp_gpu_cuda_CudaUtil_getDeviceName(JNIEnv *env, jclass cls, jint deviceId)
    {
        cudaDeviceProp deviceProp;
        cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);

        if (error != cudaSuccess)
        {
            return env->NewStringUTF("Unknown");
        }

        return env->NewStringUTF(deviceProp.name);
    }

    JNIEXPORT jlong JNICALL Java_org_apache_opennlp_gpu_cuda_CudaUtil_getDeviceMemory(JNIEnv *env, jclass cls, jint deviceId)
    {
        cudaDeviceProp deviceProp;
        cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);

        if (error != cudaSuccess)
        {
            return 0;
        }

        return deviceProp.totalGlobalMem;
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_cuda_CudaUtil_getComputeCapability(JNIEnv *env, jclass cls, jint deviceId)
    {
        cudaDeviceProp deviceProp;
        cudaError_t error = cudaGetDeviceProperties(&deviceProp, deviceId);

        if (error != cudaSuccess)
        {
            return 0;
        }

        return deviceProp.major * 10 + deviceProp.minor;
    }

    // Resource management
    JNIEXPORT jlong JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_allocateDeviceMemory
      (JNIEnv *env, jobject obj, jlong size) {
        void* devicePtr = nullptr;
        cudaError_t error = cudaMalloc(&devicePtr, size);
        if (error != cudaSuccess) {
            return 0;
        }
        return (jlong)devicePtr;
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_freeDeviceMemory
      (JNIEnv *env, jobject obj, jlong devicePtr) {
        if (devicePtr != 0) {
            cudaFree((void*)devicePtr);
        }
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_copyHostToDevice
      (JNIEnv *env, jobject obj, jfloatArray hostArray, jlong devicePtr, jint size) {
        jfloat* hostData = env->GetFloatArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy((void*)devicePtr, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_CUDA_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_copyDeviceToHost
      (JNIEnv *env, jobject obj, jlong devicePtr, jfloatArray hostArray, jint size) {
        jfloat* hostData = env->GetFloatArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy(hostData, (void*)devicePtr, size * sizeof(float), cudaMemcpyDeviceToHost);
        env->ReleaseFloatArrayElements(hostArray, hostData, 0);
        CHECK_CUDA_ERROR(error);
    }

    // Matrix operations
    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_cudaMatrixMultiply
      (JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint rowsA, jint colsB, jint sharedDim) {
        launchMatrixMultiply((const float*)aPtr, (const float*)bPtr, (float*)cPtr, rowsA, colsB, sharedDim);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_cudaMatrixAdd
      (JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint size) {
        launchMatrixAdd((const float*)aPtr, (const float*)bPtr, (float*)cPtr, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_cudaMatrixSubtract
      (JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jlong cPtr, jint size) {
        launchMatrixSubtract((const float*)aPtr, (const float*)bPtr, (float*)cPtr, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_cudaMatrixScalarMultiply
      (JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jfloat scalar, jint size) {
        launchMatrixScalarMultiply((const float*)aPtr, (float*)bPtr, scalar, size);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaMatrixOperation_cudaMatrixTranspose
      (JNIEnv *env, jobject obj, jlong aPtr, jlong bPtr, jint rows, jint cols) {
        launchMatrixTranspose((const float*)aPtr, (float*)bPtr, rows, cols);
    }

    // Feature extraction operations
    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_copyIntHostToDevice
      (JNIEnv *env, jobject obj, jintArray hostArray, jlong devicePtr, jint size) {
        jint* hostData = env->GetIntArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy((void*)devicePtr, hostData, size * sizeof(int), cudaMemcpyHostToDevice);
        env->ReleaseIntArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_CUDA_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_copyIntDeviceToHost
      (JNIEnv *env, jobject obj, jlong devicePtr, jintArray hostArray, jint size) {
        jint* hostData = env->GetIntArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy(hostData, (void*)devicePtr, size * sizeof(int), cudaMemcpyDeviceToHost);
        env->ReleaseIntArrayElements(hostArray, hostData, 0);
        CHECK_CUDA_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_copyFloatHostToDevice
      (JNIEnv *env, jobject obj, jfloatArray hostArray, jlong devicePtr, jint size) {
        jfloat* hostData = env->GetFloatArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy((void*)devicePtr, hostData, size * sizeof(float), cudaMemcpyHostToDevice);
        env->ReleaseFloatArrayElements(hostArray, hostData, JNI_ABORT);
        CHECK_CUDA_ERROR(error);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_copyFloatDeviceToHost
      (JNIEnv *env, jobject obj, jlong devicePtr, jfloatArray hostArray, jint size) {
        jfloat* hostData = env->GetFloatArrayElements(hostArray, nullptr);
        cudaError_t error = cudaMemcpy(hostData, (void*)devicePtr, size * sizeof(float), cudaMemcpyDeviceToHost);
        env->ReleaseFloatArrayElements(hostArray, hostData, 0);
        CHECK_CUDA_ERROR(error);
    }

    JNIEXPORT jint JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_cudaExtractNGrams
      (JNIEnv *env, jobject obj, jlong tokensPtr, jint numTokens, jint maxNGramLength, jlong featureMapPtr, jint featureMapSize) {
        return launchExtractNGrams((const int*)tokensPtr, (int*)featureMapPtr, numTokens, maxNGramLength, featureMapSize);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_cudaComputeTfIdf
      (JNIEnv *env, jobject obj, jlong termFreqPtr, jlong docFreqPtr, jint numDocs, jlong tfidfPtr, jint numTerms) {
        launchTfIdf((const float*)termFreqPtr, (const float*)docFreqPtr, (float*)tfidfPtr, numTerms, numDocs);
    }

    JNIEXPORT void JNICALL Java_org_apache_opennlp_gpu_compute_CudaFeatureExtractionOperation_cudaComputeCosineSimilarity
      (JNIEnv *env, jobject obj, jlong docVectorsPtr, jint numDocs, jint vectorSize, jlong similaritiesPtr) {
        launchCosineSimilarity((const float*)docVectorsPtr, (float*)similaritiesPtr, numDocs, vectorSize);
    }

} // extern "C"
