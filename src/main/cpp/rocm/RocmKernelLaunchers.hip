#include <hip/hip_runtime.h>
#include <stdio.h>

// HIP kernel declarations (defined in kernels.hip)
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k);
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int size);
__global__ void matrixSubtractKernel(const float* A, const float* B, float* C, int size);
__global__ void matrixScalarMultiplyKernel(const float* A, float* B, float scalar, int size);
__global__ void matrixTransposeKernel(const float* A, float* B, int rows, int cols);
__global__ void tfIdfKernel(const float* termFreq, const float* docFreq, float* tfidf, int numTerms, int numDocs);
__global__ void normalizeVectorsKernel(float* vectors, float* norms, int numVectors, int vectorSize);
__global__ void cosineSimilarityKernel(const float* normalizedVectors, float* similarities, int numVectors, int vectorSize);
__global__ void extractNGramsKernel(const int* tokens, int* featureCounts, int numTokens, int maxNGramLength, int vocabularySize);

// Launch configurations and helpers
#define BLOCK_SIZE 16
#define THREADS_PER_BLOCK 256

// Error checking macro
#define CHECK_HIP_ERROR(err) do { \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error: %s at %s:%d\n", hipGetErrorString(err), __FILE__, __LINE__); \
    } \
} while(0)

// Calculate grid dimensions
inline dim3 getGrid1D(int size) {
    return dim3((size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
}

inline dim3 getGrid2D(int rows, int cols) {
    return dim3((cols + BLOCK_SIZE - 1) / BLOCK_SIZE, (rows + BLOCK_SIZE - 1) / BLOCK_SIZE);
}

// Kernel launchers
void launchMatrixMultiply(const float* A, const float* B, float* C, int m, int n, int k) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim = getGrid2D(m, n);
    
    hipLaunchKernelGGL(matrixMultiplyKernel, gridDim, blockDim, 0, 0, A, B, C, m, n, k);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchMatrixAdd(const float* A, const float* B, float* C, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    hipLaunchKernelGGL(matrixAddKernel, gridDim, blockDim, 0, 0, A, B, C, size);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchMatrixSubtract(const float* A, const float* B, float* C, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    hipLaunchKernelGGL(matrixSubtractKernel, gridDim, blockDim, 0, 0, A, B, C, size);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchMatrixScalarMultiply(const float* A, float* B, float scalar, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    hipLaunchKernelGGL(matrixScalarMultiplyKernel, gridDim, blockDim, 0, 0, A, B, scalar, size);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchMatrixTranspose(const float* A, float* B, int rows, int cols) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim = getGrid2D(rows, cols);
    
    hipLaunchKernelGGL(matrixTransposeKernel, gridDim, blockDim, 0, 0, A, B, rows, cols);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchTfIdf(const float* termFreq, const float* docFreq, float* tfidf, int numTerms, int numDocs) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(numTerms);
    
    hipLaunchKernelGGL(tfIdfKernel, gridDim, blockDim, 0, 0, termFreq, docFreq, tfidf, numTerms, numDocs);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

void launchCosineSimilarity(const float* vectors, float* similarities, int numVectors, int vectorSize) {
    // Allocate device memory for normalized vectors and norms
    float* d_normalizedVectors;
    float* d_norms;
    
    hipMalloc(&d_normalizedVectors, numVectors * vectorSize * sizeof(float));
    hipMalloc(&d_norms, numVectors * sizeof(float));
    
    // Copy input vectors to device
    hipMemcpy(d_normalizedVectors, vectors, numVectors * vectorSize * sizeof(float), hipMemcpyDeviceToDevice);
    
    // Normalize vectors
    dim3 normBlockDim(THREADS_PER_BLOCK);
    dim3 normGridDim = getGrid1D(numVectors);
    hipLaunchKernelGGL(normalizeVectorsKernel, normGridDim, normBlockDim, 0, 0, 
                    d_normalizedVectors, d_norms, numVectors, vectorSize);
    
    // Compute cosine similarities
    dim3 simBlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 simGridDim = getGrid2D(numVectors, numVectors);
    hipLaunchKernelGGL(cosineSimilarityKernel, simGridDim, simBlockDim, 0, 0, 
                    d_normalizedVectors, similarities, numVectors, vectorSize);
    
    // Free temporary device memory
    hipFree(d_normalizedVectors);
    hipFree(d_norms);
    
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
}

int launchExtractNGrams(const int* tokens, int* featureCounts, int numTokens, int maxNGramLength, int vocabularySize) {
    // Initialize feature counts to zero
    hipMemset(featureCounts, 0, vocabularySize * sizeof(int));
    
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(getGrid1D(numTokens).x, maxNGramLength);
    
    hipLaunchKernelGGL(extractNGramsKernel, gridDim, blockDim, 0, 0, 
                    tokens, featureCounts, numTokens, maxNGramLength, vocabularySize);
    hipDeviceSynchronize();
    CHECK_HIP_ERROR(hipGetLastError());
    
    // Count non-zero features (this could be done on GPU for better performance)
    int* h_featureCounts = new int[vocabularySize];
    hipMemcpy(h_featureCounts, featureCounts, vocabularySize * sizeof(int), hipMemcpyDeviceToHost);
    
    int nonZeroFeatures = 0;
    for (int i = 0; i < vocabularySize; i++) {
        if (h_featureCounts[i] > 0) {
            nonZeroFeatures++;
        }
    }
    
    delete[] h_featureCounts;
    return nonZeroFeatures;
}
