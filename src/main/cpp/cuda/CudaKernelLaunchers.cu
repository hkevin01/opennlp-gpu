#include <cuda_runtime.h>
#include <stdio.h>

// CUDA kernel declarations (defined in kernels.cu)
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
#define CHECK_CUDA_ERROR(err) do { \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error: %s at %s:%d\n", cudaGetErrorString(err), __FILE__, __LINE__); \
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
    
    matrixMultiplyKernel<<<gridDim, blockDim>>>(A, B, C, m, n, k);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchMatrixAdd(const float* A, const float* B, float* C, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    matrixAddKernel<<<gridDim, blockDim>>>(A, B, C, size);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchMatrixSubtract(const float* A, const float* B, float* C, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    matrixSubtractKernel<<<gridDim, blockDim>>>(A, B, C, size);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchMatrixScalarMultiply(const float* A, float* B, float scalar, int size) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(size);
    
    matrixScalarMultiplyKernel<<<gridDim, blockDim>>>(A, B, scalar, size);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchMatrixTranspose(const float* A, float* B, int rows, int cols) {
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim = getGrid2D(rows, cols);
    
    matrixTransposeKernel<<<gridDim, blockDim>>>(A, B, rows, cols);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchTfIdf(const float* termFreq, const float* docFreq, float* tfidf, int numTerms, int numDocs) {
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim = getGrid1D(numTerms);
    
    tfIdfKernel<<<gridDim, blockDim>>>(termFreq, docFreq, tfidf, numTerms, numDocs);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

void launchCosineSimilarity(const float* vectors, float* similarities, int numVectors, int vectorSize) {
    // Allocate device memory for normalized vectors and norms
    float* d_normalizedVectors;
    float* d_norms;
    
    cudaMalloc(&d_normalizedVectors, numVectors * vectorSize * sizeof(float));
    cudaMalloc(&d_norms, numVectors * sizeof(float));
    
    // Copy input vectors to device
    cudaMemcpy(d_normalizedVectors, vectors, numVectors * vectorSize * sizeof(float), cudaMemcpyDeviceToDevice);
    
    // Normalize vectors
    dim3 normBlockDim(THREADS_PER_BLOCK);
    dim3 normGridDim = getGrid1D(numVectors);
    normalizeVectorsKernel<<<normGridDim, normBlockDim>>>(d_normalizedVectors, d_norms, numVectors, vectorSize);
    
    // Compute cosine similarities
    dim3 simBlockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 simGridDim = getGrid2D(numVectors, numVectors);
    cosineSimilarityKernel<<<simGridDim, simBlockDim>>>(d_normalizedVectors, similarities, numVectors, vectorSize);
    
    // Free temporary device memory
    cudaFree(d_normalizedVectors);
    cudaFree(d_norms);
    
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
}

int launchExtractNGrams(const int* tokens, int* featureCounts, int numTokens, int maxNGramLength, int vocabularySize) {
    // Initialize feature counts to zero
    cudaMemset(featureCounts, 0, vocabularySize * sizeof(int));
    
    dim3 blockDim(THREADS_PER_BLOCK);
    dim3 gridDim(getGrid1D(numTokens).x, maxNGramLength);
    
    extractNGramsKernel<<<gridDim, blockDim>>>(tokens, featureCounts, numTokens, maxNGramLength, vocabularySize);
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Count non-zero features (this could be done on GPU for better performance)
    int* h_featureCounts = new int[vocabularySize];
    cudaMemcpy(h_featureCounts, featureCounts, vocabularySize * sizeof(int), cudaMemcpyDeviceToHost);
    
    int nonZeroFeatures = 0;
    for (int i = 0; i < vocabularySize; i++) {
        if (h_featureCounts[i] > 0) {
            nonZeroFeatures++;
        }
    }
    
    delete[] h_featureCounts;
    return nonZeroFeatures;
}
