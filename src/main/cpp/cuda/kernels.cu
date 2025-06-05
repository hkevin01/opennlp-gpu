#include <cuda_runtime.h>

/**
 * CUDA kernel for matrix multiplication: C = A * B
 * 
 * @param A Input matrix A (m x k)
 * @param B Input matrix B (k x n)
 * @param C Output matrix C (m x n)
 * @param m Number of rows in A and C
 * @param n Number of columns in B and C
 * @param k Number of columns in A and rows in B
 */
__global__ void matrixMultiplyKernel(const float* A, const float* B, float* C, int m, int n, int k) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < m && col < n) {
        float sum = 0.0f;
        for (int i = 0; i < k; i++) {
            sum += A[row * k + i] * B[i * n + col];
        }
        C[row * n + col] = sum;
    }
}

/**
 * CUDA kernel for matrix addition: C = A + B
 */
__global__ void matrixAddKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] + B[idx];
    }
}

/**
 * CUDA kernel for matrix subtraction: C = A - B
 */
__global__ void matrixSubtractKernel(const float* A, const float* B, float* C, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        C[idx] = A[idx] - B[idx];
    }
}

/**
 * CUDA kernel for scalar multiplication: B = A * scalar
 */
__global__ void matrixScalarMultiplyKernel(const float* A, float* B, float scalar, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        B[idx] = A[idx] * scalar;
    }
}

/**
 * CUDA kernel for matrix transpose: B = A^T
 */
__global__ void matrixTransposeKernel(const float* A, float* B, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        B[col * rows + row] = A[row * cols + col];
    }
}

/**
 * CUDA kernel for TF-IDF computation
 */
__global__ void tfIdfKernel(const float* termFreq, const float* docFreq, 
                            float* tfidf, int numTerms, int numDocs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numTerms) {
        // TF-IDF = term frequency * log(total docs / document frequency)
        float idf = logf((float)numDocs / (docFreq[idx] + 1.0f));
        tfidf[idx] = termFreq[idx] * idf;
    }
}

/**
 * CUDA kernel for vector normalization (used in cosine similarity)
 */
__global__ void normalizeVectorsKernel(float* vectors, float* norms, int numVectors, int vectorSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numVectors) {
        float sum = 0.0f;
        
        // Calculate Euclidean norm
        for (int i = 0; i < vectorSize; i++) {
            float val = vectors[idx * vectorSize + i];
            sum += val * val;
        }
        norms[idx] = sqrtf(sum);
        
        // Normalize the vector
        for (int i = 0; i < vectorSize; i++) {
            if (norms[idx] > 0.0f) {
                vectors[idx * vectorSize + i] /= norms[idx];
            }
        }
    }
}

/**
 * CUDA kernel for cosine similarity computation
 */
__global__ void cosineSimilarityKernel(const float* normalizedVectors, 
                                      float* similarities, 
                                      int numVectors, int vectorSize) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < numVectors && col < numVectors) {
        float dotProduct = 0.0f;
        
        // Calculate dot product of normalized vectors (cosine similarity)
        for (int i = 0; i < vectorSize; i++) {
            dotProduct += normalizedVectors[row * vectorSize + i] * 
                          normalizedVectors[col * vectorSize + i];
        }
        
        similarities[row * numVectors + col] = dotProduct;
    }
}

/**
 * CUDA kernel for n-gram extraction
 * A simplified version that counts n-grams
 */
__global__ void extractNGramsKernel(const int* tokens, int* featureCounts, 
                                   int numTokens, int maxNGramLength,
                                   int vocabularySize) {
    int tokenPos = blockIdx.x * blockDim.x + threadIdx.x;
    int nGramLength = blockIdx.y + 1; // 1-indexed n-gram length
    
    if (tokenPos < numTokens && nGramLength <= maxNGramLength && 
        tokenPos + nGramLength <= numTokens) {
        
        // This is a simplified version that just creates a hash for the n-gram
        // In a real implementation, you would use a more sophisticated feature mapping
        unsigned int hash = 0;
        for (int i = 0; i < nGramLength; i++) {
            int token = tokens[tokenPos + i];
            if (token >= 0 && token < vocabularySize) {
                hash = hash * 31 + token;
            }
        }
        
        // Use modulo to fit within feature map size
        hash = hash % vocabularySize;
        
        // Atomic increment to handle concurrent updates
        atomicAdd(&featureCounts[hash], 1);
    }
}
