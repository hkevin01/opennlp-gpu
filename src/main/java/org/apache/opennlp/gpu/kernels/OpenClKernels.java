package org.apache.opennlp.gpu.kernels;

/**
 * OpenCL kernel source code for GPU operations
 * Contains optimized kernels for matrix operations and neural networks
 */
public class OpenClKernels {
    
    /**
     * Matrix multiplication kernel: C = A * B
     * Optimized for GPU parallelization
     */
    public static final String MATRIX_MULTIPLY_KERNEL = 
        "__kernel void matrixMultiply(\n" +
        "    __global const float* A,\n" +
        "    __global const float* B,\n" +
        "    __global float* C,\n" +
        "    const int M,\n" +
        "    const int N,\n" +
        "    const int K)\n" +
        "{\n" +
        "    int row = get_global_id(0);\n" +
        "    int col = get_global_id(1);\n" +
        "    \n" +
        "    if (row < M && col < N) {\n" +
        "        float sum = 0.0f;\n" +
        "        for (int k = 0; k < K; k++) {\n" +
        "            sum += A[row * K + k] * B[k * N + col];\n" +
        "        }\n" +
        "        C[row * N + col] = sum;\n" +
        "    }\n" +
        "}";
    
    /**
     * Optimized matrix multiplication with local memory
     * Uses work-group local memory for better performance
     */
    public static final String MATRIX_MULTIPLY_OPTIMIZED_KERNEL = 
        "#define TILE_SIZE 16\n" +
        "\n" +
        "__kernel void matrixMultiplyOptimized(\n" +
        "    __global const float* A,\n" +
        "    __global const float* B,\n" +
        "    __global float* C,\n" +
        "    const int M,\n" +
        "    const int N,\n" +
        "    const int K)\n" +
        "{\n" +
        "    __local float tileA[TILE_SIZE][TILE_SIZE];\n" +
        "    __local float tileB[TILE_SIZE][TILE_SIZE];\n" +
        "    \n" +
        "    int row = get_global_id(0);\n" +
        "    int col = get_global_id(1);\n" +
        "    int localRow = get_local_id(0);\n" +
        "    int localCol = get_local_id(1);\n" +
        "    \n" +
        "    float sum = 0.0f;\n" +
        "    \n" +
        "    for (int t = 0; t < K; t += TILE_SIZE) {\n" +
        "        // Load tile into local memory\n" +
        "        if (row < M && t + localCol < K) {\n" +
        "            tileA[localRow][localCol] = A[row * K + t + localCol];\n" +
        "        } else {\n" +
        "            tileA[localRow][localCol] = 0.0f;\n" +
        "        }\n" +
        "        \n" +
        "        if (t + localRow < K && col < N) {\n" +
        "            tileB[localRow][localCol] = B[(t + localRow) * N + col];\n" +
        "        } else {\n" +
        "            tileB[localRow][localCol] = 0.0f;\n" +
        "        }\n" +
        "        \n" +
        "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
        "        \n" +
        "        // Compute partial sum\n" +
        "        for (int k = 0; k < TILE_SIZE; k++) {\n" +
        "            sum += tileA[localRow][k] * tileB[k][localCol];\n" +
        "        }\n" +
        "        \n" +
        "        barrier(CLK_LOCAL_MEM_FENCE);\n" +
        "    }\n" +
        "    \n" +
        "    if (row < M && col < N) {\n" +
        "        C[row * N + col] = sum;\n" +
        "    }\n" +
        "}";
    
    /**
     * Element-wise matrix addition kernel: C = A + B
     */
    public static final String MATRIX_ADD_KERNEL = 
        "__kernel void matrixAdd(\n" +
        "    __global const float* A,\n" +
        "    __global const float* B,\n" +
        "    __global float* C,\n" +
        "    const int size)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < size) {\n" +
        "        C[gid] = A[gid] + B[gid];\n" +
        "    }\n" +
        "}";
    
    /**
     * Element-wise matrix subtraction kernel: C = A - B
     */
    public static final String MATRIX_SUB_KERNEL = 
        "__kernel void matrixSub(\n" +
        "    __global const float* A,\n" +
        "    __global const float* B,\n" +
        "    __global float* C,\n" +
        "    const int size)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < size) {\n" +
        "        C[gid] = A[gid] - B[gid];\n" +
        "    }\n" +
        "}";
    
    /**
     * Sigmoid activation function kernel
     */
    public static final String SIGMOID_KERNEL = 
        "__kernel void sigmoid(\n" +
        "    __global const float* input,\n" +
        "    __global float* output,\n" +
        "    const int size)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < size) {\n" +
        "        output[gid] = 1.0f / (1.0f + exp(-input[gid]));\n" +
        "    }\n" +
        "}";
    
    /**
     * ReLU activation function kernel
     */
    public static final String RELU_KERNEL = 
        "__kernel void relu(\n" +
        "    __global const float* input,\n" +
        "    __global float* output,\n" +
        "    const int size)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < size) {\n" +
        "        output[gid] = fmax(0.0f, input[gid]);\n" +
        "    }\n" +
        "}";
    
    /**
     * Tanh activation function kernel
     */
    public static final String TANH_KERNEL = 
        "__kernel void tanhActivation(\n" +
        "    __global const float* input,\n" +
        "    __global float* output,\n" +
        "    const int size)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < size) {\n" +
        "        output[gid] = tanh(input[gid]);\n" +
        "    }\n" +
        "}";
    
    /**
     * Softmax activation function kernel (two-pass implementation)
     * Pass 1: Find maximum value for numerical stability
     */
    public static final String SOFTMAX_MAX_KERNEL = 
        "__kernel void softmaxMax(\n" +
        "    __global const float* input,\n" +
        "    __global float* maxValues,\n" +
        "    const int batchSize,\n" +
        "    const int vectorSize)\n" +
        "{\n" +
        "    int batch = get_global_id(0);\n" +
        "    \n" +
        "    if (batch < batchSize) {\n" +
        "        float maxVal = input[batch * vectorSize];\n" +
        "        \n" +
        "        for (int i = 1; i < vectorSize; i++) {\n" +
        "            maxVal = fmax(maxVal, input[batch * vectorSize + i]);\n" +
        "        }\n" +
        "        \n" +
        "        maxValues[batch] = maxVal;\n" +
        "    }\n" +
        "}";
    
    /**
     * Softmax activation function kernel (two-pass implementation)
     * Pass 2: Compute softmax using max values
     */
    public static final String SOFTMAX_COMPUTE_KERNEL = 
        "__kernel void softmaxCompute(\n" +
        "    __global const float* input,\n" +
        "    __global const float* maxValues,\n" +
        "    __global float* output,\n" +
        "    const int batchSize,\n" +
        "    const int vectorSize)\n" +
        "{\n" +
        "    int batch = get_global_id(0);\n" +
        "    int idx = get_global_id(1);\n" +
        "    \n" +
        "    if (batch < batchSize && idx < vectorSize) {\n" +
        "        int globalIdx = batch * vectorSize + idx;\n" +
        "        float maxVal = maxValues[batch];\n" +
        "        \n" +
        "        // Compute exp(x - max) for numerical stability\n" +
        "        float expVal = exp(input[globalIdx] - maxVal);\n" +
        "        \n" +
        "        // Compute sum of exponentials\n" +
        "        float sum = 0.0f;\n" +
        "        for (int i = 0; i < vectorSize; i++) {\n" +
        "            sum += exp(input[batch * vectorSize + i] - maxVal);\n" +
        "        }\n" +
        "        \n" +
        "        output[globalIdx] = expVal / sum;\n" +
        "    }\n" +
        "}";
    
    /**
     * Vector dot product kernel for attention mechanisms
     */
    public static final String DOT_PRODUCT_KERNEL = 
        "__kernel void dotProduct(\n" +
        "    __global const float* A,\n" +
        "    __global const float* B,\n" +
        "    __global float* result,\n" +
        "    const int vectorSize,\n" +
        "    const int numVectors)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < numVectors) {\n" +
        "        float sum = 0.0f;\n" +
        "        \n" +
        "        for (int i = 0; i < vectorSize; i++) {\n" +
        "            sum += A[gid * vectorSize + i] * B[gid * vectorSize + i];\n" +
        "        }\n" +
        "        \n" +
        "        result[gid] = sum;\n" +
        "    }\n" +
        "}";
    
    /**
     * Vector normalization kernel (L2 norm)
     */
    public static final String VECTOR_NORMALIZE_KERNEL = 
        "__kernel void vectorNormalize(\n" +
        "    __global float* vectors,\n" +
        "    const int vectorSize,\n" +
        "    const int numVectors)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < numVectors) {\n" +
        "        float sum = 0.0f;\n" +
        "        \n" +
        "        // Calculate L2 norm\n" +
        "        for (int i = 0; i < vectorSize; i++) {\n" +
        "            float val = vectors[gid * vectorSize + i];\n" +
        "            sum += val * val;\n" +
        "        }\n" +
        "        \n" +
        "        float norm = sqrt(sum);\n" +
        "        \n" +
        "        // Normalize vector\n" +
        "        if (norm > 0.0f) {\n" +
        "            for (int i = 0; i < vectorSize; i++) {\n" +
        "                vectors[gid * vectorSize + i] /= norm;\n" +
        "            }\n" +
        "        }\n" +
        "    }\n" +
        "}";
    
    /**
     * TF-IDF feature extraction kernel
     */
    public static final String TFIDF_KERNEL = 
        "__kernel void tfidf(\n" +
        "    __global const float* termFreq,\n" +
        "    __global const float* idf,\n" +
        "    __global float* tfidf,\n" +
        "    const int numDocs,\n" +
        "    const int vocabSize)\n" +
        "{\n" +
        "    int doc = get_global_id(0);\n" +
        "    int term = get_global_id(1);\n" +
        "    \n" +
        "    if (doc < numDocs && term < vocabSize) {\n" +
        "        int idx = doc * vocabSize + term;\n" +
        "        tfidf[idx] = termFreq[idx] * idf[term];\n" +
        "    }\n" +
        "}";
    
    /**
     * N-gram extraction kernel for text processing
     */
    public static final String NGRAM_KERNEL = 
        "__kernel void extractNgrams(\n" +
        "    __global const int* tokens,\n" +
        "    __global int* ngrams,\n" +
        "    const int numTokens,\n" +
        "    const int ngramSize,\n" +
        "    const int maxNgrams)\n" +
        "{\n" +
        "    int gid = get_global_id(0);\n" +
        "    \n" +
        "    if (gid < maxNgrams && gid + ngramSize <= numTokens) {\n" +
        "        for (int i = 0; i < ngramSize; i++) {\n" +
        "            ngrams[gid * ngramSize + i] = tokens[gid + i];\n" +
        "        }\n" +
        "    }\n" +
        "}";
}
