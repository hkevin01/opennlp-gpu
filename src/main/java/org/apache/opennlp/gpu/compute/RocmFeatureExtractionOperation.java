package org.apache.opennlp.gpu.compute;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.rocm.RocmUtil;
import lombok.Getter;
import lombok.extern.slf4j.Slf4j;

/**
 * ROCm implementation of feature extraction operations.
 * This class uses AMD's ROCm platform for GPU-accelerated feature extraction.
 */
@Slf4j
public class RocmFeatureExtractionOperation implements FeatureExtractionOperation {
    private static final org.slf4j.Logger log = org.slf4j.LoggerFactory.getLogger(RocmFeatureExtractionOperation.class);
    
    @Getter
    private final ComputeProvider provider;
    private boolean initialized = false;
    private int deviceId = 0;
    
    // JNI method declarations for ROCm feature extraction operations
    private native long allocateDeviceMemory(long size);
    private native void freeDeviceMemory(long devicePtr);
    private native void copyIntHostToDevice(int[] hostArray, long devicePtr, int size);
    private native void copyIntDeviceToHost(long devicePtr, int[] hostArray, int size);
    private native void copyFloatHostToDevice(float[] hostArray, long devicePtr, int size);
    private native void copyFloatDeviceToHost(long devicePtr, float[] hostArray, int size);
    private native int rocmExtractNGrams(long tokensPtr, int numTokens, int maxNGramLength, long featureMapPtr, int featureMapSize);
    private native void rocmComputeTfIdf(long termFreqPtr, long docFreqPtr, int numDocs, long tfidfPtr, int numTerms);
    private native void rocmComputeCosineSimilarity(long docVectorsPtr, int numDocs, int vectorSize, long similaritiesPtr);
    
    /**
     * Creates a new ROCm feature extraction operation with the specified provider.
     *
     * @param provider the compute provider to use
     */
    public RocmFeatureExtractionOperation(ComputeProvider provider) {
        this.provider = provider;
        log.info("Initializing ROCm feature extraction with provider: {}", provider.getName());
        
        // Initialize ROCm
        if (!RocmUtil.isAvailable()) {
            throw new RuntimeException("ROCm is not available");
        }
        
        try {
            // Load the native library for ROCm feature extraction operations
            System.loadLibrary("opennlp_rocm_features");
            initialized = true;
            log.info("ROCm feature extraction operations initialized successfully");
        } catch (UnsatisfiedLinkError e) {
            log.error("Failed to load ROCm feature extraction library", e);
            throw new RuntimeException("Failed to initialize ROCm feature extraction operations", e);
        }
    }
    
    @Override
    public int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap) {
        if (!initialized) {
            throw new IllegalStateException("ROCm feature extraction operations not initialized");
        }
        
        log.debug("ROCm extracting n-grams: {} tokens, max length {}", numTokens, maxNGramLength);
        
        // Allocate device memory
        long tokensPtr = allocateDeviceMemory(numTokens * Integer.BYTES);
        long featureMapPtr = allocateDeviceMemory(featureMap.length * Integer.BYTES);
        
        try {
            // Copy input data to device
            copyIntHostToDevice(tokens, tokensPtr, numTokens);
            
            // Extract n-grams
            int numFeatures = rocmExtractNGrams(tokensPtr, numTokens, maxNGramLength, featureMapPtr, featureMap.length);
            
            // Copy result back to host
            copyIntDeviceToHost(featureMapPtr, featureMap, featureMap.length);
            
            return numFeatures;
        } finally {
            // Free device memory
            freeDeviceMemory(tokensPtr);
            freeDeviceMemory(featureMapPtr);
        }
    }
    
    @Override
    public void computeTfIdf(float[] termFreq, float[] docFreq, int numDocs, float[] tfidf, int numTerms) {
        if (!initialized) {
            throw new IllegalStateException("ROCm feature extraction operations not initialized");
        }
        
        log.debug("ROCm computing TF-IDF: {} terms, {} docs", numTerms, numDocs);
        
        // Allocate device memory
        long termFreqPtr = allocateDeviceMemory(numTerms * Float.BYTES);
        long docFreqPtr = allocateDeviceMemory(numTerms * Float.BYTES);
        long tfidfPtr = allocateDeviceMemory(numTerms * Float.BYTES);
        
        try {
            // Copy input data to device
            copyFloatHostToDevice(termFreq, termFreqPtr, numTerms);
            copyFloatHostToDevice(docFreq, docFreqPtr, numTerms);
            
            // Compute TF-IDF
            rocmComputeTfIdf(termFreqPtr, docFreqPtr, numDocs, tfidfPtr, numTerms);
            
            // Copy result back to host
            copyFloatDeviceToHost(tfidfPtr, tfidf, numTerms);
        } finally {
            // Free device memory
            freeDeviceMemory(termFreqPtr);
            freeDeviceMemory(docFreqPtr);
            freeDeviceMemory(tfidfPtr);
        }
    }
    
    @Override
    public void computeCosineSimilarity(float[] docVectors, int numDocs, int vectorSize, float[] similarities) {
        if (!initialized) {
            throw new IllegalStateException("ROCm feature extraction operations not initialized");
        }
        
        log.debug("ROCm computing cosine similarity: {} docs, vector size {}", numDocs, vectorSize);
        
        // Allocate device memory
        long docVectorsPtr = allocateDeviceMemory(numDocs * vectorSize * Float.BYTES);
        long similaritiesPtr = allocateDeviceMemory(numDocs * numDocs * Float.BYTES);
        
        try {
            // Copy input data to device
            copyFloatHostToDevice(docVectors, docVectorsPtr, numDocs * vectorSize);
            
            // Compute cosine similarity
            rocmComputeCosineSimilarity(docVectorsPtr, numDocs, vectorSize, similaritiesPtr);
            
            // Copy result back to host
            copyFloatDeviceToHost(similaritiesPtr, similarities, numDocs * numDocs);
        } finally {
            // Free device memory
            freeDeviceMemory(docVectorsPtr);
            freeDeviceMemory(similaritiesPtr);
        }
    }
    
    @Override
    public void release() {
        log.info("Releasing ROCm feature extraction resources");
        // No resources to release at this level
        // Native resources are managed per-operation
    }
}
