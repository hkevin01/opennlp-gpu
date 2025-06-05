package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

/**
 * CPU-based implementation of feature extraction operations.
 * This class provides fallback implementations when GPU acceleration is not available.
 */
@Slf4j
@RequiredArgsConstructor
public class CpuFeatureExtractionOperation implements FeatureExtractionOperation {
    
    @Getter
    private final ComputeProvider provider;
    
    /**
     * Creates a new CPU feature extraction operation with the specified provider.
     *
     * @param provider the compute provider to use
     */
    public CpuFeatureExtractionOperation(ComputeProvider provider) {
        this.provider = provider;
        logger.info("Initializing CPU feature extraction with provider: {}", provider.getName());
    }
    
    @Override
    public int extractNGrams(int[] tokens, int numTokens, int maxNGramLength, int[] featureMap) {
        logger.debug("CPU extracting n-grams: {} tokens, max length {}", numTokens, maxNGramLength);
        
        int featureCount = 0;
        
        // For each n-gram length
        for (int ngramLength = 1; ngramLength <= maxNGramLength; ngramLength++) {
            // For each possible starting position
            for (int startPos = 0; startPos <= numTokens - ngramLength; startPos++) {
                // Compute feature index for this n-gram
                int featureIndex = 0;
                for (int i = 0; i < ngramLength; i++) {
                    featureIndex = 31 * featureIndex + tokens[startPos + i];
                }
                featureIndex = Math.abs(featureIndex) % featureMap.length;
                
                // Increment feature count
                featureMap[featureIndex]++;
                featureCount++;
            }
        }
        
        return featureCount;
    }
    
    @Override
    public void computeTfIdf(float[] termFreq, float[] docFreq, int numDocs, float[] tfidf, int numTerms) {
        logger.debug("CPU computing TF-IDF: {} terms, {} docs", numTerms, numDocs);
        
        for (int i = 0; i < numTerms; i++) {
            // Avoid division by zero
            if (docFreq[i] > 0) {
                float idf = (float) Math.log(numDocs / docFreq[i]);
                tfidf[i] = termFreq[i] * idf;
            } else {
                tfidf[i] = 0.0f;
            }
        }
    }
    
    @Override
    public void computeCosineSimilarity(float[] docVectors, int numDocs, int vectorSize, float[] similarities) {
        logger.debug("CPU computing cosine similarity: {} docs, vector size {}", numDocs, vectorSize);
        
        // Pre-compute document vector norms
        float[] norms = new float[numDocs];
        for (int docIdx = 0; docIdx < numDocs; docIdx++) {
            float norm = 0.0f;
            for (int termIdx = 0; termIdx < vectorSize; termIdx++) {
                float value = docVectors[docIdx * vectorSize + termIdx];
                norm += value * value;
            }
            norms[docIdx] = (float) Math.sqrt(norm);
        }
        
        // Compute pairwise similarities
        for (int i = 0; i < numDocs; i++) {
            // Set diagonal to 1.0 (self-similarity)
            similarities[i * numDocs + i] = 1.0f;
            
            // Compute similarity with other documents
            for (int j = i + 1; j < numDocs; j++) {
                float dotProduct = 0.0f;
                for (int termIdx = 0; termIdx < vectorSize; termIdx++) {
                    dotProduct += docVectors[i * vectorSize + termIdx] * docVectors[j * vectorSize + termIdx];
                }
                
                // Compute cosine similarity
                float similarity = 0.0f;
                if (norms[i] > 0 && norms[j] > 0) {
                    similarity = dotProduct / (norms[i] * norms[j]);
                }
                
                // Store the similarity (symmetric matrix)
                similarities[i * numDocs + j] = similarity;
                similarities[j * numDocs + i] = similarity;
            }
        }
    }
    
    @Override
    public void release() {
        logger.info("Releasing CPU feature extraction resources");
        // No resources to release for CPU implementation
    }
}
