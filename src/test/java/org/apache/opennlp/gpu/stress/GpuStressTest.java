package org.apache.opennlp.gpu.stress;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.gpu.ml.perceptron.GpuPerceptronModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * Comprehensive stress testing for GPU acceleration components
 * Tests memory management, concurrent access, large datasets, and resource limits
 */
public class GpuStressTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuStressTest.class);
    
    private GpuConfig config;
    private MatrixOperation matrixOp;
    private GpuFeatureExtractor featureExtractor;
    private ExecutorService executorService;
    
    // Test configuration constants
    private static final int STRESS_TEST_DURATION_SECONDS = 30;
    private static final int LARGE_MATRIX_SIZE = 512;
    private static final int CONCURRENT_THREADS = 8;
    private static final int MEMORY_STRESS_ITERATIONS = 10;
    private static final long MAX_MEMORY_MB = 1024; // 1GB limit
    
    @BeforeEach
    void setUp() {
        config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setMemoryPoolSizeMB(512);
        config.setBatchSize(64);
        
        // Initialize compute infrastructure
        try {
            if (GpuComputeProvider.isGpuAvailable()) {
                GpuComputeProvider provider = new GpuComputeProvider(config);
                matrixOp = new GpuMatrixOperation(provider, config);
                featureExtractor = new GpuFeatureExtractor(provider, config, matrixOp);
                logger.info("Stress testing with GPU acceleration");
            } else {
                CpuComputeProvider provider = new CpuComputeProvider();
                matrixOp = new CpuMatrixOperation(provider);
                featureExtractor = new GpuFeatureExtractor(provider, config, matrixOp);
                logger.info("GPU not available, stress testing with CPU fallback");
            }
        } catch (Exception e) {
            logger.error("Failed to initialize compute provider: " + e.getMessage());
            // Fall back to CPU
            CpuComputeProvider provider = new CpuComputeProvider();
            matrixOp = new CpuMatrixOperation(provider);
            featureExtractor = new GpuFeatureExtractor(provider, config, matrixOp);
        }
        
        executorService = Executors.newFixedThreadPool(CONCURRENT_THREADS);
    }
    
    @AfterEach
    void tearDown() {
        if (executorService != null) {
            executorService.shutdown();
            try {
                if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            } catch (InterruptedException e) {
                executorService.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        if (featureExtractor != null) {
            featureExtractor.release();
        }
        if (matrixOp != null) {
            matrixOp.release();
        }
    }
    
    @Test
    @DisplayName("Memory Stress Test - Large Matrix Operations")
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void testMemoryStressLargeMatrices() {
        logger.info("Starting memory stress test with large matrices");
        
        long initialMemory = getUsedMemory();
        int successfulOperations = 0;
        List<Exception> errors = new ArrayList<>();
        
        for (int i = 0; i < MEMORY_STRESS_ITERATIONS; i++) {
            try {
                // Create large matrices
                float[] matrixA = createRandomMatrix(LARGE_MATRIX_SIZE * LARGE_MATRIX_SIZE);
                float[] matrixB = createRandomMatrix(LARGE_MATRIX_SIZE * LARGE_MATRIX_SIZE);
                float[] result = new float[LARGE_MATRIX_SIZE * LARGE_MATRIX_SIZE];
                
                // Perform matrix multiplication
                matrixOp.multiply(matrixA, matrixB, result, LARGE_MATRIX_SIZE, LARGE_MATRIX_SIZE, LARGE_MATRIX_SIZE);
                
                // Check memory usage
                long currentMemory = getUsedMemory();
                long memoryDelta = currentMemory - initialMemory;
                
                if (memoryDelta > MAX_MEMORY_MB * 1024 * 1024) {
                    logger.warn("Memory usage exceeded limit: " + (memoryDelta / 1024 / 1024) + "MB");
                }
                
                successfulOperations++;
                
                // Force garbage collection periodically
                if (i % 100 == 0) {
                    System.gc();
                    Thread.sleep(10);
                }
                
            } catch (Exception e) {
                errors.add(e);
                logger.debug("Operation " + i + " failed: " + e.getMessage());
            }
        }
        
        long finalMemory = getUsedMemory();
        long totalMemoryGrowth = finalMemory - initialMemory;
        
        logger.info("Memory stress test completed:");
        logger.info("  Successful operations: " + successfulOperations + "/" + MEMORY_STRESS_ITERATIONS);
        logger.info("  Total memory growth: " + (totalMemoryGrowth / 1024 / 1024) + "MB");
        logger.info("  Error count: " + errors.size());
        
        // Assertions
        assertTrue(successfulOperations > MEMORY_STRESS_ITERATIONS * 0.9, 
                  "At least 90% of operations should succeed");
        assertTrue(totalMemoryGrowth < MAX_MEMORY_MB * 1024 * 1024 * 2, 
                  "Memory growth should not exceed 2x the limit");
        assertTrue(errors.size() < MEMORY_STRESS_ITERATIONS * 0.1, 
                  "Error rate should be less than 10%");
    }
    
    @Test
    @DisplayName("Concurrent Access Stress Test")
    @Timeout(value = 45, unit = TimeUnit.SECONDS)
    void testConcurrentAccessStress() {
        logger.info("Starting concurrent access stress test");
        
        AtomicInteger successfulOperations = new AtomicInteger(0);
        AtomicInteger failedOperations = new AtomicInteger(0);
        AtomicLong totalExecutionTime = new AtomicLong(0);
        CountDownLatch latch = new CountDownLatch(CONCURRENT_THREADS);
        
        // Launch concurrent threads
        for (int threadId = 0; threadId < CONCURRENT_THREADS; threadId++) {
            final int id = threadId;
            executorService.submit(() -> {
                try {
                    for (int i = 0; i < 100; i++) {
                        long startTime = System.nanoTime();
                        
                        try {
                            // Perform various matrix operations concurrently
                            float[] matrixA = createRandomMatrix(512 * 512);
                            float[] matrixB = createRandomMatrix(512 * 512);
                            float[] result = new float[512 * 512];
                            
                            matrixOp.multiply(matrixA, matrixB, result, 512, 512, 512);
                            
                            // Feature extraction
                            String[] documents = {
                                "thread " + id + " document " + i,
                                "concurrent test sample " + (i % 10),
                                "stress testing gpu operations"
                            };
                            float[][] features = featureExtractor.extractNGramFeatures(documents, 2, 100);
                            
                            long executionTime = System.nanoTime() - startTime;
                            totalExecutionTime.addAndGet(executionTime);
                            successfulOperations.incrementAndGet();
                            
                        } catch (Exception e) {
                            failedOperations.incrementAndGet();
                            logger.debug("Thread " + id + " operation " + i + " failed: " + e.getMessage());
                        }
                    }
                } finally {
                    latch.countDown();
                }
            });
        }
        
        // Wait for all threads to complete
        try {
            assertTrue(latch.await(40, TimeUnit.SECONDS), "All threads should complete within timeout");
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Test interrupted");
        }
        
        int totalOperations = successfulOperations.get() + failedOperations.get();
        double averageExecutionTime = totalExecutionTime.get() / (double) successfulOperations.get() / 1_000_000; // ms
        
        logger.info("Concurrent access stress test completed:");
        logger.info("  Total operations: " + totalOperations);
        logger.info("  Successful operations: " + successfulOperations.get());
        logger.info("  Failed operations: " + failedOperations.get());
        logger.info("  Success rate: " + String.format("%.2f%%", (successfulOperations.get() * 100.0 / totalOperations)));
        logger.info("  Average execution time: " + String.format("%.2f ms", averageExecutionTime));
        
        // Assertions
        assertTrue(successfulOperations.get() > totalOperations * 0.95, 
                  "At least 95% of concurrent operations should succeed");
        assertTrue(failedOperations.get() < totalOperations * 0.05, 
                  "Failure rate should be less than 5%");
        assertTrue(averageExecutionTime < 1000, 
                  "Average execution time should be reasonable (<1s)");
    }
    
    @Test
    @DisplayName("Long Running Stability Test")
    @Timeout(value = 60, unit = TimeUnit.SECONDS)
    void testLongRunningStability() {
        logger.info("Starting long running stability test");
        
        long startTime = System.currentTimeMillis();
        long totalOperations = 0;
        List<Exception> errors = new ArrayList<>();
        
        while (System.currentTimeMillis() - startTime < STRESS_TEST_DURATION_SECONDS * 1000) {
            try {
                // Alternate between different operations
                if (totalOperations % 2 == 0) {
                    float[] matrix = createRandomMatrix(1024 * 1024);
                    matrixOp.add(matrix, matrix, matrix, 1024 * 1024);
                } else {
                    String[] docs = generateRandomDocuments(10);
                    featureExtractor.extractNGramFeatures(docs, 2, 100);
                }
                totalOperations++;
            } catch (Exception e) {
                errors.add(e);
            }
        }
        
        logger.info("Long running stability test completed:");
        logger.info("  Total operations: " + totalOperations);
        logger.info("  Error count: " + errors.size());
        
        assertTrue(errors.isEmpty(), "No errors should occur during long running test");
    }
    
    @Test
    @DisplayName("ML Model Stress Test")
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void testMlModelStress() {
        testPerceptronStress();
        testMaxEntStress();
    }
    
    private void testPerceptronStress() {
        System.out.println("--- Stress testing Perceptron model ---");
        
        // Create a large Perceptron model
        GpuPerceptronModel perceptron = new GpuPerceptronModel(config, 0.01f, 10);
        
        // Generate a large dataset
        int numSamples = 10000;
        int numFeatures = 2000;
        float[][] features = new float[numSamples][numFeatures];
        int[] labels = new int[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                features[i][j] = (float) (Math.random() * 2.0 - 1.0);
            }
            labels[i] = (int) (Math.random() * 2);
        }
        
        try {
            // Train for a few epochs
            for (int epoch = 0; epoch < 3; epoch++) {
                perceptron.train(features, labels);
            }
            
            // Perform batch predictions
            int batchSize = 128;
            for (int i = 0; i < numSamples; i += batchSize) {
                int end = Math.min(i + batchSize, numSamples);
                float[][] batchFeatures = new float[end - i][numFeatures];
                System.arraycopy(features, i, batchFeatures, 0, end - i);
                
                perceptron.predictBatch(batchFeatures);
            }
            
        } finally {
            perceptron.cleanup();
        }
    }
    
    private void testMaxEntStress() {
        System.out.println("--- Stress testing MaxEnt model ---");

        // Create a large, realistic MaxEnt model
        int numOutcomes = 20;
        int numPreds = 10000;
        String[] outcomes = new String[numOutcomes];
        for (int i = 0; i < numOutcomes; i++) {
            outcomes[i] = "outcome" + i;
        }

        String[] predLabels = new String[numPreds];
        for (int i = 0; i < numPreds; i++) {
            predLabels[i] = "pred" + i;
        }

        double[] params = new double[numOutcomes * numPreds];
        for (int i = 0; i < params.length; i++) {
            params[i] = Math.random() * 2.0 - 1.0;
        }
        
        Context[] contexts = new Context[predLabels.length];
        int[] outcomePattern = new int[outcomes.length];
        for (int i = 0; i < outcomes.length; i++) {
            outcomePattern[i] = i;
        }

        for (int i = 0; i < predLabels.length; i++) {
            double[] paramsForPred = new double[outcomes.length];
            for (int j = 0; j < outcomes.length; j++) {
                paramsForPred[j] = params[i * outcomes.length + j];
            }
            contexts[i] = new Context(outcomePattern, paramsForPred);
        }

        MaxentModel cpuModel = new GISModel(contexts, predLabels, outcomes);

        // Create GPU-accelerated wrapper
        GpuMaxentModel gpuModel = new GpuMaxentModel(cpuModel, config);

        try {
            // Stress test with many evaluations
            for (int iteration = 0; iteration < 1000; iteration++) {
                String[] context = generateRandomContext(predLabels, 10);
                double[] probs = gpuModel.eval(context);
                
                assertNotNull(probs);
                assertEquals(outcomes.length, probs.length);
                
                // Verify probabilities sum to approximately 1
                double sum = 0;
                for (double prob : probs) {
                    sum += prob;
                    assertTrue(prob >= 0, "Probabilities should be non-negative");
                }
                assertTrue(Math.abs(sum - 1.0) < 0.01, "Probabilities should sum to approximately 1");
            }
            
            // Test batch evaluation under stress
            String[][] batchContexts = new String[50][];
            for (int i = 0; i < batchContexts.length; i++) {
                batchContexts[i] = generateRandomContext(predLabels, 5);
            }
            
            // Test individual evaluations instead of batch
            for (int i = 0; i < batchContexts.length; i++) {
                double[] probs = gpuModel.eval(batchContexts[i]);
                assertNotNull(probs);
                assertEquals(outcomes.length, probs.length);
            }
            
        } finally {
            gpuModel.cleanup();
        }
    }
    
    private float[] createRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = (float) (Math.random() * 2.0 - 1.0);
        }
        return matrix;
    }
    
    private String[] generateRandomDocuments(int count) {
        String[] docs = new String[count];
        for (int i = 0; i < count; i++) {
            StringBuilder doc = new StringBuilder();
            int numWords = 10 + (int) (Math.random() * 20);
            for (int j = 0; j < numWords; j++) {
                doc.append("word").append((int) (Math.random() * 1000)).append(" ");
            }
            docs[i] = doc.toString();
        }
        return docs;
    }
    
    private String[] generateRandomContext(String[] predLabels, int maxFeatures) {
        int numFeatures = 1 + (int) (Math.random() * (maxFeatures - 1));
        String[] context = new String[numFeatures];
        for (int i = 0; i < numFeatures; i++) {
            context[i] = predLabels[(int) (Math.random() * predLabels.length)];
        }
        return context;
    }
    
    private long getUsedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
} 