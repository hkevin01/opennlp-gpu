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
import org.apache.opennlp.maxent.GisModel;
import org.apache.opennlp.maxent.MaxentModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;

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
    private static final int LARGE_MATRIX_SIZE = 2048;
    private static final int CONCURRENT_THREADS = 8;
    private static final int MEMORY_STRESS_ITERATIONS = 1000;
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
        long endTime = startTime + (STRESS_TEST_DURATION_SECONDS * 1000);
        
        int operationCount = 0;
        int errorCount = 0;
        long initialMemory = getUsedMemory();
        
        while (System.currentTimeMillis() < endTime) {
            try {
                // Perform various operations continuously
                operationCount++;
                
                // Matrix operations
                float[] matrix = createRandomMatrix(256 * 256);
                float[] result = new float[256 * 256];
                matrixOp.transpose(matrix, result, 256, 256);
                
                // Activation functions
                matrixOp.sigmoid(matrix, result, matrix.length);
                matrixOp.relu(result, matrix, result.length);
                
                // Feature extraction
                if (operationCount % 10 == 0) {
                    String[] docs = generateRandomDocuments(5);
                    float[][] features = featureExtractor.extractNGramFeatures(docs, 2, 50);
                    assertNotNull(features);
                }
                
                // Memory check every 100 operations
                if (operationCount % 100 == 0) {
                    long currentMemory = getUsedMemory();
                    long memoryGrowth = currentMemory - initialMemory;
                    
                    if (memoryGrowth > MAX_MEMORY_MB * 1024 * 1024) {
                        logger.warn("Memory growth detected: " + (memoryGrowth / 1024 / 1024) + "MB after " + operationCount + " operations");
                    }
                    
                    // Force GC periodically
                    if (operationCount % 500 == 0) {
                        System.gc();
                        Thread.sleep(10);
                    }
                }
                
            } catch (Exception e) {
                errorCount++;
                logger.debug("Operation " + operationCount + " failed: " + e.getMessage());
                
                // If error rate gets too high, fail the test
                if (errorCount > operationCount * 0.1) {
                    fail("Error rate exceeded 10% after " + operationCount + " operations");
                }
            }
            
            // Small delay to prevent overwhelming the system
            if (operationCount % 50 == 0) {
                try {
                    Thread.sleep(1);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    break;
                }
            }
        }
        
        long finalMemory = getUsedMemory();
        long totalTime = System.currentTimeMillis() - startTime;
        double operationsPerSecond = operationCount * 1000.0 / totalTime;
        
        logger.info("Long running stability test completed:");
        logger.info("  Duration: " + (totalTime / 1000.0) + " seconds");
        logger.info("  Total operations: " + operationCount);
        logger.info("  Error count: " + errorCount);
        logger.info("  Error rate: " + String.format("%.2f%%", (errorCount * 100.0 / operationCount)));
        logger.info("  Operations per second: " + String.format("%.2f", operationsPerSecond));
        logger.info("  Memory growth: " + ((finalMemory - initialMemory) / 1024 / 1024) + "MB");
        
        // Assertions
        assertTrue(operationCount > 1000, "Should complete at least 1000 operations");
        assertTrue(errorCount < operationCount * 0.05, "Error rate should be less than 5%");
        assertTrue(operationsPerSecond > 10, "Should maintain reasonable throughput");
        assertTrue((finalMemory - initialMemory) < MAX_MEMORY_MB * 1024 * 1024, 
                  "Memory growth should be contained");
    }
    
    @Test
    @DisplayName("ML Model Stress Test")
    @Timeout(value = 90, unit = TimeUnit.SECONDS)
    void testMlModelStress() {
        logger.info("Starting ML model stress test");
        
        try {
            // Test Perceptron under stress
            testPerceptronStress();
            
            // Test MaxEnt under stress
            testMaxEntStress();
            
        } catch (Exception e) {
            logger.error("ML model stress test failed: " + e.getMessage(), e);
            fail("ML model stress test failed: " + e.getMessage());
        }
    }
    
    private void testPerceptronStress() {
        logger.info("Testing Perceptron model under stress");
        
        GpuPerceptronModel perceptron = new GpuPerceptronModel(config, 0.01f, 100);
        
        try {
            // Generate large training dataset
            int numSamples = 5000;
            int numFeatures = 1000;
            float[][] features = new float[numSamples][numFeatures];
            int[] labels = new int[numSamples];
            
            for (int i = 0; i < numSamples; i++) {
                for (int j = 0; j < numFeatures; j++) {
                    features[i][j] = (float) (Math.random() * 2 - 1);
                }
                labels[i] = (features[i][0] + features[i][1] > 0) ? 1 : 0;
            }
            
            // Train multiple times to test stability
            for (int iteration = 0; iteration < 5; iteration++) {
                long startTime = System.currentTimeMillis();
                perceptron.train(features, labels);
                long trainingTime = System.currentTimeMillis() - startTime;
                
                logger.debug("Perceptron training iteration " + iteration + " completed in " + trainingTime + "ms");
                
                // Test prediction accuracy
                int correctPredictions = 0;
                for (int i = 0; i < Math.min(100, numSamples); i++) {
                    int prediction = perceptron.predict(features[i]);
                    if (prediction == labels[i]) {
                        correctPredictions++;
                    }
                }
                
                double accuracy = correctPredictions / 100.0;
                logger.debug("Perceptron accuracy: " + String.format("%.2f%%", accuracy * 100));
                
                assertTrue(accuracy > 0.6, "Perceptron should maintain reasonable accuracy under stress");
            }
            
        } finally {
            perceptron.cleanup();
        }
    }
    
    private void testMaxEntStress() {
        logger.info("Testing MaxEnt model under stress");
        
        try {
            // Create sample MaxEnt model
            String[] outcomes = {"class1", "class2", "class3"};
            String[] predLabels = new String[100];
            for (int i = 0; i < predLabels.length; i++) {
                predLabels[i] = "feature_" + i;
            }
            
            double[] parameters = new double[outcomes.length * predLabels.length];
            for (int i = 0; i < parameters.length; i++) {
                parameters[i] = Math.random() * 2 - 1;
            }
            
            MaxentModel cpuModel = new GisModel(outcomes, predLabels, parameters, 1, 0.0);
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
                
                double[][] batchResults = gpuModel.evalBatch(batchContexts);
                assertNotNull(batchResults);
                assertEquals(batchContexts.length, batchResults.length);
                
            } finally {
                gpuModel.cleanup();
            }
            
        } catch (Exception e) {
            logger.error("MaxEnt stress test failed: " + e.getMessage(), e);
            throw e;
        }
    }
    
    // Helper methods
    
    private float[] createRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = (float) (Math.random() * 2 - 1);
        }
        return matrix;
    }
    
    private String[] generateRandomDocuments(int count) {
        String[] docs = new String[count];
        String[] words = {"test", "document", "gpu", "acceleration", "matrix", "feature", "extraction", "performance"};
        
        for (int i = 0; i < count; i++) {
            StringBuilder doc = new StringBuilder();
            int wordCount = 5 + (int) (Math.random() * 10);
            for (int j = 0; j < wordCount; j++) {
                if (j > 0) doc.append(" ");
                doc.append(words[(int) (Math.random() * words.length)]);
            }
            docs[i] = doc.toString();
        }
        
        return docs;
    }
    
    private String[] generateRandomContext(String[] predLabels, int maxFeatures) {
        int numFeatures = 1 + (int) (Math.random() * maxFeatures);
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
