package org.apache.opennlp.gpu.stress;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.*;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Memory stress tests for GPU acceleration components
 * Tests memory leaks, large allocations, and resource management under stress
 */
@EnabledIf("isStressTestingEnabled")
public class MemoryStressTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(MemoryStressTest.class);
    
    private GpuConfig config;
    private ComputeProvider gpuProvider;
    private ComputeProvider cpuProvider;
    private MatrixOperation gpuMatrixOp;
    private MatrixOperation cpuMatrixOp;
    
    // Test configuration
    private static final int STRESS_ITERATIONS = 100;
    private static final int LARGE_MATRIX_SIZE = 2000;
    private static final int MAX_DOCUMENTS = 5000;
    
    /**
     * Check if stress testing is enabled
     */
    public static boolean isStressTestingEnabled() {
        return "true".equals(System.getProperty("gpu.stress.test.enabled", "false"));
    }
    
    @BeforeEach
    public void setUp() {
        config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setDebugMode(true);
        
        try {
            gpuProvider = new GpuComputeProvider(config);
            cpuProvider = new CpuComputeProvider();
            
            gpuMatrixOp = new GpuMatrixOperation(gpuProvider, config);
            cpuMatrixOp = new CpuMatrixOperation(cpuProvider);
            
            logger.info("Memory stress test setup completed");
        } catch (Exception e) {
            logger.warn("GPU not available for stress testing, using CPU fallback");
            gpuProvider = new CpuComputeProvider();
            gpuMatrixOp = new CpuMatrixOperation(gpuProvider);
        }
    }
    
    @AfterEach
    public void tearDown() {
        if (gpuMatrixOp != null) {
            gpuMatrixOp.release();
        }
        if (cpuMatrixOp != null) {
            cpuMatrixOp.release();
        }
        if (gpuProvider != null) {
            gpuProvider.cleanup();
        }
        if (cpuProvider != null) {
            cpuProvider.cleanup();
        }
        
        // Force garbage collection to help detect memory leaks
        System.gc();
        System.runFinalization();
        System.gc();
        
        logger.info("Memory stress test cleanup completed");
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testLargeMatrixOperationsMemoryUsage() {
        logger.info("Testing large matrix operations memory usage...");
        
        long initialMemory = getUsedMemory();
        List<Long> memorySnapshots = new ArrayList<>();
        
        for (int iteration = 0; iteration < STRESS_ITERATIONS; iteration++) {
            // Create large matrices
            int size = LARGE_MATRIX_SIZE + (iteration % 500); // Vary size slightly
            float[] matrixA = generateRandomMatrix(size * size);
            float[] matrixB = generateRandomMatrix(size * size);
            float[] result = new float[size * size];
            
            // Perform GPU operation
            gpuMatrixOp.multiply(matrixA, matrixB, result, size, size, size);
            
            // Take memory snapshot every 10 iterations
            if (iteration % 10 == 0) {
                long currentMemory = getUsedMemory();
                memorySnapshots.add(currentMemory);
                logger.debug("Iteration {}: Memory usage: {} MB", 
                           iteration, currentMemory / (1024 * 1024));
            }
            
            // Clear references to help GC
            matrixA = null;
            matrixB = null;
            result = null;
            
            // Periodic GC to detect leaks
            if (iteration % 50 == 0) {
                System.gc();
            }
        }
        
        // Final memory check
        System.gc();
        Thread.yield();
        System.gc();
        
        long finalMemory = getUsedMemory();
        long memoryIncrease = finalMemory - initialMemory;
        long maxAcceptableIncrease = 500 * 1024 * 1024; // 500MB acceptable increase
        
        logger.info("Memory usage - Initial: {} MB, Final: {} MB, Increase: {} MB",
                   initialMemory / (1024 * 1024), 
                   finalMemory / (1024 * 1024),
                   memoryIncrease / (1024 * 1024));
        
        // Analyze memory trend
        analyzeMemoryTrend(memorySnapshots);
        
        // Assert memory increase is within acceptable bounds
        if (memoryIncrease > maxAcceptableIncrease) {
            throw new AssertionError("Memory increase " + (memoryIncrease / (1024 * 1024)) + 
                                   " MB exceeds acceptable limit of " + (maxAcceptableIncrease / (1024 * 1024)) + " MB");
        }
        
        logger.info("Large matrix operations memory test passed");
    }
    
    @Test
    @Timeout(600) // 10 minutes timeout
    public void testFeatureExtractionMemoryScaling() {
        logger.info("Testing feature extraction memory scaling...");
        
        GpuFeatureExtractor extractor = new GpuFeatureExtractor(gpuProvider, config, gpuMatrixOp);
        
        try {
            long initialMemory = getUsedMemory();
            
            // Test with increasing document counts
            int[] documentCounts = {100, 500, 1000, 2000, MAX_DOCUMENTS};
            
            for (int docCount : documentCounts) {
                logger.info("Testing feature extraction with {} documents", docCount);
                
                long beforeExtraction = getUsedMemory();
                
                // Generate test documents
                String[] documents = generateTestDocuments(docCount);
                
                // Extract features
                float[][] features = extractor.extractNGramFeatures(documents, 2, 1000);
                
                long afterExtraction = getUsedMemory();
                long extractionMemory = afterExtraction - beforeExtraction;
                
                logger.info("Documents: {}, Memory for extraction: {} MB", 
                           docCount, extractionMemory / (1024 * 1024));
                
                // Validate memory scaling is reasonable
                long expectedMemoryPerDoc = 50 * 1024; // 50KB per document estimate
                long maxExpectedMemory = docCount * expectedMemoryPerDoc * 10; // 10x safety factor
                
                if (extractionMemory > maxExpectedMemory) {
                    throw new AssertionError("Memory usage " + (extractionMemory / (1024 * 1024)) + 
                                           " MB for " + docCount + " documents exceeds expected scaling");
                }
                
                // Clear references
                documents = null;
                features = null;
                System.gc();
            }
            
            long finalMemory = getUsedMemory();
            logger.info("Feature extraction scaling test completed. Memory increase: {} MB",
                       (finalMemory - initialMemory) / (1024 * 1024));
            
        } finally {
            extractor.release();
        }
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testConcurrentMemoryAccess() {
        logger.info("Testing concurrent memory access patterns...");
        
        int threadCount = 4;
        int operationsPerThread = 25;
        
        Thread[] threads = new Thread[threadCount];
        Exception[] threadExceptions = new Exception[threadCount];
        long initialMemory = getUsedMemory();
        
        // Create concurrent threads performing matrix operations
        for (int i = 0; i < threadCount; i++) {
            final int threadId = i;
            threads[i] = new Thread(() -> {
                try {
                    // Each thread gets its own matrix operation instance for thread safety
                    ComputeProvider provider = new CpuComputeProvider();
                    MatrixOperation matrixOp = new CpuMatrixOperation(provider);
                    
                    try {
                        for (int op = 0; op < operationsPerThread; op++) {
                            int size = 500 + (threadId * 100); // Different sizes per thread
                            
                            float[] a = generateRandomMatrix(size * size);
                            float[] b = generateRandomMatrix(size * size);
                            float[] result = new float[size * size];
                            
                            matrixOp.multiply(a, b, result, size, size, size);
                            
                            // Simulate some processing time
                            Thread.sleep(10);
                            
                            logger.debug("Thread {} completed operation {}", threadId, op);
                        }
                    } finally {
                        matrixOp.release();
                        provider.cleanup();
                    }
                    
                } catch (Exception e) {
                    threadExceptions[threadId] = e;
                    logger.error("Thread {} failed: {}", threadId, e.getMessage());
                }
            });
        }
        
        // Start all threads
        long startTime = System.currentTimeMillis();
        for (Thread thread : threads) {
            thread.start();
        }
        
        // Wait for completion
        try {
            for (Thread thread : threads) {
                thread.join();
            }
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Test interrupted", e);
        }
        
        long endTime = System.currentTimeMillis();
        long duration = endTime - startTime;
        
        // Check for thread exceptions
        for (int i = 0; i < threadCount; i++) {
            if (threadExceptions[i] != null) {
                throw new AssertionError("Thread " + i + " failed", threadExceptions[i]);
            }
        }
        
        // Check final memory state
        System.gc();
        long finalMemory = getUsedMemory();
        long memoryIncrease = finalMemory - initialMemory;
        
        logger.info("Concurrent memory test completed in {} ms. Memory increase: {} MB",
                   duration, memoryIncrease / (1024 * 1024));
        
        // Assert reasonable memory usage
        long maxAcceptableIncrease = 200 * 1024 * 1024; // 200MB
        if (memoryIncrease > maxAcceptableIncrease) {
            throw new AssertionError("Concurrent memory test used too much memory: " + 
                                   (memoryIncrease / (1024 * 1024)) + " MB");
        }
    }
    
    @Test
    @Timeout(180) // 3 minutes timeout
    public void testMemoryFragmentation() {
        logger.info("Testing memory fragmentation patterns...");
        
        long initialMemory = getUsedMemory();
        List<float[]> allocatedArrays = new ArrayList<>();
        
        try {
            // Allocate many small matrices to test fragmentation
            for (int i = 0; i < 1000; i++) {
                int size = 100 + (i % 50); // Varying sizes from 100 to 150
                float[] matrix = generateRandomMatrix(size * size);
                allocatedArrays.add(matrix);
                
                // Perform operation on every 10th matrix
                if (i % 10 == 0 && allocatedArrays.size() >= 2) {
                    float[] a = allocatedArrays.get(allocatedArrays.size() - 2);
                    float[] b = allocatedArrays.get(allocatedArrays.size() - 1);
                    int matrixSize = (int) Math.sqrt(a.length);
                    float[] result = new float[a.length];
                    
                    gpuMatrixOp.add(a, b, result, a.length);
                }
                
                // Randomly free some matrices to create fragmentation
                if (i % 13 == 0 && allocatedArrays.size() > 10) {
                    int indexToRemove = ThreadLocalRandom.current().nextInt(allocatedArrays.size() - 5);
                    allocatedArrays.remove(indexToRemove);
                }
                
                if (i % 100 == 0) {
                    long currentMemory = getUsedMemory();
                    logger.debug("Iteration {}: Memory usage: {} MB", 
                               i, currentMemory / (1024 * 1024));
                }
            }
            
            // Final memory check
            System.gc();
            long finalMemory = getUsedMemory();
            long memoryIncrease = finalMemory - initialMemory;
            
            logger.info("Memory fragmentation test completed. Memory increase: {} MB",
                       memoryIncrease / (1024 * 1024));
            
            // Assert memory usage is reasonable for fragmentation test
            long maxAcceptableIncrease = 1024 * 1024 * 1024; // 1GB for fragmentation test
            if (memoryIncrease > maxAcceptableIncrease) {
                throw new AssertionError("Memory fragmentation test used excessive memory: " + 
                                       (memoryIncrease / (1024 * 1024)) + " MB");
            }
            
        } finally {
            // Clear all allocated arrays
            allocatedArrays.clear();
            System.gc();
        }
    }
    
    // Utility methods
    
    private long getUsedMemory() {
        Runtime runtime = Runtime.getRuntime();
        return runtime.totalMemory() - runtime.freeMemory();
    }
    
    private float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < size; i++) {
            matrix[i] = random.nextFloat() * 2.0f - 1.0f; // Range [-1, 1]
        }
        
        return matrix;
    }
    
    private String[] generateTestDocuments(int count) {
        String[] documents = new String[count];
        String[] templates = {
            "The %s %s quickly %s through the %s forest.",
            "Machine learning with %s acceleration provides %s performance improvements.",
            "Natural language processing requires %s feature extraction and %s analysis.",
            "GPU computing enables %s parallel processing for %s applications.",
            "OpenNLP provides %s tools for %s natural language tasks."
        };
        
        String[] words = {"advanced", "efficient", "powerful", "sophisticated", "intelligent",
                         "rapid", "comprehensive", "robust", "scalable", "optimized"};
        
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < count; i++) {
            String template = templates[i % templates.length];
            String doc = String.format(template,
                words[random.nextInt(words.length)],
                words[random.nextInt(words.length)],
                words[random.nextInt(words.length)],
                words[random.nextInt(words.length)]);
            documents[i] = doc;
        }
        
        return documents;
    }
    
    private void analyzeMemoryTrend(List<Long> memorySnapshots) {
        if (memorySnapshots.size() < 3) {
            return;
        }
        
        // Calculate trend
        long totalIncrease = 0;
        int increaseCount = 0;
        
        for (int i = 1; i < memorySnapshots.size(); i++) {
            long increase = memorySnapshots.get(i) - memorySnapshots.get(i - 1);
            if (increase > 0) {
                totalIncrease += increase;
                increaseCount++;
            }
        }
        
        if (increaseCount > 0) {
            long averageIncrease = totalIncrease / increaseCount;
            logger.info("Memory trend analysis: Average increase per snapshot: {} KB",
                       averageIncrease / 1024);
            
            // Warn if trend shows consistent large increases
            if (averageIncrease > 50 * 1024 * 1024) { // 50MB per snapshot
                logger.warn("Memory trend shows large increases - potential memory leak");
            }
        }
    }
}
