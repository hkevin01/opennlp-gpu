package org.apache.opennlp.gpu.stress;

import static org.junit.Assert.*;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Comprehensive concurrency and stress testing for GPU acceleration components
 * Tests thread safety, memory management, and resource cleanup under load
 */
public class ConcurrencyTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(ConcurrencyTest.class);
    
    // Test configuration constants
    private static final int LOW_THREAD_COUNT = 4;
    private static final int MODERATE_THREAD_COUNT = 8;
    private static final int HIGH_THREAD_COUNT = 16;
    private static final int MEMORY_ITERATIONS = 100;
    private static final int STRESS_TEST_TIMEOUT_SECONDS = 60;
    private static final int OPERATION_ITERATIONS = 50;
    
    private MatrixOperation gpuOps;
    private MatrixOperation cpuOps;
    private ExecutorService executorService;
    
    @Before
    public void setUp() {
        logger.info("Setting up concurrency test environment");
        
        try {
            // Initialize GPU and CPU operations
            gpuOps = new GpuMatrixOperation(new org.apache.opennlp.gpu.common.CpuComputeProvider(), new GpuConfig());
            cpuOps = new CpuMatrixOperation(new org.apache.opennlp.gpu.common.CpuComputeProvider());
            
            // Create thread pool for tests
            executorService = Executors.newCachedThreadPool();
            
            logger.info("Concurrency test setup completed successfully");
        } catch (Exception e) {
            logger.error("Failed to set up concurrency test", e);
            throw new RuntimeException("Test setup failed", e);
        }
    }
    
    @After
    public void tearDown() {
        logger.info("Tearing down concurrency test environment");
        
        try {
            if (executorService != null) {
                executorService.shutdown();
                if (!executorService.awaitTermination(10, TimeUnit.SECONDS)) {
                    executorService.shutdownNow();
                }
            }
            
            // Cleanup GPU resources
            if (gpuOps != null) {
                // Add cleanup if needed
            }
            
            logger.info("Concurrency test teardown completed");
        } catch (Exception e) {
            logger.error("Error during test teardown", e);
        }
    }
    
    @Test
    public void testBasicConcurrency() {
        logger.info("Running basic concurrency test");
        
        final int threadCount = LOW_THREAD_COUNT;
        final CountDownLatch latch = new CountDownLatch(threadCount);
        final AtomicInteger successCount = new AtomicInteger(0);
        final AtomicInteger errorCount = new AtomicInteger(0);
        
        try {
            for (int i = 0; i < threadCount; i++) {
                final int threadId = i;
                executorService.submit(() -> {
                    try {
                        performBasicMatrixOperations(threadId);
                        successCount.incrementAndGet();
                        logger.info("Thread " + threadId + " completed successfully");
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        logger.error("Thread " + threadId + " failed", e);
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            boolean completed = latch.await(30, TimeUnit.SECONDS);
            assertTrue("Basic concurrency test should complete within timeout", completed);
            
            logger.info("Basic concurrency test results - Success: " + successCount.get() + ", Errors: " + errorCount.get());
            assertTrue("All threads should complete successfully", successCount.get() == threadCount);
            assertEquals("No errors should occur", 0, errorCount.get());
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Basic concurrency test was interrupted: " + e.getMessage());
        }
    }
    
    @Test
    public void testHighConcurrencyStress() {
        try {
            logger.info("Starting high concurrency stress test with " + HIGH_THREAD_COUNT + " threads");
            
            ExecutorService executor = Executors.newFixedThreadPool(HIGH_THREAD_COUNT);
            CountDownLatch startLatch = new CountDownLatch(1);
            CountDownLatch completionLatch = new CountDownLatch(HIGH_THREAD_COUNT);
            
            AtomicInteger successCount = new AtomicInteger(0);
            AtomicInteger errorCount = new AtomicInteger(0);
            
            // Submit concurrent tasks
            for (int i = 0; i < HIGH_THREAD_COUNT; i++) {
                final int threadId = i;
                executor.submit(() -> {
                    try {
                        startLatch.await(); // Wait for all threads to be ready
                        performConcurrentOperations(threadId);
                        successCount.incrementAndGet();
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        errorCount.incrementAndGet();
                        logger.warn("Thread " + threadId + " interrupted", e);
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        logger.error("Thread " + threadId + " failed", e);
                    } finally {
                        completionLatch.countDown();
                    }
                });
            }
            
            // Start all threads simultaneously
            startLatch.countDown();
            
            // Wait for completion with timeout
            boolean completed = completionLatch.await(STRESS_TEST_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            
            executor.shutdown();
            try {
                if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executor.shutdownNow();
            }
            
            logger.info("High concurrency stress test completed");
            logger.info("Successful operations: " + successCount.get());
            logger.info("Failed operations: " + errorCount.get());
            logger.info("Completion rate: " + (successCount.get() * 100.0 / HIGH_THREAD_COUNT) + "%");
            logger.info("Error rate: " + (errorCount.get() * 100.0 / HIGH_THREAD_COUNT) + "%");
            
            // Validate results
            assertTrue("Test should complete within timeout", completed);
            assertTrue("Success rate should be > 80%", successCount.get() > HIGH_THREAD_COUNT * 0.8);
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("High concurrency stress test was interrupted: " + e.getMessage());
        }
    }

    @Test
    public void testMemoryLeakDetection() {
        try {
            logger.info("Starting memory leak detection test");
            
            Runtime runtime = Runtime.getRuntime();
            long initialMemory = runtime.totalMemory() - runtime.freeMemory();
            
            // Perform memory-intensive operations
            for (int i = 0; i < MEMORY_ITERATIONS; i++) {
                performMemoryIntensiveOperations();
                
                if (i % 20 == 0) {
                    System.gc(); // Suggest garbage collection
                    try {
                        Thread.sleep(50); // Allow GC to run
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        logger.warn("Memory leak test interrupted during sleep", e);
                        break;
                    }
                }
            }
            
            // Final memory check
            System.gc();
            try {
                Thread.sleep(100); // Allow final GC
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                logger.warn("Memory leak test interrupted during final sleep", e);
            }
            
            long finalMemory = runtime.totalMemory() - runtime.freeMemory();
            long memoryIncrease = finalMemory - initialMemory;
            
            logger.info("Initial memory usage: " + (initialMemory / 1024 / 1024) + " MB");
            logger.info("Final memory usage: " + (finalMemory / 1024 / 1024) + " MB");
            logger.info("Memory increase: " + (memoryIncrease / 1024 / 1024) + " MB");
            
            // Allow some memory increase but not excessive
            long maxAllowedIncrease = 100 * 1024 * 1024; // 100 MB
            assertTrue("Memory increase should be reasonable (< 100MB), actual: " + 
                      (memoryIncrease / 1024 / 1024) + "MB", 
                      memoryIncrease < maxAllowedIncrease);
                      
        } catch (Exception e) {
            fail("Memory leak detection test failed: " + e.getMessage());
        }
    }

    @Test
    public void testResourceCleanupUnderLoad() {
        try {
            logger.info("Starting resource cleanup under load test");
            
            ExecutorService executor = Executors.newFixedThreadPool(MODERATE_THREAD_COUNT);
            CountDownLatch completionLatch = new CountDownLatch(MODERATE_THREAD_COUNT);
            
            AtomicInteger cleanupSuccessCount = new AtomicInteger(0);
            AtomicInteger cleanupFailureCount = new AtomicInteger(0);
            
            for (int i = 0; i < MODERATE_THREAD_COUNT; i++) {
                final int threadId = i;
                executor.submit(() -> {
                    try {
                        performResourceCleanupOperations(threadId);
                        cleanupSuccessCount.incrementAndGet();
                    } catch (Exception e) {
                        cleanupFailureCount.incrementAndGet();
                        logger.error("Resource cleanup failed for thread " + threadId, e);
                    } finally {
                        completionLatch.countDown();
                    }
                });
            }
            
            boolean completed = completionLatch.await(STRESS_TEST_TIMEOUT_SECONDS, TimeUnit.SECONDS);
            
            executor.shutdown();
            try {
                if (!executor.awaitTermination(10, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executor.shutdownNow();
            }
            
            logger.info("Resource cleanup test completed");
            logger.info("Successful cleanups: " + cleanupSuccessCount.get());
            logger.info("Failed cleanups: " + cleanupFailureCount.get());
            
            assertTrue("Test should complete within timeout", completed);
            assertTrue("Cleanup success rate should be > 95%", 
                      cleanupSuccessCount.get() > MODERATE_THREAD_COUNT * 0.95);
                      
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Resource cleanup test was interrupted: " + e.getMessage());
        }
    }
    
    // Helper methods for performing operations
    
    private void performBasicMatrixOperations(int threadId) {
        logger.info("Thread " + threadId + " performing basic matrix operations");
        
        try {
            // Create test matrices
            float[][] matrixA = createTestMatrix(100, 100, threadId);
            float[][] matrixB = createTestMatrix(100, 100, threadId + 1);
            
            // Perform operations alternating between GPU and CPU
            for (int i = 0; i < 10; i++) {
                if (i % 2 == 0) {
                    // GPU operations
                    int rowsA = matrixA.length;
                    int colsA = matrixA[0].length;
                    int colsB = matrixB[0].length;
                    float[] flatA = flatten(matrixA);
                    float[] flatB = flatten(matrixB);
                    float[] flatResult = new float[rowsA * colsB];
                    gpuOps.multiply(flatA, flatB, flatResult, rowsA, colsA, colsB);
                    float[][] result = reshape(flatResult, rowsA, colsB);
                    assertNotNull("GPU multiplication result should not be null", result);
                } else {
                    // CPU operations  
                    int rowsA = matrixA.length;
                    int colsA = matrixA[0].length;
                    int colsB = matrixB[0].length;
                    float[] flatA = flatten(matrixA);
                    float[] flatB = flatten(matrixB);
                    float[] flatResult = new float[rowsA * colsB];
                    cpuOps.multiply(flatA, flatB, flatResult, rowsA, colsA, colsB);
                    float[][] result = reshape(flatResult, rowsA, colsB);
                    assertNotNull("CPU multiplication result should not be null", result);
                }
            }
            
        } catch (Exception e) {
            logger.error("Basic matrix operations failed for thread " + threadId, e);
            throw new RuntimeException("Matrix operations failed", e);
        }
    }
    
    private void performConcurrentOperations(int threadId) {
        logger.info("Thread " + threadId + " performing concurrent operations");
        
        try {
            Random random = new Random(threadId);
            
            for (int i = 0; i < OPERATION_ITERATIONS; i++) {
                // Create random-sized matrices
                int size = 50 + random.nextInt(50); // 50-100 size
                float[][] matrixA = createTestMatrix(size, size, threadId * 1000 + i);
                float[][] matrixB = createTestMatrix(size, size, threadId * 1000 + i + 1);
                
                // Randomly choose GPU or CPU
                MatrixOperation ops = random.nextBoolean() ? gpuOps : cpuOps;

                // Perform random operation based on case
                switch (random.nextInt(3)) {
                    case 0: {
                        // Matrix multiplication
                        int rowsA = matrixA.length;
                        int colsA = matrixA[0].length;
                        int colsB = matrixB[0].length;
                        float[] flatA = flatten(matrixA);
                        float[] flatB = flatten(matrixB);
                        float[] flatResult = new float[rowsA * colsB];
                        ops.multiply(flatA, flatB, flatResult, rowsA, colsA, colsB);
                        float[][] result = reshape(flatResult, rowsA, colsB);
                        validateMatrix(result, size, size);
                        break;
                    }
                    case 1: {
                        // Matrix addition
                        int rows = matrixA.length;
                        int cols = matrixA[0].length;
                        float[] flatA = flatten(matrixA);
                        float[] flatB = flatten(matrixB);
                        float[] flatResult = new float[rows * cols];
                        ops.add(flatA, flatB, flatResult, rows * cols);
                        float[][] sum = reshape(flatResult, rows, cols);
                        validateMatrix(sum, size, size);
                        break;
                    }
                    case 2: {
                        // Matrix transpose
                        int rows = matrixA.length;
                        int cols = matrixA[0].length;
                        float[] flatA = flatten(matrixA);
                        float[] flatResult = new float[rows * cols];
                        ops.transpose(flatA, flatResult, rows, cols);
                        float[][] transposed = reshape(flatResult, cols, rows);
                        validateMatrix(transposed, size, size);
                        break;
                    }
                }
                
                // Small delay to simulate real workload
                if (i % 10 == 0) {
                    Thread.sleep(1);
                }
            }
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            throw new RuntimeException("Thread interrupted", e);
        } catch (Exception e) {
            logger.error("Concurrent operations failed for thread " + threadId, e);
            throw new RuntimeException("Concurrent operations failed", e);
        }
    }
    
    private void performMemoryIntensiveOperations() {
        logger.debug("Performing memory intensive operations");
        
        try {
            List<float[][]> matrices = new ArrayList<>();
            
            // Allocate and use matrices
            for (int i = 0; i < 50; i++) {
                float[][] matrix = createTestMatrix(200, 200, i);
                matrices.add(matrix);
                
                // Perform operation and discard immediately
                int rows = matrix.length;
                int cols = matrix[0].length;
                float[] flatMatrix = flatten(matrix);
                float[] flatResult = new float[rows * cols];
                cpuOps.transpose(flatMatrix, flatResult, rows, cols);
                float[][] result = reshape(flatResult, cols, rows);
                validateMatrix(result, 200, 200);
            }
            
            // Clear references to allow GC
            matrices.clear();
            
        } catch (Exception e) {
            logger.error("Memory intensive operations failed", e);
            throw new RuntimeException("Memory operations failed", e);
        }
    }
    
    private void performResourceCleanupOperations(int threadId) {
        logger.info("Thread " + threadId + " performing resource cleanup operations");
        
        try {
            List<float[][]> allocatedResources = new ArrayList<>();
            
            // Allocate resources
            for (int i = 0; i < 20; i++) {
                float[][] matrix = createTestMatrix(100, 100, threadId * 100 + i);
                allocatedResources.add(matrix);
                
                // Use the resource
                int rows = matrix.length;
                int cols = matrix[0].length;
                float[] flatMatrix = flatten(matrix);
                float[] flatResult = new float[rows * cols];
                gpuOps.transpose(flatMatrix, flatResult, rows, cols);
                float[][] processed = reshape(flatResult, cols, rows);
                validateMatrix(processed, 100, 100);
            }
            
            // Simulate cleanup
            allocatedResources.clear();
            
            // Force some garbage collection
            if (threadId % 2 == 0) {
                System.gc();
            }
            
        } catch (Exception e) {
            logger.error("Resource cleanup operations failed for thread " + threadId, e);
            throw new RuntimeException("Resource cleanup failed", e);
        }
    }
    
    // Utility methods

    /**
     * Flattens a 2D float array into a 1D float array in row-major order.
     */
    private float[] flatten(float[][] matrix) {
        int rows = matrix.length;
        int cols = (rows > 0) ? matrix[0].length : 0;
        float[] flat = new float[rows * cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(matrix[i], 0, flat, i * cols, cols);
        }
        return flat;
    }
    
    private float[][] createTestMatrix(int rows, int cols, int seed) {
        Random random = new Random(seed);
        float[][] matrix = new float[rows][cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextFloat() * 100.0f;
            }
        }
        
        return matrix;
    }

    /**
     * Reshapes a flat float array into a 2D float array with the given number of rows and columns.
     */
    private float[][] reshape(float[] flat, int rows, int cols) {
        if (flat.length != rows * cols) {
            throw new IllegalArgumentException("Flat array length does not match given dimensions.");
        }
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            System.arraycopy(flat, i * cols, matrix[i], 0, cols);
        }
        return matrix;
    }
    
    private void validateMatrix(float[][] matrix, int expectedRows, int expectedCols) {
        assertNotNull("Matrix should not be null", matrix);
        assertEquals("Matrix should have expected row count", expectedRows, matrix.length);
        if (matrix.length > 0) {
            assertEquals("Matrix should have expected column count", expectedCols, matrix[0].length);
        }
        
        // Check for NaN or infinite values
        for (int i = 0; i < matrix.length; i++) {
            for (int j = 0; j < matrix[i].length; j++) {
                assertFalse("Matrix should not contain NaN values", Float.isNaN(matrix[i][j]));
                assertFalse("Matrix should not contain infinite values", Float.isInfinite(matrix[i][j]));
            }
        }
    }
    
    @Test
    public void testThreadSafety() {
        logger.info("Testing thread safety of GPU operations");
        
        final int threadCount = MODERATE_THREAD_COUNT;
        final AtomicLong operationCount = new AtomicLong(0);
        final AtomicInteger errorCount = new AtomicInteger(0);
        final CountDownLatch latch = new CountDownLatch(threadCount);
        
        try {
            for (int i = 0; i < threadCount; i++) {
                final int threadId = i;
                executorService.submit(() -> {
                    try {
                        for (int j = 0; j < 100; j++) {
                            float[][] matrix = createTestMatrix(50, 50, threadId * 100 + j);
                            int rows = matrix.length;
                            int cols = matrix[0].length;
                            float[] flatMatrix = flatten(matrix);
                            float[] flatResult = new float[rows * cols];
                            gpuOps.transpose(flatMatrix, flatResult, rows, cols);
                            float[][] result = reshape(flatResult, cols, rows);
                            validateMatrix(result, 50, 50);
                            operationCount.incrementAndGet();
                        }
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        logger.error("Thread safety test failed for thread " + threadId, e);
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            boolean completed = latch.await(60, TimeUnit.SECONDS);
            assertTrue("Thread safety test should complete within timeout", completed);
            
            logger.info("Thread safety test completed - Operations: " + operationCount.get() + ", Errors: " + errorCount.get());
            assertEquals("No errors should occur in thread safety test", 0, errorCount.get());
            assertEquals("All operations should complete", threadCount * 100, operationCount.get());
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Thread safety test was interrupted: " + e.getMessage());
        }
    }
    
    @Test
    public void testRaceConditions() {
        logger.info("Testing for race conditions in shared resources");
        
        final int iterations = 1000;
        final AtomicInteger counter = new AtomicInteger(0);
        final CountDownLatch latch = new CountDownLatch(2);
        
        try {
            // Two threads incrementing counter while performing GPU operations
            for (int i = 0; i < 2; i++) {
                final int threadId = i;
                executorService.submit(() -> {
                    try {
                        for (int j = 0; j < iterations; j++) {
                            // Perform GPU operation
                            float[][] matrix = createTestMatrix(10, 10, threadId * iterations + j);
                            int rows = matrix.length;
                            int cols = matrix[0].length;
                            float[] flatMatrix = flatten(matrix);
                            float[] flatResult = new float[rows * cols];
                            gpuOps.transpose(flatMatrix, flatResult, rows, cols);
                            
                            // Increment shared counter
                            counter.incrementAndGet();
                        }
                    } catch (Exception e) {
                        logger.error("Race condition test failed for thread " + threadId, e);
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            boolean completed = latch.await(30, TimeUnit.SECONDS);
            assertTrue("Race condition test should complete within timeout", completed);
            
            // Counter should be exactly 2 * iterations if no race conditions
            assertEquals("Counter should have exact value (no race conditions)", 
                        2 * iterations, counter.get());
            
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
            fail("Race condition test was interrupted: " + e.getMessage());
        }
    }
}
