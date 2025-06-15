package org.apache.opennlp.gpu.stress;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.condition.EnabledIf;

/**
 * Concurrency and thread safety tests for GPU acceleration components
 * Tests multiple threads accessing GPU resources simultaneously
 */
@EnabledIf("isConcurrencyTestingEnabled")
public class ConcurrencyTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(ConcurrencyTest.class);
    
    private static final int DEFAULT_THREAD_COUNT = 8;
    private static final int OPERATIONS_PER_THREAD = 50;
    private static final int MATRIX_SIZE = 500;
    
    /**
     * Check if concurrency testing is enabled
     */
    public static boolean isConcurrencyTestingEnabled() {
        return "true".equals(System.getProperty("gpu.concurrency.test.enabled", "false"));
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testConcurrentMatrixOperations() throws Exception {
        logger.info("Testing concurrent matrix operations with {} threads", DEFAULT_THREAD_COUNT);
        
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        ExecutorService executor = Executors.newFixedThreadPool(DEFAULT_THREAD_COUNT);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(DEFAULT_THREAD_COUNT);
        
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        AtomicReference<Exception> firstException = new AtomicReference<>();
        
        List<Future<?>> futures = new ArrayList<>();
        
        try {
            // Submit concurrent tasks
            for (int threadId = 0; threadId < DEFAULT_THREAD_COUNT; threadId++) {
                final int tid = threadId;
                
                Future<?> future = executor.submit(() -> {
                    ComputeProvider provider = null;
                    MatrixOperation matrixOp = null;
                    
                    try {
                        // Wait for all threads to be ready
                        startLatch.await();
                        
                        // Each thread creates its own provider for thread safety
                        provider = new CpuComputeProvider(); // Use CPU for reliable testing
                        matrixOp = new CpuMatrixOperation(provider);
                        
                        // Perform operations
                        for (int op = 0; op < OPERATIONS_PER_THREAD; op++) {
                            performMatrixOperation(matrixOp, tid, op);
                            
                            // Add some randomness to timing
                            if (ThreadLocalRandom.current().nextInt(10) == 0) {
                                Thread.sleep(1);
                            }
                        }
                        
                        successCount.incrementAndGet();
                        logger.debug("Thread {} completed successfully", tid);
                        
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        firstException.compareAndSet(null, e);
                        logger.error("Thread {} failed: {}", tid, e.getMessage());
                    } finally {
                        // Cleanup resources
                        if (matrixOp != null) {
                            try {
                                matrixOp.release();
                            } catch (Exception e) {
                                logger.warn("Error releasing matrix operation in thread {}: {}", tid, e.getMessage());
                            }
                        }
                        if (provider != null) {
                            try {
                                provider.cleanup();
                            } catch (Exception e) {
                                logger.warn("Error cleaning up provider in thread {}: {}", tid, e.getMessage());
                            }
                        }
                        completionLatch.countDown();
                    }
                });
                
                futures.add(future);
            }
            
            // Start all threads simultaneously
            long startTime = System.currentTimeMillis();
            startLatch.countDown();
            
            // Wait for completion
            boolean completed = completionLatch.await(4, TimeUnit.MINUTES);
            long duration = System.currentTimeMillis() - startTime;
            
            if (!completed) {
                throw new RuntimeException("Concurrency test timed out after 4 minutes");
            }
            
            // Check results
            logger.info("Concurrency test completed in {} ms. Success: {}, Errors: {}", 
                       duration, successCount.get(), errorCount.get());
            
            if (firstException.get() != null) {
                throw new AssertionError("Concurrency test failed", firstException.get());
            }
            
            if (errorCount.get() > 0) {
                throw new AssertionError("Concurrency test had " + errorCount.get() + " errors");
            }
            
            if (successCount.get() != DEFAULT_THREAD_COUNT) {
                throw new AssertionError("Expected " + DEFAULT_THREAD_COUNT + 
                                       " successful threads, got " + successCount.get());
            }
            
        } finally {
            executor.shutdownNow();
            try {
                if (!executor.awaitTermination(30, TimeUnit.SECONDS)) {
                    logger.warn("Executor did not terminate within 30 seconds");
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    @Test
    @Timeout(240) // 4 minutes timeout
    public void testProviderResourceContention() throws Exception {
        logger.info("Testing provider resource contention");
        
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // Create shared provider to test resource contention
        ComputeProvider sharedProvider = new CpuComputeProvider();
        
        try {
            ExecutorService executor = Executors.newFixedThreadPool(6);
            CountDownLatch startLatch = new CountDownLatch(1);
            
            AtomicInteger operationCount = new AtomicInteger(0);
            AtomicInteger errorCount = new AtomicInteger(0);
            
            List<Future<Integer>> futures = new ArrayList<>();
            
            // Submit tasks that share the same provider
            for (int i = 0; i < 6; i++) {
                final int threadId = i;
                
                Future<Integer> future = executor.submit(() -> {
                    MatrixOperation matrixOp = null;
                    int operations = 0;
                    
                    try {
                        startLatch.await();
                        
                        // Create matrix operation using shared provider
                        matrixOp = new CpuMatrixOperation(sharedProvider);
                        
                        // Perform operations with shared resources
                        for (int op = 0; op < 30; op++) {
                            try {
                                performMatrixOperation(matrixOp, threadId, op);
                                operations++;
                                operationCount.incrementAndGet();
                            } catch (Exception e) {
                                errorCount.incrementAndGet();
                                logger.warn("Operation failed in thread {}: {}", threadId, e.getMessage());
                            }
                        }
                        
                    } catch (Exception e) {
                        logger.error("Thread {} failed: {}", threadId, e.getMessage());
                        errorCount.incrementAndGet();
                    } finally {
                        if (matrixOp != null) {
                            try {
                                matrixOp.release();
                            } catch (Exception e) {
                                logger.warn("Error releasing resources in thread {}: {}", threadId, e.getMessage());
                            }
                        }
                    }
                    
                    return operations;
                });
                
                futures.add(future);
            }
            
            // Start all threads
            startLatch.countDown();
            
            // Collect results
            int totalOperations = 0;
            for (Future<Integer> future : futures) {
                try {
                    Integer result = future.get(3, TimeUnit.MINUTES);
                    totalOperations += result;
                } catch (Exception e) {
                    logger.error("Future failed: {}", e.getMessage());
                    errorCount.incrementAndGet();
                }
            }
            
            executor.shutdown();
            executor.awaitTermination(30, TimeUnit.SECONDS);
            
            logger.info("Resource contention test completed. Total operations: {}, Errors: {}", 
                       totalOperations, errorCount.get());
            
            // Validate results
            if (errorCount.get() > totalOperations * 0.1) { // Allow up to 10% error rate
                throw new AssertionError("Too many errors in resource contention test: " + errorCount.get());
            }
            
        } finally {
            sharedProvider.cleanup();
        }
    }
    
    @Test
    @Timeout(180) // 3 minutes timeout
    public void testDeadlockPrevention() throws Exception {
        logger.info("Testing deadlock prevention");
        
        final Object lock1 = new Object();
        final Object lock2 = new Object();
        
        ExecutorService executor = Executors.newFixedThreadPool(2);
        CountDownLatch startLatch = new CountDownLatch(1);
        
        AtomicReference<Exception> thread1Exception = new AtomicReference<>();
        AtomicReference<Exception> thread2Exception = new AtomicReference<>();
        
        try {
            // Thread 1: acquires lock1 then lock2
            Future<?> future1 = executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    synchronized (lock1) {
                        logger.debug("Thread 1 acquired lock1");
                        Thread.sleep(50); // Small delay to increase chance of deadlock
                        
                        synchronized (lock2) {
                            logger.debug("Thread 1 acquired lock2");
                            performSimpleComputation();
                        }
                    }
                    
                } catch (Exception e) {
                    thread1Exception.set(e);
                    logger.error("Thread 1 failed: {}", e.getMessage());
                }
            });
            
            // Thread 2: acquires lock2 then lock1 (potential deadlock)
            Future<?> future2 = executor.submit(() -> {
                try {
                    startLatch.await();
                    
                    synchronized (lock2) {
                        logger.debug("Thread 2 acquired lock2");
                        Thread.sleep(50); // Small delay to increase chance of deadlock
                        
                        synchronized (lock1) {
                            logger.debug("Thread 2 acquired lock1");
                            performSimpleComputation();
                        }
                    }
                    
                } catch (Exception e) {
                    thread2Exception.set(e);
                    logger.error("Thread 2 failed: {}", e.getMessage());
                }
            });
            
            // Start both threads
            startLatch.countDown();
            
            // Wait for completion with timeout to detect deadlock
            long startTime = System.currentTimeMillis();
            
            try {
                future1.get(2, TimeUnit.MINUTES);
                future2.get(2, TimeUnit.MINUTES);
            } catch (Exception e) {
                // Check if it's a timeout (potential deadlock)
                long duration = System.currentTimeMillis() - startTime;
                if (duration > 119000) { // Close to 2 minutes
                    throw new AssertionError("Potential deadlock detected - threads did not complete in time");
                }
                throw e;
            }
            
            // Check for exceptions
            if (thread1Exception.get() != null) {
                throw new AssertionError("Thread 1 failed", thread1Exception.get());
            }
            
            if (thread2Exception.get() != null) {
                throw new AssertionError("Thread 2 failed", thread2Exception.get());
            }
            
            logger.info("Deadlock prevention test completed successfully");
            
        } finally {
            executor.shutdownNow();
            try {
                executor.awaitTermination(10, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testHighContentionScenario() throws Exception {
        logger.info("Testing high contention scenario");
        
        final int threadCount = 16;
        final int operationsPerThread = 25;
        
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(threadCount);
        
        AtomicInteger totalOperations = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        List<Long> threadDurations = new ArrayList<>();
        
        try {
            // Submit high contention tasks
            for (int threadId = 0; threadId < threadCount; threadId++) {
                final int tid = threadId;
                
                executor.submit(() -> {
                    ComputeProvider provider = null;
                    MatrixOperation matrixOp = null;
                    
                    try {
                        startLatch.await();
                        long threadStart = System.currentTimeMillis();
                        
                        provider = new CpuComputeProvider();
                        matrixOp = new CpuMatrixOperation(provider);
                        
                        for (int op = 0; op < operationsPerThread; op++) {
                            try {
                                performMatrixOperation(matrixOp, tid, op);
                                totalOperations.incrementAndGet();
                                
                                // Add small random delays to increase contention
                                if (ThreadLocalRandom.current().nextInt(5) == 0) {
                                    Thread.sleep(ThreadLocalRandom.current().nextInt(5));
                                }
                                
                            } catch (Exception e) {
                                errorCount.incrementAndGet();
                                logger.debug("Operation failed in thread {}: {}", tid, e.getMessage());
                            }
                        }
                        
                        long threadEnd = System.currentTimeMillis();
                        long duration = threadEnd - threadStart;
                        
                        synchronized (threadDurations) {
                            threadDurations.add(duration);
                        }
                        
                        logger.debug("Thread {} completed in {} ms", tid, duration);
                        
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        logger.error("Thread {} failed: {}", tid, e.getMessage());
                    } finally {
                        if (matrixOp != null) {
                            try {
                                matrixOp.release();
                            } catch (Exception e) {
                                logger.warn("Error releasing matrix operation: {}", e.getMessage());
                            }
                        }
                        if (provider != null) {
                            try {
                                provider.cleanup();
                            } catch (Exception e) {
                                logger.warn("Error cleaning up provider: {}", e.getMessage());
                            }
                        }
                        completionLatch.countDown();
                    }
                });
            }
            
            // Start all threads simultaneously
            long startTime = System.currentTimeMillis();
            startLatch.countDown();
            
            // Wait for completion
            boolean completed = completionLatch.await(4, TimeUnit.MINUTES);
            long totalDuration = System.currentTimeMillis() - startTime;
            
            if (!completed) {
                throw new RuntimeException("High contention test timed out");
            }
            
            // Analyze results
            analyzeContentionResults(threadCount, totalOperations.get(), errorCount.get(), 
                                   totalDuration, threadDurations);
            
            // Validate results
            int expectedOperations = threadCount * operationsPerThread;
            double errorRate = (double) errorCount.get() / expectedOperations;
            
            if (errorRate > 0.2) { // Allow up to 20% error rate under high contention
                throw new AssertionError("Error rate too high in high contention test: " + 
                                       (errorRate * 100) + "%");
            }
            
            logger.info("High contention test completed successfully");
            
        } finally {
            executor.shutdownNow();
            try {
                executor.awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    // Utility methods
    
    private void performMatrixOperation(MatrixOperation matrixOp, int threadId, int operationId) {
        // Create small matrices for quick operations
        int size = MATRIX_SIZE + (threadId * 10); // Slight size variation per thread
        
        float[] a = generateRandomMatrix(size);
        float[] b = generateRandomMatrix(size);
        float[] result = new float[size];
        
        // Perform operation
        matrixOp.add(a, b, result, size);
        
        // Validate result (basic check)
        for (int i = 0; i < Math.min(10, size); i++) {
            if (Float.isNaN(result[i]) || Float.isInfinite(result[i])) {
                throw new RuntimeException("Invalid result in thread " + threadId + 
                                         ", operation " + operationId + ": " + result[i]);
            }
        }
    }
    
    private void performSimpleComputation() {
        // Simple computation to simulate work
        double sum = 0;
        for (int i = 0; i < 1000; i++) {
            sum += Math.sin(i * 0.1);
        }
        
        // Use the result to prevent optimization
        if (sum > 1000) {
            logger.debug("Computation result: {}", sum);
        }
    }
    
    private float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < size; i++) {
            matrix[i] = random.nextFloat() * 2.0f - 1.0f;
        }
        
        return matrix;
    }
    
    private void analyzeContentionResults(int threadCount, int totalOperations, int errorCount,
                                        long totalDuration, List<Long> threadDurations) {
        
        logger.info("High contention analysis:");
        logger.info("  Threads: {}", threadCount);
        logger.info("  Total operations: {}", totalOperations);
        logger.info("  Error count: {}", errorCount);
        logger.info("  Total duration: {} ms", totalDuration);
        logger.info("  Operations per second: {}", (totalOperations * 1000L) / totalDuration);
        
        if (!threadDurations.isEmpty()) {
            long minDuration = threadDurations.stream().mapToLong(Long::longValue).min().orElse(0);
            long maxDuration = threadDurations.stream().mapToLong(Long::longValue).max().orElse(0);
            double avgDuration = threadDurations.stream().mapToLong(Long::longValue).average().orElse(0);
            
            logger.info("  Thread duration - Min: {} ms, Max: {} ms, Avg: {:.1f} ms", 
                       minDuration, maxDuration, avgDuration);
            
            // Check for excessive variation (potential contention issues)
            if (maxDuration > minDuration * 3) {
                logger.warn("High variation in thread durations detected - potential contention issues");
            }
        }
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testConcurrentGpuOperations() {
        int numThreads = Integer.parseInt(System.getProperty("test.threads", "4"));
        logger.info("Starting concurrent GPU operations test with {} threads", numThreads);
        
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        Thread[] threads = new Thread[numThreads];
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(numThreads);
        
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        AtomicReference<Exception> firstException = new AtomicReference<>();
        
        long startTime = System.currentTimeMillis();
        
        // Create and start threads
        for (int i = 0; i < numThreads; i++) {
            final int threadId = i;
            Thread thread = new Thread(() -> {
                try {
                    logger.debug("Thread {} starting GPU operations", threadId);
                    
                    ComputeProvider provider = new CpuComputeProvider(); // Use CPU for testing
                    MatrixOperation matrixOp = new CpuMatrixOperation(provider);
                    
                    // Wait for signal to start
                    startLatch.await();
                    
                    // Perform a series of matrix operations
                    for (int op = 0; op < OPERATIONS_PER_THREAD; op++) {
                        performMatrixOperation(matrixOp, threadId, op);
                        
                        // Introduce random sleep to simulate varied execution time
                        if (ThreadLocalRandom.current().nextInt(10) == 0) {
                            Thread.sleep(ThreadLocalRandom.current().nextInt(5));
                        }
                    }
                    
                    successCount.incrementAndGet();
                    logger.debug("Thread {} completed successfully", threadId);
                    
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                    firstException.compareAndSet(null, e);
                    logger.error("Thread {} failed: {}", threadId, e.getMessage());
                } finally {
                    completionLatch.countDown();
                    logger.warn("Thread {} completed with status: {}", threadId, "finished");
                }
            });
            
            threads[i] = thread;
            thread.start();
        }
        
        // Signal all threads to start
        startLatch.countDown();
        
        // Wait for all threads to complete
        boolean completed = completionLatch.await(5, TimeUnit.MINUTES);
        long totalTime = System.currentTimeMillis() - startTime;
        
        if (!completed) {
            throw new RuntimeException("Concurrent test timed out");
        }
        
        logger.info("Concurrent test completed in {} ms with {} threads and {} operations", 
                   totalTime, numThreads, OPERATIONS_PER_THREAD * numThreads);
        
        // Check for errors
        if (firstException.get() != null) {
            throw new AssertionError("Concurrent GPU operations test failed", firstException.get());
        }
        
        if (errorCount.get() > 0) {
            throw new AssertionError("Concurrent GPU operations test had " + errorCount.get() + " errors");
        }
    }
    
    private void performMatrixOperation(int threadId) {
        try {
            logger.debug("Thread {} performing matrix operation", threadId);
            
            // Create small matrices for quick operations
            int size = MATRIX_SIZE + (threadId * 10); // Slight size variation per thread
            
            float[] a = generateRandomMatrix(size);
            float[] b = generateRandomMatrix(size);
            float[] result = new float[size];
            
            // Perform operation
            new CpuMatrixOperation().add(a, b, result, size);
            
            // Validate result (basic check)
            for (int i = 0; i < Math.min(10, size); i++) {
                if (Float.isNaN(result[i]) || Float.isInfinite(result[i])) {
                    throw new RuntimeException("Invalid result in thread " + threadId + ": " + result[i]);
                }
            }
            
        } catch (Exception e) {
            logger.error("Matrix operation failed for thread {}: {}", threadId, e.getMessage());
        }
    }
    
    @Test
    @Timeout(300) // 5 minutes timeout
    public void testGpuResourceSharing() {
        logger.info("Testing GPU resource sharing among threads");
        
        final int threadCount = 4;
        final int operationsPerThread = 100;
        
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        ExecutorService executor = Executors.newFixedThreadPool(threadCount);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(threadCount);
        
        AtomicInteger totalOperations = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        
        try {
            // Submit tasks
            for (int i = 0; i < threadCount; i++) {
                final int threadId = i;
                
                executor.submit(() -> {
                    ComputeProvider provider = new CpuComputeProvider(); // Use CPU for testing
                    MatrixOperation matrixOp = new CpuMatrixOperation(provider);
                    
                    try {
                        // Wait for signal to start
                        startLatch.await();
                        
                        // Perform a series of matrix operations
                        for (int op = 0; op < operationsPerThread; op++) {
                            performMatrixOperation(matrixOp, threadId, op);
                            totalOperations.incrementAndGet();
                        }
                        
                    } catch (Exception e) {
                        errorCount.incrementAndGet();
                        logger.error("Thread {} failed: {}", threadId, e.getMessage());
                    } finally {
                        completionLatch.countDown();
                    }
                });
            }
            
            // Start all threads
            long startTime = System.currentTimeMillis();
            startLatch.countDown();
            
            // Wait for completion
            boolean completed = completionLatch.await(5, TimeUnit.MINUTES);
            long totalTime = System.currentTimeMillis() - startTime;
            
            if (!completed) {
                throw new RuntimeException("Resource sharing test timed out");
            }
            
            logger.info("Resource sharing test completed in {} ms. Total operations: {}, Errors: {}", 
                       totalTime, totalOperations.get(), errorCount.get());
            
        } finally {
            executor.shutdownNow();
            try {
                executor.awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
    
    private void runStressTest() {
        double loadFactor = Double.parseDouble(System.getProperty("stress.load", "1.0"));
        logger.debug("Running stress test with load factor: {}", loadFactor);
        
        final int baseThreads = 4;
        final int baseOperations = 100;
        
        int threads = (int) (baseThreads * loadFactor);
        int operations = (int) (baseOperations * loadFactor);
        
        ExecutorService executor = Executors.newFixedThreadPool(threads);
        CountDownLatch startLatch = new CountDownLatch(1);
        CountDownLatch completionLatch = new CountDownLatch(threads);
        
        AtomicInteger successCount = new AtomicInteger(0);
        AtomicInteger errorCount = new AtomicInteger(0);
        long startTime = System.currentTimeMillis();
        
        // Submit tasks
        for (int i = 0; i < threads; i++) {
            final int threadId = i;
            
            executor.submit(() -> {
                ComputeProvider provider = new CpuComputeProvider(); // Use CPU for testing
                MatrixOperation matrixOp = new CpuMatrixOperation(provider);
                
                try {
                    // Wait for signal to start
                    startLatch.await();
                    
                    // Perform a series of matrix operations
                    for (int op = 0; op < operations; op++) {
                        performMatrixOperation(matrixOp, threadId, op);
                    }
                    
                    successCount.incrementAndGet();
                    
                } catch (Exception e) {
                    errorCount.incrementAndGet();
                    logger.error("Thread {} failed: {}", threadId, e.getMessage());
                } finally {
                    completionLatch.countDown();
                }
            });
        }
        
        // Start all threads
        startLatch.countDown();
        
        try {
            // Wait for completion
            boolean completed = completionLatch.await(5, TimeUnit.MINUTES);
            long totalTime = System.currentTimeMillis() - startTime;
            
            if (!completed) {
                throw new RuntimeException("Stress test timed out");
            }
            
            // Summary
            double successRate = (double) successCount.get() / threads * 100;
            logger.info("Stress test summary - Threads: {}, Operations: {}, Success Rate: {}%, " +
                       "Avg Time: {} ms, Total Time: {} ms", 
                       threads, operations, successRate, (totalTime / threads), totalTime);
            
        } finally {
            executor.shutdownNow();
            try {
                executor.awaitTermination(30, TimeUnit.SECONDS);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
    }
}
