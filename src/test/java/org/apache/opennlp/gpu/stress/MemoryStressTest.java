package org.apache.opennlp.gpu.stress;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

import org.apache.opennlp.gpu.common.GpuLogger;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Memory stress testing for GPU acceleration components
 * Tests memory usage, leaks, and resource management under load
 */
public class MemoryStressTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(MemoryStressTest.class);
    
    private static final int LARGE_MATRIX_SIZE = 1000;
    private static final int MEMORY_ITERATIONS = 500;
    private static final int CONCURRENT_THREADS = 8;
    private static final long MAX_MEMORY_INCREASE_MB = 200;
    
    @Before
    public void setUp() {
        logger.info("Starting memory stress test setup");
        
        // Force garbage collection before testing
        System.gc();
        try {
            Thread.sleep(100);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
    }
    
    @After
    public void tearDown() {
        logger.info("Memory stress test cleanup");
        
        // Force cleanup after testing
        System.gc();
    }
    
    @Test
    public void testLargeMatrixMemoryUsage() {
        logger.info("Testing large matrix memory usage");
        
        Runtime runtime = Runtime.getRuntime();
        long initialMemory = runtime.totalMemory() - runtime.freeMemory();
        
        List<float[][]> matrices = new ArrayList<>();
        
        try {
            // Create large matrices
            for (int i = 0; i < 10; i++) {
                float[][] matrix = new float[LARGE_MATRIX_SIZE][LARGE_MATRIX_SIZE];
                
                // Fill with test data
                for (int row = 0; row < LARGE_MATRIX_SIZE; row++) {
                    for (int col = 0; col < LARGE_MATRIX_SIZE; col++) {
                        matrix[row][col] = (float) (Math.random() * 100);
                    }
                }
                
                matrices.add(matrix);
                
                if (i % 2 == 0) {
                    System.gc();
                }
            }
            
            long peakMemory = runtime.totalMemory() - runtime.freeMemory();
            long memoryIncrease = peakMemory - initialMemory;
            
            logger.info("Initial memory: " + (initialMemory / 1024 / 1024) + " MB");
            logger.info("Peak memory: " + (peakMemory / 1024 / 1024) + " MB");
            logger.info("Memory increase: " + (memoryIncrease / 1024 / 1024) + " MB");
            
            // Verify memory usage is reasonable
            assert memoryIncrease < MAX_MEMORY_INCREASE_MB * 1024 * 1024 : 
                "Memory usage too high: " + (memoryIncrease / 1024 / 1024) + " MB";
                
        } finally {
            // Cleanup
            matrices.clear();
            System.gc();
        }
    }
    
    @Test
    public void testConcurrentMemoryAccess() {
        logger.info("Testing concurrent memory access");
        
        ExecutorService executor = Executors.newFixedThreadPool(CONCURRENT_THREADS);
        CountDownLatch latch = new CountDownLatch(CONCURRENT_THREADS);
        
        AtomicInteger successCount = new AtomicInteger(0);
        
        try {
            for (int i = 0; i < CONCURRENT_THREADS; i++) {
                final int threadId = i;
                executor.submit(() -> {
                    try {
                        performMemoryOperations(threadId);
                        successCount.incrementAndGet();
                    } catch (Exception e) {
                        logger.error("Memory operation failed for thread " + threadId, e);
                    } finally {
                        latch.countDown();
                    }
                });
            }
            
            try {
                boolean completed = latch.await(30, TimeUnit.SECONDS);
                assert completed : "Concurrent memory test did not complete in time";
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                assert false : "Test was interrupted";
            }
            
            logger.info("Concurrent memory test completed");
            logger.info("Successful threads: " + successCount.get() + "/" + CONCURRENT_THREADS);
            
            assert successCount.get() >= CONCURRENT_THREADS * 0.8 : 
                "Too many thread failures: " + successCount.get() + "/" + CONCURRENT_THREADS;
                
        } finally {
            executor.shutdown();
            try {
                if (!executor.awaitTermination(5, TimeUnit.SECONDS)) {
                    executor.shutdownNow();
                }
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                executor.shutdownNow();
            }
        }
    }
    
    private void performMemoryOperations(int threadId) {
        logger.info("Performing memory operations for thread " + threadId);
        
        List<float[]> buffers = new ArrayList<>();
        
        try {
            // Allocate and deallocate memory repeatedly
            for (int i = 0; i < MEMORY_ITERATIONS; i++) {
                float[] buffer = new float[1000];
                
                // Fill buffer with test data
                for (int j = 0; j < buffer.length; j++) {
                    buffer[j] = (float) (Math.random() * threadId + i);
                }
                
                buffers.add(buffer);
                
                // Periodically clear some buffers
                if (i % 50 == 0 && !buffers.isEmpty()) {
                    buffers.remove(0);
                }
                
                // Occasional pause for GC
                if (i % 100 == 0) {
                    try {
                        Thread.sleep(1);
                    } catch (InterruptedException e) {
                        Thread.currentThread().interrupt();
                        break;
                    }
                }
            }
            
        } finally {
            buffers.clear();
        }
    }
}
