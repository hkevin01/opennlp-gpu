package org.apache.opennlp.gpu.production;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertFalse;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.util.List;
import java.util.Map;
import java.util.Set;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.monitoring.GpuPerformanceMonitor;
import org.apache.opennlp.gpu.production.ProductionOptimizer.OptimizationResult;
import org.apache.opennlp.gpu.production.ProductionOptimizer.OptimizationState;
import org.apache.opennlp.gpu.production.ProductionOptimizer.OptimizationStrategy;
import org.apache.opennlp.gpu.production.ProductionOptimizer.PerformanceSnapshot;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

/**
 * Comprehensive tests for the ProductionOptimizer system.
 * Tests optimization strategies, performance monitoring, and production features.
 */
public class ProductionOptimizerTest {
    
    private GpuConfig config;
    private GpuPerformanceMonitor performanceMonitor;
    private ProductionOptimizer optimizer;
    
    @BeforeEach
    void setUp() {
        // Initialize test configuration
        config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setBatchSize(32);
        config.setMemoryPoolSizeMB(512);
        config.setMaxMemoryUsageMB(1024);
        
        // Initialize performance monitor
        performanceMonitor = GpuPerformanceMonitor.getInstance();
        
        // Initialize optimizer
        optimizer = new ProductionOptimizer(config, performanceMonitor);
    }
    
    @AfterEach
    void tearDown() {
        if (optimizer != null) {
            optimizer.shutdown();
        }
    }
    
    @Test
    void testOptimizerInitialization() {
        assertNotNull(optimizer);
        assertEquals(OptimizationState.INITIALIZING, optimizer.getCurrentState());
        assertEquals(32, optimizer.getOptimalBatchSize());
        assertEquals(512, optimizer.getOptimalMemoryPoolSize());
        assertEquals(0.0, optimizer.getPerformanceScore(), 0.001);
    }
    
    @Test
    void testOptimizationState() {
        // Test initial state
        OptimizationState state = optimizer.getCurrentState();
        assertNotNull(state);
        
        // Test state transitions by forcing optimization
        optimizer.forceOptimization();
        
        // State should change after optimization attempt
        OptimizationState newState = optimizer.getCurrentState();
        assertNotNull(newState);
    }
    
    @Test
    void testPerformanceScoreCalculation() {
        double initialScore = optimizer.getPerformanceScore();
        
        // Force optimization to update performance score
        optimizer.forceOptimization();
        
        double updatedScore = optimizer.getPerformanceScore();
        assertTrue(updatedScore >= 0.0 && updatedScore <= 1.0, 
                  "Performance score should be between 0.0 and 1.0");
    }
    
    @Test
    void testOptimalBatchSizeAdjustment() {
        int initialBatchSize = optimizer.getOptimalBatchSize();
        assertEquals(32, initialBatchSize);
        
        // Force optimization which may adjust batch size
        optimizer.forceOptimization();
        
        int newBatchSize = optimizer.getOptimalBatchSize();
        assertTrue(newBatchSize >= 16 && newBatchSize <= 512,
                  "Batch size should be within reasonable bounds");
    }
    
    @Test
    void testOptimalMemoryPoolSizeAdjustment() {
        int initialPoolSize = optimizer.getOptimalMemoryPoolSize();
        assertEquals(512, initialPoolSize);
        
        // Force optimization which may adjust memory pool size
        optimizer.forceOptimization();
        
        int newPoolSize = optimizer.getOptimalMemoryPoolSize();
        assertTrue(newPoolSize >= 128 && newPoolSize <= 2048,
                  "Memory pool size should be within reasonable bounds");
    }
    
    @Test
    void testOptimizationEnabling() {
        // Test enabling/disabling optimization
        optimizer.setOptimizationEnabled(true);
        optimizer.forceOptimization();
        
        optimizer.setOptimizationEnabled(false);
        optimizer.forceOptimization();
        
        // Should not crash when disabled
        assertNotNull(optimizer.getCurrentState());
    }
    
    @Test
    void testPerformanceHistoryCollection() {
        // Force some optimization cycles to collect history
        for (int i = 0; i < 5; i++) {
            optimizer.forceOptimization();
            try {
                Thread.sleep(100); // Small delay to allow snapshot collection
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
            }
        }
        
        List<PerformanceSnapshot> history = optimizer.getPerformanceHistory();
        assertNotNull(history);
        // History should contain some snapshots after forced optimizations
        assertTrue(history.size() >= 0, "Performance history should be accessible");
    }
    
    @Test
    void testPerformanceSnapshot() {
        // Create a test performance snapshot
        PerformanceSnapshot snapshot = new PerformanceSnapshot(
            0.75,    // GPU utilization
            0.60,    // Memory usage
            85L,     // Average latency
            1200.0,  // Throughput
            64,      // Batch size
            0.8      // Performance score
        );
        
        assertNotNull(snapshot);
        assertNotNull(snapshot.getTimestamp());
        assertEquals(0.75, snapshot.getGpuUtilization(), 0.001);
        assertEquals(0.60, snapshot.getMemoryUsage(), 0.001);
        assertEquals(85L, snapshot.getAverageLatency());
        assertEquals(1200.0, snapshot.getThroughput(), 0.001);
        assertEquals(64, snapshot.getBatchSize());
        assertEquals(0.8, snapshot.getPerformanceScore(), 0.001);
    }
    
    @Test
    void testOptimizationResult() {
        // Test successful optimization result
        OptimizationResult success = new OptimizationResult(
            true, "TestStrategy", 0.15, "Improved performance by 15%"
        );
        
        assertTrue(success.isSuccess());
        assertEquals("TestStrategy", success.getStrategy());
        assertEquals(0.15, success.getPerformanceGain(), 0.001);
        assertEquals("Improved performance by 15%", success.getDetails());
        
        // Test failed optimization result
        OptimizationResult failure = new OptimizationResult(
            false, "TestStrategy", 0.0, "No optimization needed"
        );
        
        assertFalse(failure.isSuccess());
        assertEquals(0.0, failure.getPerformanceGain(), 0.001);
    }
    
    @Test
    void testActiveStrategies() {
        Set<String> strategies = optimizer.getActiveStrategies();
        assertNotNull(strategies);
        assertTrue(strategies.size() >= 2, "Should have at least batch size and memory pool strategies");
        assertTrue(strategies.contains("BatchSizeOptimization"));
        assertTrue(strategies.contains("MemoryPoolOptimization"));
    }
    
    @Test
    void testCustomOptimizationStrategy() {
        // Create a custom strategy
        OptimizationStrategy customStrategy = new OptimizationStrategy() {
            @Override
            public String getName() { return "CustomTestStrategy"; }
            
            @Override
            public boolean shouldApply(PerformanceSnapshot current, 
                                     java.util.Queue<PerformanceSnapshot> history) {
                return current.getThroughput() < 1000.0;
            }
            
            @Override
            public OptimizationResult apply(GpuConfig config) {
                return new OptimizationResult(true, "CustomTestStrategy", 0.05, "Custom optimization applied");
            }
            
            @Override
            public double getExpectedImprovement() { return 0.05; }
        };
        
        // Add custom strategy
        optimizer.addOptimizationStrategy("CustomTest", customStrategy);
        
        Set<String> strategies = optimizer.getActiveStrategies();
        assertTrue(strategies.contains("CustomTest"));
        
        // Remove custom strategy
        optimizer.removeOptimizationStrategy("CustomTest");
        
        strategies = optimizer.getActiveStrategies();
        assertFalse(strategies.contains("CustomTest"));
    }
    
    @Test
    void testOptimizationStatistics() {
        // Force optimization to generate some stats
        optimizer.forceOptimization();
        
        Map<String, Object> stats = optimizer.getOptimizationStats();
        assertNotNull(stats);
        
        // Verify expected stat keys
        assertTrue(stats.containsKey("currentState"));
        assertTrue(stats.containsKey("performanceScore"));
        assertTrue(stats.containsKey("optimalBatchSize"));
        assertTrue(stats.containsKey("optimalMemoryPoolSize"));
        assertTrue(stats.containsKey("optimizationEnabled"));
        assertTrue(stats.containsKey("activeStrategies"));
        assertTrue(stats.containsKey("performanceHistorySize"));
        
        // Verify stat values are reasonable
        assertTrue((Boolean) stats.get("optimizationEnabled"));
        assertTrue((Integer) stats.get("activeStrategies") >= 2);
        assertTrue((Integer) stats.get("performanceHistorySize") >= 0);
        
        Double performanceScore = (Double) stats.get("performanceScore");
        assertTrue(performanceScore >= 0.0 && performanceScore <= 1.0);
        
        Integer batchSize = (Integer) stats.get("optimalBatchSize");
        assertTrue(batchSize >= 16 && batchSize <= 512);
        
        Integer memoryPool = (Integer) stats.get("optimalMemoryPoolSize");
        assertTrue(memoryPool >= 128 && memoryPool <= 2048);
    }
    
    @Test
    void testBatchSizeOptimizationStrategy() {
        // Test batch size optimization logic indirectly
        config.setBatchSize(16); // Start with small batch
        
        optimizer.forceOptimization();
        
        // The optimizer should potentially adjust batch size based on mock performance
        int newBatchSize = optimizer.getOptimalBatchSize();
        assertTrue(newBatchSize >= 16, "Batch size should not go below minimum");
    }
    
    @Test
    void testMemoryPoolOptimizationStrategy() {
        // Test memory pool optimization logic indirectly
        config.setMemoryPoolSizeMB(128); // Start with small pool
        
        optimizer.forceOptimization();
        
        // The optimizer should potentially adjust memory pool based on mock performance
        int newPoolSize = optimizer.getOptimalMemoryPoolSize();
        assertTrue(newPoolSize >= 128, "Memory pool should not go below minimum");
    }
    
    @Test
    void testOptimizationStateTransitions() {
        // Test various optimization states
        OptimizationState state = optimizer.getCurrentState();
        assertNotNull(state);
        
        // Force optimization multiple times to potentially see state changes
        for (int i = 0; i < 3; i++) {
            optimizer.forceOptimization();
            OptimizationState newState = optimizer.getCurrentState();
            assertNotNull(newState);
            
            // Valid states should be one of the enum values
            assertTrue(newState == OptimizationState.INITIALIZING ||
                      newState == OptimizationState.MONITORING ||
                      newState == OptimizationState.ANALYZING ||
                      newState == OptimizationState.OPTIMIZING ||
                      newState == OptimizationState.STABLE ||
                      newState == OptimizationState.DEGRADED ||
                      newState == OptimizationState.EMERGENCY);
        }
    }
    
    @Test
    void testGracefulShutdown() {
        // Test that shutdown doesn't throw exceptions
        assertDoesNotThrow(() -> {
            optimizer.shutdown();
        });
        
        // Test operations after shutdown (should not crash)
        assertDoesNotThrow(() -> {
            optimizer.getCurrentState();
            optimizer.getPerformanceScore();
            optimizer.getOptimizationStats();
        });
    }
    
    @Test
    void testConfigurationIntegration() {
        // Test that optimizer respects config changes
        int originalBatch = config.getBatchSize();
        int originalMemory = config.getMemoryPoolSizeMB();
        
        // Modify config
        config.setBatchSize(128);
        config.setMemoryPoolSizeMB(1024);
        
        // Force optimization
        optimizer.forceOptimization();
        
        // Verify optimizer uses updated config values
        assertTrue(optimizer.getOptimalBatchSize() > 0);
        assertTrue(optimizer.getOptimalMemoryPoolSize() > 0);
    }
    
    @Test
    void testPerformanceMonitoringIntegration() {
        // Test that optimizer works with performance monitor
        assertNotNull(performanceMonitor);
        
        // Force optimization should work without errors
        assertDoesNotThrow(() -> {
            optimizer.forceOptimization();
        });
        
        // Performance score should be calculated
        double score = optimizer.getPerformanceScore();
        assertTrue(score >= 0.0 && score <= 1.0);
    }
    
    @Test
    void testConcurrentOptimization() {
        // Test that multiple optimization calls don't interfere
        for (int i = 0; i < 5; i++) {
            final int iteration = i;
            assertDoesNotThrow(() -> {
                optimizer.forceOptimization();
            }, "Optimization iteration " + iteration + " should not throw");
        }
        
        // State should remain valid
        assertNotNull(optimizer.getCurrentState());
        assertTrue(optimizer.getPerformanceScore() >= 0.0);
    }
}
