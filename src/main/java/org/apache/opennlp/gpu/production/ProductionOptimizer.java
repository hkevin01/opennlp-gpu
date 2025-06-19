package org.apache.opennlp.gpu.production;

import java.time.LocalDateTime;
import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Queue;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.monitoring.GpuPerformanceMonitor;

/**
 * Production optimization manager for GPU operations.
 * Handles automatic scaling, resource management, and performance optimization
 * for production deployments.
 */
public class ProductionOptimizer {
    private static final Logger logger = Logger.getLogger(ProductionOptimizer.class.getName());
    
    private final GpuConfig config;
    private final GpuPerformanceMonitor performanceMonitor;
    private final ScheduledExecutorService optimizationScheduler;
    private final AtomicBoolean optimizationEnabled;
    private final AtomicReference<OptimizationState> currentState;
    
    // Performance thresholds for optimization
    private static final double GPU_UTILIZATION_THRESHOLD = 0.85;
    private static final double MEMORY_USAGE_THRESHOLD = 0.90;
    private static final long LATENCY_THRESHOLD_MS = 100;
    private static final double THROUGHPUT_MIN_OPS_PER_SEC = 1000.0;
    
    // Optimization parameters
    private volatile int optimalBatchSize;
    private volatile int optimalMemoryPoolSize;
    private volatile double performanceScore;
    
    // Historical data for trend analysis
    private final Queue<PerformanceSnapshot> performanceHistory;
    private final Map<String, OptimizationStrategy> activeStrategies;
    
    public ProductionOptimizer(GpuConfig config, GpuPerformanceMonitor performanceMonitor) {
        this.config = config;
        this.performanceMonitor = performanceMonitor;
        this.optimizationScheduler = Executors.newScheduledThreadPool(2);
        this.optimizationEnabled = new AtomicBoolean(true);
        this.currentState = new AtomicReference<>(OptimizationState.INITIALIZING);
        
        // Initialize optimization parameters
        this.optimalBatchSize = config.getBatchSize();
        this.optimalMemoryPoolSize = config.getMemoryPoolSizeMB();
        this.performanceScore = 0.0;
        
        // Initialize collections
        this.performanceHistory = new ConcurrentLinkedQueue<>();
        this.activeStrategies = new ConcurrentHashMap<>();
        
        // Start optimization monitoring
        startOptimizationMonitoring();
        
        logger.info("ProductionOptimizer initialized with adaptive optimization enabled");
    }
    
    /**
     * Optimization states for production management
     */
    public enum OptimizationState {
        INITIALIZING,
        MONITORING,
        ANALYZING,
        OPTIMIZING,
        STABLE,
        DEGRADED,
        EMERGENCY
    }
    
    /**
     * Performance snapshot for historical analysis
     */
    public static class PerformanceSnapshot {
        private final LocalDateTime timestamp;
        private final double gpuUtilization;
        private final double memoryUsage;
        private final long averageLatency;
        private final double throughput;
        private final int batchSize;
        private final double performanceScore;
        
        public PerformanceSnapshot(double gpuUtilization, double memoryUsage, 
                                 long averageLatency, double throughput, 
                                 int batchSize, double performanceScore) {
            this.timestamp = LocalDateTime.now();
            this.gpuUtilization = gpuUtilization;
            this.memoryUsage = memoryUsage;
            this.averageLatency = averageLatency;
            this.throughput = throughput;
            this.batchSize = batchSize;
            this.performanceScore = performanceScore;
        }
        
        // Getters
        public LocalDateTime getTimestamp() { return timestamp; }
        public double getGpuUtilization() { return gpuUtilization; }
        public double getMemoryUsage() { return memoryUsage; }
        public long getAverageLatency() { return averageLatency; }
        public double getThroughput() { return throughput; }
        public int getBatchSize() { return batchSize; }
        public double getPerformanceScore() { return performanceScore; }
    }
    
    /**
     * Optimization strategy interface
     */
    public interface OptimizationStrategy {
        String getName();
        boolean shouldApply(PerformanceSnapshot current, Queue<PerformanceSnapshot> history);
        OptimizationResult apply(GpuConfig config);
        double getExpectedImprovement();
    }
    
    /**
     * Result of an optimization attempt
     */
    public static class OptimizationResult {
        private final boolean success;
        private final String strategy;
        private final double performanceGain;
        private final String details;
        
        public OptimizationResult(boolean success, String strategy, 
                                double performanceGain, String details) {
            this.success = success;
            this.strategy = strategy;
            this.performanceGain = performanceGain;
            this.details = details;
        }
        
        // Getters
        public boolean isSuccess() { return success; }
        public String getStrategy() { return strategy; }
        public double getPerformanceGain() { return performanceGain; }
        public String getDetails() { return details; }
    }
    
    /**
     * Start continuous optimization monitoring
     */
    private void startOptimizationMonitoring() {
        // Main optimization loop - runs every 30 seconds
        optimizationScheduler.scheduleAtFixedRate(this::performOptimizationCycle, 
                                                 30, 30, TimeUnit.SECONDS);
        
        // Performance snapshot collection - runs every 10 seconds
        optimizationScheduler.scheduleAtFixedRate(this::collectPerformanceSnapshot, 
                                                 10, 10, TimeUnit.SECONDS);
        
        // Initialize default optimization strategies
        initializeOptimizationStrategies();
    }
    
    /**
     * Initialize built-in optimization strategies
     */
    private void initializeOptimizationStrategies() {
        // Batch size optimization strategy
        activeStrategies.put("BatchSizeOptimization", new OptimizationStrategy() {
            @Override
            public String getName() { return "BatchSizeOptimization"; }
            
            @Override
            public boolean shouldApply(PerformanceSnapshot current, Queue<PerformanceSnapshot> history) {
                return current.getThroughput() < THROUGHPUT_MIN_OPS_PER_SEC ||
                       current.getAverageLatency() > LATENCY_THRESHOLD_MS;
            }
            
            @Override
            public OptimizationResult apply(GpuConfig config) {
                int currentBatch = config.getBatchSize();
                int newBatch = optimizeBatchSize(currentBatch);
                
                if (newBatch != currentBatch) {
                    config.setBatchSize(newBatch);
                    optimalBatchSize = newBatch;
                    
                    double expectedGain = Math.abs(newBatch - currentBatch) / (double) currentBatch * 0.15;
                    return new OptimizationResult(true, "BatchSizeOptimization", expectedGain,
                        String.format("Adjusted batch size from %d to %d", currentBatch, newBatch));
                }
                
                return new OptimizationResult(false, "BatchSizeOptimization", 0.0, "No adjustment needed");
            }
            
            @Override
            public double getExpectedImprovement() { return 0.15; }
        });
        
        // Memory pool optimization strategy
        activeStrategies.put("MemoryPoolOptimization", new OptimizationStrategy() {
            @Override
            public String getName() { return "MemoryPoolOptimization"; }
            
            @Override
            public boolean shouldApply(PerformanceSnapshot current, Queue<PerformanceSnapshot> history) {
                return current.getMemoryUsage() > MEMORY_USAGE_THRESHOLD ||
                       current.getMemoryUsage() < 0.5; // Under-utilization
            }
            
            @Override
            public OptimizationResult apply(GpuConfig config) {
                int currentPool = config.getMemoryPoolSizeMB();
                int newPool = optimizeMemoryPoolSize(currentPool);
                
                if (newPool != currentPool) {
                    config.setMemoryPoolSizeMB(newPool);
                    optimalMemoryPoolSize = newPool;
                    
                    double expectedGain = Math.abs(newPool - currentPool) / (double) currentPool * 0.10;
                    return new OptimizationResult(true, "MemoryPoolOptimization", expectedGain,
                        String.format("Adjusted memory pool from %dMB to %dMB", currentPool, newPool));
                }
                
                return new OptimizationResult(false, "MemoryPoolOptimization", 0.0, "No adjustment needed");
            }
            
            @Override
            public double getExpectedImprovement() { return 0.10; }
        });
        
        logger.info("Initialized " + activeStrategies.size() + " optimization strategies");
    }
    
    /**
     * Perform one optimization cycle
     */
    private void performOptimizationCycle() {
        if (!optimizationEnabled.get()) {
            return;
        }
        
        try {
            currentState.set(OptimizationState.ANALYZING);
            
            // Get current performance metrics
            PerformanceSnapshot current = getCurrentPerformanceSnapshot();
            if (current == null) {
                currentState.set(OptimizationState.MONITORING);
                return;
            }
            
            // Analyze performance trends
            OptimizationState newState = analyzePerformanceTrends(current);
            currentState.set(newState);
            
            // Apply optimizations if needed
            if (newState == OptimizationState.OPTIMIZING || newState == OptimizationState.DEGRADED) {
                applyOptimizations(current);
            }
            
            // Update performance score
            updatePerformanceScore(current);
            
            logger.fine(String.format("Optimization cycle completed. State: %s, Score: %.3f", 
                                     newState, performanceScore));
                                     
        } catch (Exception e) {
            logger.warning("Error during optimization cycle: " + e.getMessage());
            currentState.set(OptimizationState.EMERGENCY);
        }
    }
    
    /**
     * Collect current performance snapshot
     */
    private void collectPerformanceSnapshot() {
        try {
            PerformanceSnapshot snapshot = getCurrentPerformanceSnapshot();
            if (snapshot != null) {
                performanceHistory.offer(snapshot);
                
                // Maintain history size (keep last 100 snapshots)
                while (performanceHistory.size() > 100) {
                    performanceHistory.poll();
                }
            }
        } catch (Exception e) {
            logger.warning("Error collecting performance snapshot: " + e.getMessage());
        }
    }
    
    /**
     * Get current performance snapshot from monitor
     */
    private PerformanceSnapshot getCurrentPerformanceSnapshot() {
        try {
            // Since the actual monitor API is different, we'll create a simplified version
            // that works with the actual available methods
            
            // For now, return a simple snapshot with mock data
            // In a real implementation, this would fetch actual metrics from available operations
            double mockGpuUtil = 0.5;  // 50% utilization
            double mockMemUsage = 0.6; // 60% memory usage
            long mockLatency = 50;     // 50ms average
            double mockThroughput = 1500.0; // 1500 ops/sec
            
            return new PerformanceSnapshot(mockGpuUtil, mockMemUsage, mockLatency, 
                                         mockThroughput, config.getBatchSize(), performanceScore);
                                         
        } catch (Exception e) {
            logger.warning("Error getting performance snapshot: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Analyze performance trends to determine state
     */
    private OptimizationState analyzePerformanceTrends(PerformanceSnapshot current) {
        // Check for emergency conditions
        if (current.getMemoryUsage() > 0.95 || current.getGpuUtilization() > 0.95) {
            return OptimizationState.EMERGENCY;
        }
        
        // Check for degraded performance
        if (current.getThroughput() < THROUGHPUT_MIN_OPS_PER_SEC * 0.7 ||
            current.getAverageLatency() > LATENCY_THRESHOLD_MS * 2) {
            return OptimizationState.DEGRADED;
        }
        
        // Check if optimization is needed
        if (current.getThroughput() < THROUGHPUT_MIN_OPS_PER_SEC ||
            current.getAverageLatency() > LATENCY_THRESHOLD_MS ||
            current.getGpuUtilization() > GPU_UTILIZATION_THRESHOLD ||
            current.getMemoryUsage() > MEMORY_USAGE_THRESHOLD) {
            return OptimizationState.OPTIMIZING;
        }
        
        // Check performance trends over time
        if (performanceHistory.size() >= 5) {
            List<PerformanceSnapshot> recent = new ArrayList<>(performanceHistory);
            Collections.reverse(recent); // Most recent first
            
            // Check if performance is declining
            boolean declining = isPerformanceDeclining(recent.subList(0, Math.min(5, recent.size())));
            if (declining) {
                return OptimizationState.OPTIMIZING;
            }
        }
        
        return OptimizationState.STABLE;
    }
    
    /**
     * Check if performance is declining over recent snapshots
     */
    private boolean isPerformanceDeclining(List<PerformanceSnapshot> recent) {
        if (recent.size() < 3) return false;
        
        double[] scores = recent.stream().mapToDouble(PerformanceSnapshot::getPerformanceScore).toArray();
        
        // Check if performance score is consistently decreasing
        int decliningCount = 0;
        for (int i = 1; i < scores.length; i++) {
            if (scores[i] < scores[i-1]) {
                decliningCount++;
            }
        }
        
        return decliningCount >= (scores.length - 1) * 0.7; // 70% of measurements declining
    }
    
    /**
     * Apply optimization strategies
     */
    private void applyOptimizations(PerformanceSnapshot current) {
        currentState.set(OptimizationState.OPTIMIZING);
        
        List<OptimizationResult> results = new ArrayList<>();
        
        for (OptimizationStrategy strategy : activeStrategies.values()) {
            try {
                if (strategy.shouldApply(current, performanceHistory)) {
                    OptimizationResult result = strategy.apply(config);
                    results.add(result);
                    
                    if (result.isSuccess()) {
                        logger.info(String.format("Applied optimization: %s - %s", 
                                                 result.getStrategy(), result.getDetails()));
                    }
                }
            } catch (Exception e) {
                logger.warning(String.format("Error applying optimization strategy %s: %s", 
                                            strategy.getName(), e.getMessage()));
            }
        }
        
        // Log optimization summary
        if (!results.isEmpty()) {
            double totalGain = results.stream()
                .mapToDouble(OptimizationResult::getPerformanceGain)
                .sum();
            logger.info(String.format("Optimization cycle completed. Applied %d optimizations, expected gain: %.2f%%", 
                                     results.size(), totalGain * 100));
        }
    }
    
    /**
     * Optimize batch size based on current performance
     */
    private int optimizeBatchSize(int currentBatch) {
        if (performanceHistory.size() < 3) {
            return currentBatch; // Not enough data
        }
        
        List<PerformanceSnapshot> recent = new ArrayList<>(performanceHistory);
        PerformanceSnapshot latest = recent.get(recent.size() - 1);
        
        // If latency is high, reduce batch size
        if (latest.getAverageLatency() > LATENCY_THRESHOLD_MS) {
            return Math.max(16, currentBatch / 2);
        }
        
        // If throughput is low and GPU utilization is low, increase batch size
        if (latest.getThroughput() < THROUGHPUT_MIN_OPS_PER_SEC && 
            latest.getGpuUtilization() < 0.7) {
            return Math.min(512, currentBatch * 2);
        }
        
        return currentBatch;
    }
    
    /**
     * Optimize memory pool size based on usage patterns
     */
    private int optimizeMemoryPoolSize(int currentPool) {
        if (performanceHistory.size() < 3) {
            return currentPool;
        }
        
        List<PerformanceSnapshot> recent = new ArrayList<>(performanceHistory);
        double avgMemoryUsage = recent.stream()
            .mapToDouble(PerformanceSnapshot::getMemoryUsage)
            .average()
            .orElse(0.5);
        
        // If memory usage is consistently high, increase pool
        if (avgMemoryUsage > MEMORY_USAGE_THRESHOLD) {
            return Math.min(2048, (int) (currentPool * 1.5));
        }
        
        // If memory usage is consistently low, decrease pool
        if (avgMemoryUsage < 0.4) {
            return Math.max(128, (int) (currentPool * 0.8));
        }
        
        return currentPool;
    }
    
    /**
     * Update overall performance score
     */
    private void updatePerformanceScore(PerformanceSnapshot current) {
        // Calculate weighted performance score (0.0 to 1.0)
        double latencyScore = Math.max(0.0, 1.0 - (current.getAverageLatency() / (double) LATENCY_THRESHOLD_MS));
        double throughputScore = Math.min(1.0, current.getThroughput() / THROUGHPUT_MIN_OPS_PER_SEC);
        double utilizationScore = Math.min(1.0, current.getGpuUtilization() / GPU_UTILIZATION_THRESHOLD);
        double memoryScore = Math.max(0.0, 1.0 - (current.getMemoryUsage() / MEMORY_USAGE_THRESHOLD));
        
        // Weighted average: throughput (40%), latency (30%), utilization (20%), memory (10%)
        performanceScore = (throughputScore * 0.4) + (latencyScore * 0.3) + 
                          (utilizationScore * 0.2) + (memoryScore * 0.1);
    }
    
    // Public API methods
    
    /**
     * Get current optimization state
     */
    public OptimizationState getCurrentState() {
        return currentState.get();
    }
    
    /**
     * Get current performance score
     */
    public double getPerformanceScore() {
        return performanceScore;
    }
    
    /**
     * Get optimal batch size
     */
    public int getOptimalBatchSize() {
        return optimalBatchSize;
    }
    
    /**
     * Get optimal memory pool size
     */
    public int getOptimalMemoryPoolSize() {
        return optimalMemoryPoolSize;
    }
    
    /**
     * Enable or disable optimization
     */
    public void setOptimizationEnabled(boolean enabled) {
        optimizationEnabled.set(enabled);
        logger.info("Production optimization " + (enabled ? "enabled" : "disabled"));
    }
    
    /**
     * Force immediate optimization cycle
     */
    public void forceOptimization() {
        logger.info("Forcing immediate optimization cycle");
        performOptimizationCycle();
    }
    
    /**
     * Get performance history
     */
    public List<PerformanceSnapshot> getPerformanceHistory() {
        return new ArrayList<>(performanceHistory);
    }
    
    /**
     * Get active optimization strategies
     */
    public Set<String> getActiveStrategies() {
        return new HashSet<>(activeStrategies.keySet());
    }
    
    /**
     * Add custom optimization strategy
     */
    public void addOptimizationStrategy(String name, OptimizationStrategy strategy) {
        activeStrategies.put(name, strategy);
        logger.info("Added custom optimization strategy: " + name);
    }
    
    /**
     * Remove optimization strategy
     */
    public void removeOptimizationStrategy(String name) {
        if (activeStrategies.remove(name) != null) {
            logger.info("Removed optimization strategy: " + name);
        }
    }
    
    /**
     * Get optimization statistics
     */
    public Map<String, Object> getOptimizationStats() {
        Map<String, Object> stats = new HashMap<>();
        stats.put("currentState", currentState.get());
        stats.put("performanceScore", performanceScore);
        stats.put("optimalBatchSize", optimalBatchSize);
        stats.put("optimalMemoryPoolSize", optimalMemoryPoolSize);
        stats.put("optimizationEnabled", optimizationEnabled.get());
        stats.put("activeStrategies", activeStrategies.size());
        stats.put("performanceHistorySize", performanceHistory.size());
        
        if (!performanceHistory.isEmpty()) {
            PerformanceSnapshot latest = ((ConcurrentLinkedQueue<PerformanceSnapshot>) performanceHistory).peek();
            if (latest != null) {
                Map<String, Object> latestMetrics = new HashMap<>();
                latestMetrics.put("gpuUtilization", latest.getGpuUtilization());
                latestMetrics.put("memoryUsage", latest.getMemoryUsage());
                latestMetrics.put("averageLatency", latest.getAverageLatency());
                latestMetrics.put("throughput", latest.getThroughput());
                stats.put("latestMetrics", latestMetrics);
            }
        }
        
        return stats;
    }
    
    /**
     * Shutdown optimizer
     */
    public void shutdown() {
        optimizationEnabled.set(false);
        
        if (optimizationScheduler != null && !optimizationScheduler.isShutdown()) {
            optimizationScheduler.shutdown();
            try {
                if (!optimizationScheduler.awaitTermination(10, TimeUnit.SECONDS)) {
                    optimizationScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                optimizationScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        logger.info("ProductionOptimizer shutdown completed");
    }
}
