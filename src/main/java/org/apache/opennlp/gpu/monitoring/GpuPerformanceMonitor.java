/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.opennlp.gpu.monitoring;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;
import java.util.concurrent.atomic.AtomicReference;

import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * Advanced performance monitoring system for GPU operations.
 * Provides real-time metrics, performance analysis, and automated optimization suggestions.
 * 
 * Features:
 * - Real-time performance tracking
 * - Memory usage monitoring
 * - Operation timing and throughput analysis
 * - Automatic performance alerts
 * - Historical trend analysis
 * - Resource utilization tracking
 * 
 * @author OpenNLP GPU Team
 * @since 2.0.0
 */
public class GpuPerformanceMonitor {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuPerformanceMonitor.class);
    
    // Singleton instance
    private static final AtomicReference<GpuPerformanceMonitor> instance = 
        new AtomicReference<GpuPerformanceMonitor>();
    
    // Performance metrics storage
    private final Map<String, OperationMetrics> operationMetrics = new ConcurrentHashMap<String, OperationMetrics>();
    private final Map<String, ResourceMetrics> resourceMetrics = new ConcurrentHashMap<String, ResourceMetrics>();
    private final Map<String, List<OperationMetrics>> operationHistory = new ConcurrentHashMap<String, List<OperationMetrics>>();
    private final List<PerformanceAlert> activeAlerts = Collections.synchronizedList(new ArrayList<PerformanceAlert>());
    
    // System-wide counters
    private final AtomicLong totalOperations = new AtomicLong(0);
    private final AtomicLong totalGpuTime = new AtomicLong(0);
    private final AtomicLong totalCpuFallbacks = new AtomicLong(0);
    private final AtomicLong totalMemoryAllocated = new AtomicLong(0);
    
    // Configuration
    private volatile boolean enabled = true;
    private volatile long alertThresholdMs = 1000; // 1 second
    private volatile double memoryAlertThreshold = 0.8; // 80%
    private volatile int maxHistorySize = 1000;
    
    public void setMaxHistorySize(int size) {
        this.maxHistorySize = Math.max(100, size);
        // Trim existing histories if needed
        trimHistories();
    }
    
    private void trimHistories() {
        for (Map.Entry<String, List<OperationMetrics>> entry : operationHistory.entrySet()) {
            List<OperationMetrics> history = entry.getValue();
            if (history.size() > maxHistorySize) {
                history.subList(0, history.size() - maxHistorySize).clear();
            }
        }
    }
    
    // Performance analysis
    private final PerformanceAnalyzer analyzer = new PerformanceAnalyzer();
    
    private GpuPerformanceMonitor() {
        logger.info("GPU Performance Monitor initialized");
    }
    
    /**
     * Get the singleton instance of the performance monitor.
     */
    public static GpuPerformanceMonitor getInstance() {
        GpuPerformanceMonitor monitor = instance.get();
        if (monitor == null) {
            monitor = new GpuPerformanceMonitor();
            if (!instance.compareAndSet(null, monitor)) {
                monitor = instance.get();
            }
        }
        return monitor;
    }
    
    /**
     * Record the start of an operation.
     * 
     * @param operationName the name of the operation
     * @param operationType the type of operation (GPU, CPU, MEMORY)
     * @param dataSize the size of data being processed
     * @return a timing context for ending the operation
     */
    public TimingContext startOperation(String operationName, OperationType operationType, long dataSize) {
        if (!enabled) {
            return TimingContext.NOOP;
        }
        
        totalOperations.incrementAndGet();
        
        return new TimingContext(operationName, operationType, dataSize, System.nanoTime());
    }
    
    /**
     * Record the completion of an operation.
     * 
     * @param context the timing context from startOperation
     * @param success whether the operation completed successfully
     * @param errorMessage error message if the operation failed
     */
    public void endOperation(TimingContext context, boolean success, String errorMessage) {
        if (!enabled || context == TimingContext.NOOP) {
            return;
        }
        
        long endTime = System.nanoTime();
        long duration = endTime - context.startTime;
        
        // Update operation metrics
        OperationMetrics metrics = operationMetrics.computeIfAbsent(
            context.operationName, 
            k -> new OperationMetrics(context.operationName)
        );
        
        metrics.recordExecution(duration, context.dataSize, success, context.operationType);
        
        // Update system counters
        if (context.operationType == OperationType.GPU) {
            totalGpuTime.addAndGet(duration);
        } else if (context.operationType == OperationType.CPU_FALLBACK) {
            totalCpuFallbacks.incrementAndGet();
        }
        
        // Check for performance alerts
        checkPerformanceAlerts(context.operationName, duration, success, errorMessage);
        
        // Log slow operations
        if (duration > alertThresholdMs * 1_000_000) { // Convert to nanoseconds
            logger.warn("Slow operation detected: {} took {}ms", 
                       context.operationName, duration / 1_000_000);
        }
    }
    
    /**
     * Record memory allocation.
     * 
     * @param deviceId the device ID
     * @param size the size of memory allocated
     * @param allocationType the type of allocation (GPU, CPU, SHARED)
     */
    public void recordMemoryAllocation(String deviceId, long size, MemoryType allocationType) {
        if (!enabled) {
            return;
        }
        
        ResourceMetrics metrics = resourceMetrics.computeIfAbsent(
            deviceId, 
            k -> new ResourceMetrics(deviceId)
        );
        
        metrics.recordMemoryAllocation(size, allocationType);
        totalMemoryAllocated.addAndGet(size);
        
        // Check memory usage alerts
        checkMemoryAlerts(deviceId, metrics);
    }
    
    /**
     * Record memory deallocation.
     * 
     * @param deviceId the device ID
     * @param size the size of memory deallocated
     * @param allocationType the type of allocation being freed
     */
    public void recordMemoryDeallocation(String deviceId, long size, MemoryType allocationType) {
        if (!enabled) {
            return;
        }
        
        ResourceMetrics metrics = resourceMetrics.get(deviceId);
        if (metrics != null) {
            metrics.recordMemoryDeallocation(size, allocationType);
        }
    }
    
    /**
     * Get performance summary for all operations.
     */
    public PerformanceSummary getPerformanceSummary() {
        PerformanceSummary summary = new PerformanceSummary();
        
        summary.totalOperations = totalOperations.get();
        summary.totalGpuTimeMs = totalGpuTime.get() / 1_000_000;
        summary.totalCpuFallbacks = totalCpuFallbacks.get();
        summary.totalMemoryAllocatedMB = totalMemoryAllocated.get() / (1024 * 1024);
        
        // Calculate overall statistics
        long totalTime = 0;
        long totalDataProcessed = 0;
        
        for (OperationMetrics metrics : operationMetrics.values()) {
            totalTime += metrics.getTotalTime();
            totalDataProcessed += metrics.getTotalDataProcessed();
            summary.operationSummaries.put(metrics.getOperationName(), metrics.getSummary());
        }
        
        summary.averageOperationTimeMs = summary.totalOperations > 0 ? 
            totalTime / 1_000_000 / summary.totalOperations : 0;
        summary.totalDataProcessedMB = totalDataProcessed / (1024 * 1024);
        
        // Add resource summaries
        for (ResourceMetrics metrics : resourceMetrics.values()) {
            summary.resourceSummaries.put(metrics.getDeviceId(), metrics.getSummary());
        }
        
        // Add active alerts
        summary.activeAlerts.addAll(activeAlerts);
        
        // Generate recommendations
        List<String> generatedRecommendations = analyzer.generateRecommendations(operationMetrics, resourceMetrics);
        summary.recommendations.clear();
        summary.recommendations.addAll(generatedRecommendations);
        
        return summary;
    }
    
    /**
     * Get performance metrics for a specific operation.
     */
    public OperationMetrics getOperationMetrics(String operationName) {
        return operationMetrics.get(operationName);
    }
    
    /**
     * Get resource metrics for a specific device.
     */
    public ResourceMetrics getResourceMetrics(String deviceId) {
        return resourceMetrics.get(deviceId);
    }
    
    /**
     * Get current active alerts.
     */
    public List<PerformanceAlert> getActiveAlerts() {
        return new ArrayList<PerformanceAlert>(activeAlerts);
    }
    
    /**
     * Clear all performance metrics and alerts.
     */
    public void reset() {
        operationMetrics.clear();
        resourceMetrics.clear();
        activeAlerts.clear();
        totalOperations.set(0);
        totalGpuTime.set(0);
        totalCpuFallbacks.set(0);
        totalMemoryAllocated.set(0);
        
        logger.info("Performance metrics reset");
    }
    
    /**
     * Enable or disable performance monitoring.
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        logger.info("Performance monitoring {}", enabled ? "enabled" : "disabled");
    }
    
    /**
     * Set the alert threshold for slow operations.
     */
    public void setAlertThresholdMs(long thresholdMs) {
        this.alertThresholdMs = thresholdMs;
    }
    
    /**
     * Set the memory usage alert threshold (0.0 to 1.0).
     */
    public void setMemoryAlertThreshold(double threshold) {
        this.memoryAlertThreshold = threshold;
    }
    
    // Private helper methods
    
    private void checkPerformanceAlerts(String operationName, long duration, boolean success, String errorMessage) {
        // Check for slow operations
        if (duration > alertThresholdMs * 1_000_000) {
            PerformanceAlert alert = new PerformanceAlert(
                AlertType.SLOW_OPERATION,
                "Operation " + operationName + " took " + (duration / 1_000_000) + "ms",
                AlertSeverity.WARNING,
                System.currentTimeMillis()
            );
            addAlert(alert);
        }
        
        // Check for failures
        if (!success) {
            PerformanceAlert alert = new PerformanceAlert(
                AlertType.OPERATION_FAILURE,
                "Operation " + operationName + " failed: " + errorMessage,
                AlertSeverity.ERROR,
                System.currentTimeMillis()
            );
            addAlert(alert);
        }
        
        // Check CPU fallback rate
        OperationMetrics metrics = operationMetrics.get(operationName);
        if (metrics != null && metrics.getExecutionCount() > 10) {
            double fallbackRate = (double) metrics.getCpuFallbackCount() / metrics.getExecutionCount();
            if (fallbackRate > 0.5) {
                PerformanceAlert alert = new PerformanceAlert(
                    AlertType.HIGH_CPU_FALLBACK_RATE,
                    "Operation " + operationName + " has high CPU fallback rate: " + 
                    String.format("%.1f%%", fallbackRate * 100),
                    AlertSeverity.WARNING,
                    System.currentTimeMillis()
                );
                addAlert(alert);
            }
        }
    }
    
    private void checkMemoryAlerts(String deviceId, ResourceMetrics metrics) {
        double memoryUsage = metrics.getMemoryUsageRatio();
        
        if (memoryUsage > memoryAlertThreshold) {
            PerformanceAlert alert = new PerformanceAlert(
                AlertType.HIGH_MEMORY_USAGE,
                "Device " + deviceId + " memory usage is " + 
                String.format("%.1f%%", memoryUsage * 100),
                memoryUsage > 0.9 ? AlertSeverity.ERROR : AlertSeverity.WARNING,
                System.currentTimeMillis()
            );
            addAlert(alert);
        }
    }
    
    private void addAlert(PerformanceAlert alert) {
        // Remove old alerts of the same type for the same resource
        activeAlerts.removeIf(existing -> 
            existing.getType() == alert.getType() && 
            existing.getMessage().contains(extractResourceFromMessage(alert.getMessage()))
        );
        
        activeAlerts.add(alert);
        
        // Limit the number of active alerts
        if (activeAlerts.size() > 100) {
            activeAlerts.subList(0, activeAlerts.size() - 100).clear();
        }
        
        logger.warn("Performance alert: {}", alert.getMessage());
    }
    
    private String extractResourceFromMessage(String message) {
        // Simple extraction of resource name from alert message
        if (message.contains("Operation ")) {
            int start = message.indexOf("Operation ") + "Operation ".length();
            int end = message.indexOf(" ", start);
            return end > start ? message.substring(start, end) : "";
        } else if (message.contains("Device ")) {
            int start = message.indexOf("Device ") + "Device ".length();
            int end = message.indexOf(" ", start);
            return end > start ? message.substring(start, end) : "";
        }
        return "";
    }
    
    // Inner classes and enums
    
    public enum OperationType {
        GPU, CPU_FALLBACK, MEMORY, OTHER
    }
    
    public enum MemoryType {
        GPU, CPU, SHARED
    }
    
    public enum AlertType {
        SLOW_OPERATION, OPERATION_FAILURE, HIGH_MEMORY_USAGE, HIGH_CPU_FALLBACK_RATE, RESOURCE_EXHAUSTION
    }
    
    public enum AlertSeverity {
        INFO, WARNING, ERROR, CRITICAL
    }
    
    /**
     * Context for timing operations.
     */
    public static class TimingContext {
        public static final TimingContext NOOP = new TimingContext("", OperationType.OTHER, 0, 0);
        
        final String operationName;
        final OperationType operationType;
        final long dataSize;
        final long startTime;
        
        TimingContext(String operationName, OperationType operationType, long dataSize, long startTime) {
            this.operationName = operationName;
            this.operationType = operationType;
            this.dataSize = dataSize;
            this.startTime = startTime;
        }
    }
    
    /**
     * Performance alert information.
     */
    public static class PerformanceAlert {
        private final AlertType type;
        private final String message;
        private final AlertSeverity severity;
        private final long timestamp;
        
        public PerformanceAlert(AlertType type, String message, AlertSeverity severity, long timestamp) {
            this.type = type;
            this.message = message;
            this.severity = severity;
            this.timestamp = timestamp;
        }
        
        public AlertType getType() { return type; }
        public String getMessage() { return message; }
        public AlertSeverity getSeverity() { return severity; }
        public long getTimestamp() { return timestamp; }
        
        @Override
        public String toString() {
            return String.format("[%s] %s: %s", severity, type, message);
        }
    }
    
    /**
     * Overall performance summary.
     */
    public static class PerformanceSummary {
        public long totalOperations;
        public long totalGpuTimeMs;
        public long totalCpuFallbacks;
        public long totalMemoryAllocatedMB;
        public long averageOperationTimeMs;
        public long totalDataProcessedMB;
        
        public final Map<String, OperationSummary> operationSummaries = new ConcurrentHashMap<String, OperationSummary>();
        public final Map<String, ResourceSummary> resourceSummaries = new ConcurrentHashMap<String, ResourceSummary>();
        public final List<PerformanceAlert> activeAlerts = new ArrayList<PerformanceAlert>();
        public final List<String> recommendations = new ArrayList<String>();
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append("=== GPU Performance Summary ===\n");
            sb.append(String.format("Total Operations: %d\n", totalOperations));
            sb.append(String.format("Total GPU Time: %d ms\n", totalGpuTimeMs));
            sb.append(String.format("CPU Fallbacks: %d\n", totalCpuFallbacks));
            sb.append(String.format("Memory Allocated: %d MB\n", totalMemoryAllocatedMB));
            sb.append(String.format("Average Operation Time: %d ms\n", averageOperationTimeMs));
            sb.append(String.format("Data Processed: %d MB\n", totalDataProcessedMB));
            
            if (!activeAlerts.isEmpty()) {
                sb.append("\n=== Active Alerts ===\n");
                for (PerformanceAlert alert : activeAlerts) {
                    sb.append(alert.toString()).append("\n");
                }
            }
            
            if (!recommendations.isEmpty()) {
                sb.append("\n=== Recommendations ===\n");
                for (String recommendation : recommendations) {
                    sb.append("- ").append(recommendation).append("\n");
                }
            }
            
            return sb.toString();
        }
    }
    
    /**
     * Summary for a specific operation.
     */
    public static class OperationSummary {
        public final String operationName;
        public final long executionCount;
        public final long totalTimeMs;
        public final long averageTimeMs;
        public final long minTimeMs;
        public final long maxTimeMs;
        public final double successRate;
        public final long totalDataProcessed;
        public final double averageThroughputMBps;
        
        public OperationSummary(String operationName, long executionCount, long totalTimeMs,
                               long averageTimeMs, long minTimeMs, long maxTimeMs,
                               double successRate, long totalDataProcessed, double averageThroughputMBps) {
            this.operationName = operationName;
            this.executionCount = executionCount;
            this.totalTimeMs = totalTimeMs;
            this.averageTimeMs = averageTimeMs;
            this.minTimeMs = minTimeMs;
            this.maxTimeMs = maxTimeMs;
            this.successRate = successRate;
            this.totalDataProcessed = totalDataProcessed;
            this.averageThroughputMBps = averageThroughputMBps;
        }
    }
    
    /**
     * Summary for a specific resource/device.
     */
    public static class ResourceSummary {
        public final String deviceId;
        public final long totalMemoryMB;
        public final long usedMemoryMB;
        public final double memoryUsageRatio;
        public final long allocationCount;
        public final long deallocationCount;
        
        public ResourceSummary(String deviceId, long totalMemoryMB, long usedMemoryMB,
                              double memoryUsageRatio, long allocationCount, long deallocationCount) {
            this.deviceId = deviceId;
            this.totalMemoryMB = totalMemoryMB;
            this.usedMemoryMB = usedMemoryMB;
            this.memoryUsageRatio = memoryUsageRatio;
            this.allocationCount = allocationCount;
            this.deallocationCount = deallocationCount;
        }
    }
}
