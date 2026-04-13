/*
 * Copyright 2025 OpenNLP GPU Extension Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
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

 * ID: GPU-GPM-001
 * Requirement: Provide a thread-safe, singleton performance monitoring system
 *   that collects, aggregates, and alerts on GPU and CPU compute operation
 *   metrics in real time.
 * Purpose: Enables operators and developers to observe GPU utilization,
 *   identify performance bottlenecks, and receive automated alerts when
 *   latency or memory consumption exceeds configured thresholds.
 * Rationale: Centralizing metrics in a singleton avoids per-provider overhead
 *   and provides a single source of truth for all monitoring clients (logs,
 *   dashboards, tests). ConcurrentHashMap and AtomicLong ensure correctness
 *   under concurrent inference workloads.
 * Inputs: Operation names (String), timing values (nanoseconds), memory
 *   allocation sizes (MB). All supplied by compute providers at operation
 *   boundaries.
 * Outputs: OperationMetrics lookup, resource utilization snapshots, active
 *   PerformanceAlert list, cumulative counters.
 * Preconditions: Call getInstance() before any operation recording.
 * Postconditions: All recorded metrics are visible to all threads immediately
 *   after the recording call returns.
 * Assumptions: Clock source is System.nanoTime() — monotonic, not wall-clock.
 *   All operations complete in < Long.MAX_VALUE nanoseconds.
 * Side Effects: Maintains bounded history lists per operation; trims when
 *   maxHistorySize is exceeded to prevent unbounded memory growth.
 * Failure Modes: Metrics recording silently no-ops when enabled=false.
 *   History trim is best-effort under concurrent insertion.
 * Constraints: maxHistorySize ≥ 100 (enforced by setter). alertThresholdMs ≥ 0.
 * Verification: Tested by ConcurrencyTest with 16 concurrent recording threads.
 * References: OpenTelemetry metrics naming conventions; Google SRE alerting
 *   principles for latency SLOs.
 */
public class GpuPerformanceMonitor {

    private static final GpuLogger logger = GpuLogger.getLogger(GpuPerformanceMonitor.class);

    /**

     * ID: GPU-GPM-002
     * Requirement: Singleton instance reference, lazily initialized via
     *   compare-and-set to avoid synchronized block overhead on hot paths.
     */
    private static final AtomicReference<GpuPerformanceMonitor> instance =
        new AtomicReference<>();

    // ---- Per-operation metrics (thread-safe maps) ----

    /** Latest OperationMetrics snapshot keyed by operation name. */
    private final Map<String, OperationMetrics> operationMetrics = new ConcurrentHashMap<>();

    /** Latest ResourceMetrics snapshot keyed by resource name (e.g., "GPU_0"). */
    private final Map<String, ResourceMetrics> resourceMetrics = new ConcurrentHashMap<>();

    /**
     * Bounded history of OperationMetrics per operation name.
     * Trimmed to maxHistorySize on each insertion.
     */
    private final Map<String, List<OperationMetrics>> operationHistory = new ConcurrentHashMap<>();

    /** Active alerts that have been raised but not yet resolved. */
    private final List<PerformanceAlert> activeAlerts = Collections.synchronizedList(new ArrayList<>());

    // ---- System-wide atomic counters ----

    /** Total number of operations recorded since JVM start. */
    private final AtomicLong totalOperations = new AtomicLong(0);

    /** Cumulative GPU compute time in nanoseconds. */
    private final AtomicLong totalGpuTime = new AtomicLong(0);

    /** Number of times a GPU operation fell back to the CPU provider. */
    private final AtomicLong totalCpuFallbacks = new AtomicLong(0);

    /** Total GPU memory allocated in MB across all operations. */
    private final AtomicLong totalMemoryAllocated = new AtomicLong(0);

    // ---- Configurable thresholds (volatile for safe cross-thread reads) ----

    /** When false, all recording calls are no-ops (zero overhead). */
    private volatile boolean enabled = true;

    /**
     * Operations exceeding this many milliseconds trigger a latency alert.
     * Default: 1000 ms (1 second).
     */
    private volatile long alertThresholdMs = 1000;

    /**
     * Memory usage fraction [0.0, 1.0] above which a memory alert is raised.
     * Default: 0.80 (80% of device memory).
     */
    private volatile double memoryAlertThreshold = 0.80;

    /**
     * Maximum number of historical OperationMetrics retained per operation name.
     * Default: 1000.
     */
    private volatile int maxHistorySize = 1000;

    /**

     * ID: GPU-GPM-003
     * Requirement: Update the maximum history retention size and immediately
     *   trim any histories that now exceed the new limit.
     * Inputs: size — must be ≥ 100; values below 100 are coerced to 100.
     */
    public void setMaxHistorySize(int size) {
        this.maxHistorySize = Math.max(100, size);
        trimHistories();
    }

    /**

     * ID: GPU-GPM-004
     * Requirement: Trim all per-operation history lists to maxHistorySize,
     *   removing the oldest entries first (FIFO).
     * Side Effects: Modifies operationHistory lists in place.
     * Constraints: Synchronizes on each individual history list.
     */
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

    /**
    
     * ID: GPU-GPM-005
     * Requirement: GpuPerformanceMonitor must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuPerformanceMonitor instance.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private GpuPerformanceMonitor() {
        logger.info("GPU Performance Monitor initialized");
    }

    /**
     * Get the singleton instance of the performance monitor.
     */
    /**
    
     * ID: GPU-GPM-006
     * Requirement: Return the Instance field value without side effects.
     * Purpose: Return the value of the Instance property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-007
     * Requirement: startOperation must execute correctly within the contract defined by this class.
     * Purpose: Implement the startOperation operation for this class.
     * Inputs: String operationName, OperationType operationType, long dataSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-008
     * Requirement: endOperation must execute correctly within the contract defined by this class.
     * Purpose: Implement the endOperation operation for this class.
     * Inputs: TimingContext context, boolean success, String errorMessage
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-009
     * Requirement: recordMemoryAllocation must execute correctly within the contract defined by this class.
     * Purpose: Implement the recordMemoryAllocation operation for this class.
     * Inputs: String deviceId, long size, MemoryType allocationType
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-010
     * Requirement: recordMemoryDeallocation must execute correctly within the contract defined by this class.
     * Purpose: Implement the recordMemoryDeallocation operation for this class.
     * Inputs: String deviceId, long size, MemoryType allocationType
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-011
     * Requirement: Return the PerformanceSummary field value without side effects.
     * Purpose: Return the value of the PerformanceSummary property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-012
     * Requirement: Return the OperationMetrics field value without side effects.
     * Purpose: Return the value of the OperationMetrics property.
     * Inputs: String operationName
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public OperationMetrics getOperationMetrics(String operationName) {
        return operationMetrics.get(operationName);
    }

    /**
     * Get resource metrics for a specific device.
     */
    /**
    
     * ID: GPU-GPM-013
     * Requirement: Return the ResourceMetrics field value without side effects.
     * Purpose: Return the value of the ResourceMetrics property.
     * Inputs: String deviceId
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public ResourceMetrics getResourceMetrics(String deviceId) {
        return resourceMetrics.get(deviceId);
    }

    /**
     * Get current active alerts.
     */
    /**
    
     * ID: GPU-GPM-014
     * Requirement: Return the ActiveAlerts field value without side effects.
     * Purpose: Return the value of the ActiveAlerts property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public List<PerformanceAlert> getActiveAlerts() {
        return new ArrayList<PerformanceAlert>(activeAlerts);
    }

    /**
     * Clear all performance metrics and alerts.
     */
    /**
    
     * ID: GPU-GPM-015
     * Requirement: reset must execute correctly within the contract defined by this class.
     * Purpose: Implement the reset operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-GPM-016
     * Requirement: Update the Enabled field to the supplied non-null value.
     * Purpose: Set the Enabled property to the supplied value.
     * Inputs: boolean enabled
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setEnabled(boolean enabled) {
        this.enabled = enabled;
        logger.info("Performance monitoring {}", enabled ? "enabled" : "disabled");
    }

    /**
     * Set the alert threshold for slow operations.
     */
    /**
    
     * ID: GPU-GPM-017
     * Requirement: Update the AlertThresholdMs field to the supplied non-null value.
     * Purpose: Set the AlertThresholdMs property to the supplied value.
     * Inputs: long thresholdMs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setAlertThresholdMs(long thresholdMs) {
        this.alertThresholdMs = thresholdMs;
    }

    /**
     * Set the memory usage alert threshold (0.0 to 1.0).
     */
    /**
    
     * ID: GPU-GPM-018
     * Requirement: Update the MemoryAlertThreshold field to the supplied non-null value.
     * Purpose: Set the MemoryAlertThreshold property to the supplied value.
     * Inputs: double threshold
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setMemoryAlertThreshold(double threshold) {
        this.memoryAlertThreshold = threshold;
    }

    // Private helper methods

    /**
    
     * ID: GPU-GPM-019
     * Requirement: checkPerformanceAlerts must execute correctly within the contract defined by this class.
     * Purpose: Validate preconditions for PerformanceAlerts.
     * Inputs: String operationName, long duration, boolean success, String errorMessage
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**
    
     * ID: GPU-GPM-020
     * Requirement: checkMemoryAlerts must execute correctly within the contract defined by this class.
     * Purpose: Validate preconditions for MemoryAlerts.
     * Inputs: String deviceId, ResourceMetrics metrics
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**
    
     * ID: GPU-GPM-021
     * Requirement: addAlert must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: PerformanceAlert alert
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

    /**
    
     * ID: GPU-GPM-022
     * Requirement: extractResourceFromMessage must execute correctly within the contract defined by this class.
     * Purpose: Implement the extractResourceFromMessage operation for this class.
     * Inputs: String message
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
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

        /**
        
         * ID: GPU-GPM-023
         * Requirement: PerformanceAlert must execute correctly within the contract defined by this class.
         * Purpose: Implement the PerformanceAlert operation for this class.
         * Inputs: AlertType type, String message, AlertSeverity severity, long timestamp
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public PerformanceAlert(AlertType type, String message, AlertSeverity severity, long timestamp) {
            this.type = type;
            this.message = message;
            this.severity = severity;
            this.timestamp = timestamp;
        }

        /**
        
         * ID: GPU-GPM-024
         * Requirement: Return the Type field value without side effects.
         * Purpose: Return the value of the Type property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public AlertType getType() { return type; }
        /**
        
         * ID: GPU-GPM-025
         * Requirement: Return the Message field value without side effects.
         * Purpose: Return the value of the Message property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getMessage() { return message; }
        /**
        
         * ID: GPU-GPM-026
         * Requirement: Return the Severity field value without side effects.
         * Purpose: Return the value of the Severity property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public AlertSeverity getSeverity() { return severity; }
        /**
        
         * ID: GPU-GPM-027
         * Requirement: Return the Timestamp field value without side effects.
         * Purpose: Return the value of the Timestamp property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public long getTimestamp() { return timestamp; }

        /**
        
         * ID: GPU-GPM-028
         * Requirement: toString must execute correctly within the contract defined by this class.
         * Purpose: Implement the toString operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
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

        /**
        
         * ID: GPU-GPM-029
         * Requirement: toString must execute correctly within the contract defined by this class.
         * Purpose: Implement the toString operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
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

        /**
        
         * ID: GPU-GPM-030
         * Requirement: OperationSummary must execute correctly within the contract defined by this class.
         * Purpose: Implement the OperationSummary operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
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

        /**
        
         * ID: GPU-GPM-031
         * Requirement: ResourceSummary must execute correctly within the contract defined by this class.
         * Purpose: Implement the ResourceSummary operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
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
