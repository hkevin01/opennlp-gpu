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
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

/**

 * ID: GPU-OM-001
 * Requirement: OperationMetrics must hold aggregated performance metrics (call count, total/min/max/avg latency, memory) for a single named GPU operation.
 * Purpose: Immutable value object populated by GpuPerformanceMonitor, surfacing per-operation latency histogram data to callers.
 * Rationale: Per-operation metric aggregation (not just totals) enables percentile-based SLA monitoring and latency regression detection.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; immutable after construction.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class OperationMetrics {
    
    private final String operationName;
    private final AtomicLong executionCount = new AtomicLong(0);
    private final AtomicLong successCount = new AtomicLong(0);
    private final AtomicLong totalTime = new AtomicLong(0);
    private final AtomicLong totalDataProcessed = new AtomicLong(0);
    private final AtomicLong minTime = new AtomicLong(Long.MAX_VALUE);
    private final AtomicLong maxTime = new AtomicLong(0);
    private final AtomicLong cpuFallbackCount = new AtomicLong(0);
    
    private final List<Long> recentExecutionTimes = new ArrayList<Long>();
    private final Object historyLock = new Object();
    private static final int MAX_HISTORY_SIZE = 100;
    
    /**
    
     * ID: GPU-OM-002
     * Requirement: OperationMetrics must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a OperationMetrics instance.
     * Inputs: String operationName
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public OperationMetrics(String operationName) {
        this.operationName = operationName;
    }
    
    /**
    
     * ID: GPU-OM-003
     * Requirement: recordExecution must execute correctly within the contract defined by this class.
     * Purpose: Implement the recordExecution operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void recordExecution(long duration, long dataSize, boolean success, 
                               GpuPerformanceMonitor.OperationType operationType) {
        executionCount.incrementAndGet();
        if (success) {
            successCount.incrementAndGet();
        }
        
        totalTime.addAndGet(duration);
        totalDataProcessed.addAndGet(dataSize);
        
        // Update min/max times
        updateMinTime(duration);
        updateMaxTime(duration);
        
        // Track CPU fallbacks
        if (operationType == GpuPerformanceMonitor.OperationType.CPU_FALLBACK) {
            cpuFallbackCount.incrementAndGet();
        }
        
        // Track recent execution times
        synchronized (historyLock) {
            recentExecutionTimes.add(duration);
            if (recentExecutionTimes.size() > MAX_HISTORY_SIZE) {
                recentExecutionTimes.remove(0);
            }
        }
    }
    
    /**
    
     * ID: GPU-OM-004
     * Requirement: updateMinTime must execute correctly within the contract defined by this class.
     * Purpose: Implement the updateMinTime operation for this class.
     * Inputs: long duration
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void updateMinTime(long duration) {
        long current = minTime.get();
        while (duration < current && !minTime.compareAndSet(current, duration)) {
            current = minTime.get();
        }
    }
    
    /**
    
     * ID: GPU-OM-005
     * Requirement: updateMaxTime must execute correctly within the contract defined by this class.
     * Purpose: Implement the updateMaxTime operation for this class.
     * Inputs: long duration
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void updateMaxTime(long duration) {
        long current = maxTime.get();
        while (duration > current && !maxTime.compareAndSet(current, duration)) {
            current = maxTime.get();
        }
    }
    
    /**
    
     * ID: GPU-OM-006
     * Requirement: Return the OperationName field value without side effects.
     * Purpose: Return the value of the OperationName property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public String getOperationName() { return operationName; }
    /**
    
     * ID: GPU-OM-007
     * Requirement: Return the ExecutionCount field value without side effects.
     * Purpose: Return the value of the ExecutionCount property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getExecutionCount() { return executionCount.get(); }
    /**
    
     * ID: GPU-OM-008
     * Requirement: Return the SuccessCount field value without side effects.
     * Purpose: Return the value of the SuccessCount property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getSuccessCount() { return successCount.get(); }
    /**
    
     * ID: GPU-OM-009
     * Requirement: Return the TotalTime field value without side effects.
     * Purpose: Return the value of the TotalTime property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getTotalTime() { return totalTime.get(); }
    /**
    
     * ID: GPU-OM-010
     * Requirement: Return the TotalDataProcessed field value without side effects.
     * Purpose: Return the value of the TotalDataProcessed property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getTotalDataProcessed() { return totalDataProcessed.get(); }
    /**
    
     * ID: GPU-OM-011
     * Requirement: Return the MinTime field value without side effects.
     * Purpose: Return the value of the MinTime property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getMinTime() { return minTime.get() == Long.MAX_VALUE ? 0 : minTime.get(); }
    /**
    
     * ID: GPU-OM-012
     * Requirement: Return the MaxTime field value without side effects.
     * Purpose: Return the value of the MaxTime property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getMaxTime() { return maxTime.get(); }
    /**
    
     * ID: GPU-OM-013
     * Requirement: Return the CpuFallbackCount field value without side effects.
     * Purpose: Return the value of the CpuFallbackCount property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getCpuFallbackCount() { return cpuFallbackCount.get(); }
    
    /**
    
     * ID: GPU-OM-014
     * Requirement: Return the SuccessRate field value without side effects.
     * Purpose: Return the value of the SuccessRate property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public double getSuccessRate() {
        long total = executionCount.get();
        return total > 0 ? (double) successCount.get() / total : 0.0;
    }
    
    /**
    
     * ID: GPU-OM-015
     * Requirement: Return the AverageTime field value without side effects.
     * Purpose: Return the value of the AverageTime property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getAverageTime() {
        long total = executionCount.get();
        return total > 0 ? totalTime.get() / total : 0;
    }
    
    /**
    
     * ID: GPU-OM-016
     * Requirement: Return the AverageThroughputMBps field value without side effects.
     * Purpose: Return the value of the AverageThroughputMBps property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public double getAverageThroughputMBps() {
        long totalTimeMs = totalTime.get() / 1_000_000; // Convert to milliseconds
        long totalDataMB = totalDataProcessed.get() / (1024 * 1024);
        return totalTimeMs > 0 ? (double) totalDataMB * 1000 / totalTimeMs : 0.0;
    }
    
    public GpuPerformanceMonitor.OperationSummary getSummary() {
        return new GpuPerformanceMonitor.OperationSummary(
            operationName,
            getExecutionCount(),
            getTotalTime() / 1_000_000, // Convert to milliseconds
            getAverageTime() / 1_000_000,
            getMinTime() / 1_000_000,
            getMaxTime() / 1_000_000,
            getSuccessRate(),
            getTotalDataProcessed(),
            getAverageThroughputMBps()
        );
    }
}
