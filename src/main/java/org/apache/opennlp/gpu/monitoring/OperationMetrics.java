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
 * Metrics tracking for specific GPU operations.
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
    
    public OperationMetrics(String operationName) {
        this.operationName = operationName;
    }
    
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
    
    private void updateMinTime(long duration) {
        long current = minTime.get();
        while (duration < current && !minTime.compareAndSet(current, duration)) {
            current = minTime.get();
        }
    }
    
    private void updateMaxTime(long duration) {
        long current = maxTime.get();
        while (duration > current && !maxTime.compareAndSet(current, duration)) {
            current = maxTime.get();
        }
    }
    
    public String getOperationName() { return operationName; }
    public long getExecutionCount() { return executionCount.get(); }
    public long getSuccessCount() { return successCount.get(); }
    public long getTotalTime() { return totalTime.get(); }
    public long getTotalDataProcessed() { return totalDataProcessed.get(); }
    public long getMinTime() { return minTime.get() == Long.MAX_VALUE ? 0 : minTime.get(); }
    public long getMaxTime() { return maxTime.get(); }
    public long getCpuFallbackCount() { return cpuFallbackCount.get(); }
    
    public double getSuccessRate() {
        long total = executionCount.get();
        return total > 0 ? (double) successCount.get() / total : 0.0;
    }
    
    public long getAverageTime() {
        long total = executionCount.get();
        return total > 0 ? totalTime.get() / total : 0;
    }
    
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
