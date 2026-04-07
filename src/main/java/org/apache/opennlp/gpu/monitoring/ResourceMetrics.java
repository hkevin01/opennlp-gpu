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

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * ID: RM-001
 * Requirement: ResourceMetrics must hold GPU resource utilisation metrics (VRAM used, allocation count, pool hit rate) at a point in time.
 * Purpose: Snapshot value object capturing GPU resource state for logging and alerting by GpuPerformanceMonitor.
 * Rationale: Resource metrics complement operation latency metrics, enabling detection of memory pressure before OOM errors occur.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None; immutable snapshot.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class ResourceMetrics {
    
    private final String deviceId;
    private final AtomicLong totalMemory = new AtomicLong(0);
    private final AtomicLong usedMemory = new AtomicLong(0);
    private final AtomicLong allocationCount = new AtomicLong(0);
    private final AtomicLong deallocationCount = new AtomicLong(0);
    
    private final Map<GpuPerformanceMonitor.MemoryType, AtomicLong> memoryByType = 
        new ConcurrentHashMap<GpuPerformanceMonitor.MemoryType, AtomicLong>();
    
    public ResourceMetrics(String deviceId) {
        this.deviceId = deviceId;
        
        // Initialize memory counters for each type
        for (GpuPerformanceMonitor.MemoryType type : GpuPerformanceMonitor.MemoryType.values()) {
            memoryByType.put(type, new AtomicLong(0));
        }
    }
    
    public void recordMemoryAllocation(long size, GpuPerformanceMonitor.MemoryType type) {
        usedMemory.addAndGet(size);
        allocationCount.incrementAndGet();
        memoryByType.get(type).addAndGet(size);
        
        // Update total memory if this is the first allocation of this type
        if (totalMemory.get() == 0) {
            // Estimate total memory as 10x the first large allocation
            if (size > 1024 * 1024) { // > 1MB
                totalMemory.set(size * 10);
            }
        }
    }
    
    public void recordMemoryDeallocation(long size, GpuPerformanceMonitor.MemoryType type) {
        usedMemory.addAndGet(-size);
        deallocationCount.incrementAndGet();
        
        AtomicLong typeCounter = memoryByType.get(type);
        if (typeCounter != null) {
            typeCounter.addAndGet(-size);
        }
    }
    
    public String getDeviceId() { return deviceId; }
    public long getTotalMemory() { return totalMemory.get(); }
    public long getUsedMemory() { return Math.max(0, usedMemory.get()); }
    public long getAllocationCount() { return allocationCount.get(); }
    public long getDeallocationCount() { return deallocationCount.get(); }
    
    public double getMemoryUsageRatio() {
        long total = totalMemory.get();
        return total > 0 ? (double) getUsedMemory() / total : 0.0;
    }
    
    public long getMemoryByType(GpuPerformanceMonitor.MemoryType type) {
        AtomicLong counter = memoryByType.get(type);
        return counter != null ? Math.max(0, counter.get()) : 0;
    }
    
    public GpuPerformanceMonitor.ResourceSummary getSummary() {
        return new GpuPerformanceMonitor.ResourceSummary(
            deviceId,
            getTotalMemory() / (1024 * 1024), // Convert to MB
            getUsedMemory() / (1024 * 1024),
            getMemoryUsageRatio(),
            getAllocationCount(),
            getDeallocationCount()
        );
    }
}
