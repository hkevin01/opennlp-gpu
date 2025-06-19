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
import java.util.Map;

/**
 * Analyzes performance data and generates optimization recommendations.
 */
public class PerformanceAnalyzer {
    
    /**
     * Generate performance optimization recommendations based on collected metrics.
     */
    public List<String> generateRecommendations(Map<String, OperationMetrics> operationMetrics,
                                               Map<String, ResourceMetrics> resourceMetrics) {
        List<String> recommendations = new ArrayList<String>();
        
        // Analyze operation performance
        analyzeOperationPerformance(operationMetrics, recommendations);
        
        // Analyze resource utilization
        analyzeResourceUtilization(resourceMetrics, recommendations);
        
        // Analyze CPU fallback patterns
        analyzeCpuFallbacks(operationMetrics, recommendations);
        
        // Analyze memory usage patterns
        analyzeMemoryUsage(resourceMetrics, recommendations);
        
        return recommendations;
    }
    
    private void analyzeOperationPerformance(Map<String, OperationMetrics> operationMetrics,
                                           List<String> recommendations) {
        for (OperationMetrics metrics : operationMetrics.values()) {
            String operationName = metrics.getOperationName();
            
            // Check for slow operations
            if (metrics.getAverageTime() > 100_000_000) { // > 100ms
                recommendations.add(String.format(
                    "Operation '%s' is slow (avg: %dms). Consider optimization or GPU acceleration.",
                    operationName, metrics.getAverageTime() / 1_000_000
                ));
            }
            
            // Check for high variance in execution times
            long avgTime = metrics.getAverageTime();
            long maxTime = metrics.getMaxTime();
            if (avgTime > 0 && maxTime > avgTime * 5) {
                recommendations.add(String.format(
                    "Operation '%s' has inconsistent performance (max: %dms, avg: %dms). " +
                    "Check for memory pressure or resource contention.",
                    operationName, maxTime / 1_000_000, avgTime / 1_000_000
                ));
            }
            
            // Check for low success rates
            if (metrics.getSuccessRate() < 0.95 && metrics.getExecutionCount() > 10) {
                recommendations.add(String.format(
                    "Operation '%s' has low success rate (%.1f%%). Check error handling and resource availability.",
                    operationName, metrics.getSuccessRate() * 100
                ));
            }
            
            // Check for low throughput
            double throughput = metrics.getAverageThroughputMBps();
            if (throughput > 0 && throughput < 10) { // < 10 MB/s
                recommendations.add(String.format(
                    "Operation '%s' has low throughput (%.1f MB/s). Consider batch processing or GPU optimization.",
                    operationName, throughput
                ));
            }
        }
    }
    
    private void analyzeResourceUtilization(Map<String, ResourceMetrics> resourceMetrics,
                                          List<String> recommendations) {
        for (ResourceMetrics metrics : resourceMetrics.values()) {
            String deviceId = metrics.getDeviceId();
            double memoryUsage = metrics.getMemoryUsageRatio();
            
            // Check for high memory usage
            if (memoryUsage > 0.8) {
                recommendations.add(String.format(
                    "Device '%s' has high memory usage (%.1f%%). Consider memory optimization or cleanup.",
                    deviceId, memoryUsage * 100
                ));
            }
            
            // Check for memory leaks
            long allocations = metrics.getAllocationCount();
            long deallocations = metrics.getDeallocationCount();
            if (allocations > 100 && deallocations < allocations * 0.8) {
                recommendations.add(String.format(
                    "Device '%s' may have memory leaks (%d allocations, %d deallocations). " +
                    "Check resource cleanup.",
                    deviceId, allocations, deallocations
                ));
            }
            
            // Check for fragmented allocations
            if (allocations > deallocations * 2) {
                recommendations.add(String.format(
                    "Device '%s' has many small allocations. Consider memory pooling or batch allocation.",
                    deviceId
                ));
            }
        }
    }
    
    private void analyzeCpuFallbacks(Map<String, OperationMetrics> operationMetrics,
                                   List<String> recommendations) {
        long totalOperations = 0;
        long totalFallbacks = 0;
        
        for (OperationMetrics metrics : operationMetrics.values()) {
            totalOperations += metrics.getExecutionCount();
            totalFallbacks += metrics.getCpuFallbackCount();
            
            // Check per-operation fallback rates
            if (metrics.getExecutionCount() > 10) {
                double fallbackRate = (double) metrics.getCpuFallbackCount() / metrics.getExecutionCount();
                if (fallbackRate > 0.3) {
                    recommendations.add(String.format(
                        "Operation '%s' has high CPU fallback rate (%.1f%%). " +
                        "Check GPU availability and operation size thresholds.",
                        metrics.getOperationName(), fallbackRate * 100
                    ));
                }
            }
        }
        
        // Check overall fallback rate
        if (totalOperations > 0) {
            double overallFallbackRate = (double) totalFallbacks / totalOperations;
            if (overallFallbackRate > 0.2) {
                recommendations.add(String.format(
                    "High overall CPU fallback rate (%.1f%%). " +
                    "Consider GPU driver updates or hardware diagnostics.",
                    overallFallbackRate * 100
                ));
            }
        }
    }
    
    private void analyzeMemoryUsage(Map<String, ResourceMetrics> resourceMetrics,
                                  List<String> recommendations) {
        long totalMemoryUsed = 0;
        long totalMemoryAvailable = 0;
        
        for (ResourceMetrics metrics : resourceMetrics.values()) {
            totalMemoryUsed += metrics.getUsedMemory();
            totalMemoryAvailable += metrics.getTotalMemory();
            
            // Check for GPU vs CPU memory balance
            long gpuMemory = metrics.getMemoryByType(GpuPerformanceMonitor.MemoryType.GPU);
            long cpuMemory = metrics.getMemoryByType(GpuPerformanceMonitor.MemoryType.CPU);
            
            if (gpuMemory > 0 && cpuMemory > gpuMemory * 10) {
                recommendations.add(String.format(
                    "Device '%s' is using mostly CPU memory (%d MB) vs GPU memory (%d MB). " +
                    "Consider moving more operations to GPU.",
                    metrics.getDeviceId(), cpuMemory / (1024 * 1024), gpuMemory / (1024 * 1024)
                ));
            }
        }
        
        // Overall memory efficiency
        if (totalMemoryAvailable > 0) {
            double overallUsage = (double) totalMemoryUsed / totalMemoryAvailable;
            if (overallUsage < 0.1) {
                recommendations.add(
                    "Low overall memory utilization. Consider increasing batch sizes or operation complexity."
                );
            } else if (overallUsage > 0.9) {
                recommendations.add(
                    "Very high memory utilization. Consider reducing batch sizes or implementing memory pooling."
                );
            }
        }
    }
}
