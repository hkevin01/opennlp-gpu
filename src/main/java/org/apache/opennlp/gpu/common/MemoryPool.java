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

/**
 * PHASE 2: CORE IMPLEMENTATION - MEMORY MANAGEMENT
 * 
 * Memory pooling implementation for efficient GPU memory management.
 * This class provides pooling capabilities to reduce the overhead of
 * repeated memory allocations and deallocations on the GPU.
 * 
 * Part of the OpenNLP GPU acceleration project.
 */
package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.LinkedList;
import java.util.Map;
import java.util.Queue;

import org.jocl.cl_mem;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Memory pooling for GPU buffers to reduce allocation overhead.
 * This class maintains pools of pre-allocated buffers categorized by size
 * for efficient reuse, reducing the overhead of frequent GPU memory operations.
 */
public class MemoryPool {
    private static final Logger logger = LoggerFactory.getLogger(MemoryPool.class);
    
    // Resource manager that handles actual memory allocation
    private final ResourceManager resourceManager;
    
    // Pools of buffers by size in bytes
    private final Map<Long, Queue<cl_mem>> bufferPools = new HashMap<>();
    
    // Statistics for performance monitoring
    private long totalAllocations = 0;
    private long reuseCount = 0;
    private long totalBytesAllocated = 0;
    private long currentBytesAllocated = 0;
    private long maxBytesAllocated = 0;
    
    // Configuration
    private final long maxPoolSize;
    private final boolean trackStatistics;
    
    /**
     * Creates a new memory pool with the specified resource manager.
     * 
     * @param resourceManager The resource manager to use for allocations
     * @param maxPoolSize The maximum size of the pool in bytes, or -1 for unlimited
     * @param trackStatistics Whether to track allocation statistics
     */
    public MemoryPool(ResourceManager resourceManager, long maxPoolSize, boolean trackStatistics) {
        this.resourceManager = resourceManager;
        this.maxPoolSize = maxPoolSize;
        this.trackStatistics = trackStatistics;
        
        MemoryPool.logger.info("Created memory pool with max size: {}", 
                   maxPoolSize == -1 ? "unlimited" : MemoryPool.humanReadableByteCount(maxPoolSize));
    }
    
    /**
     * Get or allocate a buffer of the specified size.
     * 
     * @param sizeInBytes The size of the buffer in bytes
     * @param readOnly Whether the buffer is read-only
     * @return A buffer of the specified size
     */
    public cl_mem getBuffer(long sizeInBytes, boolean readOnly) {
        if (sizeInBytes <= 0) {
            throw new IllegalArgumentException("Buffer size must be positive");
        }
        
        // Round up to the nearest power of 2 for better pooling
        long poolSize = MemoryPool.nextPowerOfTwo(sizeInBytes);
        
        // Try to get from pool
        Queue<cl_mem> pool = bufferPools.computeIfAbsent(poolSize, k -> new LinkedList<>());
        cl_mem buffer = pool.poll();
        
        if (buffer != null) {
            // Found in pool, reuse
            if (trackStatistics) {
                reuseCount++;
            }
            return buffer;
        }
        
        // Not found in pool, allocate new
        buffer = resourceManager.allocateBuffer((int)sizeInBytes, readOnly);
        
        if (trackStatistics) {
            totalAllocations++;
            totalBytesAllocated += sizeInBytes;
            currentBytesAllocated += sizeInBytes;
            if (currentBytesAllocated > maxBytesAllocated) {
                maxBytesAllocated = currentBytesAllocated;
            }
        }
        
        return buffer;
    }
    
    /**
     * Return a buffer to the pool for reuse.
     * 
     * @param buffer The buffer to return
     * @param sizeInBytes The size of the buffer in bytes
     */
    public void returnBuffer(cl_mem buffer, long sizeInBytes) {
        if (buffer == null) {
            return;
        }
        
        // Round up to the nearest power of 2 for better pooling
        long poolSize = MemoryPool.nextPowerOfTwo(sizeInBytes);
        
        // Check if we're exceeding max pool size
        if (maxPoolSize != -1 && currentBytesAllocated + sizeInBytes > maxPoolSize) {
            // Pool full, just release the buffer
            resourceManager.releaseBuffer(buffer);
            
            if (trackStatistics) {
                currentBytesAllocated -= sizeInBytes;
            }
            
            return;
        }
        
        // Add to pool
        Queue<cl_mem> pool = bufferPools.computeIfAbsent(poolSize, k -> new LinkedList<>());
        pool.offer(buffer);
    }
    
    /**
     * Release all buffers in the pool.
     */
    public void releaseAll() {
        MemoryPool.logger.info("Releasing all pooled memory");
        
        for (Queue<cl_mem> pool : bufferPools.values()) {
            for (cl_mem buffer : pool) {
                resourceManager.releaseBuffer(buffer);
            }
            pool.clear();
        }
        
        bufferPools.clear();
        
        if (trackStatistics) {
            currentBytesAllocated = 0;
        }
    }
    
    /**
     * Get statistics about memory usage.
     * 
     * @return A map of statistics
     */
    public Map<String, Object> getStatistics() {
        Map<String, Object> stats = new HashMap<>();
        
        if (!trackStatistics) {
            stats.put("trackingEnabled", false);
            return stats;
        }
        
        stats.put("totalAllocations", totalAllocations);
        stats.put("reuseCount", reuseCount);
        stats.put("totalBytesAllocated", totalBytesAllocated);
        stats.put("totalBytesAllocatedHuman", MemoryPool.humanReadableByteCount(totalBytesAllocated));
        stats.put("currentBytesAllocated", currentBytesAllocated);
        stats.put("currentBytesAllocatedHuman", MemoryPool.humanReadableByteCount(currentBytesAllocated));
        stats.put("maxBytesAllocated", maxBytesAllocated);
        stats.put("maxBytesAllocatedHuman", MemoryPool.humanReadableByteCount(maxBytesAllocated));
        stats.put("poolCount", bufferPools.size());
        
        double reuseRate = totalAllocations == 0 ? 0 : (double) reuseCount / (totalAllocations + reuseCount);
        stats.put("reuseRate", reuseRate);
        stats.put("reuseRatePercent", String.format("%.2f%%", reuseRate * 100));
        
        return stats;
    }
    
    /**
     * Get the next power of two greater than or equal to the given value.
     * 
     * @param value The input value
     * @return The next power of two
     */
    private static long nextPowerOfTwo(long value) {
        value--;
        value |= value >> 1;
        value |= value >> 2;
        value |= value >> 4;
        value |= value >> 8;
        value |= value >> 16;
        value |= value >> 32;
        value++;
        return value;
    }
    
    /**
     * Convert bytes to a human-readable string (KB, MB, GB, etc.).
     * 
     * @param bytes The number of bytes
     * @return A human-readable string
     */
    private static String humanReadableByteCount(long bytes) {
        if (bytes < 1024) {
            return bytes + " B";
        }
        int exp = (int) (Math.log(bytes) / Math.log(1024));
        String pre = "KMGTPE".charAt(exp - 1) + "i";
        return String.format("%.1f %sB", bytes / Math.pow(1024, exp), pre);
    }
}
