package org.apache.opennlp.gpu.common;

import java.util.Map;

/**
 * Manages hardware resources such as memory buffers, contexts, and compiled kernels.
 * Provides caching and resource pooling to minimize allocation overhead.
 */
public interface ResourceManager {
    
    /**
     * Allocate a memory buffer of the specified size.
     *
     * @param size the size in bytes
     * @param type the type of memory (e.g., "host", "device")
     * @return a handle to the allocated buffer
     */
    Object allocateBuffer(long size, String type);
    
    /**
     * Release a previously allocated buffer.
     *
     * @param buffer the buffer to release
     */
    void releaseBuffer(Object buffer);
    
    /**
     * Get or create a compiled kernel.
     *
     * @param kernelName the name of the kernel
     * @param kernelSource the source code of the kernel
     * @return a handle to the compiled kernel
     */
    Object getOrCreateKernel(String kernelName, String kernelSource);
    
    /**
     * Cache data for frequent use.
     *
     * @param key the cache key
     * @param data the data to cache
     */
    void cacheData(String key, Object data);
    
    /**
     * Retrieve cached data.
     *
     * @param key the cache key
     * @return the cached data, or null if not found
     */
    Object getCachedData(String key);
    
    /**
     * Clear all cached data.
     */
    void clearCache();
    
    /**
     * Get statistics about resource usage.
     *
     * @return a map of statistics names to values
     */
    Map<String, Object> getStatistics();
    
    /**
     * Release all resources managed by this resource manager.
     */
    void releaseAll();
}
