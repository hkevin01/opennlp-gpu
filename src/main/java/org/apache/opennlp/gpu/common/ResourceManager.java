package org.apache.opennlp.gpu.common;

/**
 * Interface for managing GPU resources.
 */
public interface ResourceManager {
    
    /**
     * Initialize the resource manager.
     * 
     * @return true if initialization was successful
     */
    boolean initialize();
    
    /**
     * Release resources.
     */
    void release();
    
    /**
     * Get a memory manager.
     * 
     * @return the memory manager
     */
    MemoryManager getMemoryManager();
    
    /**
     * Release all resources.
     */
    void releaseAll();
}
