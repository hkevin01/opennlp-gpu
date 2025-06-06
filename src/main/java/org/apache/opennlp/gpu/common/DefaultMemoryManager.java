package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.Map;

public class DefaultMemoryManager implements MemoryManager {
    private final Map<Long, byte[]> memoryBlocks = new HashMap<>();
    private long nextHandle = 1;
    
    public DefaultMemoryManager() {
        // Default constructor
    }
    
    @Override
    public int allocate(long size) {
        if (size > Integer.MAX_VALUE) {
            throw new IllegalArgumentException("Size too large");
        }
        byte[] block = new byte[(int)size];
        long handle = nextHandle++;
        memoryBlocks.put(handle, block);
        return (int)handle;
    }
    
    @Override
    public void free(long ptr) {
        memoryBlocks.remove(ptr);
    }
    
    @Override
    public void copyHostToDevice(long devicePtr, byte[] hostData, long size) {
        byte[] deviceMem = memoryBlocks.get(devicePtr);
        if (deviceMem != null && hostData != null) {
            int copySize = (int)Math.min(size, Math.min(deviceMem.length, hostData.length));
            System.arraycopy(hostData, 0, deviceMem, 0, copySize);
        }
    }
    
    @Override
    public void copyDeviceToHost(long devicePtr, byte[] hostData, long size) {
        byte[] deviceMem = memoryBlocks.get(devicePtr);
        if (deviceMem != null && hostData != null) {
            int copySize = (int)Math.min(size, Math.min(deviceMem.length, hostData.length));
            System.arraycopy(deviceMem, 0, hostData, 0, copySize);
        }
    }
    
    @Override
    public void releaseAll() {
        memoryBlocks.clear();
        nextHandle = 1;
    }
}
