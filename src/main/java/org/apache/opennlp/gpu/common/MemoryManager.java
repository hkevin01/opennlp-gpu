package org.apache.opennlp.gpu.common;

public interface MemoryManager {
    int allocate(long size);
    void free(long ptr);
    void copyHostToDevice(long devicePtr, byte[] hostData, long size);
    void copyDeviceToHost(long devicePtr, byte[] hostData, long size);
    void releaseAll();
}
