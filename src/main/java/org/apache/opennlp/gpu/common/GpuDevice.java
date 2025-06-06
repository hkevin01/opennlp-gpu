package org.apache.opennlp.gpu.common;

/**
 * Interface representing a GPU device.
 */
public interface GpuDevice {
    String getName();
    int getDeviceId();
    
    // Add missing method referenced in GpuDemoMain
    long getMemoryMB();
}
