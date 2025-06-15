package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * CPU compute provider implementation
 */
public class CpuComputeProvider implements ComputeProvider {
    
    private static final GpuLogger logger = GpuLogger.getLogger(CpuComputeProvider.class);
    
    @Override
    public boolean isGpuProvider() {
        return false;
    }
    
    @Override
    public void cleanup() {
        CpuComputeProvider.logger.debug("CPU compute provider cleaned up");
    }
    
    @Override
    public String getName() {
        return "CPU Provider";
    }
    
    @Override
    public Type getType() {
        return Type.CPU;
    }
    
    @Override
    public boolean isAvailable() {
        return true; // CPU is always available
    }
    
    @Override
    public long getMaxMemoryMB() {
        return Runtime.getRuntime().maxMemory() / (1024 * 1024);
    }
    
    @Override
    public long getCurrentMemoryUsageMB() {
        return (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory()) / (1024 * 1024);
    }
    
    @Override
    public Object getResourceManager() {
        return null; // CPU doesn't need resource management
    }
    
    @Override
    public void matrixMultiply(float[] a, float[] b, float[] result, int m, int n, int k) {
        // Simple CPU matrix multiplication
        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                float sum = 0.0f;
                for (int l = 0; l < k; l++) {
                    sum += a[i * k + l] * b[l * n + j];
                }
                result[i * n + j] = sum;
            }
        }
    }
    
    @Override
    public void matrixAdd(float[] a, float[] b, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = a[i] + b[i];
        }
    }
    
    @Override
    public void matrixTranspose(float[] input, float[] output, int rows, int cols) {
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                output[j * rows + i] = input[i * cols + j];
            }
        }
    }
    
    @Override
    public void extractFeatures(String[] text, float[] features) {
        // Simple feature extraction placeholder
        for (int i = 0; i < features.length && i < text.length; i++) {
            features[i] = text[i].length(); // Use text length as simple feature
        }
    }
    
    @Override
    public void computeTfIdf(float[] termFreq, float[] docFreq, float[] result, int size) {
        for (int i = 0; i < size; i++) {
            result[i] = termFreq[i] * (float) Math.log(1.0 + 1.0 / (docFreq[i] + 1e-10));
        }
    }
    
    @Override
    public void initialize() {
        CpuComputeProvider.logger.debug("Initializing CPU compute provider");
        // CPU provider doesn't need special initialization
    }
    
    @Override
    public void initialize(GpuConfig config) {
        initialize();
    }
    
    @Override
    public boolean supportsOperation(String operationType) {
        return true; // CPU supports all operations
    }
}
