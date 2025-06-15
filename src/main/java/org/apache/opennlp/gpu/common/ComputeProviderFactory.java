package org.apache.opennlp.gpu.common;

public class ComputeProviderFactory {
    
    public static ComputeProvider getDefaultProvider() {
        return new CpuComputeProvider();
    }
    
    public static ComputeProvider createProvider(ComputeProvider.Type type) {
        switch (type) {
            case CPU:
                return new CpuComputeProvider();
            case OPENCL:
                return new OpenClComputeProvider();
            case CUDA:
                return new CudaComputeProvider();
            case ROCM:
                return new RocmComputeProvider();
            default:
                return new CpuComputeProvider();
        }
    }
    
    public static ComputeProvider getProvider(ComputeProvider.Type type) {
        return ComputeProviderFactory.createProvider(type);
    }
}
