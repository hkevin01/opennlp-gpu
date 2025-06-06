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
            default:
                return new CpuComputeProvider();
        }
    }
}
