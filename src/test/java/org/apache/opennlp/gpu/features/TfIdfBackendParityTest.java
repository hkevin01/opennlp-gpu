package org.apache.opennlp.gpu.features;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.apache.opennlp.gpu.common.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.CudaFeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.OpenClFeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.RocmFeatureExtractionOperation;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class TfIdfBackendParityTest {

    private static final float EPSILON = 1e-6f;

    @Test
    @DisplayName("TF-IDF parity: CPU, OpenCL, CUDA, and ROCm backends produce matching scores")
    void tfIdfParityAcrossBackends() {
        String[] corpus = {
                "GPU acceleration for Apache OpenNLP",
                "OpenNLP models with cpu fallback and robust unicode: café",
                "tf idf bm25 and ngram feature extraction"
        };

        CpuComputeProvider provider = new CpuComputeProvider();

        CpuFeatureExtractionOperation cpu = new CpuFeatureExtractionOperation(provider);
        OpenClFeatureExtractionOperation opencl = new OpenClFeatureExtractionOperation(provider);
        CudaFeatureExtractionOperation cuda = CudaFeatureExtractionOperation.createParityTestInstance(provider);
        RocmFeatureExtractionOperation rocm = RocmFeatureExtractionOperation.createParityTestInstance(provider);

        float[] cpuScores = cpu.computeTfIdf(corpus);
        float[] openclScores = opencl.computeTfIdf(corpus);
        float[] cudaScores = cuda.computeTfIdf(corpus);
        float[] rocmScores = rocm.computeTfIdf(corpus);

        assertEquals(cpuScores.length, openclScores.length);
        assertEquals(cpuScores.length, cudaScores.length);
        assertEquals(cpuScores.length, rocmScores.length);

        for (int i = 0; i < cpuScores.length; i++) {
            assertEquals(cpuScores[i], openclScores[i], EPSILON, "CPU vs OpenCL mismatch at " + i);
            assertEquals(cpuScores[i], cudaScores[i], EPSILON, "CPU vs CUDA mismatch at " + i);
            assertEquals(cpuScores[i], rocmScores[i], EPSILON, "CPU vs ROCm mismatch at " + i);
        }
    }
}
