package org.apache.opennlp.gpu.features;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import org.apache.opennlp.gpu.compute.CpuFeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.OpenClFeatureExtractionOperation;
import org.apache.opennlp.gpu.common.CpuComputeProvider;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

class TfIdfAlgorithmsTest {

    private static final float EPS = 1e-6f;

    @Test
    @DisplayName("TF-IDF scores: document with rarer terms should score higher")
    void rareTermsIncreaseDocumentScore() {
        String[] docs = {
                "common common common rare1",
                "common common common common",
                "common common rare2"
        };

        float[] scores = TfIdfAlgorithms.computeDocumentScores(docs);

        assertEquals(3, scores.length);
        assertTrue(scores[0] > scores[1], "doc containing rare1 should score above all-common doc");
        assertTrue(scores[2] > scores[1], "doc containing rare2 should score above all-common doc");
    }

    @Test
    @DisplayName("CPU and OpenCL fallback TF-IDF paths should be numerically consistent")
    void cpuAndOpenClFallbackConsistency() {
        String[] docs = {
                "nlp gpu acceleration",
                "nlp cpu fallback",
                "gpu kernels and fallback"
        };

        CpuComputeProvider provider = new CpuComputeProvider();
        CpuFeatureExtractionOperation cpu = new CpuFeatureExtractionOperation(provider);
        OpenClFeatureExtractionOperation opencl = new OpenClFeatureExtractionOperation(provider);

        float[] cpuScores = cpu.computeTfIdf(docs);
        float[] openClScores = opencl.computeTfIdf(docs);

        assertEquals(cpuScores.length, openClScores.length);
        for (int i = 0; i < cpuScores.length; i++) {
            assertEquals(cpuScores[i], openClScores[i], EPS,
                    "CPU and OpenCL fallback TF-IDF should match at index " + i);
        }
    }
}
