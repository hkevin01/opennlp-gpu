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
    @DisplayName("TF-IDF scores match smoothed formula on a small corpus")
    void tfIdfScoresMatchExpectedFormula() {
        String[] docs = {
                "a a b",
                "a c"
        };

        float[] scores = TfIdfAlgorithms.computeDocumentScores(docs);
        assertEquals(2, scores.length);

        // With N = 2 and smoothed idf(t) = ln((N+1)/(df+1)) + 1:
        // df(a)=2 => idf(a)=1
        // df(b)=1 => idf(b)=ln(3/2)+1
        // df(c)=1 => idf(c)=ln(3/2)+1
        double idfRare = Math.log(3.0 / 2.0) + 1.0;

        // doc0 = "a a b": tf(a)=2/3, tf(b)=1/3
        double expected0 = Math.sqrt(Math.pow((2.0 / 3.0) * 1.0, 2)
                                   + Math.pow((1.0 / 3.0) * idfRare, 2));

        // doc1 = "a c": tf(a)=1/2, tf(c)=1/2
        double expected1 = Math.sqrt(Math.pow(0.5 * 1.0, 2)
                                   + Math.pow(0.5 * idfRare, 2));

        assertEquals((float) expected0, scores[0], EPS);
        assertEquals((float) expected1, scores[1], EPS);
        assertTrue(scores[1] > scores[0], "Expected ordering from computed formula should hold");
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
