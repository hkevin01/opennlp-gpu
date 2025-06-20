package org.apache.opennlp.gpu.performance;

import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;
import static org.junit.jupiter.api.Assertions.fail;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.gpu.ml.perceptron.GpuPerceptronModel;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.TestInstance;

import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * Comprehensive performance benchmarking for GPU acceleration
 * Measures and validates performance improvements across different operations
 */
@TestInstance(TestInstance.Lifecycle.PER_CLASS)
public class GpuPerformanceBenchmark {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuPerformanceBenchmark.class);
    
    private GpuConfig gpuConfig;
    private GpuConfig cpuConfig;
    private MatrixOperation gpuMatrixOp;
    private MatrixOperation cpuMatrixOp;
    private GpuFeatureExtractor gpuFeatureExtractor;
    private GpuFeatureExtractor cpuFeatureExtractor;
    
    // Benchmark configuration
    private static final int WARMUP_ITERATIONS = 10;
    private static final int BENCHMARK_ITERATIONS = 50;
    private static final double MIN_SPEEDUP_THRESHOLD = 1.5; // Minimum 1.5x speedup expected
    private static final double TARGET_SPEEDUP = 3.0; // Target 3x speedup
    
    @BeforeEach
    void setUp() {
        // GPU configuration
        gpuConfig = new GpuConfig();
        gpuConfig.setGpuEnabled(true);
        
        // CPU configuration for comparison
        cpuConfig = new GpuConfig();
        cpuConfig.setGpuEnabled(false);
        
        try {
            // Initialize GPU components
            if (GpuComputeProvider.isGpuAvailable()) {
                GpuComputeProvider gpuProvider = new GpuComputeProvider(gpuConfig);
                gpuMatrixOp = new GpuMatrixOperation(gpuProvider, gpuConfig);
                gpuFeatureExtractor = new GpuFeatureExtractor(gpuProvider, gpuConfig, gpuMatrixOp);
                logger.info("GPU provider initialized for benchmarking");
            } else {
                logger.warn("GPU not available, using CPU fallback for GPU tests");
                CpuComputeProvider provider = new CpuComputeProvider();
                gpuMatrixOp = new CpuMatrixOperation(provider);
                gpuFeatureExtractor = new GpuFeatureExtractor(provider, gpuConfig, gpuMatrixOp);
            }
            
            // Initialize CPU components for comparison
            CpuComputeProvider cpuProvider = new CpuComputeProvider();
            cpuMatrixOp = new CpuMatrixOperation(cpuProvider);
            cpuFeatureExtractor = new GpuFeatureExtractor(cpuProvider, cpuConfig, cpuMatrixOp);
            
        } catch (Exception e) {
            logger.error("Failed to initialize providers: " + e.getMessage());
            fail("Setup failed: " + e.getMessage());
        }
    }
    
    @AfterEach
    void tearDown() {
        if (gpuFeatureExtractor != null) gpuFeatureExtractor.release();
        if (cpuFeatureExtractor != null) cpuFeatureExtractor.release();
        if (gpuMatrixOp != null) gpuMatrixOp.release();
        if (cpuMatrixOp != null) cpuMatrixOp.release();
    }
    
    @Test
    @DisplayName("Matrix Operations Performance Benchmark")
    void benchmarkMatrixOperations() {
        logger.info("Starting matrix operations performance benchmark");
        
        // Test different matrix sizes
        int[] matrixSizes = {64, 128, 256, 512, 1024, 2048};
        Map<String, BenchmarkResults> results = new HashMap<>();
        
        for (int size : matrixSizes) {
            logger.info("Benchmarking matrix operations for size: " + size + "x" + size);
            
            // Matrix multiplication benchmark
            BenchmarkResults multiplyResults = benchmarkMatrixMultiplication(size);
            results.put("multiply_" + size, multiplyResults);
            
            // Matrix addition benchmark
            BenchmarkResults addResults = benchmarkMatrixAddition(size);
            results.put("add_" + size, addResults);
            
            // Activation function benchmarks
            BenchmarkResults sigmoidResults = benchmarkActivationFunction(size, "sigmoid");
            results.put("sigmoid_" + size, sigmoidResults);
            
            BenchmarkResults reluResults = benchmarkActivationFunction(size, "relu");
            results.put("relu_" + size, reluResults);
        }
        
        // Print comprehensive results
        printBenchmarkSummary(results);
        
        // Validate performance improvements
        validatePerformanceThresholds(results);
    }
    
    @Test
    @DisplayName("Feature Extraction Performance Benchmark")
    void benchmarkFeatureExtraction() {
        logger.info("Starting feature extraction performance benchmark");
        
        Map<String, BenchmarkResults> results = new HashMap<>();
        
        // Test different dataset sizes
        int[] documentCounts = {100, 500, 1000, 2000, 5000};
        int[] vocabularySizes = {1000, 5000, 10000};
        
        for (int docCount : documentCounts) {
            for (int vocabSize : vocabularySizes) {
                String testName = "ngram_" + docCount + "_docs_" + vocabSize + "_vocab";
                logger.info("Benchmarking feature extraction: " + testName);
                
                BenchmarkResults ngramResults = benchmarkNGramExtraction(docCount, vocabSize);
                results.put(testName, ngramResults);
                
                String tfidfTestName = "tfidf_" + docCount + "_docs_" + vocabSize + "_vocab";
                BenchmarkResults tfidfResults = benchmarkTfIdfExtraction(docCount, vocabSize);
                results.put(tfidfTestName, tfidfResults);
            }
        }
        
        // Print results
        printBenchmarkSummary(results);
        validatePerformanceThresholds(results);
    }
    
    @Test
    @DisplayName("ML Model Training Performance Benchmark")
    void benchmarkMlModelTraining() {
        logger.info("Starting ML model training performance benchmark");
        
        Map<String, BenchmarkResults> results = new HashMap<>();
        
        // Perceptron training benchmark
        BenchmarkResults perceptronResults = benchmarkPerceptronTraining();
        results.put("perceptron_training", perceptronResults);
        
        // MaxEnt evaluation benchmark
        BenchmarkResults maxentResults = benchmarkMaxEntEvaluation();
        results.put("maxent_evaluation", maxentResults);
        
        // Print results
        printBenchmarkSummary(results);
        validatePerformanceThresholds(results);
    }
    
    private BenchmarkResults benchmarkMatrixMultiplication(int size) {
        float[] matrixA = createRandomMatrix(size * size);
        float[] matrixB = createRandomMatrix(size * size);
        float[] resultGpu = new float[size * size];
        float[] resultCpu = new float[size * size];
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuMatrixOp.multiply(matrixA, matrixB, resultGpu, size, size, size);
            cpuMatrixOp.multiply(matrixA, matrixB, resultCpu, size, size, size);
        }
        
        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            gpuMatrixOp.multiply(matrixA, matrixB, resultGpu, size, size, size);
            gpuTotalTime += System.nanoTime() - startTime;
        }
        
        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            cpuMatrixOp.multiply(matrixA, matrixB, resultCpu, size, size, size);
            cpuTotalTime += System.nanoTime() - startTime;
        }
        
        return new BenchmarkResults("Matrix Multiply " + size + "x" + size,
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private BenchmarkResults benchmarkMatrixAddition(int size) {
        float[] matrixA = createRandomMatrix(size * size);
        float[] matrixB = createRandomMatrix(size * size);
        float[] resultGpu = new float[size * size];
        float[] resultCpu = new float[size * size];
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuMatrixOp.add(matrixA, matrixB, resultGpu, size * size);
            cpuMatrixOp.add(matrixA, matrixB, resultCpu, size * size);
        }
        
        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            gpuMatrixOp.add(matrixA, matrixB, resultGpu, size * size);
            gpuTotalTime += System.nanoTime() - startTime;
        }
        
        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            cpuMatrixOp.add(matrixA, matrixB, resultCpu, size * size);
            cpuTotalTime += System.nanoTime() - startTime;
        }
        
        return new BenchmarkResults("Matrix Add " + size + "x" + size,
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private BenchmarkResults benchmarkActivationFunction(int size, String function) {
        float[] input = createRandomMatrix(size * size);
        float[] outputGpu = new float[size * size];
        float[] outputCpu = new float[size * size];
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            applyActivation(gpuMatrixOp, input, outputGpu, function);
            applyActivation(cpuMatrixOp, input, outputCpu, function);
        }
        
        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            applyActivation(gpuMatrixOp, input, outputGpu, function);
            gpuTotalTime += System.nanoTime() - startTime;
        }
        
        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            applyActivation(cpuMatrixOp, input, outputCpu, function);
            cpuTotalTime += System.nanoTime() - startTime;
        }
        
        return new BenchmarkResults(function + " " + size + "x" + size,
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private BenchmarkResults benchmarkNGramExtraction(int docCount, int vocabSize) {
        String[] documents = generateRandomDocuments(docCount, 20); // 20 words per doc
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuFeatureExtractor.extractNGramFeatures(documents, 2, vocabSize);
            cpuFeatureExtractor.extractNGramFeatures(documents, 2, vocabSize);
        }
        
        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            float[][] gpuFeatures = gpuFeatureExtractor.extractNGramFeatures(documents, 2, vocabSize);
            gpuTotalTime += System.nanoTime() - startTime;
            assertNotNull(gpuFeatures);
        }
        
        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            float[][] cpuFeatures = cpuFeatureExtractor.extractNGramFeatures(documents, 2, vocabSize);
            cpuTotalTime += System.nanoTime() - startTime;
            assertNotNull(cpuFeatures);
        }
        
        return new BenchmarkResults("N-gram extraction " + docCount + " docs",
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private BenchmarkResults benchmarkTfIdfExtraction(int docCount, int vocabSize) {
        String[] documents = generateRandomDocuments(docCount, 15);
        
        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuFeatureExtractor.extractTfIdfFeatures(documents, vocabSize, 0);
            cpuFeatureExtractor.extractTfIdfFeatures(documents, vocabSize, 0);
        }
        
        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            float[][] gpuFeatures = gpuFeatureExtractor.extractTfIdfFeatures(documents, vocabSize, 0);
            gpuTotalTime += System.nanoTime() - startTime;
            assertNotNull(gpuFeatures);
        }
        
        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            float[][] cpuFeatures = cpuFeatureExtractor.extractTfIdfFeatures(documents, vocabSize, 0);
            cpuTotalTime += System.nanoTime() - startTime;
            assertNotNull(cpuFeatures);
        }
        
        return new BenchmarkResults("TF-IDF extraction " + docCount + " docs",
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private BenchmarkResults benchmarkPerceptronTraining() {
        int numSamples = 2000;
        int numFeatures = 1000;
        
        float[][] features = new float[numSamples][numFeatures];
        int[] labels = new int[numSamples];
        
        // Generate training data
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                features[i][j] = (float) (Math.random() * 2 - 1);
            }
            labels[i] = (features[i][0] + features[i][1] > 0) ? 1 : 0;
        }
        
        // GPU training benchmark
        GpuPerceptronModel gpuPerceptron = new GpuPerceptronModel(gpuConfig, 0.01f, 50);
        long gpuStartTime = System.nanoTime();
        gpuPerceptron.train(features, labels);
        long gpuTrainingTime = System.nanoTime() - gpuStartTime;
        gpuPerceptron.cleanup();
        
        // CPU training benchmark (simulate with config)
        GpuPerceptronModel cpuPerceptron = new GpuPerceptronModel(cpuConfig, 0.01f, 50);
        long cpuStartTime = System.nanoTime();
        cpuPerceptron.train(features, labels);
        long cpuTrainingTime = System.nanoTime() - cpuStartTime;
        cpuPerceptron.cleanup();
        
        return new BenchmarkResults("Perceptron Training " + numSamples + " samples",
                                   gpuTrainingTime, cpuTrainingTime, 1);
    }
    
    private BenchmarkResults benchmarkMaxEntEvaluation() {
        // Create a large, realistic MaxEnt model for benchmarking
        int numOutcomes = 50;
        int numPreds = 5000;
        String[] outcomes = new String[numOutcomes];
        for (int i = 0; i < numOutcomes; i++) {
            outcomes[i] = "outcome" + i;
        }

        String[] predLabels = new String[numPreds];
        for (int i = 0; i < numPreds; i++) {
            predLabels[i] = "pred" + i;
        }

        double[] params = new double[numOutcomes * numPreds];
        for (int i = 0; i < params.length; i++) {
            params[i] = Math.random() * 2.0 - 1.0;
        }
        
        Context[] contexts = new Context[predLabels.length];
        int[] outcomePattern = new int[outcomes.length];
        for (int i = 0; i < outcomes.length; i++) {
            outcomePattern[i] = i;
        }

        for (int i = 0; i < predLabels.length; i++) {
            double[] paramsForPred = new double[outcomes.length];
            for (int j = 0; j < outcomes.length; j++) {
                paramsForPred[j] = params[i * outcomes.length + j];
            }
            contexts[i] = new Context(outcomePattern, paramsForPred);
        }

        MaxentModel cpuModel = new GISModel(contexts, predLabels, outcomes);

        // Create GPU and CPU models
        GpuMaxentModel gpuModel = new GpuMaxentModel(cpuModel, gpuConfig);
        GpuMaxentModel cpuGpuModel = new GpuMaxentModel(cpuModel, cpuConfig);

        // Generate sample context data
        String[][] sampleContexts = new String[1000][];
        for (int i = 0; i < sampleContexts.length; i++) {
            int numFeaturesInContext = 5 + (int) (Math.random() * 10);
            sampleContexts[i] = new String[numFeaturesInContext];
            for (int j = 0; j < numFeaturesInContext; j++) {
                sampleContexts[i][j] = predLabels[(int) (Math.random() * predLabels.length)];
            }
        }

        // Warmup
        for (int i = 0; i < WARMUP_ITERATIONS; i++) {
            gpuModel.eval(sampleContexts[i % sampleContexts.length]);
            cpuGpuModel.eval(sampleContexts[i % sampleContexts.length]);
        }

        // GPU benchmark
        long gpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            for (String[] context : sampleContexts) {
                gpuModel.eval(context);
            }
            gpuTotalTime += System.nanoTime() - startTime;
        }

        // CPU benchmark
        long cpuTotalTime = 0;
        for (int i = 0; i < BENCHMARK_ITERATIONS; i++) {
            long startTime = System.nanoTime();
            for (String[] context : sampleContexts) {
                cpuGpuModel.eval(context);
            }
            cpuTotalTime += System.nanoTime() - startTime;
        }

        gpuModel.cleanup();
        cpuGpuModel.cleanup();

        return new BenchmarkResults("MaxEnt Evaluation " + sampleContexts.length + " contexts",
                                   gpuTotalTime, cpuTotalTime, BENCHMARK_ITERATIONS);
    }
    
    private void applyActivation(MatrixOperation matrixOp, float[] input, float[] output, String function) {
        switch (function) {
            case "sigmoid":
                matrixOp.sigmoid(input, output, input.length);
                break;
            case "relu":
                matrixOp.relu(input, output, input.length);
                break;
            case "tanh":
                matrixOp.tanh(input, output, input.length);
                break;
            default:
                throw new IllegalArgumentException("Unknown activation function: " + function);
        }
    }
    
    private float[] createRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = (float) (Math.random() * 2 - 1);
        }
        return matrix;
    }
    
    private String[] generateRandomDocuments(int count, int wordsPerDoc) {
        String[] documents = new String[count];
        String[] vocabulary = {
            "machine", "learning", "gpu", "acceleration", "matrix", "computation",
            "feature", "extraction", "neural", "network", "deep", "training",
            "algorithm", "optimization", "performance", "benchmark", "test", "data"
        };
        
        for (int i = 0; i < count; i++) {
            StringBuilder doc = new StringBuilder();
            for (int j = 0; j < wordsPerDoc; j++) {
                if (j > 0) doc.append(" ");
                doc.append(vocabulary[(int) (Math.random() * vocabulary.length)]);
            }
            documents[i] = doc.toString();
        }
        
        return documents;
    }
    
    private void printBenchmarkSummary(Map<String, BenchmarkResults> results) {
        logger.info("========== PERFORMANCE BENCHMARK RESULTS ==========");
        
        List<BenchmarkResults> sortedResults = new ArrayList<>(results.values());
        sortedResults.sort((a, b) -> Double.compare(b.getSpeedup(), a.getSpeedup()));
        
        for (BenchmarkResults result : sortedResults) {
            logger.info(result.toString());
        }
        
        // Calculate summary statistics
        double avgSpeedup = sortedResults.stream()
                .mapToDouble(BenchmarkResults::getSpeedup)
                .average().orElse(0.0);
        
        double maxSpeedup = sortedResults.stream()
                .mapToDouble(BenchmarkResults::getSpeedup)
                .max().orElse(0.0);
        
        logger.info("========== SUMMARY ==========");
        logger.info("Average speedup: " + String.format("%.2fx", avgSpeedup));
        logger.info("Best performing operation: " + sortedResults.get(0).getOperationName() + 
                   " (" + String.format("%.2fx", sortedResults.get(0).getSpeedup()) + ")");
        logger.info("Total operations benchmarked: " + results.size());
        
        long operationsAboveTarget = sortedResults.stream()
                .mapToDouble(BenchmarkResults::getSpeedup)
                .mapToInt(speedup -> speedup >= TARGET_SPEEDUP ? 1 : 0)
                .sum();
        
        logger.info("Operations exceeding target speedup (" + TARGET_SPEEDUP + "x): " + 
                   operationsAboveTarget + "/" + results.size());
    }
    
    private void validatePerformanceThresholds(Map<String, BenchmarkResults> results) {
        for (BenchmarkResults result : results.values()) {
            double speedup = result.getSpeedup();
            
            // Log warning for operations below minimum threshold
            if (speedup < MIN_SPEEDUP_THRESHOLD) {
                logger.warn("Performance below threshold: " + result.getOperationName() + 
                           " achieved " + String.format("%.2fx", speedup) + 
                           " speedup (minimum: " + MIN_SPEEDUP_THRESHOLD + "x)");
            }
            
            // Assert that at least some improvement is achieved (even if below threshold)
            assertTrue(speedup > 0.8, "GPU implementation should not be significantly slower than CPU: " + 
                      result.getOperationName() + " achieved " + String.format("%.2fx", speedup));
        }
        
        // Check overall performance
        double avgSpeedup = results.values().stream()
                .mapToDouble(BenchmarkResults::getSpeedup)
                .average().orElse(0.0);
        
        assertTrue(avgSpeedup >= MIN_SPEEDUP_THRESHOLD, 
                  "Average speedup should exceed minimum threshold. Achieved: " + 
                  String.format("%.2fx", avgSpeedup) + ", Expected: " + MIN_SPEEDUP_THRESHOLD + "x");
        
        logger.info("Performance validation passed with average speedup: " + String.format("%.2fx", avgSpeedup));
    }
    
    /**
     * Container for benchmark results
     */
    private static class BenchmarkResults {
        private final String operationName;
        private final long gpuTime;
        private final long cpuTime;
        private final int iterations;
        
        public BenchmarkResults(String operationName, long gpuTime, long cpuTime, int iterations) {
            this.operationName = operationName;
            this.gpuTime = gpuTime;
            this.cpuTime = cpuTime;
            this.iterations = iterations;
        }
        
        public String getOperationName() {
            return operationName;
        }
        
        public double getGpuTimeMs() {
            return gpuTime / 1_000_000.0 / iterations;
        }
        
        public double getCpuTimeMs() {
            return cpuTime / 1_000_000.0 / iterations;
        }
        
        public double getSpeedup() {
            return (double) cpuTime / gpuTime;
        }
        
        @Override
        public String toString() {
            return String.format("%-40s | GPU: %8.2f ms | CPU: %8.2f ms | Speedup: %5.2fx",
                               operationName, getGpuTimeMs(), getCpuTimeMs(), getSpeedup());
        }
    }
}
