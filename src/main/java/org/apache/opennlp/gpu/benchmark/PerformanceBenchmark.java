package org.apache.opennlp.gpu.benchmark;

import java.util.HashMap;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.gpu.ml.neural.GpuNeuralNetwork;

/**
 * Comprehensive performance benchmarking system
 * Measures GPU vs CPU performance across all operations
 */
public class PerformanceBenchmark {
    
    private static final GpuLogger logger = GpuLogger.getLogger(PerformanceBenchmark.class);
    
    private final GpuConfig config;
    private BenchmarkResults latestResults;
    
    // Benchmark configuration
    private static final int WARMUP_ITERATIONS = 5;
    private static final int BENCHMARK_ITERATIONS = 10;
    private static final int[] MATRIX_SIZES = {100, 500, 1000, 2000, 5000};
    private static final int[] FEATURE_SIZES = {1000, 5000, 10000, 20000};
    private static final int[] NEURAL_SIZES = {100, 500, 1000, 2000};
    
    public PerformanceBenchmark() {
        this.config = new GpuConfig();
        PerformanceBenchmark.logger.info("Initialized performance benchmark suite");
    }
    
    /**
     * Run comprehensive benchmarks and return results
     */
    public BenchmarkResults runFullBenchmark() {
        PerformanceBenchmark.logger.info("Starting comprehensive performance benchmark");
        BenchmarkResults results = new BenchmarkResults();
        
        try {
            // Matrix operation benchmarks
            Map<String, BenchmarkMetrics> matrixResults = benchmarkMatrixOperations();
            results.addCategory("Matrix Operations", matrixResults);
            
            // Feature extraction benchmarks
            Map<String, BenchmarkMetrics> featureResults = benchmarkFeatureExtraction();
            results.addCategory("Feature Extraction", featureResults);
            
            // Neural network benchmarks
            Map<String, BenchmarkMetrics> neuralResults = benchmarkNeuralNetworks();
            results.addCategory("Neural Networks", neuralResults);
            
            this.latestResults = results;
            
            PerformanceBenchmark.logger.info("Benchmark completed successfully");
            PerformanceBenchmark.logger.info("Overall speedup: " + String.format("%.2f", results.getOverallSpeedup()) + "x");
            
        } catch (Exception e) {
            PerformanceBenchmark.logger.error("Benchmark failed: " + e.getMessage());
            results.addError("Benchmark execution failed", e);
        }
        
        return results;
    }
    
    public Map<String, BenchmarkMetrics> benchmarkMatrixOperations() {
        PerformanceBenchmark.logger.info("Benchmarking matrix operations...");
        Map<String, BenchmarkMetrics> results = new HashMap<String, BenchmarkMetrics>();
        
        try {
            ComputeProvider cpuProvider = new CpuComputeProvider();
            ComputeProvider gpuProvider = new GpuComputeProvider(config);
            
            MatrixOperation cpuOp = new CpuMatrixOperation(cpuProvider);
            MatrixOperation gpuOp = new GpuMatrixOperation(gpuProvider, config);
            
            // Benchmark matrix multiplication
            results.put("Matrix Multiplication", benchmarkMatrixMultiplication(cpuOp, gpuOp));
            
            // Benchmark matrix addition
            results.put("Matrix Addition", benchmarkMatrixAddition(cpuOp, gpuOp));
            
            // Benchmark activation functions
            results.put("Sigmoid Activation", benchmarkSigmoidActivation(cpuOp, gpuOp));
            results.put("ReLU Activation", benchmarkReluActivation(cpuOp, gpuOp));
            results.put("Softmax Activation", benchmarkSoftmaxActivation(cpuOp, gpuOp));
            
            // Benchmark statistical operations
            results.put("Vector Normalization", benchmarkVectorNormalization(cpuOp, gpuOp));
            
            cpuOp.release();
            gpuOp.release();
            cpuProvider.cleanup();
            gpuProvider.cleanup();
            
        } catch (Exception e) {
            PerformanceBenchmark.logger.error("Matrix operations benchmark failed: " + e.getMessage());
        }
        
        return results;
    }
    
    private BenchmarkMetrics benchmarkMatrixMultiplication(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        BenchmarkMetrics metrics = new BenchmarkMetrics("Matrix Multiplication");
        
        for (int size : PerformanceBenchmark.MATRIX_SIZES) {
            int m = size, n = size, k = size;
            
            float[] a = generateRandomMatrix(m * k);
            float[] b = generateRandomMatrix(k * n);
            float[] result = new float[m * n];
            
            // Warmup
            for (int i = 0; i < PerformanceBenchmark.WARMUP_ITERATIONS; i++) {
                cpuOp.multiply(a, b, result, m, n, k);
                gpuOp.multiply(a, b, result, m, n, k);
            }
            
            // CPU benchmark
            long cpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                cpuOp.multiply(a, b, result, m, n, k);
            }
            long cpuTime = (System.nanoTime() - cpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            // GPU benchmark
            long gpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                gpuOp.multiply(a, b, result, m, n, k);
            }
            long gpuTime = (System.nanoTime() - gpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            metrics.addDataPoint(size, cpuTime / 1_000_000.0, gpuTime / 1_000_000.0);
        }
        
        return metrics;
    }
    
    private BenchmarkMetrics benchmarkMatrixAddition(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        BenchmarkMetrics metrics = new BenchmarkMetrics("Matrix Addition");
        
        for (int size : PerformanceBenchmark.MATRIX_SIZES) {
            int elements = size * size;
            
            float[] a = generateRandomMatrix(elements);
            float[] b = generateRandomMatrix(elements);
            float[] result = new float[elements];
            
            // Warmup
            for (int i = 0; i < PerformanceBenchmark.WARMUP_ITERATIONS; i++) {
                cpuOp.add(a, b, result, elements);
                gpuOp.add(a, b, result, elements);
            }
            
            // CPU benchmark
            long cpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                cpuOp.add(a, b, result, elements);
            }
            long cpuTime = (System.nanoTime() - cpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            // GPU benchmark
            long gpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                gpuOp.add(a, b, result, elements);
            }
            long gpuTime = (System.nanoTime() - gpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            metrics.addDataPoint(size, cpuTime / 1_000_000.0, gpuTime / 1_000_000.0);
        }
        
        return metrics;
    }
    
    private BenchmarkMetrics benchmarkSigmoidActivation(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        return benchmarkActivationFunction(cpuOp, gpuOp, "sigmoid", "Sigmoid Activation");
    }
    
    private BenchmarkMetrics benchmarkReluActivation(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        return benchmarkActivationFunction(cpuOp, gpuOp, "relu", "ReLU Activation");
    }
    
    private BenchmarkMetrics benchmarkSoftmaxActivation(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        return benchmarkActivationFunction(cpuOp, gpuOp, "softmax", "Softmax Activation");
    }
    
    private BenchmarkMetrics benchmarkActivationFunction(MatrixOperation cpuOp, MatrixOperation gpuOp, 
                                                         String function, String name) {
        BenchmarkMetrics metrics = new BenchmarkMetrics(name);
        
        for (int size : PerformanceBenchmark.NEURAL_SIZES) {
            int elements = size * 100; // Simulate neural layer
            
            float[] input = generateRandomMatrix(elements);
            float[] result = new float[elements];
            
            // Warmup
            for (int i = 0; i < PerformanceBenchmark.WARMUP_ITERATIONS; i++) {
                applyActivation(cpuOp, function, input, result, elements);
                applyActivation(gpuOp, function, input, result, elements);
            }
            
            // CPU benchmark
            long cpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                applyActivation(cpuOp, function, input, result, elements);
            }
            long cpuTime = (System.nanoTime() - cpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            // GPU benchmark
            long gpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                applyActivation(gpuOp, function, input, result, elements);
            }
            long gpuTime = (System.nanoTime() - gpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            metrics.addDataPoint(size, cpuTime / 1_000_000.0, gpuTime / 1_000_000.0);
        }
        
        return metrics;
    }
    
    private void applyActivation(MatrixOperation op, String function, float[] input, float[] output, int size) {
        switch (function) {
            case "sigmoid":
                op.sigmoid(input, output, size);
                break;
            case "relu":
                op.relu(input, output, size);
                break;
            case "softmax":
                op.softmax(input, output, size);
                break;
        }
    }
    
    private BenchmarkMetrics benchmarkVectorNormalization(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        BenchmarkMetrics metrics = new BenchmarkMetrics("Vector Normalization");
        
        for (int size : PerformanceBenchmark.FEATURE_SIZES) {
            float[] input = generateRandomMatrix(size);
            float[] result = new float[size];
            
            // Warmup
            for (int i = 0; i < PerformanceBenchmark.WARMUP_ITERATIONS; i++) {
                cpuOp.normalize(input, result, size);
                gpuOp.normalize(input, result, size);
            }
            
            // CPU benchmark
            long cpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                cpuOp.normalize(input, result, size);
            }
            long cpuTime = (System.nanoTime() - cpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            // GPU benchmark
            long gpuStartTime = System.nanoTime();
            for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                gpuOp.normalize(input, result, size);
            }
            long gpuTime = (System.nanoTime() - gpuStartTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS;
            
            metrics.addDataPoint(size, cpuTime / 1_000_000.0, gpuTime / 1_000_000.0);
        }
        
        return metrics;
    }
    
    public Map<String, BenchmarkMetrics> benchmarkFeatureExtraction() {
        PerformanceBenchmark.logger.info("Benchmarking feature extraction...");
        Map<String, BenchmarkMetrics> results = new HashMap<String, BenchmarkMetrics>();
        
        try {
            ComputeProvider provider = new CpuComputeProvider();
            MatrixOperation matrixOp = new CpuMatrixOperation(provider);
            GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
            
            // Generate test documents
            String[] documents = generateTestDocuments(1000);
            
            BenchmarkMetrics ngramMetrics = new BenchmarkMetrics("N-gram Extraction");
            BenchmarkMetrics tfidfMetrics = new BenchmarkMetrics("TF-IDF Extraction");
            
            for (int docCount : new int[]{100, 500, 1000}) {
                String[] testDocs = new String[docCount];
                System.arraycopy(documents, 0, testDocs, 0, docCount);
                
                // Benchmark n-gram extraction
                long startTime = System.nanoTime();
                extractor.extractNGramFeatures(testDocs, 2, 5000);
                long ngramTime = (System.nanoTime() - startTime) / 1_000_000;
                
                // Benchmark TF-IDF extraction
                startTime = System.nanoTime();
                extractor.extractTfIdfFeatures(testDocs, 2, 5000);
                long tfidfTime = (System.nanoTime() - startTime) / 1_000_000;
                
                ngramMetrics.addDataPoint(docCount, ngramTime, ngramTime); // CPU only for now
                tfidfMetrics.addDataPoint(docCount, tfidfTime, tfidfTime); // CPU only for now
            }
            
            results.put("N-gram Extraction", ngramMetrics);
            results.put("TF-IDF Extraction", tfidfMetrics);
            
            extractor.release();
            matrixOp.release();
            provider.cleanup();
            
        } catch (Exception e) {
            PerformanceBenchmark.logger.error("Feature extraction benchmark failed: " + e.getMessage());
        }
        
        return results;
    }
    
    public Map<String, BenchmarkMetrics> benchmarkNeuralNetworks() {
        PerformanceBenchmark.logger.info("Benchmarking neural networks...");
        Map<String, BenchmarkMetrics> results = new HashMap<String, BenchmarkMetrics>();
        
        try {
            ComputeProvider provider = new CpuComputeProvider();
            MatrixOperation matrixOp = new CpuMatrixOperation(provider);
            
            BenchmarkMetrics forwardMetrics = new BenchmarkMetrics("Neural Forward Pass");
            BenchmarkMetrics batchMetrics = new BenchmarkMetrics("Neural Batch Prediction");
            
            for (int networkSize : PerformanceBenchmark.NEURAL_SIZES) {
                int[] layerSizes = {networkSize, networkSize * 2, networkSize, 10};
                String[] activations = {"relu", "relu", "softmax"};
                
                GpuNeuralNetwork network = new GpuNeuralNetwork(layerSizes, activations, config, matrixOp);
                
                // Benchmark forward pass
                float[] input = generateRandomMatrix(networkSize);
                
                // Warmup
                for (int i = 0; i < PerformanceBenchmark.WARMUP_ITERATIONS; i++) {
                    network.predict(input);
                }
                
                long startTime = System.nanoTime();
                for (int i = 0; i < PerformanceBenchmark.BENCHMARK_ITERATIONS; i++) {
                    network.predict(input);
                }
                long forwardTime = (System.nanoTime() - startTime) / PerformanceBenchmark.BENCHMARK_ITERATIONS / 1_000_000;
                
                // Benchmark batch prediction
                float[][] batchInput = new float[50][networkSize];
                for (int i = 0; i < 50; i++) {
                    batchInput[i] = generateRandomMatrix(networkSize);
                }
                
                startTime = System.nanoTime();
                network.predictBatch(batchInput);
                long batchTime = (System.nanoTime() - startTime) / 1_000_000;
                
                forwardMetrics.addDataPoint(networkSize, forwardTime, forwardTime); // CPU only for now
                batchMetrics.addDataPoint(networkSize, batchTime, batchTime); // CPU only for now
                
                network.cleanup();
            }
            
            results.put("Neural Forward Pass", forwardMetrics);
            results.put("Neural Batch Prediction", batchMetrics);
            
            matrixOp.release();
            provider.cleanup();
            
        } catch (Exception e) {
            PerformanceBenchmark.logger.error("Neural network benchmark failed: " + e.getMessage());
        }
        
        return results;
    }
    
    // Utility methods
    
    private float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = (float) (Math.random() * 2.0 - 1.0);
        }
        return matrix;
    }
    
    private String[] generateTestDocuments(int count) {
        String[] documents = new String[count];
        String[] words = {"machine", "learning", "natural", "language", "processing", "gpu", "acceleration", 
                         "neural", "network", "deep", "computer", "vision", "algorithm", "data", "science"};
        
        for (int i = 0; i < count; i++) {
            StringBuilder doc = new StringBuilder();
            int wordCount = 50 + (int)(Math.random() * 100);
            for (int j = 0; j < wordCount; j++) {
                doc.append(words[(int)(Math.random() * words.length)]).append(" ");
            }
            documents[i] = doc.toString().trim();
        }
        
        return documents;
    }
    
    public BenchmarkResults getLatestResults() {
        return latestResults;
    }
    
    /**
     * Individual benchmark metrics for a specific operation
     */
    public static class BenchmarkMetrics {
        private final String operationName;
        private final Map<Integer, Double> cpuTimes = new HashMap<Integer, Double>();
        private final Map<Integer, Double> gpuTimes = new HashMap<Integer, Double>();
        
        public BenchmarkMetrics(String operationName) {
            this.operationName = operationName;
        }
        
        public void addDataPoint(int size, double cpuTime, double gpuTime) {
            cpuTimes.put(size, cpuTime);
            gpuTimes.put(size, gpuTime);
        }
        
        public double getAverageSpeedup() {
            double totalSpeedup = 0.0;
            int count = 0;
            
            for (Integer size : cpuTimes.keySet()) {
                if (gpuTimes.containsKey(size)) {
                    double speedup = cpuTimes.get(size) / gpuTimes.get(size);
                    totalSpeedup += speedup;
                    count++;
                }
            }
            
            return count > 0 ? totalSpeedup / count : 1.0;
        }
        
        public String getOperationName() { return operationName; }
        public Map<Integer, Double> getCpuTimes() { return new HashMap<Integer, Double>(cpuTimes); }
        public Map<Integer, Double> getGpuTimes() { return new HashMap<Integer, Double>(gpuTimes); }
    }
    
    /**
     * Complete benchmark results container
     */
    public static class BenchmarkResults {
        private final Map<String, Map<String, BenchmarkMetrics>> categoryResults = new HashMap<String, Map<String, BenchmarkMetrics>>();
        private final StringBuilder errorLog = new StringBuilder();
        
        public void addCategory(String category, Map<String, BenchmarkMetrics> metrics) {
            categoryResults.put(category, metrics);
        }
        
        public void addError(String error, Exception e) {
            errorLog.append("ERROR: ").append(error).append(" - ").append(e.getMessage()).append("\n");
        }
        
        public double getOverallSpeedup() {
            double totalSpeedup = 0.0;
            int count = 0;
            
            for (Map<String, BenchmarkMetrics> category : categoryResults.values()) {
                for (BenchmarkMetrics metrics : category.values()) {
                    totalSpeedup += metrics.getAverageSpeedup();
                    count++;
                }
            }
            
            return count > 0 ? totalSpeedup / count : 1.0;
        }
        
        public boolean isValid() {
            return !categoryResults.isEmpty() && errorLog.length() == 0;
        }
        
        public Map<String, Map<String, BenchmarkMetrics>> getCategoryResults() {
            return new HashMap<String, Map<String, BenchmarkMetrics>>(categoryResults);
        }
        
        public String getErrorLog() {
            return errorLog.toString();
        }
        
        public String generateReport() {
            StringBuilder report = new StringBuilder();
            report.append("=== OpenNLP GPU Benchmark Report ===\n\n");
            
            report.append("Overall Speedup: ").append(String.format("%.2f", getOverallSpeedup())).append("x\n\n");
            
            for (Map.Entry<String, Map<String, BenchmarkMetrics>> category : categoryResults.entrySet()) {
                report.append("Category: ").append(category.getKey()).append("\n");
                
                for (BenchmarkMetrics metrics : category.getValue().values()) {
                    report.append("  ").append(metrics.getOperationName())
                          .append(": ").append(String.format("%.2f", metrics.getAverageSpeedup()))
                          .append("x speedup\n");
                }
                report.append("\n");
            }
            
            if (errorLog.length() > 0) {
                report.append("Errors:\n").append(errorLog.toString());
            }
            
            return report.toString();
        }
    }
}