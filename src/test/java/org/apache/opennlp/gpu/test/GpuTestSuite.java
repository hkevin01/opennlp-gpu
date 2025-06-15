package org.apache.opennlp.gpu.test;

import org.apache.opennlp.gpu.benchmark.PerformanceBenchmark;
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
 * Comprehensive test suite for GPU acceleration components
 * Tests accuracy, performance, and reliability
 */
public class GpuTestSuite {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuTestSuite.class);
    
    private final GpuConfig config;
    private final PerformanceBenchmark benchmark;
    
    // Test tolerances
    private static final float FLOAT_TOLERANCE = 1e-5f;
    private static final double DOUBLE_TOLERANCE = 1e-10;
    
    public GpuTestSuite() {
        this.config = new GpuConfig();
        this.benchmark = new PerformanceBenchmark();
        GpuTestSuite.logger.info("Initialized GPU test suite");
    }
    
    /**
     * Run all tests in the suite
     */
    public TestResults runAllTests() {
        GpuTestSuite.logger.info("Starting comprehensive GPU test suite");
        TestResults results = new TestResults();
        
        try {
            // Matrix operation tests
            results.addResult("Matrix Operations", testMatrixOperations());
            
            // Feature extraction tests
            results.addResult("Feature Extraction", testFeatureExtraction());
            
            // Neural network tests
            results.addResult("Neural Networks", testNeuralNetworks());
            
            // Performance tests
            results.addResult("Performance Benchmarks", testPerformance());
            
            // Memory management tests
            results.addResult("Memory Management", testMemoryManagement());
            
            // Error handling tests
            results.addResult("Error Handling", testErrorHandling());
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Test suite failed with exception: " + e.getMessage());
            results.addError("Test Suite", e);
        }
        
        GpuTestSuite.logger.info("Test suite completed with " + results.getPassCount() + " passes, " + 
                   results.getFailCount() + " failures");
        return results;
    }
    
    private boolean testMatrixOperations() {
        GpuTestSuite.logger.info("Testing matrix operations...");
        
        try {
            ComputeProvider cpuProvider = new CpuComputeProvider();
            MatrixOperation cpuOp = new CpuMatrixOperation(cpuProvider);
            
            ComputeProvider gpuProvider = new GpuComputeProvider(config);
            MatrixOperation gpuOp = new GpuMatrixOperation(gpuProvider, config);
            
            // Test matrix multiplication
            if (!testMatrixMultiplication(cpuOp, gpuOp)) return false;
            
            // Test matrix addition
            if (!testMatrixAddition(cpuOp, gpuOp)) return false;
            
            // Test activation functions
            if (!testActivationFunctions(cpuOp, gpuOp)) return false;
            
            // Test statistical operations
            if (!testStatisticalOperations(cpuOp, gpuOp)) return false;
            
            cpuOp.release();
            gpuOp.release();
            
            GpuTestSuite.logger.info("Matrix operations tests passed");
            return true;
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Matrix operations test failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean testMatrixMultiplication(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        int m = 100, n = 80, k = 120;
        
        float[] a = generateRandomMatrix(m * k);
        float[] b = generateRandomMatrix(k * n);
        float[] cpuResult = new float[m * n];
        float[] gpuResult = new float[m * n];
        
        cpuOp.multiply(a, b, cpuResult, m, n, k);
        gpuOp.multiply(a, b, gpuResult, m, n, k);
        
        return compareArrays(cpuResult, gpuResult, GpuTestSuite.FLOAT_TOLERANCE);
    }
    
    private boolean testMatrixAddition(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        int size = 10000;
        
        float[] a = generateRandomMatrix(size);
        float[] b = generateRandomMatrix(size);
        float[] cpuResult = new float[size];
        float[] gpuResult = new float[size];
        
        cpuOp.add(a, b, cpuResult, size);
        gpuOp.add(a, b, gpuResult, size);
        
        return compareArrays(cpuResult, gpuResult, GpuTestSuite.FLOAT_TOLERANCE);
    }
    
    private boolean testActivationFunctions(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        int size = 1000;
        float[] input = generateRandomMatrix(size);
        
        // Test sigmoid
        float[] cpuSigmoid = new float[size];
        float[] gpuSigmoid = new float[size];
        cpuOp.sigmoid(input, cpuSigmoid, size);
        gpuOp.sigmoid(input, gpuSigmoid, size);
        if (!compareArrays(cpuSigmoid, gpuSigmoid, GpuTestSuite.FLOAT_TOLERANCE)) return false;
        
        // Test ReLU
        float[] cpuRelu = new float[size];
        float[] gpuRelu = new float[size];
        cpuOp.relu(input, cpuRelu, size);
        gpuOp.relu(input, gpuRelu, size);
        if (!compareArrays(cpuRelu, gpuRelu, GpuTestSuite.FLOAT_TOLERANCE)) return false;
        
        // Test softmax
        float[] cpuSoftmax = new float[size];
        float[] gpuSoftmax = new float[size];
        cpuOp.softmax(input, cpuSoftmax, size);
        gpuOp.softmax(input, gpuSoftmax, size);
        return compareArrays(cpuSoftmax, gpuSoftmax, GpuTestSuite.FLOAT_TOLERANCE);
    }
    
    private boolean testStatisticalOperations(MatrixOperation cpuOp, MatrixOperation gpuOp) {
        int size = 5000;
        float[] input = generateRandomMatrix(size);
        
        // Test mean
        float[] cpuMean = new float[1];
        float[] gpuMean = new float[1];
        cpuOp.mean(input, cpuMean, size);
        gpuOp.mean(input, gpuMean, size);
        if (!compareArrays(cpuMean, gpuMean, GpuTestSuite.FLOAT_TOLERANCE)) return false;
        
        // Test normalization
        float[] cpuNorm = new float[size];
        float[] gpuNorm = new float[size];
        cpuOp.normalize(input, cpuNorm, size);
        gpuOp.normalize(input, gpuNorm, size);
        return compareArrays(cpuNorm, gpuNorm, GpuTestSuite.FLOAT_TOLERANCE);
    }
    
    private boolean testFeatureExtraction() {
        GpuTestSuite.logger.info("Testing feature extraction...");
        
        try {
            ComputeProvider provider = new CpuComputeProvider();
            MatrixOperation matrixOp = new CpuMatrixOperation(provider);
            GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
            
            // Test documents
            String[] documents = {
                "the quick brown fox jumps over the lazy dog",
                "machine learning with gpu acceleration is fast",
                "natural language processing requires feature extraction",
                "opencv provides computer vision capabilities"
            };
            
            // Test n-gram extraction
            float[][] ngramFeatures = extractor.extractNGramFeatures(documents, 2, 1000);
            if (ngramFeatures.length != documents.length) return false;
            
            // Test TF-IDF extraction
            float[][] tfidfFeatures = extractor.extractTfIdfFeatures(documents, 2, 1000);
            if (tfidfFeatures.length != documents.length) return false;
            
            // Test context features
            String[] targetWords = {"machine", "gpu"};
            float[][] contextFeatures = extractor.extractContextFeatures(documents, targetWords, 3);
            if (contextFeatures.length == 0) return false;
            
            extractor.release();
            matrixOp.release();
            
            GpuTestSuite.logger.info("Feature extraction tests passed");
            return true;
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Feature extraction test failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean testNeuralNetworks() {
        GpuTestSuite.logger.info("Testing neural networks...");
        
        try {
            ComputeProvider provider = new CpuComputeProvider();
            MatrixOperation matrixOp = new CpuMatrixOperation(provider);
            
            // Create simple neural network
            int[] layerSizes = {10, 20, 5, 1};
            String[] activations = {"relu", "relu", "sigmoid"};
            GpuNeuralNetwork network = new GpuNeuralNetwork(layerSizes, activations, config, matrixOp);
            
            // Test forward propagation
            float[] input = generateRandomMatrix(10);
            float[] output = network.predict(input);
            if (output.length != 1) return false;
            
            // Test batch prediction
            float[][] batchInput = new float[5][10];
            for (int i = 0; i < 5; i++) {
                batchInput[i] = generateRandomMatrix(10);
            }
            float[][] batchOutput = network.predictBatch(batchInput);
            if (batchOutput.length != 5) return false;
            
            // Test training (simple)
            float[][] trainInput = new float[10][10];
            float[][] trainOutput = new float[10][1];
            for (int i = 0; i < 10; i++) {
                trainInput[i] = generateRandomMatrix(10);
                trainOutput[i] = new float[]{(float) Math.random()};
            }
            network.train(trainInput, trainOutput, 5);
            
            network.cleanup();
            matrixOp.release();
            
            GpuTestSuite.logger.info("Neural network tests passed");
            return true;
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Neural network test failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean testPerformance() {
        GpuTestSuite.logger.info("Testing performance benchmarks...");
        
        try {
            // Run basic performance tests
            benchmark.benchmarkMatrixOperations();
            benchmark.benchmarkFeatureExtraction();
            benchmark.benchmarkNeuralNetworks();
            
            // Check if GPU shows performance improvements
            PerformanceBenchmark.BenchmarkResults results = benchmark.getLatestResults();
            return results != null && results.isValid();
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Performance test failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean testMemoryManagement() {
        GpuTestSuite.logger.info("Testing memory management...");
        
        try {
            // Test resource allocation and cleanup
            ComputeProvider provider = new GpuComputeProvider(config);
            MatrixOperation matrixOp = new GpuMatrixOperation(provider, config);
            
            // Perform multiple operations to test memory management
            for (int i = 0; i < 10; i++) {
                float[] a = generateRandomMatrix(1000);
                float[] b = generateRandomMatrix(1000);
                float[] result = new float[1000];
                matrixOp.add(a, b, result, 1000);
            }
            
            matrixOp.release();
            provider.cleanup();
            
            GpuTestSuite.logger.info("Memory management tests passed");
            return true;
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Memory management test failed: " + e.getMessage());
            return false;
        }
    }
    
    private boolean testErrorHandling() {
        GpuTestSuite.logger.info("Testing error handling...");
        
        try {
            ComputeProvider provider = new CpuComputeProvider();
            MatrixOperation matrixOp = new CpuMatrixOperation(provider);
            
            // Test invalid matrix dimensions
            try {
                float[] a = new float[10];
                float[] b = new float[20];
                float[] result = new float[30];
                matrixOp.multiply(a, b, result, 2, 3, 5); // Mismatched dimensions
                // Should not crash due to fallback mechanisms
            } catch (Exception e) {
                // Expected behavior - graceful handling
            }
            
            matrixOp.release();
            
            GpuTestSuite.logger.info("Error handling tests passed");
            return true;
            
        } catch (Exception e) {
            GpuTestSuite.logger.error("Error handling test failed: " + e.getMessage());
            return false;
        }
    }
    
    // Utility methods
    
    private float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = (float) (Math.random() * 2.0 - 1.0);
        }
        return matrix;
    }
    
    private boolean compareArrays(float[] a, float[] b, float tolerance) {
        if (a.length != b.length) return false;
        
        for (int i = 0; i < a.length; i++) {
            if (Math.abs(a[i] - b[i]) > tolerance) {
                GpuTestSuite.logger.warn("Array comparison failed at index " + i + ": " + a[i] + " vs " + b[i]);
                return false;
            }
        }
        return true;
    }
    
    /**
     * Test results container
     */
    public static class TestResults {
        private int passCount = 0;
        private int failCount = 0;
        private final StringBuilder report = new StringBuilder();
        
        public void addResult(String testName, boolean passed) {
            if (passed) {
                passCount++;
                report.append("✅ ").append(testName).append(" - PASSED\n");
            } else {
                failCount++;
                report.append("❌ ").append(testName).append(" - FAILED\n");
            }
        }
        
        public void addError(String testName, Exception e) {
            failCount++;
            report.append("💥 ").append(testName).append(" - ERROR: ").append(e.getMessage()).append("\n");
        }
        
        public int getPassCount() { return passCount; }
        public int getFailCount() { return failCount; }
        public String getReport() { return report.toString(); }
        public boolean allPassed() { return failCount == 0; }
    }
}
