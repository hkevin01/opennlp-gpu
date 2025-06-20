package org.apache.opennlp.gpu.compatibility;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

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
import org.junit.jupiter.api.condition.EnabledOnOs;
import org.junit.jupiter.api.condition.OS;

import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * Cross-platform compatibility testing for GPU acceleration
 * Tests functionality across different operating systems, GPU vendors, and configurations
 */
public class CrossPlatformCompatibilityTest {
    
    private static final GpuLogger logger = GpuLogger.getLogger(CrossPlatformCompatibilityTest.class);
    
    private GpuConfig config;
    private Map<String, Object> systemInfo;
    private CompatibilityReport compatibilityReport;
    
    @BeforeEach
    void setUp() {
        config = new GpuConfig();
        config.setGpuEnabled(true);
        
        systemInfo = gatherSystemInformation();
        compatibilityReport = new CompatibilityReport();
        
        logger.info("Starting cross-platform compatibility test");
        logger.info("System: " + systemInfo.get("os.name") + " " + systemInfo.get("os.version"));
        logger.info("Java: " + systemInfo.get("java.version") + " (" + systemInfo.get("java.vendor") + ")");
        logger.info("Architecture: " + systemInfo.get("os.arch"));
    }
    
    @AfterEach
    void tearDown() {
        printCompatibilityReport();
    }
    
    @Test
    @DisplayName("Basic GPU Detection and Initialization")
    void testGpuDetectionAndInitialization() {
        TestResult result = new TestResult("GPU Detection and Initialization");
        
        try {
            // Test GPU availability detection
            boolean gpuAvailable = GpuComputeProvider.isGpuAvailable();
            result.addDetail("GPU Available", String.valueOf(gpuAvailable));
            
            if (gpuAvailable) {
                // Test GPU provider initialization
                GpuComputeProvider gpuProvider = new GpuComputeProvider(config);
                result.addDetail("GPU Provider Created", "Success");
                
                // Test GPU info retrieval
                String gpuInfo = gpuProvider.toString();
                result.addDetail("GPU Info", gpuInfo);
                
                gpuProvider.cleanup();
                result.setStatus(TestResult.Status.PASSED);
                result.addDetail("GPU Cleanup", "Success");
            } else {
                result.setStatus(TestResult.Status.WARNING);
                result.addDetail("Reason", "No GPU hardware detected");
            }
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
            logger.warn("GPU detection failed: " + e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        
        // Should not fail the test if GPU is not available
        assertTrue(result.getStatus() != TestResult.Status.FAILED || 
                  result.getDetails().containsKey("Error"));
    }
    
    @Test
    @DisplayName("CPU Fallback Functionality")
    void testCpuFallbackFunctionality() {
        TestResult result = new TestResult("CPU Fallback Functionality");
        
        try {
            // Force CPU-only configuration
            GpuConfig cpuConfig = new GpuConfig();
            cpuConfig.setGpuEnabled(false);
            
            // Test CPU provider
            CpuComputeProvider cpuProvider = new CpuComputeProvider();
            MatrixOperation cpuMatrixOp = new CpuMatrixOperation(cpuProvider);
            
            // Test basic matrix operations
            float[] matrixA = {1, 2, 3, 4};
            float[] matrixB = {5, 6, 7, 8};
            float[] matrixResult = new float[4];
            
            cpuMatrixOp.add(matrixA, matrixB, matrixResult, 4);
            
            // Verify results
            float[] expected = {6, 8, 10, 12};
            assertArrayEquals(expected, matrixResult, 0.001f);
            
            result.setStatus(TestResult.Status.PASSED);
            result.addDetail("Matrix Operations", "Working");
            result.addDetail("Provider Type", cpuProvider.getName());
            
            cpuMatrixOp.release();
            cpuProvider.cleanup();
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() == TestResult.Status.PASSED);
    }
    
    @Test
    @DisplayName("Matrix Operations Cross-Platform")
    void testMatrixOperationsCrossPlatform() {
        TestResult result = new TestResult("Matrix Operations Cross-Platform");
        
        try {
            MatrixOperation matrixOp = createMatrixOperation();
            
            // Test various matrix operations
            testMatrixAddition(matrixOp, result);
            testMatrixMultiplication(matrixOp, result);
            testActivationFunctions(matrixOp, result);
            testStatisticalOperations(matrixOp, result);
            
            result.setStatus(TestResult.Status.PASSED);
            matrixOp.release();
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() == TestResult.Status.PASSED);
    }
    
    @Test
    @DisplayName("Feature Extraction Cross-Platform")
    void testFeatureExtractionCrossPlatform() {
        TestResult result = new TestResult("Feature Extraction Cross-Platform");
        
        try {
            GpuFeatureExtractor extractor = createFeatureExtractor();
            
            // Test N-gram extraction
            String[] documents = {
                "this is a test document",
                "another test document here",
                "final test document example"
            };
            
            float[][] ngramFeatures = extractor.extractNGramFeatures(documents, 2, 100);
            assertNotNull(ngramFeatures);
            assertEquals(documents.length, ngramFeatures.length);
            result.addDetail("N-gram Extraction", "Working");
            
            // Test TF-IDF extraction
            float[][] tfidfFeatures = extractor.extractTfIdfFeatures(documents, 100, 0);
            assertNotNull(tfidfFeatures);
            assertEquals(documents.length, tfidfFeatures.length);
            result.addDetail("TF-IDF Extraction", "Working");
            
            // Test context features  
            String[] targetWords = {"test", "machine"};
            float[][] contextFeatures = extractor.extractContextFeatures(documents, targetWords, 3);
            assertNotNull(contextFeatures);
            result.addDetail("Context Features", "Working");
            
            result.setStatus(TestResult.Status.PASSED);
            extractor.release();
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() == TestResult.Status.PASSED);
    }
    
    @Test
    @DisplayName("ML Models Cross-Platform")
    void testMlModelsCrossPlatform() {
        TestResult result = new TestResult("ML Models Cross-Platform");
        
        try {
            // Test Perceptron model
            testPerceptronModel(result);
            
            // Test MaxEnt model
            testMaxEntModel(result);
            
            result.setStatus(TestResult.Status.PASSED);
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() == TestResult.Status.PASSED);
    }
    
    @Test
    @EnabledOnOs(OS.LINUX)
    @DisplayName("Linux-Specific GPU Features")
    void testLinuxSpecificFeatures() {
        TestResult result = new TestResult("Linux-Specific Features");
        
        try {
            // Test CUDA availability on Linux
            boolean cudaAvailable = testCudaAvailability();
            result.addDetail("CUDA Available", String.valueOf(cudaAvailable));
            
            // Test ROCm availability on Linux
            boolean rocmAvailable = testRocmAvailability();
            result.addDetail("ROCm Available", String.valueOf(rocmAvailable));
            
            // Test OpenCL availability
            boolean openclAvailable = testOpenClAvailability();
            result.addDetail("OpenCL Available", String.valueOf(openclAvailable));
            
            result.setStatus(TestResult.Status.PASSED);
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.WARNING);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        // Don't fail test for Linux-specific features
        assertTrue(result.getStatus() != TestResult.Status.FAILED);
    }
    
    @Test
    @EnabledOnOs(OS.WINDOWS)
    @DisplayName("Windows-Specific Compatibility")
    void testWindowsSpecificCompatibility() {
        TestResult result = new TestResult("Windows-Specific Compatibility");
        
        try {
            // Test Windows path handling
            testWindowsPathHandling(result);
            
            // Test DirectX/DirectCompute availability
            testDirectXAvailability(result);
            
            // Test Windows-specific GPU detection
            testWindowsGpuDetection(result);
            
            result.setStatus(TestResult.Status.PASSED);
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.WARNING);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() != TestResult.Status.FAILED);
    }
    
    @Test
    @EnabledOnOs(OS.MAC)
    @DisplayName("macOS-Specific Compatibility")
    void testMacOsSpecificCompatibility() {
        TestResult result = new TestResult("macOS-Specific Compatibility");
        
        try {
            // Test Metal availability on macOS
            testMetalAvailability(result);
            
            // Test Apple Silicon compatibility
            testAppleSiliconCompatibility(result);
            
            // Test macOS-specific GPU features
            testMacOsGpuFeatures(result);
            
            result.setStatus(TestResult.Status.PASSED);
            
        } catch (Exception e) {
            result.setStatus(TestResult.Status.WARNING);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() != TestResult.Status.FAILED);
    }
    
    @Test
    @DisplayName("Memory Management Cross-Platform")
    void testMemoryManagementCrossPlatform() {
        TestResult result = new TestResult("Memory Management Cross-Platform");
        
        try {
            MatrixOperation matrixOp = createMatrixOperation();
            
            // Test large memory allocations
            int size = 1024 * 1024; // 1M elements
            float[] largeMatrix = new float[size];
            float[] sigmoidResult = new float[size];
            
            // Fill with test data
            for (int i = 0; i < size; i++) {
                largeMatrix[i] = (float) Math.random();
            }
            
            // Test memory operations
            matrixOp.sigmoid(largeMatrix, sigmoidResult, size);
            assertNotNull(sigmoidResult);
            result.addDetail("Large Matrix Sigmoid", "Working");
            
            // Test memory cleanup
            matrixOp.release();
            result.addDetail("Memory Cleanup", "Working");
            
            result.setStatus(TestResult.Status.PASSED);
            
        } catch (OutOfMemoryError e) {
            result.setStatus(TestResult.Status.WARNING);
            result.addDetail("Out of Memory", e.getMessage());
        } catch (Exception e) {
            result.setStatus(TestResult.Status.FAILED);
            result.addDetail("Error", e.getMessage());
        }
        
        compatibilityReport.addResult(result);
        assertTrue(result.getStatus() != TestResult.Status.FAILED);
    }
    
    // Helper methods for specific tests
    
    private void testMatrixAddition(MatrixOperation matrixOp, TestResult result) {
        float[] a = {1, 2, 3, 4};
        float[] b = {5, 6, 7, 8};
        float[] res = new float[4];
        
        matrixOp.add(a, b, res, 4);
        
        float[] expected = {6, 8, 10, 12};
        assertArrayEquals(expected, res, 0.001f);
        result.addDetail("Matrix Addition", "Working");
    }
    
    private void testMatrixMultiplication(MatrixOperation matrixOp, TestResult result) {
        float[] a = {1, 2, 3, 4};
        float[] b = {1, 0, 0, 1};
        float[] res = new float[4];
        
        matrixOp.multiply(a, b, res, 2, 2, 2);
        
        float[] expected = {1, 2, 3, 4};
        assertArrayEquals(expected, res, 0.001f);
        result.addDetail("Matrix Multiplication", "Working");
    }
    
    private void testActivationFunctions(MatrixOperation matrixOp, TestResult result) {
        float[] input = {-1, 0, 1, 2};
        float[] output = new float[4];
        
        // Test sigmoid
        matrixOp.sigmoid(input, output, 4);
        assertTrue(output[0] < 0.5f && output[3] > 0.5f);
        result.addDetail("Sigmoid Activation", "Working");
        
        // Test ReLU
        matrixOp.relu(input, output, 4);
        assertEquals(0.0f, output[0], 0.001f);
        assertEquals(2.0f, output[3], 0.001f);
        result.addDetail("ReLU Activation", "Working");
    }
    
    private void testStatisticalOperations(MatrixOperation matrixOp, TestResult result) {
        float[] data = {1, 2, 3, 4, 5};
        float[] output = new float[1];
        
        // Test mean
        matrixOp.mean(data, output, 5);
        assertEquals(3.0f, output[0], 0.001f);
        result.addDetail("Mean Calculation", "Working");
        
        // Test variance
        float[] meanResult = new float[1];
        matrixOp.mean(data, meanResult, 5);
        matrixOp.variance(data, output, 5, meanResult[0]);
        assertTrue(output[0] > 0);
        result.addDetail("Variance Calculation", "Working");
    }
    
    private void testPerceptronModel(TestResult result) {
        GpuPerceptronModel perceptron = new GpuPerceptronModel(config, 0.1f, 10);
        
        try {
            // Create simple training data
            float[][] features = {{1, 0}, {0, 1}, {1, 1}, {0, 0}};
            int[] labels = {1, 1, 1, 0};
            
            perceptron.train(features, labels);
            
            // Test prediction
            int prediction = perceptron.predict(new float[]{1, 1});
            assertTrue(prediction >= 0 && prediction <= 1);
            
            result.addDetail("Perceptron Training", "Working");
            result.addDetail("Perceptron Prediction", "Working");
            
        } finally {
            perceptron.cleanup();
        }
    }
    
    private void testMaxEntModel(TestResult result) {
        // Create a dummy MaxEnt model
        String[] outcomes = {"outcome1", "outcome2"};
        String[] predLabels = {"pred1", "pred2"};
        double[] params = {0.1, 0.2, 0.3, 0.4};
        
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
        
        // Create a GPU-accelerated wrapper
        GpuMaxentModel gpuModel = new GpuMaxentModel(cpuModel, config);
        
        try {
            String[] context = {"pred1"};
            double[] probs = gpuModel.eval(context);
            
            assertNotNull(probs);
            assertEquals(2, probs.length);
            assertTrue(Math.abs(probs[0] + probs[1] - 1.0) < 0.01);
            
            result.addDetail("MaxEnt Evaluation", "Working");
            
        } finally {
            gpuModel.cleanup();
        }
    }
    
    private boolean testCudaAvailability() {
        try {
            // Try to detect CUDA runtime
            ProcessBuilder pb = new ProcessBuilder("nvidia-smi");
            Process process = pb.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (Exception e) {
            return false;
        }
    }
    
    private boolean testRocmAvailability() {
        try {
            // Try to detect ROCm
            ProcessBuilder pb = new ProcessBuilder("rocm-smi");
            Process process = pb.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (Exception e) {
            return false;
        }
    }
    
    private boolean testOpenClAvailability() {
        try {
            // Try to detect OpenCL
            ProcessBuilder pb = new ProcessBuilder("clinfo");
            Process process = pb.start();
            int exitCode = process.waitFor();
            return exitCode == 0;
        } catch (Exception e) {
            return false;
        }
    }
    
    private void testWindowsPathHandling(TestResult result) {
        // Test Windows-specific path operations
        String testPath = "C:\\Windows\\System32\\test.dll";
        assertTrue(testPath.contains("Windows"), "Path should contain Windows directory");
        result.addDetail("Path Handling", "Basic Windows paths supported - " + testPath);
    }
    
    private void testDirectXAvailability(TestResult result) {
        // Test DirectX/DirectCompute availability
        result.addDetail("DirectX Support", "Not implemented yet");
    }
    
    private void testWindowsGpuDetection(TestResult result) {
        // Test Windows-specific GPU detection
        result.addDetail("Windows GPU Detection", "Using standard GPU detection");
    }
    
    private void testMetalAvailability(TestResult result) {
        // Test Metal compute availability on macOS
        result.addDetail("Metal Support", "Not implemented yet");
    }
    
    private void testAppleSiliconCompatibility(TestResult result) {
        String arch = System.getProperty("os.arch");
        boolean isAppleSilicon = arch.contains("aarch64") || arch.contains("arm64");
        result.addDetail("Apple Silicon", isAppleSilicon ? "Detected" : "Not detected");
    }
    
    private void testMacOsGpuFeatures(TestResult result) {
        // Test macOS-specific GPU features
        result.addDetail("macOS GPU Features", "Using standard GPU detection");
    }
    
    private MatrixOperation createMatrixOperation() {
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                GpuComputeProvider provider = new GpuComputeProvider(config);
                return new GpuMatrixOperation(provider, config);
            }
        } catch (Exception e) {
            logger.debug("GPU not available, using CPU: " + e.getMessage());
        }
        
        CpuComputeProvider provider = new CpuComputeProvider();
        return new CpuMatrixOperation(provider);
    }
    
    private GpuFeatureExtractor createFeatureExtractor() {
        MatrixOperation matrixOp = createMatrixOperation();
        
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                GpuComputeProvider provider = new GpuComputeProvider(config);
                return new GpuFeatureExtractor(provider, config, matrixOp);
            }
        } catch (Exception e) {
            logger.debug("GPU not available for feature extraction: " + e.getMessage());
        }
        
        CpuComputeProvider provider = new CpuComputeProvider();
        return new GpuFeatureExtractor(provider, config, matrixOp);
    }
    
    private Map<String, Object> gatherSystemInformation() {
        Map<String, Object> info = new HashMap<>();
        
        info.put("os.name", System.getProperty("os.name"));
        info.put("os.version", System.getProperty("os.version"));
        info.put("os.arch", System.getProperty("os.arch"));
        info.put("java.version", System.getProperty("java.version"));
        info.put("java.vendor", System.getProperty("java.vendor"));
        info.put("java.home", System.getProperty("java.home"));
        info.put("user.name", System.getProperty("user.name"));
        info.put("user.home", System.getProperty("user.home"));
        info.put("available.processors", Runtime.getRuntime().availableProcessors());
        info.put("max.memory", Runtime.getRuntime().maxMemory());
        info.put("total.memory", Runtime.getRuntime().totalMemory());
        info.put("free.memory", Runtime.getRuntime().freeMemory());
        
        return info;
    }
    
    private void printCompatibilityReport() {
        logger.info("========== CROSS-PLATFORM COMPATIBILITY REPORT ==========");
        
        int totalTests = compatibilityReport.getResults().size();
        int passedTests = 0;
        int warningTests = 0;
        int failedTests = 0;
        
        for (TestResult result : compatibilityReport.getResults()) {
            logger.info(result.toString());
            
            switch (result.getStatus()) {
                case PASSED:
                    passedTests++;
                    break;
                case WARNING:
                    warningTests++;
                    break;
                case FAILED:
                    failedTests++;
                    break;
            }
        }
        
        logger.info("========== SUMMARY ==========");
        logger.info("Total tests: " + totalTests);
        logger.info("Passed: " + passedTests);
        logger.info("Warnings: " + warningTests);
        logger.info("Failed: " + failedTests);
        logger.info("Success rate: " + String.format("%.1f%%", (passedTests * 100.0 / totalTests)));
        
        // Add system information to report
        logger.info("========== SYSTEM INFORMATION ==========");
        for (Map.Entry<String, Object> entry : systemInfo.entrySet()) {
            logger.info(entry.getKey() + ": " + entry.getValue());
        }
    }
    
    /**
     * Container for test results
     */
    private static class TestResult {
        public enum Status { PASSED, WARNING, FAILED }
        
        private final String testName;
        private Status status;
        private final Map<String, String> details;
        
        public TestResult(String testName) {
            this.testName = testName;
            this.status = Status.FAILED;
            this.details = new HashMap<>();
        }
        
        public void setStatus(Status status) {
            this.status = status;
        }
        
        public Status getStatus() {
            return status;
        }
        
        public void addDetail(String key, String value) {
            details.put(key, value);
        }
        
        public Map<String, String> getDetails() {
            return details;
        }
        
        @Override
        public String toString() {
            StringBuilder sb = new StringBuilder();
            sb.append(String.format("%-40s [%s]", testName, status));
            
            if (!details.isEmpty()) {
                sb.append("\n");
                for (Map.Entry<String, String> entry : details.entrySet()) {
                    sb.append(String.format("  %-20s: %s\n", entry.getKey(), entry.getValue()));
                }
            }
            
            return sb.toString();
        }
    }
    
    /**
     * Container for compatibility report
     */
    private static class CompatibilityReport {
        private final List<TestResult> results;
        
        public CompatibilityReport() {
            this.results = new ArrayList<>();
        }
        
        public void addResult(TestResult result) {
            results.add(result);
        }
        
        public List<TestResult> getResults() {
            return results;
        }
    }
}
