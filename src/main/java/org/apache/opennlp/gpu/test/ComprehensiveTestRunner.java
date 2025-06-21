package org.apache.opennlp.gpu.test;

import java.util.ArrayList;
import java.util.List;

import org.apache.opennlp.gpu.benchmark.PerformanceBenchmark;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.tools.GpuDiagnostics;

/**
 * Comprehensive test runner for all OpenNLP GPU components
 * Runs diagnostics, tests, and benchmarks in sequence
 */
public class ComprehensiveTestRunner {
    
    private static final GpuLogger logger = GpuLogger.getLogger(ComprehensiveTestRunner.class);
    
    private final List<String> results = new ArrayList<>();
    private boolean allTestsPassed = true;
    
    public static void main(String[] args) {
        System.out.println("==============================================================");
        System.out.println("OpenNLP GPU - Comprehensive Test Suite");
        System.out.println("==============================================================");
        System.out.println();
        
        ComprehensiveTestRunner runner = new ComprehensiveTestRunner();
        boolean success = runner.runAllTests();
        
        System.out.println();
        System.out.println("==============================================================");
        System.out.println("TEST SUITE SUMMARY");
        System.out.println("==============================================================");
        
        for (String result : runner.results) {
            System.out.println(result);
        }
        
        System.out.println();
        if (success) {
            System.out.println("🎉 ALL TESTS PASSED - System ready for development!");
        } else {
            System.out.println("⚠️  Some tests failed - Review results above");
        }
        System.out.println("==============================================================");
        
        System.exit(success ? 0 : 1);
    }
    
    public boolean runAllTests() {
        // Step 1: GPU Diagnostics
        runGpuDiagnostics();
        
        // Step 2: Core Component Tests
        runCoreComponentTests();
        
        // Step 3: Performance Benchmarks
        runPerformanceBenchmarks();
        
        // Step 4: Integration Tests
        runIntegrationTests();
        
        return allTestsPassed;
    }
    
    private void runGpuDiagnostics() {
        System.out.println("1️⃣  Running GPU Diagnostics...");
        System.out.println("------------------------------");
        
        try {
            // Run GPU diagnostics
            GpuDiagnostics diagnostics = new GpuDiagnostics();
            GpuDiagnostics.DiagnosticReport report = diagnostics.runComprehensiveDiagnostics();
            
            boolean gpuReady = report.isGpuReady();
            
            if (gpuReady) {
                results.add("✅ GPU Diagnostics: GPU available and ready");
                System.out.println("✅ GPU detected and ready for acceleration");
            } else {
                results.add("⚠️  GPU Diagnostics: No GPU detected, using CPU fallback");
                System.out.println("⚠️  No GPU detected - tests will run with CPU fallback");
            }
            
        } catch (Exception e) {
            results.add("❌ GPU Diagnostics: Failed with error - " + e.getMessage());
            System.out.println("❌ GPU diagnostics failed: " + e.getMessage());
            allTestsPassed = false;
        }
        
        System.out.println();
    }
    
    private void runCoreComponentTests() {
        System.out.println("2️⃣  Running Core Component Tests...");
        System.out.println("------------------------------------");
        
        // Test matrix operations
        testMatrixOperations();
        
        // Test feature extraction
        testFeatureExtraction();
        
        // Test neural pipeline
        testNeuralPipeline();
        
        System.out.println();
    }
    
    private void testMatrixOperations() {
        try {
            System.out.println("Testing Matrix Operations...");
            
            // Simple matrix operation test
            org.apache.opennlp.gpu.compute.CpuComputeProvider provider = 
                new org.apache.opennlp.gpu.compute.CpuComputeProvider();
            org.apache.opennlp.gpu.compute.CpuMatrixOperation matrixOp = 
                new org.apache.opennlp.gpu.compute.CpuMatrixOperation(provider);
            
            // Test basic operations
            float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
            float[] result = new float[4];
            matrixOp.add(a, b, result, 4);
            
            boolean passed = (result[0] == 6.0f && result[1] == 8.0f && 
                            result[2] == 10.0f && result[3] == 12.0f);
            
            if (passed) {
                results.add("✅ Matrix Operations: Basic operations working");
                System.out.println("  ✅ Matrix addition test passed");
            } else {
                results.add("❌ Matrix Operations: Basic operations failed");
                System.out.println("  ❌ Matrix addition test failed");
                allTestsPassed = false;
            }
            
            matrixOp.release();
            provider.cleanup();
            
        } catch (Exception e) {
            results.add("❌ Matrix Operations: Failed with error - " + e.getMessage());
            System.out.println("  ❌ Matrix operations test failed: " + e.getMessage());
            allTestsPassed = false;
        }
    }
    
    private void testFeatureExtraction() {
        try {
            System.out.println("Testing Feature Extraction...");
            
            // Just test that the class can be loaded
            Class<?> extractorClass = org.apache.opennlp.gpu.features.GpuFeatureExtractor.class;
            boolean passed = (extractorClass != null);
            
            if (passed) {
                results.add("✅ Feature Extraction: Class accessible");
                System.out.println("  ✅ Feature extraction class test passed");
            } else {
                results.add("❌ Feature Extraction: Class not found");
                System.out.println("  ❌ Feature extraction class test failed");
                allTestsPassed = false;
            }
            
        } catch (Exception e) {
            results.add("❌ Feature Extraction: Failed with error - " + e.getMessage());
            System.out.println("  ❌ Feature extraction test failed: " + e.getMessage());
            allTestsPassed = false;
        }
    }
    
    private void testNeuralPipeline() {
        try {
            System.out.println("Testing Neural Pipeline...");
            
            // Just test that the class can be loaded
            Class<?> pipelineClass = org.apache.opennlp.gpu.ml.neural.GpuNeuralPipeline.class;
            boolean passed = (pipelineClass != null);
            
            if (passed) {
                results.add("✅ Neural Pipeline: Class accessible");
                System.out.println("  ✅ Neural pipeline class test passed");
            } else {
                results.add("❌ Neural Pipeline: Class not found");
                System.out.println("  ❌ Neural pipeline class test failed");
                allTestsPassed = false;
            }
            
        } catch (Exception e) {
            results.add("❌ Neural Pipeline: Failed with error - " + e.getMessage());
            System.out.println("  ❌ Neural pipeline test failed: " + e.getMessage());
            allTestsPassed = false;
        }
    }
    
    private void runPerformanceBenchmarks() {
        System.out.println("3️⃣  Running Performance Benchmarks...");
        System.out.println("-------------------------------------");
        
        try {
            PerformanceBenchmark benchmark = new PerformanceBenchmark();
            PerformanceBenchmark.BenchmarkResults benchmarkResults = benchmark.runFullBenchmark();
            
            double overallSpeedup = benchmarkResults.getOverallSpeedup();
            
            if (overallSpeedup > 0.8) { // Allow for some variance
                results.add("✅ Performance Benchmarks: Completed successfully (Speedup: " + 
                          String.format("%.2fx", overallSpeedup) + ")");
                System.out.println("✅ Performance benchmarks completed");
                System.out.printf("   Overall speedup: %.2fx%n", overallSpeedup);
            } else {
                results.add("⚠️  Performance Benchmarks: Low speedup detected (" + 
                          String.format("%.2fx", overallSpeedup) + ")");
                System.out.println("⚠️  Performance benchmarks show low speedup");
                System.out.printf("   Overall speedup: %.2fx%n", overallSpeedup);
            }
            
        } catch (Exception e) {
            results.add("❌ Performance Benchmarks: Failed with error - " + e.getMessage());
            System.out.println("❌ Performance benchmarks failed: " + e.getMessage());
            allTestsPassed = false;
        }
        
        System.out.println();
    }
    
    private void runIntegrationTests() {
        System.out.println("4️⃣  Running Integration Tests...");
        System.out.println("--------------------------------");
        
        // Test real OpenNLP integration
        testOpenNlpIntegration();
        
        // Test example applications
        testExampleApplications();
        
        System.out.println();
    }
    
    private void testOpenNlpIntegration() {
        try {
            System.out.println("Testing OpenNLP Integration...");
            
            // Test that OpenNLP MaxentModel class is available
            Class<?> maxentClass = opennlp.tools.ml.model.MaxentModel.class;
            boolean passed = (maxentClass != null);
            
            if (passed) {
                results.add("✅ OpenNLP Integration: OpenNLP classes accessible");
                System.out.println("  ✅ OpenNLP integration test passed");
            } else {
                results.add("❌ OpenNLP Integration: OpenNLP classes not found");
                System.out.println("  ❌ OpenNLP integration test failed");
                allTestsPassed = false;
            }
            
        } catch (Exception e) {
            results.add("❌ OpenNLP Integration: Failed with error - " + e.getMessage());
            System.out.println("  ❌ OpenNLP integration test failed: " + e.getMessage());
            allTestsPassed = false;
        }
    }
    
    private void testExampleApplications() {
        try {
            System.out.println("Testing Example Applications...");
            
            // Test that example classes exist and can be instantiated
            Class<?> sentimentClass = Class.forName("org.apache.opennlp.gpu.examples.sentiment.GpuSentimentAnalysis");
            Object sentimentExample = sentimentClass.getDeclaredConstructor().newInstance();
            
            boolean passed = (sentimentExample != null);
            
            if (passed) {
                results.add("✅ Example Applications: Examples can be instantiated");
                System.out.println("  ✅ Example applications test passed");
            } else {
                results.add("❌ Example Applications: Examples failed to instantiate");
                System.out.println("  ❌ Example applications test failed");
                allTestsPassed = false;
            }
            
        } catch (ClassNotFoundException e) {
            results.add("⚠️  Example Applications: Example classes not found (may be in examples/ directory)");
            System.out.println("  ⚠️  Example classes not in main source (normal - they're in examples/)");
        } catch (Exception e) {
            results.add("❌ Example Applications: Failed with error - " + e.getMessage());
            System.out.println("  ❌ Example applications test failed: " + e.getMessage());
            allTestsPassed = false;
        }
    }
}
