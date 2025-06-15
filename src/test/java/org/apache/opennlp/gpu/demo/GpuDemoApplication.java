package org.apache.opennlp.gpu.demo;

import org.apache.opennlp.gpu.benchmark.PerformanceBenchmark;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.test.GpuTestSuite;

/**
 * Demonstration application showcasing GPU acceleration capabilities
 * Runs tests and benchmarks to validate GPU acceleration
 */
public class GpuDemoApplication {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuDemoApplication.class);
    
    public static void main(String[] args) {
        GpuDemoApplication.logger.info("🚀 Starting OpenNLP GPU Acceleration Demo");
        
        try {
            // Initialize configuration
            GpuConfig config = new GpuConfig();
            config.setGpuEnabled(true);
            
            // Run comprehensive tests
            GpuDemoApplication.runTestSuite();
            
            // Run performance benchmarks
            GpuDemoApplication.runBenchmarks();
            
            GpuDemoApplication.logger.info("🎉 Demo completed successfully!");
            
        } catch (Exception e) {
            GpuDemoApplication.logger.error("Demo failed: " + e.getMessage());
            System.exit(1);
        }
    }
    
    private static void runTestSuite() {
        GpuDemoApplication.logger.info("📋 Running comprehensive test suite...");
        
        GpuTestSuite testSuite = new GpuTestSuite();
        GpuTestSuite.TestResults results = testSuite.runAllTests();
        
        System.out.println("\n=== TEST RESULTS ===");
        System.out.println(results.getReport());
        
        if (results.allPassed()) {
            GpuDemoApplication.logger.info("✅ All tests passed!");
        } else {
            GpuDemoApplication.logger.warn("⚠️ Some tests failed - see report above");
        }
    }
    
    private static void runBenchmarks() {
        GpuDemoApplication.logger.info("📊 Running performance benchmarks...");
        
        PerformanceBenchmark benchmark = new PerformanceBenchmark();
        PerformanceBenchmark.BenchmarkResults results = benchmark.runFullBenchmark();
        
        System.out.println("\n=== BENCHMARK RESULTS ===");
        System.out.println(results.generateReport());
        
        if (results.isValid()) {
            GpuDemoApplication.logger.info("✅ Benchmarks completed successfully!");
            GpuDemoApplication.logger.info("🚀 Overall GPU speedup: " + String.format("%.2f", results.getOverallSpeedup()) + "x");
        } else {
            GpuDemoApplication.logger.warn("⚠️ Benchmark issues detected - see report above");
        }
    }
}
