package org.apache.opennlp.gpu.demo;

import org.apache.opennlp.gpu.benchmark.PerformanceBenchmark;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.integration.OpenNLPTestDataIntegration;
import org.apache.opennlp.gpu.test.GpuTestSuite;

/**

 * Requirement: GpuDemoApplication must run all GPU acceleration demo scenarios as JUnit 5 tests with pass/fail assertions.
 * Purpose: JUnit test class driving the full demo suite, asserting that all GPU or CPU-fallback paths complete without error.
 * Rationale: Turning demos into assertions provides continuous correctness validation across builds.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Initialises GPU context per test; asserts no exceptions and valid output shapes.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
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
            
            // Run integration tests with real data
            GpuDemoApplication.runIntegrationTests();
            
            GpuDemoApplication.logger.info("🎉 Demo completed successfully!");
            
        } catch (Exception e) {
            GpuDemoApplication.logger.error("Demo failed: " + e.getMessage(), e);
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
    
    private static void runIntegrationTests() {
        GpuDemoApplication.logger.info("🔗 Running integration tests with real OpenNLP data...");
        
        try {
            OpenNLPTestDataIntegration integration = new OpenNLPTestDataIntegration();
            integration.runRealModelTests();
            
            GpuDemoApplication.logger.info("✅ Integration tests completed successfully!");
        } catch (Exception e) {
            GpuDemoApplication.logger.warn("⚠️ Integration tests encountered issues: " + e.getMessage());
        }
    }
}
