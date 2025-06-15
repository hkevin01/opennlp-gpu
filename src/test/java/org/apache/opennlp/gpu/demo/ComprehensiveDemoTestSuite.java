package org.apache.opennlp.gpu.demo;

import java.io.ByteArrayOutputStream;
import java.io.PrintStream;

import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Suite;
import org.junit.runners.Suite.SuiteClasses;

/**
 * Comprehensive JUnit test suite for all GPU demo configurations
 * Tests all demo scenarios programmatically
 */
@RunWith(Suite.class)
@SuiteClasses({
    ComprehensiveDemoTestSuite.BasicDemoTest.class,
    ComprehensiveDemoTestSuite.OpenClDemoTest.class,
    ComprehensiveDemoTestSuite.DebugDemoTest.class,
    ComprehensiveDemoTestSuite.ComprehensiveDemoTest.class,
    ComprehensiveDemoTestSuite.PerformanceDemoTest.class
})
public class ComprehensiveDemoTestSuite {
    
    /**
     * Test 1: Basic Demo Application
     */
    public static class BasicDemoTest {
        private ByteArrayOutputStream outputStream;
        private PrintStream originalOut;
        
        @Before
        public void setUp() {
            outputStream = new ByteArrayOutputStream();
            originalOut = System.out;
            System.setOut(new PrintStream(outputStream));
        }
        
        @After
        public void tearDown() {
            System.setOut(originalOut);
        }
        
        @Test
        public void testBasicDemo() {
            try {
                // Clear any existing system properties
                System.clearProperty("gpu.backend");
                System.clearProperty("gpu.debug");
                System.clearProperty("comprehensive");
                System.clearProperty("performance.only");
                
                // Run basic demo
                GpuDemoApplication.main(new String[]{});
                
                String output = outputStream.toString();
                assert output.contains("Starting OpenNLP GPU Acceleration Demo") : "Demo should start properly";
                assert !output.contains("ERROR") : "Demo should not contain errors";
                
            } catch (Exception e) {
                System.setOut(originalOut);
                System.err.println("Basic demo test failed: " + e.getMessage());
                throw new AssertionError("Basic demo test failed", e);
            }
        }
    }
    
    /**
     * Test 2: OpenCL Configuration Demo
     */
    public static class OpenClDemoTest {
        private ByteArrayOutputStream outputStream;
        private PrintStream originalOut;
        
        @Before
        public void setUp() {
            outputStream = new ByteArrayOutputStream();
            originalOut = System.out;
            System.setOut(new PrintStream(outputStream));
        }
        
        @After
        public void tearDown() {
            System.setOut(originalOut);
            System.clearProperty("gpu.backend");
        }
        
        @Test
        public void testOpenClDemo() {
            try {
                // Set OpenCL backend
                System.setProperty("gpu.backend", "opencl");
                
                // Run demo
                GpuDemoApplication.main(new String[]{});
                
                String output = outputStream.toString();
                assert output.contains("Starting OpenNLP GPU Acceleration Demo") : "Demo should start properly";
                
            } catch (Exception e) {
                System.setOut(originalOut);
                System.err.println("OpenCL demo test failed: " + e.getMessage());
                throw new AssertionError("OpenCL demo test failed", e);
            }
        }
    }
    
    /**
     * Test 3: Debug Mode Demo
     */
    public static class DebugDemoTest {
        private ByteArrayOutputStream outputStream;
        private PrintStream originalOut;
        
        @Before
        public void setUp() {
            outputStream = new ByteArrayOutputStream();
            originalOut = System.out;
            System.setOut(new PrintStream(outputStream));
        }
        
        @After
        public void tearDown() {
            System.setOut(originalOut);
            System.clearProperty("gpu.debug");
        }
        
        @Test
        public void testDebugDemo() {
            try {
                // Enable debug mode
                System.setProperty("gpu.debug", "true");
                
                // Run demo
                GpuDemoApplication.main(new String[]{});
                
                String output = outputStream.toString();
                assert output.contains("Starting OpenNLP GPU Acceleration Demo") : "Demo should start properly";
                // In debug mode, we might see more detailed output
                
            } catch (Exception e) {
                System.setOut(originalOut);
                System.err.println("Debug demo test failed: " + e.getMessage());
                throw new AssertionError("Debug demo test failed", e);
            }
        }
    }
    
    /**
     * Test 4: Comprehensive Testing Demo
     */
    public static class ComprehensiveDemoTest {
        private ByteArrayOutputStream outputStream;
        private PrintStream originalOut;
        
        @Before
        public void setUp() {
            outputStream = new ByteArrayOutputStream();
            originalOut = System.out;
            System.setOut(new PrintStream(outputStream));
        }
        
        @After
        public void tearDown() {
            System.setOut(originalOut);
            System.clearProperty("comprehensive");
        }
        
        @Test
        public void testComprehensiveDemo() {
            try {
                // Enable comprehensive testing
                System.setProperty("comprehensive", "true");
                
                // Run demo
                GpuDemoApplication.main(new String[]{});
                
                String output = outputStream.toString();
                assert output.contains("Starting OpenNLP GPU Acceleration Demo") : "Demo should start properly";
                assert output.contains("test") || output.contains("benchmark") : "Should run tests or benchmarks";
                
            } catch (Exception e) {
                System.setOut(originalOut);
                System.err.println("Comprehensive demo test failed: " + e.getMessage());
                throw new AssertionError("Comprehensive demo test failed", e);
            }
        }
    }
    
    /**
     * Test 5: Performance Focus Demo
     */
    public static class PerformanceDemoTest {
        private ByteArrayOutputStream outputStream;
        private PrintStream originalOut;
        
        @Before
        public void setUp() {
            outputStream = new ByteArrayOutputStream();
            originalOut = System.out;
            System.setOut(new PrintStream(outputStream));
        }
        
        @After
        public void tearDown() {
            System.setOut(originalOut);
            System.clearProperty("performance.only");
        }
        
        @Test
        public void testPerformanceDemo() {
            try {
                // Enable performance-only mode
                System.setProperty("performance.only", "true");
                
                // Run demo
                GpuDemoApplication.main(new String[]{});
                
                String output = outputStream.toString();
                assert output.contains("Starting OpenNLP GPU Acceleration Demo") : "Demo should start properly";
                // Performance mode should focus on benchmarks
                
            } catch (Exception e) {
                System.setOut(originalOut);
                System.err.println("Performance demo test failed: " + e.getMessage());
                throw new AssertionError("Performance demo test failed", e);
            }
        }
    }
    
    /**
     * Convenience method to run all tests programmatically
     */
    public static void runAllDemoTests() {
        System.out.println("🧪 Running comprehensive demo test suite...");
        
        try {
            // Run each test class
            new BasicDemoTest().testBasicDemo();
            System.out.println("✅ Basic demo test passed");
            
            new OpenClDemoTest().testOpenClDemo();
            System.out.println("✅ OpenCL demo test passed");
            
            new DebugDemoTest().testDebugDemo();
            System.out.println("✅ Debug demo test passed");
            
            new ComprehensiveDemoTest().testComprehensiveDemo();
            System.out.println("✅ Comprehensive demo test passed");
            
            new PerformanceDemoTest().testPerformanceDemo();
            System.out.println("✅ Performance demo test passed");
            
            System.out.println("🎉 All demo tests completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ Demo test suite failed: " + e.getMessage());
            throw new RuntimeException("Demo test suite failed", e);
        }
    }
    
    /**
     * Main method for standalone execution
     */
    public static void main(String[] args) {
        System.out.println("🚀 Running OpenNLP GPU Demo Test Suite");
        System.out.println("======================================");
        
        // Add detailed compilation and dependency checks
        try {
            System.out.println("🔍 Checking project dependencies...");
            
            // Test if all classes are available
            Class.forName("org.apache.opennlp.gpu.demo.GpuDemoApplication");
            System.out.println("✅ GpuDemoApplication found");
            
            Class.forName("org.apache.opennlp.gpu.common.GpuConfig");
            System.out.println("✅ GpuConfig found");
            
            // Check for test classes with graceful fallback
            try {
                Class.forName("org.apache.opennlp.gpu.test.GpuTestSuite");
                System.out.println("✅ GpuTestSuite found");
            } catch (ClassNotFoundException e) {
                System.out.println("⚠️ GpuTestSuite not found (optional)");
            }
            
            try {
                Class.forName("org.apache.opennlp.gpu.benchmark.PerformanceBenchmark");
                System.out.println("✅ PerformanceBenchmark found");
            } catch (ClassNotFoundException e) {
                System.out.println("⚠️ PerformanceBenchmark not found (optional)");
            }
            
            System.out.println("✅ Core classes found - ready to run tests");
            
        } catch (ClassNotFoundException e) {
            System.err.println("❌ Missing required classes: " + e.getMessage());
            System.err.println("");
            System.err.println("🛠️ SOLUTION: Please compile the project first:");
            System.err.println("   1. Open terminal in project root");
            System.err.println("   2. Run: mvn clean compile test-compile");
            System.err.println("   3. Then try running this class again");
            System.err.println("");
            System.err.println("💡 Alternative: Use Maven exec plugin:");
            System.err.println("   mvn exec:java -Dexec.mainClass=\"org.apache.opennlp.gpu.demo.ComprehensiveDemoTestSuite\"");
            System.exit(1);
        } catch (Exception e) {
            System.err.println("❌ Unexpected error: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
        
        // Check Java version
        String javaVersion = System.getProperty("java.version");
        System.out.println("☕ Java version: " + javaVersion);
        
        // Run the actual tests
        try {
            ComprehensiveDemoTestSuite.runAllDemoTests();
        } catch (Exception e) {
            System.err.println("❌ Test execution failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
