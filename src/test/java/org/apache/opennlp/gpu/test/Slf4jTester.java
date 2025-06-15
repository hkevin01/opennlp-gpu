package org.apache.opennlp.gpu.test;

import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.slf4j.MDC;

/**
 * Test class to verify SLF4J logging configuration and functionality
 * Ensures proper logging setup for GPU acceleration components
 */
public class Slf4jTester {
    
    private static final Logger logger = LoggerFactory.getLogger(Slf4jTester.class);
    
    @Test
    public void testBasicLogging() {
        logger.trace("This is a TRACE message");
        logger.debug("This is a DEBUG message");
        logger.info("This is an INFO message");
        logger.warn("This is a WARN message");
        logger.error("This is an ERROR message");
    }
    
    @Test
    public void testParameterizedLogging() {
        String component = "GpuMatrixOperation";
        int operations = 42;
        double time = 123.45;
        
        logger.info("Component {} completed {} operations in {} ms", 
                   component, operations, time);
        
        logger.debug("Processing matrix of size {}x{} with {} elements", 
                    100, 100, 10000);
    }
    
    @Test
    public void testExceptionLogging() {
        try {
            throw new RuntimeException("Test exception for logging");
        } catch (Exception e) {
            logger.error("Exception occurred during GPU operation", e);
            logger.warn("Fallback to CPU implementation due to: {}", e.getMessage());
        }
    }
    
    @Test
    public void testMDCLogging() {
        MDC.put("component", "GPU");
        MDC.put("operation", "matrixMultiply");
        MDC.put("size", "1000x1000");
        
        logger.info("Starting GPU operation");
        logger.debug("Allocating GPU memory");
        logger.info("Operation completed successfully");
        
        MDC.clear();
    }
    
    @Test
    public void testLoggerHierarchy() {
        Logger gpuLogger = LoggerFactory.getLogger("org.apache.opennlp.gpu");
        Logger matrixLogger = LoggerFactory.getLogger("org.apache.opennlp.gpu.compute");
        Logger kernelLogger = LoggerFactory.getLogger("org.apache.opennlp.gpu.kernels");
        
        gpuLogger.info("GPU subsystem initialized");
        matrixLogger.debug("Matrix operations ready");
        kernelLogger.trace("OpenCL kernels compiled");
    }
    
    @Test
    public void testPerformanceLogging() {
        long startTime = System.currentTimeMillis();
        
        // Simulate some work
        try {
            Thread.sleep(10);
        } catch (InterruptedException e) {
            Thread.currentThread().interrupt();
        }
        
        long duration = System.currentTimeMillis() - startTime;
        
        logger.info("GPU operation completed in {} ms", duration);
        
        if (duration > 100) {
            logger.warn("GPU operation took longer than expected: {} ms", duration);
        }
    }
    
    /**
     * Test main method for standalone execution
     */
    public static void main(String[] args) {
        System.out.println("🧪 Testing SLF4J Logging Configuration");
        System.out.println("=====================================");
        
        Slf4jTester tester = new Slf4jTester();
        
        try {
            tester.testBasicLogging();
            System.out.println("✅ Basic logging test passed");
            
            tester.testParameterizedLogging();
            System.out.println("✅ Parameterized logging test passed");
            
            tester.testExceptionLogging();
            System.out.println("✅ Exception logging test passed");
            
            tester.testMDCLogging();
            System.out.println("✅ MDC logging test passed");
            
            tester.testLoggerHierarchy();
            System.out.println("✅ Logger hierarchy test passed");
            
            tester.testPerformanceLogging();
            System.out.println("✅ Performance logging test passed");
            
            System.out.println("🎉 All SLF4J tests completed successfully!");
            
        } catch (Exception e) {
            System.err.println("❌ SLF4J test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
