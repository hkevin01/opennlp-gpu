package org.apache.opennlp.gpu.test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Test class for demonstrating SLF4J logging functionality.
 */
public class Slf4jTester {
    // Use the correct logger name
    private static final Logger logger = LoggerFactory.getLogger(Slf4jTester.class);
    
    /**
     * Test different log levels.
     */
    public void testLogLevels() {
        logger.trace("This is a TRACE message");
        logger.debug("This is a DEBUG message");
        logger.info("This is an INFO message");
        logger.warn("This is a WARN message");
        logger.error("This is an ERROR message");
    }
    
    /**
     * Test parameterized logging.
     * 
     * @param param1 first parameter
     * @param param2 second parameter
     */
    public void testParameterizedLogging(String param1, int param2) {
        logger.info("Testing with parameters: {} and {}", param1, param2);
    }
    
    /**
     * Test logging with exception.
     * 
     * @param exception the exception to log
     */
    public void testExceptionLogging(Exception exception) {
        logger.error("An error occurred", exception);
    }
    
    /**
     * Main method to demonstrate SLF4J logging.
     * 
     * @param args command line arguments
     */
    public static void main(String[] args) {
        Slf4jTester tester = new Slf4jTester();
        
        System.out.println("Testing SLF4J logging...");
        
        tester.testLogLevels();
        tester.testParameterizedLogging("test", 123);
        
        try {
            throw new RuntimeException("Test exception");
        } catch (Exception e) {
            tester.testExceptionLogging(e);
        }
        
        System.out.println("SLF4J logging test complete");
    }
}
