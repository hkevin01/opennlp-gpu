package org.apache.opennlp.gpu.common;

// Correct JUnit 4 imports
import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotNull;

import lombok.extern.slf4j.Slf4j; // Keep for testing if @Slf4j works after fixes
import lombok.Getter;
import lombok.Setter;


/**
 * Simple test class to verify Lombok annotation processing is working.
 */
@Slf4j // This will be replaced by fix_slf4j_annotations if it runs after this
public class LombokTest {
    // If fix_slf4j_annotations runs, 'log' will become 'logger'
    // private static final org.slf4j.Logger logger = org.slf4j.LoggerFactory.getLogger(LombokTest.class);
    
    @Getter @Setter
    private String testField;
    
    @Test
    public void testLombokAnnotations() {
        // If fix_slf4j_annotations has run, use logger.info, otherwise log.info
        // For robustness, the script should ensure logger is used if @Slf4j is removed.
        // The current fix_slf4j_annotations script will change log. to logger.
        log.info("Testing Lombok annotations"); // This will be converted to logger.info by fix_slf4j_annotations
        
        // Test @Getter and @Setter
        setTestField("test value");
        assertEquals("test value", getTestField());
        
        // Verify the log field was generated (or logger if transformed)
        assertNotNull(log); // This will be converted to logger by fix_slf4j_annotations
        
        log.info("Lombok annotations working correctly"); // Converted to logger.info
    }
}
