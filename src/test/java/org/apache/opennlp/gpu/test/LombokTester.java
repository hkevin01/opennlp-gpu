package org.apache.opennlp.gpu.test;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.RequiredArgsConstructor;

@RequiredArgsConstructor
@NoArgsConstructor(force = true)
public class LombokTester {
    private static final Logger logger = LoggerFactory.getLogger(LombokTester.class);
    
    @Getter
    private final String name = "Default Name"; // Provide a default value
    
    public void testLogging() {
        logger.info("Testing Lombok logging with field name: {}", name);
    }
}
