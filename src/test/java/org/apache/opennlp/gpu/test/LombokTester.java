package org.apache.opennlp.gpu.test;

import lombok.Getter;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.RequiredArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

@RequiredArgsConstructor
public class LombokTester {
    private static final Logger logger = LoggerFactory.getLogger(LombokTester.class);
    @Getter
    private final String name;
    
    public void testLogging() {
        logger.info("Testing Lombok logging with field name: {}", name);
    }
}
