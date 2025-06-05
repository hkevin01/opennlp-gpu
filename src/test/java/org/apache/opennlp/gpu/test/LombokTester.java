package org.apache.opennlp.gpu.test;

import lombok.Getter;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;

@Slf4j
@RequiredArgsConstructor
public class LombokTester {
    @Getter
    private final String name;
    
    public void testLogging() {
        log.info("Testing Lombok logging with field name: {}", name);
    }
}
