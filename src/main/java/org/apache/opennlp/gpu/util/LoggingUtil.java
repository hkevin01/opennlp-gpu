package org.apache.opennlp.gpu.util;

import ch.qos.logback.classic.Level;
import ch.qos.logback.classic.Logger;
import ch.qos.logback.classic.LoggerContext;
import org.slf4j.LoggerFactory;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;

/**
 * Utility class for adjusting logging settings programmatically.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class LoggingUtil {

    /**
     * Sets the logging level for a specific package.
     *
     * @param packageName the package name
     * @param level the logging level
     */
    public static void setLogLevel(String packageName, Level level) {
        LoggerContext loggerContext = (LoggerContext) LoggerFactory.getILoggerFactory();
        Logger logger = loggerContext.getLogger(packageName);
        logger.setLevel(level);
    }

    /**
     * Sets the logging level for GPU operations.
     *
     * @param level the logging level
     */
    public static void setGpuLogLevel(Level level) {
        setLogLevel("org.apache.opennlp.gpu", level);
    }

    /**
     * Enables performance logging.
     */
    public static void enablePerformanceLogging() {
        setLogLevel("org.apache.opennlp.gpu.perf", Level.DEBUG);
    }

    /**
     * Disables performance logging.
     */
    public static void disablePerformanceLogging() {
        setLogLevel("org.apache.opennlp.gpu.perf", Level.INFO);
    }
}
