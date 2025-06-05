# Logging with SLF4J and Logback

## Overview

The OpenNLP GPU project uses SLF4J (Simple Logging Facade for Java) with Logback as the implementation. This document explains how logging is configured and how to use it effectively in the project.

## Architecture

### SLF4J

SLF4J serves as an abstraction layer for various logging frameworks. It allows you to write logging code that is not tied to a specific implementation. Key benefits include:

- **Compile-time binding**: The logging implementation is determined at compile time, not runtime
- **No dynamic class loading**: Improves performance and security
- **Support for parameterized logging**: Better performance than string concatenation
- **Compatibility with multiple frameworks**: Can be used with Logback, Log4j, JUL, etc.

### Logback

Logback is the implementation of the SLF4J API used in this project. It was designed by the same developer who created Log4j and offers:

- **Better performance**: Faster than Log4j 1.x and java.util.logging
- **Extensive configuration options**: XML or Groovy configuration
- **Automatic reloading**: Configuration can be reloaded automatically when changed
- **Conditional processing**: Supports conditional configuration
- **Prudent mode**: Safe writing to log files from multiple JVMs
- **File compression and rolling policies**: For log file management

## Project Configuration

### Dependencies

The project uses the following logging dependencies:

```gradle
implementation 'org.slf4j:slf4j-api:2.0.7'           // The SLF4J API
implementation 'ch.qos.logback:logback-classic:1.4.11' // Logback implementation
implementation 'ch.qos.logback:logback-core:1.4.11'    // Logback core
```

### Configuration Files

#### Main Configuration

The main Logback configuration is located at:
`src/main/resources/logback.xml`

This configuration:
- Sets up console and file appenders
- Creates specific loggers for GPU operations
- Configures a separate performance logger
- Sets appropriate log levels for development

#### Test Configuration

A separate test configuration is located at:
`src/test/resources/logback-test.xml`

This configuration:
- Sets DEBUG level for tests
- Creates a separate test log file
- Resets the log file on each test run

## Using Logging in the Code

### Basic Usage

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyClass {
    private static final Logger logger = LoggerFactory.getLogger(MyClass.class);
    
    public void doSomething() {
        logger.debug("Debug message");
        logger.info("Info message");
        logger.warn("Warning message");
        logger.error("Error message");
    }
}
```

### Parameterized Logging

Use parameterized logging for better performance:

```java
// Preferred approach
logger.debug("Processing matrix: {} x {}", rows, cols);

// Avoid string concatenation
// logger.debug("Processing matrix: " + rows + " x " + cols); // Less efficient
```

### Exception Logging

```java
try {
    // Some code that might throw an exception
} catch (Exception e) {
    logger.error("Failed to process data", e);
    // The exception stack trace will be included in the log
}
```

## Log Levels

The project uses the following log levels:

| Level | Usage |
|-------|-------|
| ERROR | Serious failures that prevent normal operation |
| WARN  | Potential issues that don't prevent operation |
| INFO  | General information about system operation |
| DEBUG | Detailed information for debugging |
| TRACE | Very detailed information (rarely used) |

## Special Loggers

### Performance Logger

A dedicated logger for performance metrics:

```java
private static final Logger perfLogger = LoggerFactory.getLogger("org.apache.opennlp.gpu.perf");

// In performance-sensitive code
long startTime = System.nanoTime();
// ...perform operation...
long endTime = System.nanoTime();
perfLogger.info("Operation completed in {} ms", (endTime - startTime) / 1_000_000.0);
```

### GPU Operation Logger

GPU operations are logged with a specialized logger:

```java
private static final Logger gpuLogger = LoggerFactory.getLogger("org.apache.opennlp.gpu");

// Log GPU operation details
gpuLogger.debug("Matrix multiply: {}x{} * {}x{}", m, k, k, n);
```

## Programmatic Configuration

The project provides a utility class for adjusting logging levels at runtime:

```java
// Enable debug logging for GPU operations
LoggingUtil.setGpuLogLevel(Level.DEBUG);

// Enable detailed performance logging
LoggingUtil.enablePerformanceLogging();

// Set log level for a specific package
LoggingUtil.setLogLevel("org.apache.opennlp.gpu.kernels", Level.TRACE);
```

## Best Practices

1. **Use appropriate log levels**: Reserve ERROR for actual errors, INFO for important events
2. **Use parameterized logging**: Better performance and cleaner code
3. **Include context**: Log enough information to understand what happened
4. **Consider performance**: Be careful with DEBUG/TRACE logging in hot paths
5. **Structure logs**: Use consistent patterns for easier parsing
6. **Don't log sensitive data**: Be careful with personal or security-sensitive information

## Log File Management

The Logback configuration in this project includes:

- Daily rolling file policy
- 30-day retention policy
- 3GB total size cap
- Separate log files for tests

Logs are stored in the `logs/` directory:
- Main log: `logs/opennlp-gpu.log`
- Daily archives: `logs/opennlp-gpu.YYYY-MM-DD.log`
- Test log: `logs/test.log`
