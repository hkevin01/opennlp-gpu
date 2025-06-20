# Using Lombok with SLF4J in OpenNLP GPU

This guide explains how to effectively use Project Lombok together with SLF4J for logging in the OpenNLP GPU project.

## Introduction

### What is Lombok?

[Project Lombok](https://projectlombok.org/) is a Java library that automatically plugs into your editor and build tools to reduce boilerplate code. Lombok uses annotations to generate common code patterns at compile time, such as:

- Getters and setters
- Constructors
- `equals()` and `hashCode()` methods
- Builder patterns
- Logging declarations

### What is SLF4J?

[SLF4J (Simple Logging Facade for Java)](http://www.slf4j.org/) is a logging abstraction that allows you to use a single logging API while being able to switch between different logging implementations (like Logback, Log4j, or JUL) without code changes.

## Benefits of Using Lombok with SLF4J

1. **Reduced boilerplate**: Eliminates verbose logger declarations
2. **Consistent logging**: Enforces consistent logger naming patterns
3. **Compile-time safety**: Ensures loggers are properly initialized
4. **Cleaner code**: Removes noise from your classes

## Setup and Configuration

### Maven Configuration

```xml
<dependencies>
    <!-- Lombok -->
    <dependency>
        <groupId>org.projectlombok</groupId>
        <artifactId>lombok</artifactId>
        <version>1.18.30</version>
        <scope>provided</scope>
    </dependency>
    
    <!-- SLF4J API -->
    <dependency>
        <groupId>org.slf4j</groupId>
        <artifactId>slf4j-api</artifactId>
        <version>2.0.9</version>
    </dependency>
    
    <!-- SLF4J Implementation (e.g., Logback) -->
    <dependency>
        <groupId>ch.qos.logback</groupId>
        <artifactId>logback-classic</artifactId>
        <version>1.4.11</version>
    </dependency>
</dependencies>
```

### IDE Configuration

For VS Code:
1. Install the "Lombok Annotations Support for VS Code" extension
2. Configure settings.json:
   ```json
   {
       "java.jdt.ls.lombokSupport": true,
       "java.configuration.updateBuildConfiguration": "automatic"
   }
   ```

## Using Lombok for SLF4J Logging

### Traditional SLF4J Declaration

Without Lombok, you would declare an SLF4J logger like this:

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyClass {
    private static final Logger logger = LoggerFactory.getLogger(MyClass.class);
    
    public void doSomething() {
        logger.info("Doing something");
    }
}
```

### With Lombok's @Slf4j

Using Lombok's `@Slf4j` annotation simplifies this to:

```java
import lombok.extern.slf4j.Slf4j;

@Slf4j
public class MyClass {
    public void doSomething() {
        log.info("Doing something");
    }
}
```

Lombok automatically generates the logger field using the class name. The field is named `log` by default.

### Customizing the Logger Name

You can customize the logger name using the `topic` parameter:

```java
@Slf4j(topic = "com.example.CustomLogger")
public class MyClass {
    // ...
}
```

## Common Usage Patterns

### Combining with Other Lombok Annotations

Lombok's logging annotations work well with other Lombok features:

```java
import lombok.extern.slf4j.Slf4j;
import lombok.Getter;
import lombok.RequiredArgsConstructor;

@Slf4j
@RequiredArgsConstructor
public class ComputeOperation {
    @Getter
    private final ComputeProvider provider;
    
    public void execute() {
        log.info("Executing operation with provider: {}", provider.getName());
        // ...
    }
}
```

### Parameterized Logging

SLF4J supports parameterized logging which is more efficient than string concatenation:

```java
// Good - uses SLF4J's parameterized logging
log.debug("Processing item {} of {}", current, total);

// Bad - performs string concatenation even if debug is disabled
log.debug("Processing item " + current + " of " + total);
```

### Log Levels

Use appropriate log levels for different types of messages:

```java
log.trace("Very detailed information for debugging");
log.debug("Detailed information useful for debugging");
log.info("General information about application progress");
log.warn("Potential issues that don't prevent operation");
log.error("Errors that prevent proper operation");
```

## Troubleshooting Lombok with SLF4J

### Common Issues

1. **"Cannot find symbol: variable log"**
   - **Cause**: Lombok annotation processing isn't working properly
   - **Solution**: Ensure Lombok is properly configured in your build system and IDE

2. **Logger not appearing in the compiled class**
   - **Cause**: Lombok may not be in the classpath during compilation
   - **Solution**: Check that Lombok is included with `provided` scope

3. **Multiple Logging Implementations Found**
   - **Cause**: Multiple SLF4J bindings on the classpath
   - **Solution**: Exclude unwanted bindings from dependencies

### Manual Fallback

If Lombok processing fails, you can always fall back to the standard SLF4J pattern:

```java
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class MyClass {
    private static final Logger log = LoggerFactory.getLogger(MyClass.class);
    // ...
}
```

## Other Lombok Logging Annotations

Lombok supports other logging frameworks too:

- `@Log`: Uses java.util.logging
- `@Log4j`: Uses Log4j 1.x
- `@Log4j2`: Uses Log4j 2.x
- `@CommonsLog`: Uses Apache Commons Logging
- `@Flogger`: Uses Google's Flogger

## Best Practices

1. **Always use parameterized logging** to avoid string concatenation overhead
2. **Use appropriate log levels** to ensure proper filtering in production
3. **Include contextual information** in log messages
4. **Add meaningful log messages** that help with debugging
5. **Consider logging performance** in hot code paths

## Conclusion

Combining Lombok with SLF4J offers a clean, concise approach to logging in Java applications. By reducing boilerplate and enforcing consistent patterns, it helps maintain a high-quality, maintainable codebase.

However, be aware of potential issues with annotation processing in certain environments and have fallback strategies ready if needed.
