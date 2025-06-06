# Lombok Removal Guide

This document explains the removal of Project Lombok from the OpenNLP GPU project.

## Why Lombok Was Removed

1. **Simplifies Development**: Eliminating Lombok reduces the complexity in the build and development process.
2. **Improves Compatibility**: Direct Java code is more compatible across different environments and tools.
3. **Reduces Learning Curve**: Contributors don't need to learn Lombok annotations and behavior.
4. **Easier Troubleshooting**: Makes it easier to debug and understand the codebase without the "magic" of Lombok's code generation.
5. **Better IDE Support**: Ensures consistent behavior across all IDEs without requiring special plugins.

## What Was Replaced

| Lombok Annotation          | Replacement                                                              |
| -------------------------- | ------------------------------------------------------------------------ |
| `@Getter`/`@Setter`        | Explicit getter and setter methods                                       |
| `@Slf4j`                   | Standard logger declaration                                              |
| `@Data`                    | Explicit implementations of getters, setters, toString, equals, hashCode |
| `@RequiredArgsConstructor` | Explicit constructor with final fields                                   |
| `@NoArgsConstructor`       | Explicit default constructor                                             |
| `@AllArgsConstructor`      | Explicit constructor with all fields                                     |
| `@Builder`                 | Manual builder pattern implementation if needed                          |

## How the Removal Was Done

1. Created a script (`scripts/remove_lombok.sh`) to automatically replace most Lombok annotations
2. Removed Lombok dependency from `pom.xml`
3. Manually adjusted complex cases where the script couldn't fully replace Lombok functionality
4. Updated IDE settings to remove Lombok-specific configurations
5. Removed Lombok configuration files (.factorypath, lombok.config)

## Coding Patterns Going Forward

When writing new code:

### For Logging

Instead of:
```java
@Slf4j
public class MyClass {
    // ...
    log.info("Hello");
}
```

Use:
```java
public class MyClass {
    private static final Logger log = LoggerFactory.getLogger(MyClass.class);
    // ...
    log.info("Hello");
}
```

### For Properties

Instead of:
```java
@Getter @Setter
private String myField;
```

Use:
```java
private String myField;

public String getMyField() {
    return myField;
}

public void setMyField(String myField) {
    this.myField = myField;
}
```

### For Data Classes

Instead of:
```java
@Data
public class MyData {
    private String field1;
    private int field2;
}
```

Use:
```java
public class MyData {
    private String field1;
    private int field2;
    
    // Getters and setters
    
    @Override
    public boolean equals(Object o) {
        // Proper implementation
    }
    
    @Override
    public int hashCode() {
        // Proper implementation
    }
    
    @Override
    public String toString() {
        // Proper implementation
    }
}
```

## Using IDE Code Generation

Most modern IDEs can generate boilerplate code for you:

- **IntelliJ IDEA**: Alt+Insert (Windows) or Cmd+N (Mac)
- **Eclipse**: Source > Generate...
- **VS Code**: With Java extensions, right-click > Source Action...

These tools can generate getters, setters, constructors, equals/hashCode, and toString methods.

## Benefits of Explicit Code

1. **Readability**: All code is visible, no "magic" happening
2. **Control**: Precise control over method implementations
3. **Debuggability**: Easier to step through and understand
4. **Customization**: Can easily customize generated methods
5. **Stability**: No dependency on annotation processors
