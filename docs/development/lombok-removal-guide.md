# Lombok Removal Guide

This document explains the process and rationale for removing Project Lombok from the OpenNLP GPU project.

## Why We Removed Lombok

As documented in [Lombok VS Code Issues](lombok-vscode-issues.md), we experienced several challenges with Lombok:

1. **Persistent syntax errors** in VS Code that didn't reflect actual compilation errors
2. **Unreliable code navigation** and refactoring features
3. **Developer productivity issues** from fighting with IDE integration
4. **Onboarding complexity** for new team members
5. **Build tool inconsistencies** across different environments

## What Was Replaced

| Lombok Feature             | Standard Java Replacement                                                |
| -------------------------- | ------------------------------------------------------------------------ |
| `@Getter`/`@Setter`        | Explicit getter and setter methods                                       |
| `@Slf4j`                   | Standard SLF4J logger declaration                                        |
| `@Data`                    | Explicit implementations of getters, setters, equals, hashCode, toString |
| `@RequiredArgsConstructor` | Explicit constructor with final fields                                   |
| `@Builder`                 | Standard builder pattern implementation                                  |
| Other annotations          | Explicit implementations                                                 |

## The Removal Process

We used an automated script (`scripts/remove_lombok.sh`) that:

1. Removed all Lombok imports
2. Replaced `@Slf4j` with standard logger declarations
3. Generated getters and setters for fields with `@Getter`/`@Setter`
4. Generated constructors for classes with `@RequiredArgsConstructor`
5. Removed other Lombok annotations with notes for manual follow-up
6. Removed Lombok from the project dependencies

## Manual Follow-up Tasks

Some Lombok features couldn't be fully automated and require manual implementation:

1. **Complex builders**: Classes using `@Builder` need manual builder implementation
2. **equals/hashCode/toString**: Classes using `@Data` need these methods implemented
3. **Custom annotation configurations**: Any special Lombok settings need manual replication

## Using IDE Features

Modern IDEs provide code generation features that can help create the code Lombok previously generated:

- **VS Code**: Right-click > Source Action... > Generate...
- **IntelliJ IDEA**: Alt+Insert (Windows/Linux) or Cmd+N (Mac)
- **Eclipse**: Source > Generate...

## Benefits of Explicit Code

Now that we've removed Lombok:

1. **No false errors** in VS Code or other IDEs
2. **Reliable code navigation** and refactoring
3. **Easier debugging** with clear method implementations
4. **Simpler onboarding** for new developers
5. **Consistent build process** with fewer dependencies

## Lombok-free Coding Standards

Going forward:

1. Use standard Java code patterns instead of annotation magic
2. Use IDE code generation features when appropriate
3. Follow consistent naming conventions for getters/setters
4. Use standard SLF4J logger declaration patterns
5. Consider using Java 16+ records for simple data classes

## Conclusion

While Lombok provided brevity in our code, the maintenance and tooling challenges outweighed the benefits. Moving to standard Java code has improved our development experience with better IDE support, clearer code paths, and a more approachable codebase for all developers.
