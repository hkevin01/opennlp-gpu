# Common Gradle Commands

This document lists common Gradle commands used in the OpenNLP GPU project.

## Basic Commands

| Command | Description |
|---------|-------------|
| `./gradlew build` | Compile, test, and package the project |
| `./gradlew clean` | Remove build directories |
| `./gradlew test` | Run all tests |
| `./gradlew assemble` | Create output artifacts (without running tests) |
| `./gradlew check` | Run verification tasks (tests and other checks) |
| `./gradlew run` | Run the main application |

## Build Modifiers

| Command | Description |
|---------|-------------|
| `./gradlew build -x test` | Build without running tests |
| `./gradlew build --parallel` | Build using parallel execution |
| `./gradlew build --info` | Build with detailed information output |
| `./gradlew build --debug` | Build with debug information output |

## Project Inspection

| Command | Description |
|---------|-------------|
| `./gradlew projects` | Display information about projects |
| `./gradlew tasks` | List available tasks |
| `./gradlew dependencies` | Show project dependencies |
| `./gradlew properties` | Show project properties |

## Specific Tasks

| Command | Description |
|---------|-------------|
| `./gradlew testClasses` | Compile test classes only |
| `./gradlew jacocoTestReport` | Generate test coverage report |
| `./gradlew javadoc` | Generate Javadoc documentation |
| `./gradlew wrapper --gradle-version=8.10` | Upgrade Gradle wrapper version |

## Running Specific Tests

| Command | Description |
|---------|-------------|
| `./gradlew test --tests "org.apache.opennlp.gpu.compute.*"` | Run all tests in a package |
| `./gradlew test --tests "*MatrixOpsTest"` | Run tests matching pattern |
| `./gradlew test --tests "*MatrixOpsTest.testMatrixMultiply"` | Run a specific test method |

## Working with the OpenNLP GPU Project

### CUDA Support

```bash
# Build with CUDA support
./gradlew build -PenableCuda=true

# Run benchmarks with CUDA
./gradlew run --args="--benchmark --cuda --iterations=100"
```

### OpenCL Support

```bash
# List available OpenCL devices
./gradlew run --args="--list-devices"

# Benchmark specific OpenCL device
./gradlew run --args="--benchmark --opencl --device=0"
```

### Common Issues

If you encounter "Could not find or load main class org.gradle.wrapper.GradleWrapperMain":

```bash
# Fix Gradle wrapper
./scripts/upgrade_gradle.sh
```

Or manually download the wrapper:

```bash
mkdir -p gradle/wrapper
curl -L -o gradle/wrapper/gradle-wrapper.jar https://raw.githubusercontent.com/gradle/gradle/v8.6.0/gradle/wrapper/gradle-wrapper.jar
chmod +x gradlew
```
