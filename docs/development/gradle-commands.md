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

### CUDA Support (NVIDIA GPUs)

```bash
# Build with CUDA support
./gradlew build -PenableCuda=true

# Run benchmarks with CUDA
./gradlew run --args="--benchmark --cuda --iterations=100"
```

### ROCm Support (AMD GPUs)

```bash
# Build with ROCm support
./gradlew build -PenableRocm=true

# Run benchmarks with ROCm
./gradlew run --args="--benchmark --rocm --iterations=100"

# Specify a particular AMD GPU device
./gradlew run --args="--benchmark --rocm --device=0"

# Run with specific ROCm environment variables
./gradlew run -Dorg.gradle.jvmargs="-DROCM_PATH=/opt/rocm -DHIP_PLATFORM=amd" --args="--benchmark --rocm"
```

### OpenCL Support (Cross-platform)

```bash
# List available OpenCL devices
./gradlew run --args="--list-devices"

# Benchmark specific OpenCL device
./gradlew run --args="--benchmark --opencl --device=0"
```

### GPU Platform Comparison

```bash
# Run benchmark comparison across available platforms
./gradlew run --args="--benchmark --compare-all"

# Compare specific platforms
./gradlew run --args="--benchmark --compare --platforms=cuda,rocm,opencl"
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

## ROCm-Specific Troubleshooting

If you encounter issues with ROCm:

```bash
# Verify ROCm installation
./gradlew run --args="--verify-rocm"

# Update ROCm path (if installed in non-default location)
./gradlew build -PROCM_PATH=/path/to/rocm

# Clear ROCm cache
./gradlew cleanRocmCache
```

### Environment Setup for ROCm

For AMD GPU support, ensure your environment is properly configured:

1. Install the ROCm stack (version 5.0 or higher recommended)
2. Set environment variables:
   ```bash
   export ROCM_PATH=/opt/rocm              # Default ROCm installation path
   export HIP_PLATFORM=amd                 # Use AMD platform
   export HSA_OVERRIDE_GFX_VERSION=10.3.0  # Optional: Override GPU architecture
   ```
3. For Java integration, ensure the JNI libraries can locate ROCm:
   ```bash
   export LD_LIBRARY_PATH=$ROCM_PATH/lib:$LD_LIBRARY_PATH
   ```
