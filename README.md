# OpenNLP GPU Acceleration

This project adds GPU acceleration capabilities to [Apache OpenNLP](https://github.com/apache/opennlp) using JOCL (Java bindings for OpenCL). The goal is to significantly improve the performance of computationally intensive NLP tasks such as model training and inference.

## Features

- GPU-accelerated matrix operations for machine learning algorithms
- Support for MaxEnt model acceleration
- Automatic fallback to CPU when GPU is not available
- Compatible with existing OpenNLP APIs

## Prerequisites

- Java 11 or higher (Java 17 recommended)
- Gradle 8.4 or higher 
- CUDA Toolkit 11.0 or higher (for NVIDIA GPUs)
- OpenCL 1.2 or higher compatible GPU

## Getting Started

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu

# Build the project
./gradlew build
```

### Troubleshooting Build Issues

If you encounter Java version compatibility issues:

1. Make sure you have Java 11 or Java 17 installed
2. Set the `JAVA_HOME` environment variable to point to this installation
3. Run `./gradlew --version` to verify Gradle is using the correct Java version
4. If needed, update `gradle.properties` to set the correct JVM arguments

## Troubleshooting

### Gradle Wrapper Issues

If you encounter issues with the Gradle wrapper (`./gradlew`):

```bash
# Manually download the wrapper JAR if needed
mkdir -p gradle/wrapper
curl -L -o gradle/wrapper/gradle-wrapper.jar https://raw.githubusercontent.com/gradle/gradle/v8.6.0/gradle/wrapper/gradle-wrapper.jar
chmod +x gradlew
```

### Java Version Issues

If you see "Unsupported Java Runtime":

1. Install Java 11 or 17 (instead of Java 21)
2. Set `JAVA_HOME` to point to this installation 
3. Run: `./gradlew --version` to verify Gradle is using the correct Java

Alternatively, run the script to update Gradle to support Java 21:

```bash
./scripts/upgrade_gradle.sh
```

## Usage Examples

```java
// Initialize GPU device
List<GpuDevice> devices = GpuDevice.getAvailableDevices();
if (devices.isEmpty()) {
    System.out.println("No GPU devices available, falling back to CPU");
    // Use regular OpenNLP implementation
} else {
    GpuDevice device = devices.get(0);
    System.out.println("Using GPU: " + device.getName());
    
    // Create GPU-accelerated MaxEnt model
    GpuMaxentModel model = new GpuMaxentModel(device, weights, numOutcomes, numFeatures);
    
    // Evaluate context
    float[] probabilities = model.eval(context);
    
    // Don't forget to release resources when done
    model.release();
}
```

## Project Structure

- `src/main/java/org/apache/opennlp/gpu/common`: Common utilities
- `src/main/java/org/apache/opennlp/gpu/kernels`: GPU kernel implementations
- `src/main/java/org/apache/opennlp/gpu/ml`: Machine learning components
- `src/main/resources/opencl`: OpenCL kernel source files
- `docs`: Documentation

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the Apache License 2.0 - see the LICENSE file for details.
