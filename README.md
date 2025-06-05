# OpenNLP GPU Acceleration

This project adds GPU acceleration capabilities to [Apache OpenNLP](https://github.com/apache/opennlp) using JOCL (Java bindings for OpenCL). The goal is to significantly improve the performance of computationally intensive NLP tasks such as model training and inference.

## Features

- GPU-accelerated matrix operations for machine learning algorithms
- Support for MaxEnt model acceleration
- Automatic fallback to CPU when GPU is not available
- Compatible with existing OpenNLP APIs

## Prerequisites

- Java 11 or higher
- Apache Maven 3.6 or higher
- CUDA Toolkit 11.0 or higher (for NVIDIA GPUs)
- OpenCL 1.2 or higher compatible GPU

## Getting Started

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu

# Build the project
mvn clean install
```

### Including in Your Project

```xml
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-gpu</artifactId>
    <version>0.1.0-SNAPSHOT</version>
</dependency>
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
