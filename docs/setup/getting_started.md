# Getting Started with OpenNLP GPU Acceleration

## Overview

OpenNLP GPU is an extension of Apache OpenNLP that provides GPU acceleration for machine learning operations, significantly improving performance for large-scale natural language processing tasks.

## Prerequisites

### System Requirements

- **Java**: Java 11 or higher (Java 17+ recommended)
- **GPU**: NVIDIA GPU with CUDA support, AMD GPU with ROCm support, or Intel GPU with OpenCL support
- **Memory**: Minimum 4GB RAM, 8GB+ recommended for large models
- **Storage**: 2GB+ free space for models and dependencies

### GPU Drivers and Runtime

#### NVIDIA (CUDA)
```bash
# Install NVIDIA drivers
sudo apt update
sudo apt install nvidia-driver-535

# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Verify installation
nvidia-smi
nvcc --version
```

#### AMD (ROCm)
```bash
# Add ROCm repository
wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
echo 'deb [arch=amd64] https://repo.radeon.com/rocm/apt/debian ubuntu main' | sudo tee /etc/apt/sources.list.d/rocm.list

# Install ROCm
sudo apt update
sudo apt install rocm-dev rocm-libs

# Verify installation
rocm-smi --showproductname
```

#### Intel (OpenCL)
```bash
# Install Intel OpenCL runtime
sudo apt install intel-opencl-icd

# Verify installation
clinfo | grep "Intel"
```

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/apache/opennlp-gpu.git
cd opennlp-gpu
```

### 2. Build the Project

```bash
# Using Maven
mvn clean install

# Or using the provided build script
./build.sh
```

### 3. Verify Installation

```bash
# Run GPU diagnostics
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"

# Run basic tests
mvn test
```

## Quick Start Examples

### 1. GPU-Accelerated Named Entity Recognition

```java
import org.apache.opennlp.gpu.examples.ner.GpuNamedEntityRecognition;

public class QuickNER {
    public static void main(String[] args) {
        GpuNamedEntityRecognition ner = new GpuNamedEntityRecognition();
        
        String text = "John Smith works at Microsoft in Seattle.";
        String[] entities = ner.findEntities(text);
        
        for (String entity : entities) {
            System.out.println("Found entity: " + entity);
        }
    }
}
```

### 2. GPU-Accelerated Sentiment Analysis

```java
import org.apache.opennlp.gpu.examples.sentiment_analysis.GpuSentimentAnalysis;

public class QuickSentiment {
    public static void main(String[] args) {
        GpuSentimentAnalysis sentiment = new GpuSentimentAnalysis();
        
        String text = "This product is absolutely amazing!";
        double score = sentiment.analyzeSentiment(text);
        
        System.out.println("Sentiment score: " + score);
    }
}
```

### 3. GPU-Accelerated Document Classification

```java
import org.apache.opennlp.gpu.examples.classification.GpuDocumentClassification;

public class QuickClassification {
    public static void main(String[] args) {
        GpuDocumentClassification classifier = new GpuDocumentClassification();
        
        String document = "This is a technical document about machine learning.";
        String category = classifier.classifyDocument(document);
        
        System.out.println("Document category: " + category);
    }
}
```

## Configuration

### GPU Configuration

Create a `gpu-config.properties` file:

```properties
# Enable GPU acceleration
gpu.enabled=true

# Memory pool size in MB
gpu.memory.pool.size=512

# Batch size for operations
gpu.batch.size=64

# Compute provider (cuda, rocm, opencl, cpu)
gpu.provider=cuda

# Performance logging
gpu.logging.performance=true
```

### Performance Tuning

```java
import org.apache.opennlp.gpu.common.GpuConfig;

GpuConfig config = new GpuConfig();
config.setGpuEnabled(true);
config.setMemoryPoolSizeMB(1024);  // 1GB memory pool
config.setBatchSize(128);          // Larger batch size for better GPU utilization
config.setPerformanceLogging(true);
```

## Performance Benchmarks

### Speedup Comparison

| Operation | CPU Time | GPU Time | Speedup |
|-----------|----------|----------|---------|
| NER (1000 docs) | 45.2s | 8.7s | 5.2x |
| Sentiment (10000 docs) | 123.4s | 18.9s | 6.5x |
| Classification (5000 docs) | 67.8s | 12.3s | 5.5x |

### Memory Usage

| Model Size | CPU Memory | GPU Memory | Efficiency |
|------------|------------|------------|------------|
| Small (10MB) | 45MB | 32MB | 1.4x |
| Medium (100MB) | 180MB | 95MB | 1.9x |
| Large (1GB) | 1.8GB | 1.1GB | 1.6x |

## Troubleshooting

### Common Issues

1. **GPU not detected**
   ```bash
   # Run diagnostics
   mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
   ```

2. **Out of memory errors**
   - Reduce batch size in configuration
   - Increase GPU memory pool size
   - Use CPU fallback for large models

3. **Performance issues**
   - Ensure GPU drivers are up to date
   - Check for thermal throttling
   - Monitor GPU utilization with `nvidia-smi` or `rocm-smi`

### Debug Mode

Enable debug logging:

```properties
# In logback.xml or application.properties
logging.level.org.apache.opennlp.gpu=DEBUG
```

## Next Steps

- Explore the [API Reference](api/quick_reference.md)
- Check [Performance Benchmarks](performance/performance_benchmarks.md)
- Review [GPU Prerequisites](setup/gpu_prerequisites_guide.md)
- Join the [Community](https://opennlp.apache.org/community.html)

## Support

- **Documentation**: [https://opennlp.apache.org/docs/](https://opennlp.apache.org/docs/)
- **Issues**: [GitHub Issues](https://github.com/apache/opennlp-gpu/issues)
- **Mailing List**: [dev@opennlp.apache.org](mailto:dev@opennlp.apache.org)
- **Discord**: [OpenNLP Community](https://discord.gg/opennlp)
