# OpenNLP GPU Acceleration User Guide

This guide explains how to integrate and use GPU acceleration with Apache OpenNLP to improve performance for computationally intensive NLP tasks.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Basic Usage](#basic-usage)
- [Advanced Configuration](#advanced-configuration)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Prerequisites

Before using OpenNLP GPU, ensure you have the following:

- Java Development Kit (JDK) 8 or higher
- Apache OpenNLP 2.0.0 or higher
- For GPU acceleration:
  - CUDA-compatible GPU (for CUDA provider)
  - OpenCL-compatible GPU (for OpenCL provider)
  - ROCm-compatible GPU (for AMD ROCm provider)
- Appropriate GPU drivers installed

## Installation

### Maven Integration

Add the OpenNLP GPU dependency to your Maven project:

```xml
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-gpu</artifactId>
    <version>1.0.0</version>
</dependency>

<!-- Original OpenNLP dependencies -->
<dependency>
    <groupId>org.apache.opennlp</groupId>
    <artifactId>opennlp-tools</artifactId>
    <version>2.0.0</version>
</dependency>
```

### Manual Installation

1. Download the OpenNLP GPU JAR file
2. Add it to your classpath along with OpenNLP
3. Include the necessary native libraries for your platform

## Basic Usage

OpenNLP GPU provides accelerated implementations of key NLP operations. Here's how to use them:

### 1. Initialize the Compute Provider

```java
import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;

// Auto-select the best available provider
ComputeProvider provider = ComputeProviderFactory.getDefaultProvider();

// Or specify a particular provider type
ComputeProvider openclProvider = ComputeProviderFactory.getProvider(ComputeProvider.Type.OPENCL);
ComputeProvider cudaProvider = ComputeProviderFactory.getProvider(ComputeProvider.Type.CUDA);
ComputeProvider cpuProvider = ComputeProviderFactory.getProvider(ComputeProvider.Type.CPU);
```

### 2. Use Accelerated Matrix Operations

```java
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;

// Create matrix operation using the provider
MatrixOperation matrixOp = OperationFactory.createMatrixOperation(provider);

// Use matrix operations
float[] matrixA = new float[rows * cols];
float[] matrixB = new float[cols * dims];
float[] result = new float[rows * dims];

// Matrix multiplication
matrixOp.multiply(matrixA, matrixB, result, rows, dims, cols);

// Matrix addition
float[] sum = new float[size];
matrixOp.add(matrixA, matrixB, sum);

// Don't forget to release resources when done
matrixOp.release();
```

### 3. Use Accelerated Feature Extraction

```java
import org.apache.opennlp.gpu.common.FeatureExtractionOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;

// Create feature extraction operation
FeatureExtractionOperation featureOp = OperationFactory.createFeatureExtraction(provider);

// Extract features
String[] tokens = {"This", "is", "a", "sample", "text"};
float[] features = featureOp.extractFeatures(tokens);

// Calculate TF-IDF
String[] documents = {"Document one text", "Document two text", "Third document"};
float[] tfidf = featureOp.computeTfIdf(documents);

// Release resources
featureOp.release();
```

### 4. Replace Standard OpenNLP Operations

To integrate with existing OpenNLP code, use the adapter classes:

```java
import opennlp.tools.ml.maxent.GISTrainer;
import org.apache.opennlp.gpu.ml.maxent.GpuGISTrainer;

// Instead of standard GIS trainer
// GISTrainer trainer = new GISTrainer();

// Use GPU-accelerated GIS trainer
GpuGISTrainer trainer = new GpuGISTrainer();
trainer.setComputeProvider(provider);

// Then use as normal
trainer.trainModel(...);
```

## Advanced Configuration

### Custom Provider Configuration

You can customize the behavior of compute providers:

```java
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
import org.apache.opennlp.gpu.common.ComputeProviderConfig;

ComputeProviderConfig config = new ComputeProviderConfig();
config.setProperty("memoryUsagePercentage", "75");
config.setProperty("preferredDevice", "0");
config.setProperty("kernelCacheSize", "100");

ComputeProvider provider = ComputeProviderFactory.getProvider(ComputeProvider.Type.OPENCL, config);
```

### Performance Thresholds

Set thresholds for when to use GPU vs CPU:

```java
ComputeProviderConfig config = new ComputeProviderConfig();
config.setProperty("matrixSizeThreshold", "1000");  // Matrices smaller than 1000 elements use CPU
config.setProperty("batchSizeThreshold", "32");     // Batches smaller than 32 use CPU

ComputeProvider provider = ComputeProviderFactory.getProvider(config);
```

## Performance Tuning

### Matrix Operations

- **Batch Processing**: Process multiple inputs together when possible
- **Memory Management**: Reuse output arrays to minimize allocations
- **Provider Selection**: Use benchmarking to select the best provider for your workload

```java
// Benchmark different providers
for (ComputeProvider.Type type : ComputeProvider.Type.values()) {
    ComputeProvider testProvider = ComputeProviderFactory.getProvider(type);
    if (testProvider.isAvailable()) {
        double score = testProvider.getPerformanceScore("matrixMultiply", dataSize);
        System.out.println(type + " score: " + score);
    }
}
```

### Feature Extraction

- **Caching**: Enable feature caching for repeated operations
- **Batch Size**: Adjust batch sizes based on your GPU memory

## Troubleshooting

### Common Issues

1. **Provider Not Available**
   - Check that GPU drivers are installed and working
   - Verify GPU compatibility with provider type
   - Try a different provider type

2. **Out of Memory Errors**
   - Reduce batch sizes
   - Free unused resources with `release()` calls
   - Adjust memory usage percentage in provider config

3. **Performance Issues**
   - Ensure data size is large enough to benefit from GPU
   - Check for excessive CPU-GPU data transfers
   - Use profiling to identify bottlenecks

### Diagnostic Tools

```java
// Check available providers
for (ComputeProvider.Type type : ComputeProvider.Type.values()) {
    ComputeProvider provider = ComputeProviderFactory.getProvider(type);
    System.out.println(type + " available: " + provider.isAvailable());
    if (provider.isAvailable()) {
        System.out.println("  Name: " + provider.getName());
        System.out.println("  Capabilities: " + provider.getComputeCapability());
    }
}
```

## Examples

### Accelerated MaxEnt Model Training

```java
import opennlp.tools.ml.maxent.GISModel;
import org.apache.opennlp.gpu.ml.maxent.GpuGISTrainer;

// Prepare your training data
DataIndexer indexer = new OnePassDataIndexer(events, cutoff);

// Create GPU-accelerated trainer
GpuGISTrainer trainer = new GpuGISTrainer();

// Train model with GPU acceleration
GISModel model = trainer.trainModel(indexer);
```

### Accelerated Document Classification

```java
import opennlp.tools.doccat.DocumentCategorizerME;
import opennlp.tools.doccat.DoccatModel;
import org.apache.opennlp.gpu.doccat.GpuDocumentCategorizerME;

// Load your existing model
DoccatModel model = new DoccatModel(new FileInputStream("en-doccat.bin"));

// Create GPU-accelerated categorizer instead of standard one
// DocumentCategorizerME categorizer = new DocumentCategorizerME(model);
GpuDocumentCategorizerME categorizer = new GpuDocumentCategorizerME(model);

// Use as normal with GPU acceleration
double[] probabilities = categorizer.categorize("This is a sample text");
String category = categorizer.getBestCategory(probabilities);
```

### Complete Application Example

See the [example projects](/examples) directory for complete working examples of:
- GPU-accelerated document classification
- GPU-accelerated named entity recognition
- GPU-accelerated sentiment analysis
- Performance comparison benchmarks

## Further Resources

- [API Documentation](./api/index.html)
- [Performance Benchmarks](./benchmarks.md)
- [Contributing to OpenNLP GPU](../development/CONTRIBUTING.md)
