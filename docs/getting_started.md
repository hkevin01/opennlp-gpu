# OpenNLP GPU Acceleration - Getting Started Guide

## Quick Start Demo

This guide shows how to use OpenNLP GPU acceleration in your projects with simple "Hello World" examples.

### Prerequisites

- Java 8 or higher
- Maven 3.6+
- GPU drivers installed (optional - falls back to CPU)

### 1. Build the Project

```bash
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu
mvn clean compile
```

### 2. Run the Demo Application

```bash
# Run comprehensive demo with tests and benchmarks
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication"

# Or run individual components
mvn test -Dtest=GpuQuickStartDemo
```

## Hello World Examples

### Example 1: Basic GPU Matrix Operations

```java
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.*;

public class MatrixHelloWorld {
    public static void main(String[] args) {
        // Configure GPU acceleration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // Create matrix operation provider
        ComputeProvider provider = new GpuComputeProvider(config);
        MatrixOperation matrixOp = new GpuMatrixOperation(provider, config);
        
        // Create sample matrices
        float[] matrixA = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2 matrix
        float[] matrixB = {5.0f, 6.0f, 7.0f, 8.0f};  // 2x2 matrix
        float[] result = new float[4];
        
        // Perform GPU-accelerated matrix multiplication
        matrixOp.multiply(matrixA, matrixB, result, 2, 2, 2);
        
        System.out.println("GPU Matrix Multiplication Result:");
        System.out.printf("[%.1f, %.1f]\n[%.1f, %.1f]\n", 
                         result[0], result[1], result[2], result[3]);
        
        // Cleanup
        matrixOp.release();
        provider.cleanup();
    }
}
```

### Example 2: GPU-Accelerated Feature Extraction

```java
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.*;

public class FeatureExtractionHelloWorld {
    public static void main(String[] args) {
        // Setup GPU configuration
        GpuConfig config = new GpuConfig();
        ComputeProvider provider = new CpuComputeProvider(); // Use CPU for demo
        MatrixOperation matrixOp = new CpuMatrixOperation(provider);
        
        // Create feature extractor
        GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
        
        // Sample documents
        String[] documents = {
            "OpenNLP provides natural language processing",
            "GPU acceleration makes machine learning faster",
            "Feature extraction is important for NLP tasks"
        };
        
        // Extract n-gram features
        System.out.println("Extracting n-gram features...");
        float[][] ngramFeatures = extractor.extractNGramFeatures(documents, 2, 100);
        
        System.out.printf("Extracted features for %d documents\n", ngramFeatures.length);
        System.out.printf("Feature vector size: %d\n", ngramFeatures[0].length);
        System.out.printf("Vocabulary size: %d\n", extractor.getVocabularySize());
        
        // Extract TF-IDF features
        System.out.println("\nExtracting TF-IDF features...");
        float[][] tfidfFeatures = extractor.extractTfIdfFeatures(documents, 2, 100);
        
        System.out.println("TF-IDF extraction completed!");
        
        // Cleanup
        extractor.release();
        matrixOp.release();
    }
}
```

### Example 3: GPU-Accelerated Neural Network

```java
import org.apache.opennlp.gpu.ml.neural.GpuNeuralNetwork;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.*;

public class NeuralNetworkHelloWorld {
    public static void main(String[] args) {
        // Setup
        GpuConfig config = new GpuConfig();
        ComputeProvider provider = new CpuComputeProvider();
        MatrixOperation matrixOp = new CpuMatrixOperation(provider);
        
        // Create neural network: 4 inputs -> 6 hidden -> 3 outputs
        int[] layerSizes = {4, 6, 3};
        String[] activations = {"relu", "softmax"};
        
        GpuNeuralNetwork network = new GpuNeuralNetwork(layerSizes, activations, config, matrixOp);
        
        // Sample input (4 features)
        float[] input = {0.5f, -0.2f, 0.8f, 0.1f};
        
        // Predict
        System.out.println("Neural Network Prediction:");
        float[] output = network.predict(input);
        
        for (int i = 0; i < output.length; i++) {
            System.out.printf("Output[%d]: %.4f\n", i, output[i]);
        }
        
        // Batch prediction
        float[][] batchInput = {{0.1f, 0.2f, 0.3f, 0.4f}, 
                               {0.5f, 0.6f, 0.7f, 0.8f}};
        
        System.out.println("\nBatch Prediction:");
        float[][] batchOutput = network.predictBatch(batchInput);
        
        for (int i = 0; i < batchOutput.length; i++) {
            System.out.printf("Sample %d: [%.3f, %.3f, %.3f]\n", 
                             i, batchOutput[i][0], batchOutput[i][1], batchOutput[i][2]);
        }
        
        // Cleanup
        network.cleanup();
        matrixOp.release();
    }
}
```

## Integration with Existing OpenNLP Code

### Replacing Standard MaxEnt with GPU-Accelerated Version

```java
import org.apache.opennlp.maxent.MaxentModel;
import org.apache.opennlp.gpu.ml.GpuModelFactory;
import org.apache.opennlp.gpu.common.GpuConfig;

public class OpenNLPGpuIntegration {
    public static void main(String[] args) {
        // Your existing OpenNLP MaxEnt model
        MaxentModel existingModel = loadYourExistingModel();
        
        // Create GPU configuration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // Create GPU model factory
        GpuModelFactory factory = new GpuModelFactory(config);
        
        // Wrap existing model with GPU acceleration
        MaxentModel gpuAcceleratedModel = factory.createGpuMaxentModel(existingModel);
        
        // Use exactly like your original model - same interface!
        String[] context = {"word=hello", "pos=NN", "prev=the"};
        double[] probabilities = gpuAcceleratedModel.eval(context);
        
        System.out.println("GPU-accelerated prediction probabilities:");
        for (int i = 0; i < probabilities.length; i++) {
            String outcome = gpuAcceleratedModel.getOutcome(i);
            System.out.printf("%s: %.4f\n", outcome, probabilities[i]);
        }
        
        // Batch processing for better GPU utilization
        String[][] batchContexts = {
            {"word=hello", "pos=NN"},
            {"word=world", "pos=NN"},
            {"word=gpu", "pos=NN"}
        };
        
        System.out.println("\nBatch processing:");
        for (String[] batchContext : batchContexts) {
            double[] batchProbs = gpuAcceleratedModel.eval(batchContext);
            System.out.printf("Context %s -> %.4f\n", 
                             String.join(",", batchContext), batchProbs[0]);
        }
    }
    
    private static MaxentModel loadYourExistingModel() {
        // Replace with your actual model loading logic
        return new org.apache.opennlp.maxent.GisModel();
    }
}
```

### Drop-in Replacement Pattern

```java
// Before (standard OpenNLP)
MaxentModel model = new GisModel();
double[] results = model.eval(context);

// After (GPU-accelerated) - same interface!
GpuConfig config = new GpuConfig();
GpuModelFactory factory = new GpuModelFactory(config);
MaxentModel model = factory.createGpuMaxentModel(new GisModel());
double[] results = model.eval(context); // Same call, GPU acceleration!
```

## Performance Testing

### Run Benchmarks

```java
import org.apache.opennlp.gpu.benchmark.PerformanceBenchmark;

public class PerformanceDemo {
    public static void main(String[] args) {
        PerformanceBenchmark benchmark = new PerformanceBenchmark();
        
        // Run comprehensive benchmarks
        PerformanceBenchmark.BenchmarkResults results = benchmark.runFullBenchmark();
        
        // Print detailed report
        System.out.println(results.generateReport());
        
        // Check overall speedup
        double speedup = results.getOverallSpeedup();
        System.out.printf("Overall GPU speedup: %.2fx\n", speedup);
    }
}
```

### Quick Performance Test

```java
import org.apache.opennlp.gpu.test.GpuTestSuite;

public class QuickTest {
    public static void main(String[] args) {
        GpuTestSuite testSuite = new GpuTestSuite();
        GpuTestSuite.TestResults results = testSuite.runAllTests();
        
        System.out.println("=== Test Results ===");
        System.out.println(results.getReport());
        
        if (results.allPassed()) {
            System.out.println("✅ All tests passed - GPU acceleration working!");
        } else {
            System.out.println("⚠️ Some tests failed - check configuration");
        }
    }
}
```

## Configuration Options

### GPU Configuration

```java
GpuConfig config = new GpuConfig();

// Enable/disable GPU acceleration
config.setGpuEnabled(true);

// Set performance thresholds
config.setMatrixSizeThreshold(1000);    // Use GPU for matrices > 1000 elements
config.setFeatureCountThreshold(500);   // Use GPU for > 500 features
config.setBatchSizeThreshold(10);       // Use GPU for batches > 10

// Memory management
config.setMaxGpuMemoryMB(2048);         // Limit GPU memory usage
config.setBufferPoolSize(100);          // Buffer pool size

// Fallback behavior
config.setFallbackToCpu(true);          // Always fallback to CPU on GPU errors
config.setLogPerformanceStats(true);    // Log performance statistics
```

### Provider Selection

```java
// Automatic provider selection
ComputeProvider provider = ComputeProviderFactory.createBestProvider();

// Manual provider selection
ComputeProvider cpuProvider = new CpuComputeProvider();
ComputeProvider gpuProvider = new GpuComputeProvider(config);

// Check availability
if (GpuComputeProvider.isGpuAvailable()) {
    System.out.println("GPU acceleration available!");
} else {
    System.out.println("Using CPU fallback");
}
```

## Troubleshooting

### Common Issues

1. **GPU not detected**: Ensure GPU drivers are installed
2. **Out of memory**: Reduce batch sizes or matrix dimensions
3. **Slow performance**: Check if GPU thresholds are set appropriately

### Debug Mode

```java
// Enable debug logging
GpuConfig config = new GpuConfig();
config.setDebugMode(true);
config.setLogLevel("DEBUG");

// This will print detailed GPU operation logs
```

### Performance Tips

1. **Use larger batch sizes** for better GPU utilization
2. **Pre-allocate matrices** when possible
3. **Profile your workload** to find optimal thresholds
4. **Enable GPU memory pooling** for repeated operations

## Next Steps

1. **Explore the examples** in `src/test/java/org/apache/opennlp/gpu/examples/`
2. **Run benchmarks** to measure performance on your hardware
3. **Integrate incrementally** - start with one model at a time
4. **Monitor performance** and adjust thresholds as needed

## API Reference

See the complete API documentation in:
- [Matrix Operations API](api/matrix_operations.md)
- [Feature Extraction API](api/feature_extraction.md)
- [Neural Networks API](api/neural_networks.md)
- [Configuration Reference](api/configuration.md)

## Support

- **Issues**: Report bugs on GitHub Issues
- **Documentation**: Check the `docs/` folder
- **Examples**: See `src/test/java/org/apache/opennlp/gpu/examples/`
- **Benchmarks**: Run the demo application for performance metrics
