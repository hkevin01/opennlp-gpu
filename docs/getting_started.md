# OpenNLP GPU Acceleration - Getting Started Guide

## Quick Start Demo

This guide shows how to use OpenNLP GPU acceleration in your projects with simple "Hello World" examples.

### Prerequisites

- Java 8 or higher (Java 17+ recommended)
- Maven 3.6+
- GPU drivers installed (optional - falls back to CPU)

### 0. Check GPU Prerequisites (Important!)

**Before starting**, verify your system is ready for GPU acceleration:

#### Option A: Quick Check (No Build Required)
```bash
# Instant check without downloading the full project
curl -fsSL https://raw.githubusercontent.com/yourusername/opennlp-gpu/main/scripts/check_gpu_prerequisites.sh | bash
```

#### Option B: Comprehensive Check (Recommended)
```bash
# Full diagnostics with detailed analysis
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu
mvn clean compile

# Run comprehensive GPU diagnostics
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

Both will check:
- ✅ GPU hardware detection (NVIDIA, AMD, Intel, Apple)
- ✅ Driver installation and compatibility
- ✅ Runtime environments (CUDA, ROCm, OpenCL)
- ✅ Java environment setup
- ✅ Performance validation (comprehensive only)

**If GPU is not ready:** Don't worry! All examples will automatically fall back to CPU implementations.

### 1. Build the Project

```bash
git clone https://github.com/yourusername/opennlp-gpu.git
cd opennlp-gpu
mvn clean compile
```

### 2. Run the Demo Application

```bash
# FIRST: Check your GPU setup
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"

# THEN: Run comprehensive demo with tests and benchmarks
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

## Testing with Real OpenNLP Data

### Use Official OpenNLP Test Data

```java
import org.apache.opennlp.gpu.util.TestDataLoader;
import org.apache.opennlp.gpu.integration.OpenNLPTestDataIntegration;

public class RealDataDemo {
    public static void main(String[] args) {
        // Load real OpenNLP test sentences
        List<String> sentences = TestDataLoader.loadSentences();
        System.out.println("Loaded " + sentences.size() + " real test sentences");
        
        // Load POS tagging data
        List<String> posData = TestDataLoader.loadPosTaggingData();
        System.out.println("Loaded " + posData.size() + " POS tagging examples");
        
        // Test GPU acceleration with real data
        String[] documents = sentences.toArray(new String[sentences.size()]);
        
        GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
        float[][] features = extractor.extractNGramFeatures(documents, 2, 1000);
        
        System.out.println("Extracted features from real OpenNLP data!");
        System.out.printf("Feature matrix: %d documents × %d features\n", 
                         features.length, features[0].length);
    }
}
```

### Performance Testing with Varying Data Sizes

```java
// Create test sets of different sizes
List<List<String>> testSets = TestDataLoader.createPerformanceTestSets();

for (List<String> testSet : testSets) {
    System.out.println("Testing with " + testSet.size() + " documents...");
    
    // Convert to array
    String[] documents = testSet.toArray(new String[testSet.size()]);
    
    // Benchmark GPU vs CPU performance
    long startTime = System.currentTimeMillis();
    float[][] gpuFeatures = gpuExtractor.extractNGramFeatures(documents, 2, 1000);
    long gpuTime = System.currentTimeMillis() - startTime;
    
    startTime = System.currentTimeMillis();
    float[][] cpuFeatures = cpuExtractor.extractNGramFeatures(documents, 2, 1000);
    long cpuTime = System.currentTimeMillis() - startTime;
    
    double speedup = (double) cpuTime / gpuTime;
    System.out.printf("Dataset size %d: GPU=%dms, CPU=%dms, Speedup=%.2fx\n", 
                     documents.length, gpuTime, cpuTime, speedup);
}
```

### Complete Integration Test

```bash
# Run comprehensive tests with real OpenNLP data
mvn test -Dtest=OpenNLPTestDataIntegration

# Run just the real data demo
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.integration.OpenNLPTestDataIntegration"
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

## Available OpenNLP Test Datasets

The integration automatically downloads and uses:

1. **Sentence Detection**: Real sentences from OpenNLP test suite
2. **Tokenization**: Complex tokenization examples with punctuation, URLs, emails
3. **POS Tagging**: Part-of-speech tagged sentences for testing
4. **Named Entity Recognition**: Text with person, location, organization entities
5. **Large Datasets**: Automatically generated datasets of 10-10,000 documents

### Dataset Examples

```java
// Load specific datasets
List<String> sentences = TestDataLoader.loadDataset("sentences");
List<String> nerData = TestDataLoader.loadDataset("ner");
List<String> largeDataset = TestDataLoader.loadLargeDataset(1000);

// Use with your GPU acceleration
testGpuAcceleration(sentences.toArray(new String[sentences.size()]));
```

## Troubleshooting

### Step 1: Run GPU Diagnostics

If you encounter any issues, start with our comprehensive diagnostics:

```bash
# Get detailed GPU environment report
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

This will detect and report:
- GPU hardware (NVIDIA, AMD, Intel, Apple Silicon)
- Driver versions and compatibility
- Runtime environments (CUDA, ROCm, OpenCL)
- Java environment issues
- Performance baseline tests

### Common Issues & Solutions

**🔧 Issue**: "No GPU detected"
```bash
# Check GPU hardware
lspci | grep -i gpu           # Linux
system_profiler SPDisplaysDataType  # macOS

# Install appropriate drivers
sudo apt install nvidia-driver-535  # NVIDIA
sudo apt install rocm-dkms         # AMD
sudo apt install intel-opencl-icd  # Intel
```

**🔧 Issue**: "CUDA/OpenCL not found" 
```bash
# Install CUDA toolkit (NVIDIA)
sudo apt install nvidia-cuda-toolkit

# Install ROCm (AMD)
curl -fsSL https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -
sudo apt install rocm-dev

# Install OpenCL
sudo apt install ocl-icd-opencl-dev
```

**🔧 Issue**: "Permission denied accessing GPU"
```bash
# Add user to GPU groups
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout and login to apply changes
```

**🔧 Issue**: "Out of memory errors"
```java
// Reduce memory usage
GpuConfig config = new GpuConfig();
config.setMemoryPoolSizeMB(256);    // Reduce from default 512MB
config.setBatchSize(32);            // Reduce from default 64
config.setMatrixSize(512);          // Reduce from default 1024
```

**🔧 Issue**: "Slow performance"
```java
// Check GPU utilization
GpuMonitor monitor = new GpuMonitor(config);
monitor.startMonitoring();

// Your operations here...

PerformanceReport report = monitor.getReport();
if (report.getGpuUtilization() < 50) {
    // Increase batch size for better GPU utilization
    config.setBatchSize(128);
}
```

### Debug Mode

```java
// Enable comprehensive debug logging
GpuConfig config = new GpuConfig();
config.setDebugMode(true);
config.setLogLevel("DEBUG");
config.setPerformanceMonitoring(true);

// This will print detailed GPU operation logs
ComputeProvider provider = new GpuComputeProvider(config);
```

### Getting Help

1. **Run diagnostics** first: `GpuDiagnostics` tool
2. **Check logs** with debug mode enabled
3. **Review GPU utilization** with performance monitoring
4. **Verify drivers** are up to date
5. **Test CPU fallback** to isolate GPU issues

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
