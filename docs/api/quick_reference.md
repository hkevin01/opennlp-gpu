# OpenNLP GPU API Quick Reference

## Core Classes

### Matrix Operations
```java
// Create matrix operation provider
ComputeProvider provider = new GpuComputeProvider(config);
MatrixOperation matrixOp = new GpuMatrixOperation(provider, config);

// Basic operations
matrixOp.multiply(a, b, result, m, n, k);     // Matrix multiplication
matrixOp.add(a, b, result, size);             // Element-wise addition
matrixOp.transpose(input, output, rows, cols); // Matrix transpose

// Activation functions
matrixOp.sigmoid(input, output, size);        // Sigmoid activation
matrixOp.relu(input, output, size);           // ReLU activation
matrixOp.softmax(input, output, size);        // Softmax activation

// Statistical operations
matrixOp.mean(input, result, size);           // Calculate mean
matrixOp.normalize(input, output, size);      // L2 normalization
```

### Feature Extraction
```java
// Create feature extractor
GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);

// Extract features
float[][] ngramFeatures = extractor.extractNGramFeatures(documents, ngramSize, maxFeatures);
float[][] tfidfFeatures = extractor.extractTfIdfFeatures(documents, ngramSize, maxFeatures);
float[][] contextFeatures = extractor.extractContextFeatures(documents, targetWords, windowSize);

// Normalize features
extractor.normalizeFeatures(features);
```

### Neural Networks
```java
// Create neural network
int[] layerSizes = {inputSize, hiddenSize, outputSize};
String[] activations = {"relu", "softmax"};
GpuNeuralNetwork network = new GpuNeuralNetwork(layerSizes, activations, config, matrixOp);

// Prediction
float[] output = network.predict(input);
float[][] batchOutput = network.predictBatch(batchInput);

// Training
network.train(trainInputs, trainOutputs, epochs);
```

### OpenNLP Integration
```java
// Create GPU model factory
GpuModelFactory factory = new GpuModelFactory(config);

// Wrap existing MaxEnt model
MaxentModel gpuModel = factory.createGpuMaxentModel(existingModel);
MaxentModel adaptedModel = factory.createGpuModelAdapter(existingModel);

// Use same interface as original
double[] probs = gpuModel.eval(context);
```

## Configuration
```java
GpuConfig config = new GpuConfig();
config.setGpuEnabled(true);
config.setMatrixSizeThreshold(1000);
config.setFallbackToCpu(true);
```

## Testing & Benchmarks
```java
// Run tests
GpuTestSuite testSuite = new GpuTestSuite();
TestResults results = testSuite.runAllTests();

// Run benchmarks
PerformanceBenchmark benchmark = new PerformanceBenchmark();
BenchmarkResults results = benchmark.runFullBenchmark();
```
