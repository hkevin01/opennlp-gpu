# OpenNLP GPU Acceleration

**Experimental GPU acceleration research project for Apache OpenNLP** to explore performance improvements for natural language processing tasks through GPU computing.

## 🎯 Project Overview

This is a **research and development project** that demonstrates GPU acceleration concepts for NLP tasks. The project provides working examples that showcase potential performance benefits and serves as a foundation for future OpenNLP GPU integration research.

**Current Status**: Working GPU-accelerated examples with custom APIs. Future seamless OpenNLP integration is planned but not yet implemented.

## ⚡ GPU Prerequisites Check

**IMPORTANT:** Before using GPU acceleration, verify your system is ready:

### Quick GPU Readiness Check (No Build Required)

> **Note**: GitHub URLs in this section are placeholders for future Apache OpenNLP integration. 
> For now, use the local scripts in your cloned project directory.

Run our lightweight prerequisites check:

```bash
# Quick check without building the project (once in Apache OpenNLP)
curl -fsSL https://raw.githubusercontent.com/apache/opennlp-gpu/main/scripts/check_gpu_prerequisites.sh | bash

# Or download and run locally (once in Apache OpenNLP)
wget https://raw.githubusercontent.com/apache/opennlp-gpu/main/scripts/check_gpu_prerequisites.sh
chmod +x check_gpu_prerequisites.sh
./check_gpu_prerequisites.sh

# For now, run from your local project directory:
cd /path/to/your/opennlp-gpu
./scripts/check_gpu_prerequisites.sh
```

### Comprehensive GPU Diagnostics

For detailed analysis, build the project and run our comprehensive diagnostics tool:

```bash
# Clone and build the project (once in Apache OpenNLP)
git clone https://github.com/apache/opennlp-gpu.git
cd opennlp-gpu
mvn clean compile

# For now, use your current project directory:
cd /path/to/your/opennlp-gpu
mvn clean compile

# Run comprehensive GPU diagnostics
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

Both tools check for:
- ✅ **GPU Hardware** (NVIDIA, AMD, Intel, Apple Silicon)
- ✅ **GPU Drivers** (NVIDIA, ROCm, Intel drivers)
- ✅ **GPU Runtimes** (CUDA, ROCm, OpenCL)
- ✅ **Java Environment** compatibility
- ✅ **Performance** baseline test (comprehensive tool only)

### Manual Prerequisites

If you prefer to check manually:

**For NVIDIA GPUs:**
```bash
# Check NVIDIA driver
nvidia-smi

# Install CUDA Toolkit (if needed)
sudo apt install nvidia-cuda-toolkit
```

**For AMD GPUs:**
```bash
# Check ROCm installation
rocm-smi

# Install ROCm (if needed)
sudo apt install rocm-dkms
```

**For Intel GPUs:**
```bash
# Check Intel GPU tools
intel_gpu_top

# Install Intel compute runtime
sudo apt install intel-opencl-icd
```

### CPU Fallback

✅ **No GPU? No problem!** Examples include CPU fallback implementations when GPU is unavailable.

📖 **Detailed Setup Guide**: See [`docs/setup/gpu_prerequisites_guide.md`](docs/setup/gpu_prerequisites_guide.md) for comprehensive GPU setup instructions.

### 📦 What This Project Provides

- ✅ **Working GPU-accelerated examples** for major NLP operations
- ✅ **Research foundation** for future OpenNLP GPU integration
- ✅ **Performance benchmarking** tools and demonstrations
- ✅ **GPU diagnostics** and compatibility checking
- ✅ **Cross-platform testing** infrastructure (Linux, macOS, Windows)
- ✅ **CPU fallback** implementations for compatibility
- ✅ **Comprehensive test suite** with automated validation

📖 **Examples Overview**: See [`examples/README.md`](examples/README.md) for complete example documentation and usage instructions.

**Note**: This is experimental research code, not a production-ready library.

## 🚀 Quick Start (Working Examples)

### Current Status: Research Examples Available

This project provides **working GPU-accelerated examples** that demonstrate potential performance benefits. These examples use custom APIs designed for research purposes and are not integrated with standard OpenNLP APIs.

**Important**: This is not a drop-in replacement for OpenNLP. It's a research project exploring GPU acceleration concepts.

### Step 0: Verify GPU Support (Recommended)

Run the GPU diagnostics to ensure your system is ready:

```bash
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"
```

If GPU is not available, the examples will automatically use CPU fallback.

### Step 1: Add Dependencies

Add to your existing `pom.xml`:

```xml
<dependencies>
    <!-- Your existing OpenNLP dependency -->
    <dependency>
        <groupId>org.apache.opennlp</groupId>
        <artifactId>opennlp-tools</artifactId>
        <version>2.3.3</version>
    </dependency>
    
    <!-- Add GPU acceleration (when integrated) -->
    <dependency>
        <groupId>org.apache.opennlp</groupId>
        <artifactId>opennlp-gpu</artifactId>
        <version>1.0-SNAPSHOT</version>
    </dependency>
    
    <!-- GPU runtime (choose one) -->
    <!-- For NVIDIA GPUs -->
    <dependency>
        <groupId>org.jocl</groupId>
        <artifactId>jocl</artifactId>
        <version>2.0.4</version>
    </dependency>
    
    <!-- For AMD GPUs -->
    <dependency>
        <groupId>org.aparapi</groupId>
        <artifactId>aparapi</artifactId>
        <version>3.0.0</version>
    </dependency>
</dependencies>
```

### Step 2: Use GPU Examples (Available Now)

Run the working GPU-accelerated examples:

```java
// Current working examples
// 1. Sentiment Analysis - Twitter sentiment with GPU acceleration
GpuSentimentAnalysis analyzer = new GpuSentimentAnalysis();
SentimentResult[] results = analyzer.analyzeBatch(socialMediaPosts);

// 2. Named Entity Recognition - High-speed entity extraction
GpuNamedEntityRecognition ner = new GpuNamedEntityRecognition();
EntityResult[] entities = ner.extractEntitiesBatch(documents);

// 3. Document Classification - Large-scale document categorization
GpuDocumentClassification classifier = new GpuDocumentClassification();
ClassificationResult[] categories = classifier.classifyBatch(documents);

// 4. Language Detection - Multi-language processing
GpuLanguageDetection detector = new GpuLanguageDetection();
LanguageResult[] languages = detector.detectLanguageBatch(texts);

// 5. Question Answering - Neural QA with attention mechanisms
GpuQuestionAnswering qa = new GpuQuestionAnswering();
QAResult[] answers = qa.answerQuestionsBatch(questionPairs);
```

📖 **Example Documentation:**
- [Sentiment Analysis](examples/sentiment_analysis/README.md) - Twitter sentiment with GPU acceleration
- [Named Entity Recognition](examples/ner/README.md) - High-speed entity extraction  
- [Document Classification](examples/classification/README.md) - Large-scale document categorization
- [Language Detection](examples/language_detection/README.md) - Multi-language processing
- [Question Answering](examples/question_answering/README.md) - Neural QA with attention mechanisms

### Step 3: Configuration (Available)

```java
// Configure GPU settings (works now)
GpuConfig config = new GpuConfig();
config.setGpuEnabled(true);
config.setBatchSize(64);        // Optimize for your workload
config.setMemoryPoolSizeMB(512); // Adjust based on GPU memory

// Examples use this configuration
GpuSentimentAnalysis analyzer = new GpuSentimentAnalysis(config);
```

## 🔧 Research Example Patterns

### Pattern 1: Current GPU Examples (Working)

Available now - explicit GPU-accelerated examples:

```java
// Create and configure GPU example
GpuConfig config = new GpuConfig();
config.setGpuEnabled(true);

GpuSentimentAnalysis analyzer = new GpuSentimentAnalysis(config);
SentimentResult[] results = analyzer.analyzeBatch(texts);

// Process results
for (SentimentResult result : results) {
    System.out.printf("%s: %.2f confidence%n", 
                     result.getSentiment(), result.getConfidence());
}
```

### Pattern 2: Planned OpenNLP Integration (Future)

**⚠️ The following APIs are planned for future development and do not currently exist:**

```java
// ⚠️ FUTURE API - NOT YET IMPLEMENTED
GpuConfig config = new GpuConfig();
GpuMaxentModelFactory factory = new GpuMaxentModelFactory(config); // Doesn't exist

// Convert existing model to GPU version
MaxentModel cpuModel = new MaxentModel(modelFile);
MaxentModel gpuModel = factory.createGpuAcceleratedModel(cpuModel); // Planned API
```

### Pattern 3: Neural Network Integration (Research Concept)

**⚠️ The following represents research concepts and target architecture:**

```java
// ⚠️ RESEARCH CONCEPT - NOT YET IMPLEMENTED
GpuConfig config = new GpuConfig();
GpuNeuralPipeline pipeline = new GpuNeuralPipeline(config); // Basic implementation exists

// Advanced features are planned but not implemented:
// - Multi-head attention layers
// - Feed-forward networks
// - Production optimization
```

### Pattern 4: Production Optimization (Future Vision)

**⚠️ The following represents the future vision for production optimization:**

```java
// ⚠️ FUTURE VISION - NOT YET IMPLEMENTED
// Production optimization APIs are not available
// Current implementation focuses on research examples
```

## 📋 Current vs. Planned Features

### ✅ What's Currently Working

**Available Now - Research Examples:**
```java
// Working GPU examples (available now)
// 1. Sentiment Analysis - Twitter sentiment with GPU acceleration
GpuSentimentAnalysis analyzer = new GpuSentimentAnalysis();
SentimentResult[] results = analyzer.analyzeBatch(texts);

// 2. Named Entity Recognition - High-speed entity extraction
GpuNamedEntityRecognition ner = new GpuNamedEntityRecognition();
EntityResult[] entities = ner.extractEntitiesBatch(documents);

// 3. Document Classification - Large-scale document categorization
GpuDocumentClassification classifier = new GpuDocumentClassification();
ClassificationResult[] categories = classifier.classifyBatch(documents);

// 4. Language Detection - Multi-language processing
GpuLanguageDetection detector = new GpuLanguageDetection();
LanguageResult[] languages = detector.detectLanguageBatch(texts);

// 5. Question Answering - Neural QA with attention mechanisms
GpuQuestionAnswering qa = new GpuQuestionAnswering();
QAResult[] answers = qa.answerQuestionsBatch(questionPairs);
```

📖 **Complete Example Documentation:**
- [Sentiment Analysis](examples/sentiment_analysis/README.md) - Twitter sentiment with GPU acceleration
- [Named Entity Recognition](examples/ner/README.md) - High-speed entity extraction
- [Document Classification](examples/classification/README.md) - Large-scale document categorization
- [Language Detection](examples/language_detection/README.md) - Multi-language processing
- [Question Answering](examples/question_answering/README.md) - Neural QA with attention mechanisms

### 🔮 Planned Future Integration

**Target Architecture for OpenNLP Integration (Not Yet Implemented):**

The following APIs represent the target design for future seamless OpenNLP integration. **These classes do not currently exist:**

```java
// ⚠️ PLANNED FUTURE API - NOT YET IMPLEMENTED
GpuConfigurationManager.initializeGpuSupport(); // Doesn't exist yet

// Target: Existing OpenNLP code would work unchanged
TokenizerModel model = new TokenizerModel(modelInputStream);
TokenizerME tokenizer = new TokenizerME(model); // Would be auto GPU-accelerated
```

## 🔍 GPU Diagnostics & Troubleshooting

### Comprehensive GPU Health Check

Our diagnostics tool provides detailed analysis of your GPU setup:

```bash
# Run comprehensive diagnostics
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"

# Sample output:
# 🔍 OpenNLP GPU Acceleration - Hardware Diagnostics
# ==================================================
# ✅ System Information
#   OS: Linux 5.15.0
#   Java Version: 17.0.7 ✅ Compatible
#   Available Processors: 16
# 
# ✅ GPU Hardware
#   NVIDIA RTX 4090: ✅ Detected
#   GPU Memory: 24GB
# 
# ✅ NVIDIA Drivers
#   Driver Version: 535.104.05 ✅ Compatible
#   CUDA Version: 12.2 ✅ Ready
# 
# 🎉 GPU acceleration is ready to use!
```

### Common Issues & Solutions

**Issue**: "No GPU detected"
```bash
# Check GPU hardware
lspci | grep -i gpu

# Install drivers
sudo apt install nvidia-driver-535  # NVIDIA
sudo apt install rocm-dkms         # AMD
```

**Issue**: "CUDA/OpenCL not found"
```bash
# Install CUDA toolkit
sudo apt install nvidia-cuda-toolkit

# Install OpenCL
sudo apt install ocl-icd-opencl-dev
```

**Issue**: "Permission denied" accessing GPU
```bash
# Add user to GPU group
sudo usermod -a -G video $USER
sudo usermod -a -G render $USER

# Logout/login to apply changes
```

### Performance Troubleshooting

```java
// Enable detailed performance logging
GpuConfig config = new GpuConfig();
config.setLoggingLevel(LogLevel.DEBUG);
config.setPerformanceMonitoring(true);

// Check GPU utilization
GpuMonitor monitor = new GpuMonitor(config);
monitor.startMonitoring();

// Your GPU operations here...

PerformanceReport report = monitor.getReport();
System.out.println("GPU Utilization: " + report.getGpuUtilization() + "%");
System.out.println("Memory Usage: " + report.getMemoryUsage() + "MB");
```

## 🏗️ System Requirements

### Minimum Requirements
- **Java**: 11+ (Java 17+ recommended)
- **Memory**: 4GB RAM (8GB+ recommended)
- **GPU**: Any OpenCL 1.2+ compatible GPU
- **OS**: Linux, Windows, macOS

### Supported GPUs
- ✅ **NVIDIA**: GTX 1060+, RTX series, Tesla, Quadro
- ✅ **AMD**: RX 580+, Vega series, RDNA series
- ✅ **Intel**: Iris Pro, Arc series, Xe series
- ✅ **Apple**: M1/M2 with Metal Performance Shaders

### Performance Expectations

**Note**: These are theoretical performance targets based on GPU acceleration research. Actual performance varies significantly based on hardware, data size, and specific operations.

| Workload Type          | CPU Baseline | Research Target | Status         |
| ---------------------- | ------------ | --------------- | -------------- |
| **Example Processing** | 1x           | 2-4x potential  | Research phase |
| **Feature Extraction** | 1x           | 3-6x potential  | Early testing  |
| **Batch Processing**   | 1x           | 5-10x potential | Concept stage  |

**Important**: Performance benefits depend heavily on:
- GPU hardware capabilities
- Data batch sizes (GPU benefits larger batches)
- Specific operation types
- Memory bandwidth and latency

## 🔧 Installation & Setup

### Step 1: Clone or Download

```bash
# Clone repository (once contributed to Apache OpenNLP)
git clone https://github.com/apache/opennlp-gpu.git
cd opennlp-gpu

# For now, use your current project directory:
cd /path/to/your/opennlp-gpu

# Build the project
mvn clean compile
```

> **Note**: Repository URL will be updated once this project is contributed to Apache OpenNLP.
> For now, build from source using your local project directory.

### Step 2: Build from Source

```bash
# Build the project
mvn clean compile package

# Run tests to verify installation
mvn test

# Install to local repository
mvn install
```

### Step 3: Verify GPU Setup

```bash
# Check GPU availability
java -cp target/opennlp-gpu-1.0-SNAPSHOT.jar \
     org.apache.opennlp.gpu.tools.GpuDiagnostics

# Expected output:
# ✅ GPU detected: AMD Radeon RX 5700 XT
# ✅ OpenCL runtime: Available
# ✅ Memory available: 8192 MB
# ✅ Compute units: 40
# ✅ Ready for acceleration
```

### Step 4: Run Demo

```bash
# Run GPU acceleration demo
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.demo.GpuDemoApplication"

# Or use jar directly
java -jar target/opennlp-gpu-1.0-SNAPSHOT.jar demo
```

## 📊 Benchmarking Your Integration

### Performance Testing

```java
// Benchmark your specific workload
GpuPerformanceBenchmark benchmark = new GpuPerformanceBenchmark();

// Test your models
BenchmarkResult cpuResult = benchmark.benchmarkCpu(yourModel, testData);
BenchmarkResult gpuResult = benchmark.benchmarkGpu(yourModel, testData);

// Compare results
double speedup = cpuResult.getExecutionTime() / gpuResult.getExecutionTime();
System.out.println("GPU Speedup: " + speedup + "x");
System.out.println("Accuracy maintained: " + 
    Math.abs(cpuResult.getAccuracy() - gpuResult.getAccuracy()) < 0.001);
```

### Memory Usage Analysis

```java
// Monitor memory usage
ResourceMetrics metrics = monitor.getResourceMetrics("gpu-0");
System.out.println("GPU Memory Used: " + metrics.getUsedMemory() + " MB");
System.out.println("GPU Utilization: " + metrics.getMemoryUsageRatio() * 100 + "%");
```

## 🚀 Research Deployment & Testing

**Note**: This is experimental research code. The following deployment examples are conceptual and for research purposes.

### Research Testing Framework

```java
// Research performance analysis (conceptual)
// Note: Actual production deployment APIs are not implemented
GpuConfig config = new GpuConfig();
// Use the working examples for actual testing
```

### Performance Testing Suite

**🔬 Want to Test Performance Yourself?**
The project includes a comprehensive GPU diagnostics tool and performance benchmarking suite:

```bash
# Check your GPU capabilities
mvn exec:java -Dexec.mainClass="org.apache.opennlp.gpu.tools.GpuDiagnostics"

# Run all examples with timing benchmarks
./scripts/run_all_demos.sh
```

You can also scale the test datasets - each example supports batch sizes from 1K to 1M+ documents, so you can test exactly the data volumes you work with.

### AWS Deployment Example

```bash
# Launch p3.2xlarge instance with GPU support
# Install CUDA drivers (automated in our setup scripts)
./scripts/setup_aws_gpu_environment.sh

# Deploy with AWS Batch for large-scale processing
# Process documents from S3, output results back to S3
# Automatically scale based on queue depth
```

**AWS Cost Calculator:**
• **Traditional CPU processing**: 1M documents on c5.4xlarge = ~$24/hour
• **GPU acceleration**: Same workload on p3.2xlarge = ~$8/hour (3x faster + lower cost)
• **Spot pricing**: Further 50-70% reduction = ~$2.40-4/hour

## 🌐 Development & Testing Options

**Note**: This is experimental research code. Deployment examples are for development and testing purposes.

### Local Development
```bash
# Works on any system with Java 11+
mvn clean install
# Run the working examples
./scripts/run_all_demos.sh
```

### Docker Testing
```bash
# Test in containerized environments
docker-compose up test-ubuntu
docker-compose up test-windows
```

### Cross-Platform Testing
```bash
# Test compatibility across platforms
./scripts/test_cross_platform_compatibility.sh
```

**Development Focus**: This project demonstrates GPU acceleration concepts through working examples and provides a foundation for future research and integration efforts.
