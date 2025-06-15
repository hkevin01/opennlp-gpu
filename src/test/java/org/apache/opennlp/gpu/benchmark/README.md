# Performance Benchmark Tests

This directory contains comprehensive performance benchmarking tools for measuring GPU acceleration effectiveness across all OpenNLP components.

## 🎯 Benchmark Files Overview

### `PerformanceBenchmark.java`
**Purpose**: Comprehensive performance measurement system comparing GPU vs CPU performance  
**What it benchmarks**: Matrix operations, feature extraction, neural networks, statistical functions

**Run complete benchmark suite:**
```bash
# Run all benchmarks with default settings
mvn test -Dtest=PerformanceBenchmark

# Run with custom iteration counts
mvn test -Dtest=PerformanceBenchmark -Dbenchmark.iterations=20 -Dwarmup.iterations=10

# Run with specific data sizes
mvn test -Dtest=PerformanceBenchmark -Dbenchmark.sizes=100,500,1000,5000
```

**Run from Java:**
```java
// Execute programmatically
PerformanceBenchmark benchmark = new PerformanceBenchmark();
PerformanceBenchmark.BenchmarkResults results = benchmark.runFullBenchmark();
System.out.println(results.generateReport());

// Check overall speedup
double speedup = results.getOverallSpeedup();
System.out.printf("Overall GPU speedup: %.2fx\n", speedup);
```

**Expected Output:**
