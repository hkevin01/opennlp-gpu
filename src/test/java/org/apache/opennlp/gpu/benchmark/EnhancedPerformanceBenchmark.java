package org.apache.opennlp.gpu.benchmark;

import java.util.Random;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.kernels.MatrixOps;

/**
 * Enhanced performance benchmarking for GPU acceleration
 * Tests real-world NLP workloads and memory usage
 */
public class EnhancedPerformanceBenchmark {
    
    private final GpuConfig gpuConfig;
    private final Random random;
    
    public EnhancedPerformanceBenchmark() {
        this.gpuConfig = new GpuConfig();
        this.random = new Random(42); // Fixed seed for reproducible results
    }
    
    public static void main(String[] args) {
        System.out.println("📊 Enhanced Performance Benchmark Suite");
        System.out.println("==========================================");
        
        EnhancedPerformanceBenchmark benchmark = new EnhancedPerformanceBenchmark();
        
        benchmark.runMatrixOperationBenchmarks();
        benchmark.runMemoryUsageBenchmarks();
        benchmark.runScalabilityBenchmarks();
        benchmark.runRealWorldNlpBenchmarks();
        
        System.out.println("\n🏆 Enhanced performance benchmarking completed!");
    }
    
    private void runMatrixOperationBenchmarks() {
        System.out.println("\n🔢 Matrix Operation Performance Tests");
        System.out.println("------------------------------------");
        
        int[] sizes = {64, 128, 256, 512, 1024};
        
        for (int size : sizes) {
            System.out.printf("\nTesting %dx%d matrices:%n", size, size);
            
            // Generate test data
            float[] a = generateRandomMatrix(size * size);
            float[] b = generateRandomMatrix(size * size);
            float[] result = new float[size * size];
            
            // Warm up JVM
            for (int i = 0; i < 3; i++) {
                MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            }
            
            // Measure performance
            long startTime = System.nanoTime();
            int iterations = Math.max(1, 1000 / (size / 64)); // Adjust iterations based on size
            
            for (int i = 0; i < iterations; i++) {
                MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            }
            
            long duration = System.nanoTime() - startTime;
            double avgTime = (duration / iterations) / 1_000_000.0; // Convert to milliseconds
            double gflops = (2.0 * size * size * size) / ((duration / iterations) / 1_000_000_000.0 * 1_000_000_000.0);
            
            System.out.printf("  Average time: %.3f ms%n", avgTime);
            System.out.printf("  Performance: %.2f GFLOPS%n", gflops);
            System.out.printf("  Iterations: %d%n", iterations);
        }
    }
    
    private void runMemoryUsageBenchmarks() {
        System.out.println("\n💾 Memory Usage Benchmarks");
        System.out.println("---------------------------");
        
        Runtime runtime = Runtime.getRuntime();
        
        // Measure baseline memory
        System.gc();
        long baselineMemory = runtime.totalMemory() - runtime.freeMemory();
        System.out.printf("Baseline memory usage: %.2f MB%n", baselineMemory / 1024.0 / 1024.0);
        
        // Test memory usage with different matrix sizes
        int[] sizes = {256, 512, 1024, 2048};
        
        for (int size : sizes) {
            System.out.printf("%nTesting memory usage for %dx%d matrices:%n", size, size);
            
            long beforeAllocation = runtime.totalMemory() - runtime.freeMemory();
            
            // Allocate matrices
            float[] a = generateRandomMatrix(size * size);
            float[] b = generateRandomMatrix(size * size);
            float[] result = new float[size * size];
            
            long afterAllocation = runtime.totalMemory() - runtime.freeMemory();
            
            // Perform operations
            long beforeOperation = System.nanoTime();
            MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            long operationTime = System.nanoTime() - beforeOperation;
            
            long afterOperation = runtime.totalMemory() - runtime.freeMemory();
            
            double allocationMemory = (afterAllocation - beforeAllocation) / 1024.0 / 1024.0;
            double operationMemory = (afterOperation - afterAllocation) / 1024.0 / 1024.0;
            double expectedMemory = (3.0 * size * size * 4) / 1024.0 / 1024.0; // 3 float arrays * 4 bytes
            
            System.out.printf("  Expected memory: %.2f MB%n", expectedMemory);
            System.out.printf("  Actual allocation: %.2f MB%n", allocationMemory);
            System.out.printf("  Operation overhead: %.2f MB%n", operationMemory);
            System.out.printf("  Operation time: %.3f ms%n", operationTime / 1_000_000.0);
            
            // Clean up
            a = null;
            b = null;
            result = null;
            System.gc();
        }
    }
    
    private void runScalabilityBenchmarks() {
        System.out.println("\n📈 Scalability Benchmarks");
        System.out.println("-------------------------");
        
        // Test how performance scales with problem size
        int[] problemSizes = {100, 200, 500, 1000, 1500, 2000};
        double[] times = new double[problemSizes.length];
        
        for (int i = 0; i < problemSizes.length; i++) {
            int size = problemSizes[i];
            
            float[] a = generateRandomMatrix(size * size);
            float[] b = generateRandomMatrix(size * size);
            float[] result = new float[size * size];
            
            // Warm up
            MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            
            // Measure
            long startTime = System.nanoTime();
            MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            long duration = System.nanoTime() - startTime;
            
            times[i] = duration / 1_000_000.0; // Convert to milliseconds
            
            System.out.printf("Size %4d: %.3f ms%n", size, times[i]);
        }
        
        // Analyze scaling behavior
        System.out.println("\nScaling analysis:");
        for (int i = 1; i < times.length; i++) {
            double sizeRatio = (double) problemSizes[i] / problemSizes[i-1];
            double timeRatio = times[i] / times[i-1];
            double theoreticalRatio = Math.pow(sizeRatio, 3); // O(n³) expected for matrix multiplication
            
            System.out.printf("  %4d -> %4d: %.2fx size, %.2fx time (theoretical: %.2fx)%n",
                             problemSizes[i-1], problemSizes[i], sizeRatio, timeRatio, theoreticalRatio);
        }
    }
    
    private void runRealWorldNlpBenchmarks() {
        System.out.println("\n🗣️ Real-World NLP Workload Benchmarks");
        System.out.println("-------------------------------------");
        
        // Simulate real NLP workloads
        simulateTokenizationWorkload();
        simulateFeatureExtractionWorkload();
        simulateClassificationWorkload();
    }
    
    private void simulateTokenizationWorkload() {
        System.out.println("\n📝 Tokenization Workload Simulation:");
        
        String[] documents = generateTestDocuments(1000, 100); // 1000 docs, ~100 words each
        
        long startTime = System.nanoTime();
        int totalTokens = 0;
        
        for (String doc : documents) {
            String[] tokens = doc.split("\\s+");
            totalTokens += tokens.length;
            
            // Simulate feature extraction for each token
            if (tokens.length > 0) {
                float[] features = new float[tokens.length * 10]; // 10 features per token
                for (int i = 0; i < features.length; i++) {
                    features[i] = random.nextFloat();
                }
            }
        }
        
        long duration = System.nanoTime() - startTime;
        double seconds = duration / 1_000_000_000.0;
        
        System.out.printf("  Processed: %d documents, %d tokens%n", documents.length, totalTokens);
        System.out.printf("  Time: %.3f ms%n", seconds * 1000);
        System.out.printf("  Throughput: %.1f docs/sec, %.1f tokens/sec%n", 
                         documents.length / seconds, totalTokens / seconds);
    }
    
    private void simulateFeatureExtractionWorkload() {
        System.out.println("\n🎯 Feature Extraction Workload Simulation:");
        
        String[] sentences = generateTestSentences(500, 20); // 500 sentences, ~20 words each
        int featureDim = 256;
        
        long startTime = System.nanoTime();
        
        for (String sentence : sentences) {
            String[] tokens = sentence.split("\\s+");
            
            // Simulate TF-IDF feature extraction
            float[][] tfidfMatrix = new float[tokens.length][featureDim];
            for (int i = 0; i < tokens.length; i++) {
                for (int j = 0; j < featureDim; j++) {
                    tfidfMatrix[i][j] = random.nextFloat();
                }
            }
            
            // Simulate matrix operations for feature combination
            if (tokens.length > 1) {
                float[] weights = new float[featureDim];
                float[] result = new float[featureDim];
                
                // Simulate weighted combination
                for (int i = 0; i < featureDim; i++) {
                    weights[i] = random.nextFloat();
                    result[i] = 0.0f;
                    for (int j = 0; j < tokens.length; j++) {
                        result[i] += tfidfMatrix[j][i] * weights[i];
                    }
                }
            }
        }
        
        long duration = System.nanoTime() - startTime;
        double seconds = duration / 1_000_000_000.0;
        
        System.out.printf("  Processed: %d sentences%n", sentences.length);
        System.out.printf("  Feature dimension: %d%n", featureDim);
        System.out.printf("  Time: %.3f ms%n", seconds * 1000);
        System.out.printf("  Throughput: %.1f sentences/sec%n", sentences.length / seconds);
    }
    
    private void simulateClassificationWorkload() {
        System.out.println("\n📊 Classification Workload Simulation:");
        
        int numSamples = 1000;
        int featureDim = 512;
        int numClasses = 10;
        
        // Generate feature matrices
        float[] features = generateRandomMatrix(numSamples * featureDim);
        float[] weights = generateRandomMatrix(featureDim * numClasses);
        float[] results = new float[numSamples * numClasses];
        
        long startTime = System.nanoTime();
        
        // Simulate classification matrix multiplication
        MatrixOps.multiplyOptimized(features, weights, results, numSamples, featureDim, numClasses);
        
        // Simulate softmax activation
        for (int i = 0; i < numSamples; i++) {
            float maxVal = Float.NEGATIVE_INFINITY;
            for (int j = 0; j < numClasses; j++) {
                maxVal = Math.max(maxVal, results[i * numClasses + j]);
            }
            
            float sum = 0.0f;
            for (int j = 0; j < numClasses; j++) {
                results[i * numClasses + j] = (float) Math.exp(results[i * numClasses + j] - maxVal);
                sum += results[i * numClasses + j];
            }
            
            for (int j = 0; j < numClasses; j++) {
                results[i * numClasses + j] /= sum;
            }
        }
        
        long duration = System.nanoTime() - startTime;
        double seconds = duration / 1_000_000_000.0;
        
        System.out.printf("  Classified: %d samples%n", numSamples);
        System.out.printf("  Feature dimension: %d%n", featureDim);
        System.out.printf("  Number of classes: %d%n", numClasses);
        System.out.printf("  Time: %.3f ms%n", seconds * 1000);
        System.out.printf("  Throughput: %.1f samples/sec%n", numSamples / seconds);
    }
    
    private float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < size; i++) {
            matrix[i] = random.nextFloat() * 2.0f - 1.0f; // Range: -1 to 1
        }
        return matrix;
    }
    
    private String[] generateTestDocuments(int count, int avgWordsPerDoc) {
        String[] words = {"the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", 
                         "natural", "language", "processing", "machine", "learning", "artificial", 
                         "intelligence", "computer", "science", "algorithm", "data", "analysis"};
        
        String[] documents = new String[count];
        
        for (int i = 0; i < count; i++) {
            StringBuilder doc = new StringBuilder();
            int numWords = avgWordsPerDoc + random.nextInt(avgWordsPerDoc / 2) - avgWordsPerDoc / 4;
            
            for (int j = 0; j < numWords; j++) {
                if (j > 0) doc.append(" ");
                doc.append(words[random.nextInt(words.length)]);
            }
            
            documents[i] = doc.toString();
        }
        
        return documents;
    }
    
    private String[] generateTestSentences(int count, int avgWordsPerSentence) {
        String[] words = {"this", "is", "a", "test", "sentence", "for", "performance", "benchmarking",
                         "gpu", "acceleration", "improves", "processing", "speed", "significantly",
                         "matrix", "operations", "are", "optimized", "using", "opencl", "kernels"};
        
        String[] sentences = new String[count];
        
        for (int i = 0; i < count; i++) {
            StringBuilder sentence = new StringBuilder();
            int numWords = avgWordsPerSentence + random.nextInt(avgWordsPerSentence / 2) - avgWordsPerSentence / 4;
            
            for (int j = 0; j < numWords; j++) {
                if (j > 0) sentence.append(" ");
                sentence.append(words[random.nextInt(words.length)]);
            }
            
            sentences[i] = sentence.toString();
        }
        
        return sentences;
    }
}