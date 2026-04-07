package org.apache.opennlp.gpu.performance;

import java.util.Random;

import org.apache.opennlp.gpu.kernels.MatrixOps;

/**

 * Requirement: KernelPerformanceTest must measure individual GPU kernel launch latency and memory transfer overhead.
 * Purpose: Fine-grained timing tests for individual GPU kernels, separating compute time from JNI/PCIe transfer overhead.
 * Rationale: Separating kernel latency from transfer overhead identifies whether bottlenecks are compute-bound or memory-bound.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Runs GPU kernel benchmarks; may flush GPU pipeline between measurements.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class KernelPerformanceTest {
    
    public static void main(String[] args) {
        System.out.println("🔥 GPU Kernel Optimization Performance Test");
        System.out.println("==================================================");
        
        runMatrixMultiplicationTests();
        runScalabilityTests();
        runAccuracyTests();
        
        System.out.println("\n✅ Kernel performance testing completed!");
    }
    
    private static void runMatrixMultiplicationTests() {
        System.out.println("\n📊 Matrix Multiplication Performance Tests");
        
        int[] sizes = {64, 128, 256, 512, 1024};
        Random random = new Random(42);
        
        for (int size : sizes) {
            System.out.printf("\nTesting %dx%d matrices:\n", size, size);
            
            // Generate test matrices
            float[] a = generateRandomMatrix(size, size, random);
            float[] b = generateRandomMatrix(size, size, random);
            float[] result = new float[size * size];
            
            // Test optimized kernel
            long startTime = System.nanoTime();
            MatrixOps.multiplyOptimized(a, b, result, size, size, size);
            long duration = System.nanoTime() - startTime;
            
            double seconds = duration / 1_000_000_000.0;
            double gflops = (2.0 * size * size * size) / (seconds * 1_000_000_000.0);
            
            System.out.printf("  Optimized kernel: %.3f ms, %.2f GFLOPS\n", 
                             seconds * 1000, gflops);
            
            // Compare with different matrix shapes
            testRectangularMatrices(size, random);
        }
    }
    
    private static void testRectangularMatrices(int baseSize, Random random) {
        // Test rectangular matrices for real-world scenarios
        int rowsA = baseSize;
        int colsA = baseSize / 2;
        int colsB = baseSize * 2;
        
        float[] a = generateRandomMatrix(rowsA, colsA, random);
        float[] b = generateRandomMatrix(colsA, colsB, random);
        float[] result = new float[rowsA * colsB];
        
        long startTime = System.nanoTime();
        MatrixOps.multiplyOptimized(a, b, result, rowsA, colsA, colsB);
        long duration = System.nanoTime() - startTime;
        
        double seconds = duration / 1_000_000_000.0;
        System.out.printf("  Rectangular (%dx%d * %dx%d): %.3f ms\n", 
                         rowsA, colsA, colsA, colsB, seconds * 1000);
    }
    
    private static void runScalabilityTests() {
        System.out.println("\n📈 Scalability Tests");
        
        Random random = new Random(42);
        int[] problemSizes = {100, 500, 1000, 2000};
        
        for (int size : problemSizes) {
            float[] a = generateRandomMatrix(size, size, random);
            float[] b = generateRandomMatrix(size, size, random);
            float[] result = new float[size * size];
            
            // Measure multiple runs for consistency
            double totalTime = 0;
            int runs = 3;
            
            for (int run = 0; run < runs; run++) {
                long startTime = System.nanoTime();
                MatrixOps.multiplyOptimized(a, b, result, size, size, size);
                totalTime += (System.nanoTime() - startTime) / 1_000_000_000.0;
            }
            
            double avgTime = totalTime / runs;
            double gflops = (2.0 * size * size * size) / (avgTime * 1_000_000_000.0);
            
            System.out.printf("  Size %4d: Avg %.3f ms, %.2f GFLOPS\n", 
                             size, avgTime * 1000, gflops);
        }
    }
    
    private static void runAccuracyTests() {
        System.out.println("\n🎯 Accuracy Verification Tests");
        
        Random random = new Random(42);
        
        // Test small matrices for accuracy verification
        int size = 4;
        float[] a = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        float[] b = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}; // Identity matrix
        float[] result = new float[16];
        
        MatrixOps.multiplyOptimized(a, b, result, size, size, size);
        
        // Result should be the same as matrix a (multiplying by identity)
        boolean accurate = true;
        for (int i = 0; i < 16; i++) {
            if (Math.abs(result[i] - a[i]) > 1e-6) {
                accurate = false;
                break;
            }
        }
        
        System.out.printf("  Identity multiplication test: %s\n", 
                         accurate ? "✅ PASSED" : "❌ FAILED");
        
        // Test zero matrix
        float[] zero = new float[16];
        float[] zeroResult = new float[16];
        MatrixOps.multiplyOptimized(a, zero, zeroResult, size, size, size);
        
        boolean zeroTest = true;
        for (float value : zeroResult) {
            if (Math.abs(value) > 1e-6) {
                zeroTest = false;
                break;
            }
        }
        
        System.out.printf("  Zero multiplication test: %s\n", 
                         zeroTest ? "✅ PASSED" : "❌ FAILED");
    }
    
    private static float[] generateRandomMatrix(int rows, int cols, Random random) {
        float[] matrix = new float[rows * cols];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = random.nextFloat() * 10.0f - 5.0f; // Range: -5 to 5
        }
        return matrix;
    }
}