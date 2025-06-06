package org.apache.opennlp.gpu;

import java.util.Arrays;
import java.util.List;
import java.util.Random;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
import org.apache.opennlp.gpu.common.GpuDevice;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Main demo class for OpenNLP GPU operations.
 * This class demonstrates the usage of GPU acceleration for various NLP operations.
 */
public class GpuDemoMain {
    private static final Logger logger = LoggerFactory.getLogger(GpuDemoMain.class);
    
    // Constants for demo
    private static final int MATRIX_SIZE = 1000;
    private static final int ITERATIONS = 5;
    
    public static void main(String[] args) {
        logger.info("Starting OpenNLP GPU Demo");
        
        try {
            // Parse command line arguments
            boolean listDevices = Arrays.asList(args).contains("--list-devices");
            boolean benchmark = Arrays.asList(args).contains("--benchmark");
            boolean useRocm = Arrays.asList(args).contains("--rocm");
            boolean useCuda = Arrays.asList(args).contains("--cuda");
            boolean useOpenCL = Arrays.asList(args).contains("--opencl");
            
            // Get available devices
            List<GpuDevice> availableDevices = GpuDevice.getAvailableDevices();
            
            if (availableDevices.isEmpty()) {
                logger.warn("No GPU devices found. Using CPU fallback.");
            } else {
                logger.info("Found {} GPU devices:", availableDevices.size());
                for (int i = 0; i < availableDevices.size(); i++) {
                    GpuDevice device = availableDevices.get(i);
                    logger.info("  Device {}: {} with {} MB memory", 
                               i, device.getName(), device.getMemoryMB());
                }
            }
            
            if (listDevices) {
                // Just list devices and exit
                return;
            }
            
            // Create the appropriate compute provider
            ComputeProviderFactory providerFactory = ComputeProviderFactory.getInstance(); // Get factory instance
            ComputeProvider provider;
            if (useCuda) {
                provider = providerFactory.getProvider(ComputeProvider.Type.CUDA);
            } else if (useRocm) {
                provider = providerFactory.getProvider(ComputeProvider.Type.ROCM);
            } else if (useOpenCL) {
                provider = providerFactory.getProvider(ComputeProvider.Type.OPENCL);
            } else {
                // Auto-select best provider - provide the required arguments
                // First argument: operation type (e.g., "matrixOperations")
                // Second argument: problem size (using the matrix size from the benchmark)
                provider = providerFactory.getBestProvider("matrixOperations", MATRIX_SIZE * MATRIX_SIZE);
            }
            
            if (provider == null) {
                logger.error("No suitable compute provider found. Exiting.");
                return;
            }
            
            logger.info("Using compute provider: {}", provider.getName());
            
            // Initialize provider
            if (!provider.initialize()) {
                logger.error("Failed to initialize compute provider. Exiting.");
                return;
            }
            
            if (benchmark) {
                runMatrixBenchmark(provider);
            } else {
                runSimpleDemo(provider);
            }
            
            // Cleanup
            provider.release();
            logger.info("Demo completed successfully");
            
        } catch (Exception e) {
            logger.error("Error running demo", e);
        }
    }
    
    /**
     * Run a simple matrix multiplication demo.
     */
    private static void runSimpleDemo(ComputeProvider provider) {
        logger.info("Running simple matrix operation demo");
        
        // Pass a device index (0 for primary device) instead of the provider
        OperationFactory operationFactory = new OperationFactory();
        MatrixOperation matrixOp = operationFactory.createMatrixOperation(0); // Use device index 0
        
        // Create simple matrices for demo
        float[] matrixA = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] matrixB = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] result = new float[4];
        
        logger.info("Matrix A: {}", Arrays.toString(matrixA));
        logger.info("Matrix B: {}", Arrays.toString(matrixB));
        
        // Call the appropriate matrix multiplication method
        // Different interfaces might use different method names
        try {
            // Try executing the matrix multiplication using reflection to be flexible
            java.lang.reflect.Method multiplyMethod = findMatrixMultiplyMethod(matrixOp.getClass());
            if (multiplyMethod != null) {
                multiplyMethod.invoke(matrixOp, matrixA, matrixB, result, 2, 2, 2);
                logger.info("Result (A*B): {}", Arrays.toString(result));
            } else {
                logger.error("No suitable matrix multiplication method found");
            }
        } catch (Exception e) {
            logger.error("Error performing matrix multiplication", e);
        }
        
        // Release resources
        matrixOp.release();
    }
    
    /**
     * Run a benchmark of matrix operations.
     */
    private static void runMatrixBenchmark(ComputeProvider provider) {
        logger.info("Running matrix operation benchmark");
        
        // Pass a device index (0 for primary device) instead of the provider
        OperationFactory operationFactory = new OperationFactory();
        MatrixOperation matrixOp = operationFactory.createMatrixOperation(0); // Use device index 0
        
        // Create large random matrices for benchmark
        float[] matrixA = generateRandomMatrix(MATRIX_SIZE * MATRIX_SIZE);
        float[] matrixB = generateRandomMatrix(MATRIX_SIZE * MATRIX_SIZE);
        float[] result = new float[MATRIX_SIZE * MATRIX_SIZE];
        
        logger.info("Matrix size: {}x{}", MATRIX_SIZE, MATRIX_SIZE);
        logger.info("Warming up...");
        
        // Find the appropriate matrix multiplication method
        java.lang.reflect.Method multiplyMethod;
        try {
            multiplyMethod = findMatrixMultiplyMethod(matrixOp.getClass());
            if (multiplyMethod == null) {
                logger.error("No suitable matrix multiplication method found");
                return;
            }
            
            // Warm-up run
            multiplyMethod.invoke(matrixOp, matrixA, matrixB, result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
            
            // Benchmark
            logger.info("Running benchmark ({} iterations)...", ITERATIONS);
            long totalTime = 0;
            
            for (int i = 0; i < ITERATIONS; i++) {
                long startTime = System.currentTimeMillis();
                multiplyMethod.invoke(matrixOp, matrixA, matrixB, result, MATRIX_SIZE, MATRIX_SIZE, MATRIX_SIZE);
                long endTime = System.currentTimeMillis();
                long iterationTime = endTime - startTime;
                
                totalTime += iterationTime;
                logger.info("  Iteration {}: {} ms", i + 1, iterationTime);
            }
            
            // Report results
            double avgTime = (double) totalTime / ITERATIONS;
            logger.info("Benchmark results:");
            logger.info("  Average time: {:.2f} ms", avgTime);
            logger.info("  Estimated GFLOPS: {:.2f}", 
                       (2.0 * MATRIX_SIZE * MATRIX_SIZE * MATRIX_SIZE / (avgTime / 1000.0)) / 1e9);
        } catch (Exception e) {
            logger.error("Error performing matrix multiplication", e);
        }
        
        // Release resources
        matrixOp.release();
    }
    
    /**
     * Generate a random matrix of specified size.
     */
    private static float[] generateRandomMatrix(int size) {
        float[] matrix = new float[size];
        Random random = new Random(42); // Fixed seed for reproducibility
        
        for (int i = 0; i < size; i++) {
            matrix[i] = random.nextFloat();
        }
        
        return matrix;
    }
    
    /**
     * Find a method that looks like matrix multiplication in the given class.
     * 
     * @param clazz The class to examine
     * @return A method that can be used for matrix multiplication, or null if none found
     */
    private static java.lang.reflect.Method findMatrixMultiplyMethod(Class<?> clazz) {
        // Try common method names for matrix multiplication
        String[] methodNames = {"multiply", "matrixMultiply", "multiplyMatrices", "mul"};
        
        for (String methodName : methodNames) {
            try {
                return clazz.getMethod(methodName, float[].class, float[].class, float[].class, 
                                     int.class, int.class, int.class);
            } catch (NoSuchMethodException e) {
                // Try the next name
            }
        }
        
        // If we get here, we couldn't find a suitable method
        logger.warn("Available methods on {}: {}", clazz.getName(), 
                   Arrays.toString(clazz.getMethods()));
        return null;
    }
}
