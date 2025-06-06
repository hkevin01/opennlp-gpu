package org.apache.opennlp.gpu;

import java.util.Arrays;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
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
    
    public static void main(String[] args) {
        GpuDemoMain.logger.info("Starting OpenNLP GPU Demo");
        
        try {
            // Initialize compute provider
            ComputeProvider provider = ComputeProviderFactory.getDefaultProvider();
            GpuDemoMain.logger.info("Using compute provider: {}", provider.getName());
            
            // Demo matrix operations
            GpuDemoMain.demoMatrixOperations(provider);
            
            GpuDemoMain.logger.info("Demo completed successfully");
        } catch (Exception e) {
            GpuDemoMain.logger.error("Error running demo", e);
        }
    }
    
    /**
     * Run a simple matrix multiplication demo.
     */
    private static void demoMatrixOperations(ComputeProvider provider) {
        // Create a matrix operation
        MatrixOperation matrixOp = OperationFactory.createMatrixOperation();
        
        // Use fixed 2x2 matrices for demo
        float[] matrixA = {1.0f, 2.0f, 3.0f, 4.0f};
        float[] matrixB = {5.0f, 6.0f, 7.0f, 8.0f};
        float[] result = new float[4];
        
        GpuDemoMain.logger.info("Matrix A: {}", Arrays.toString(matrixA));
        GpuDemoMain.logger.info("Matrix B: {}", Arrays.toString(matrixB));
        
        try {
            // Reflection call to find and invoke the multiply method using dimensions 2,2,2
            java.lang.reflect.Method multiplyMethod = GpuDemoMain.findMatrixMultiplyMethod(matrixOp.getClass());
            if (multiplyMethod != null) {
                multiplyMethod.invoke(matrixOp, matrixA, matrixB, result, 2, 2, 2);
                GpuDemoMain.logger.info("Result (A*B): {}", Arrays.toString(result));
            } else {
                GpuDemoMain.logger.error("No suitable matrix multiplication method found");
            }
        } catch (Exception e) {
            GpuDemoMain.logger.error("Error performing matrix multiplication", e);
        }
        
        // Clean up
        matrixOp.release();
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
        GpuDemoMain.logger.warn("Available methods on {}: {}", clazz.getName(), 
                   Arrays.toString(clazz.getMethods()));
        return null;
    }
}
