package org.apache.opennlp.gpu.stress;

import java.util.Random;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.junit.jupiter.api.Assertions;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;

public class PerformanceBenchmark {

    private static MatrixOperation gpuOps;
    private static MatrixOperation cpuOps;
    
    @BeforeAll
    static void setup() {
        ComputeProvider provider = new ComputeProvider() {
            @Override
            public String getName() {
                return "BenchmarkProvider";
            }
            
            @Override
            public ComputeProvider.Type getType() {
                return ComputeProvider.Type.CPU;
            }
            
            @Override
            public boolean isAvailable() {
                return true;
            }
            
            @Override
            public void initialize() {
                // No initialization needed
            }
            
            @Override
            public void initialize(GpuConfig config) {
                // No initialization needed
            }
            
            @Override
            public long getMaxMemoryMB() {
                return 1024L;
            }
            
            @Override
            public long getCurrentMemoryUsageMB() {
                return 0L;
            }
            
            @Override
            public Object getResourceManager() {
                return null;
            }
            
            @Override
            public boolean supportsOperation(String operation) {
                return true;
            }
            
            @Override
            public void matrixMultiply(float[] a, float[] b, float[] c, int m, int n, int k) {
                // Not needed for benchmark test
            }
            
            @Override
            public void matrixTranspose(float[] input, float[] output, int rows, int cols) {
                // Not needed for benchmark test
            }
            
            @Override
            public void matrixAdd(float[] a, float[] b, float[] c, int size) {
                // Not needed for benchmark test
            }
            
            @Override
            public void computeTfIdf(float[] tf, float[] idf, float[] result, int size) {
                // Not needed for benchmark test
            }
            
            @Override
            public void extractFeatures(String[] text, float[] features) {
                // Not needed for benchmark test
            }
        };
        GpuConfig config = new GpuConfig();
        gpuOps = new GpuMatrixOperation(provider, config);
        cpuOps = new CpuMatrixOperation(provider);
    }

    @Test
    void benchmarkMatrixMultiply() {
        int size = 512;
        float[][] matrixA = createTestMatrix(size, size, 123);
        float[][] matrixB = createTestMatrix(size, size, 456);
        float[] a = flattenMatrix(matrixA);
        float[] b = flattenMatrix(matrixB);
        float[] c = new float[size * size];

        long gpuTime = timeExecution(() -> gpuOps.multiply(a, b, c, size, size, size));
        long cpuTime = timeExecution(() -> cpuOps.multiply(a, b, c, size, size, size));

        System.out.println("GPU Time (Multiply): " + gpuTime + "ms");
        System.out.println("CPU Time (Multiply): " + cpuTime + "ms");
    }
    
    @Test
    void benchmarkMatrixTranspose() {
        int size = 1024;
        float[][] matrix = createTestMatrix(size, size, 789);
        float[] input = flattenMatrix(matrix);
        float[] output = new float[size * size];

        long gpuTime = timeExecution(() -> gpuOps.transpose(input, output, size, size));
        long cpuTime = timeExecution(() -> cpuOps.transpose(input, output, size, size));

        System.out.println("GPU Time (Transpose): " + gpuTime + "ms");
        System.out.println("CPU Time (Transpose): " + cpuTime + "ms");
        Assertions.assertTrue(gpuTime < cpuTime, "GPU should be faster than CPU for matrix transpose");
    }

    private long timeExecution(Runnable operation) {
        long startTime = System.currentTimeMillis();
        operation.run();
        return System.currentTimeMillis() - startTime;
    }
    
    private float[][] createTestMatrix(int rows, int cols, int seed) {
        Random random = new Random(seed);
        float[][] matrix = new float[rows][cols];
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                matrix[i][j] = random.nextFloat();
            }
        }
        return matrix;
    }
    
    private float[] flattenMatrix(float[][] matrix) {
        int rows = matrix.length;
        int cols = matrix[0].length;
        float[] flattened = new float[rows * cols];
        
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                flattened[i * cols + j] = matrix[i][j];
            }
        }
        
        return flattened;
    }
}
