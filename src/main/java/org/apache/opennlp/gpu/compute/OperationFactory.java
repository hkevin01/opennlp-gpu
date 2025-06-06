package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

public class OperationFactory {
    
    public static MatrixOperation createMatrixOperation() {
        return new DummyMatrixOperation();
    }
    
    public static MatrixOperation createMatrixOperation(ComputeProvider provider) {
        return new DummyMatrixOperation();
    }
    
    private static class DummyMatrixOperation implements MatrixOperation {
        public ComputeProvider getProvider() { 
            return null; 
        }
        
        public void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim) { 
            // CPU fallback implementation
            for (int i = 0; i < rowsA; i++) {
                for (int j = 0; j < colsB; j++) {
                    float sum = 0.0f;
                    for (int k = 0; k < sharedDim; k++) {
                        sum += a[i * sharedDim + k] * b[k * colsB + j];
                    }
                    c[i * colsB + j] = sum;
                }
            }
        }
        
        public void add(float[] a, float[] b, float[] c, int elements) { 
            for (int i = 0; i < elements; i++) {
                c[i] = a[i] + b[i];
            }
        }
        
        public void subtract(float[] a, float[] b, float[] c, int elements) { 
            for (int i = 0; i < elements; i++) {
                c[i] = a[i] - b[i];
            }
        }
        
        public void scalarMultiply(float[] a, float[] b, float scalar, int elements) { 
            for (int i = 0; i < elements; i++) {
                b[i] = a[i] * scalar;
            }
        }
        
        public void transpose(float[] a, float[] b, int rows, int cols) { 
            for (int i = 0; i < rows; i++) {
                for (int j = 0; j < cols; j++) {
                    b[j * rows + i] = a[i * cols + j];
                }
            }
        }
        
        public void release() { 
            // No-op
        }
    }
}
