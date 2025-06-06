package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

public interface MatrixOperation {
    ComputeProvider getProvider();
    void multiply(float[] a, float[] b, float[] c, int rowsA, int colsB, int sharedDim);
    void add(float[] a, float[] b, float[] c, int elements);
    void subtract(float[] a, float[] b, float[] c, int elements);
    void scalarMultiply(float[] a, float[] b, float scalar, int elements);
    void transpose(float[] a, float[] b, int rows, int cols);
    void release();
}
