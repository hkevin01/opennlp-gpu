package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Stub OpenCL matrix operation implementing MatrixOperation
 */
public class OpenClMatrixOperation implements MatrixOperation {

    private final ComputeProvider provider;

    public OpenClMatrixOperation(ComputeProvider provider) {
        this.provider = provider;
    }

    @Override
    public ComputeProvider getProvider() {
        return provider;
    }

    @Override
    public void release() {
        // no-op
    }

    @Override
    public void multiply(float[] a, float[] b, float[] result, int m, int n, int k) {
        // no-op
    }

    @Override
    public void transpose(float[] input, float[] output, int rows, int cols) {
        // no-op
    }

    @Override
    public void scalarMultiply(float[] input, float[] output, float scalar, int length) {
        // no-op
    }

    @Override
    public void subtract(float[] a, float[] b, float[] result, int size) {
        // no-op
    }

    @Override
    public void add(float[] a, float[] b, float[] result, int size) {
        // no-op
    }
}
