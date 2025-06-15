package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;

/**
 * Minimal OpenCL matrix operation implementing MatrixOperation
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
    public void add(float[] a, float[] b, float[] result, int size) {
        // no-op
    }

    @Override
    public void subtract(float[] a, float[] b, float[] result, int size) {
        // no-op
    }

    @Override
    public void dotProduct(float[] a, float[] b, float[] result, int length) {
        // no-op
    }

    @Override
    public void vectorNorm(float[] input, float[] result, int length) {
        // no-op
    }

    @Override
    public void elementWiseMultiply(float[] a, float[] b, float[] result, int size) {
        // no-op
    }

    @Override
    public void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols) {
        // no-op
    }

    @Override
    public void sigmoid(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void tanh(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void relu(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void softmax(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void mean(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void variance(float[] input, float[] result, int size, float mean) {
        // no-op
    }

    @Override
    public void normalize(float[] input, float[] result, int size) {
        // no-op
    }

    @Override
    public void copyArray(float[] source, float[] destination, int size) {
        // no-op
    }

    @Override
    public void fillArray(float[] array, float value, int size) {
        // no-op
    }

    @Override
    public void findMax(float[] input, int[] maxIndex, float[] maxValue, int size) {
        // no-op
    }

    @Override
    public void findMin(float[] input, int[] minIndex, float[] minValue, int size) {
        // no-op
    }
}
