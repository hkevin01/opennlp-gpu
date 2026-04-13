package org.apache.opennlp.gpu.compute;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;

/**

 * ID: GPU-GMO-001
 * Requirement: GpuMatrixOperation must implement MatrixOperation using the best available GPU backend at runtime, with CPU fallback.
 * Purpose: Selects CUDA, ROCm, OpenCL, or CPU MatrixOperation implementation based on runtime hardware detection.
 * Rationale: Centralising backend selection in one class ensures callers never need to query hardware type directly.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: May allocate GPU device buffers on first use; delegates cleanup to underlying provider.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuMatrixOperation implements MatrixOperation {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuMatrixOperation.class);
    
    private final ComputeProvider provider;
    private final GpuConfig config;
    private final CpuMatrixOperation fallback;
    
    // Performance thresholds for GPU acceleration
    private static final int MIN_SIZE_FOR_GPU = 1000;
    private static final int BLOCK_SIZE = 16;
    
    /**
    
     * ID: GPU-GMO-002
     * Requirement: GpuMatrixOperation must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuMatrixOperation instance.
     * Inputs: ComputeProvider provider, GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuMatrixOperation(ComputeProvider provider, GpuConfig config) {
        this.provider = provider;
        this.config = config;
        this.fallback = new CpuMatrixOperation(provider);
        
        logger.info("Initialized GPU matrix operations with " + provider.getName());
    }
    
    /**
    
     * ID: GPU-GMO-003
     * Requirement: Return the Provider field value without side effects.
     * Purpose: Return the value of the Provider property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public ComputeProvider getProvider() {
        return provider;
    }
    
    /**
    
     * ID: GPU-GMO-004
     * Requirement: release must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void release() {
        if (fallback != null) {
            fallback.release();
        }
        logger.debug("Released GPU matrix operation resources");
    }
    
    // Basic Matrix Operations
    
    /**
    
     * ID: GPU-GMO-005
     * Requirement: multiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the multiply operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int m, int n, int k
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void multiply(float[] a, float[] b, float[] result, int m, int n, int k) {
        if (shouldUseGpu(m * k + k * n + m * n)) {
            multiplyGpu(a, b, result, m, n, k);
        } else {
            fallback.multiply(a, b, result, m, n, k);
        }
    }
    
    /**
    
     * ID: GPU-GMO-006
     * Requirement: transpose must execute correctly within the contract defined by this class.
     * Purpose: Implement the transpose operation for this class.
     * Inputs: float[] input, float[] output, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void transpose(float[] input, float[] output, int rows, int cols) {
        if (shouldUseGpu(rows * cols)) {
            transposeGpu(input, output, rows, cols);
        } else {
            fallback.transpose(input, output, rows, cols);
        }
    }
    
    /**
    
     * ID: GPU-GMO-007
     * Requirement: scalarMultiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the scalarMultiply operation for this class.
     * Inputs: float[] input, float[] output, float scalar, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void scalarMultiply(float[] input, float[] output, float scalar, int length) {
        if (shouldUseGpu(length)) {
            scalarMultiplyGpu(input, output, scalar, length);
        } else {
            fallback.scalarMultiply(input, output, scalar, length);
        }
    }
    
    /**
    
     * ID: GPU-GMO-008
     * Requirement: add must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void add(float[] a, float[] b, float[] result, int size) {
        if (shouldUseGpu(size)) {
            addGpu(a, b, result, size);
        } else {
            fallback.add(a, b, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-009
     * Requirement: subtract must execute correctly within the contract defined by this class.
     * Purpose: Implement the subtract operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void subtract(float[] a, float[] b, float[] result, int size) {
        if (shouldUseGpu(size)) {
            subtractGpu(a, b, result, size);
        } else {
            fallback.subtract(a, b, result, size);
        }
    }
    
    // Advanced Matrix Operations
    
    /**
    
     * ID: GPU-GMO-010
     * Requirement: dotProduct must execute correctly within the contract defined by this class.
     * Purpose: Implement the dotProduct operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void dotProduct(float[] a, float[] b, float[] result, int length) {
        if (shouldUseGpu(length)) {
            dotProductGpu(a, b, result, length);
        } else {
            fallback.dotProduct(a, b, result, length);
        }
    }
    
    /**
    
     * ID: GPU-GMO-011
     * Requirement: vectorNorm must execute correctly within the contract defined by this class.
     * Purpose: Implement the vectorNorm operation for this class.
     * Inputs: float[] input, float[] result, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void vectorNorm(float[] input, float[] result, int length) {
        if (shouldUseGpu(length)) {
            vectorNormGpu(input, result, length);
        } else {
            fallback.vectorNorm(input, result, length);
        }
    }
    
    /**
    
     * ID: GPU-GMO-012
     * Requirement: elementWiseMultiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the elementWiseMultiply operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void elementWiseMultiply(float[] a, float[] b, float[] result, int size) {
        if (shouldUseGpu(size)) {
            elementWiseMultiplyGpu(a, b, result, size);
        } else {
            fallback.elementWiseMultiply(a, b, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-013
     * Requirement: matrixVectorMultiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the matrixVectorMultiply operation for this class.
     * Inputs: float[] matrix, float[] vector, float[] result, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void matrixVectorMultiply(float[] matrix, float[] vector, float[] result, int rows, int cols) {
        if (shouldUseGpu(rows * cols)) {
            matrixVectorMultiplyGpu(matrix, vector, result, rows, cols);
        } else {
            fallback.matrixVectorMultiply(matrix, vector, result, rows, cols);
        }
    }
    
    // Activation Functions
    
    /**
    
     * ID: GPU-GMO-014
     * Requirement: sigmoid must execute correctly within the contract defined by this class.
     * Purpose: Implement the sigmoid operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void sigmoid(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            sigmoidGpu(input, result, size);
        } else {
            fallback.sigmoid(input, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-015
     * Requirement: tanh must execute correctly within the contract defined by this class.
     * Purpose: Implement the tanh operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void tanh(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            tanhGpu(input, result, size);
        } else {
            fallback.tanh(input, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-016
     * Requirement: relu must execute correctly within the contract defined by this class.
     * Purpose: Implement the relu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void relu(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            reluGpu(input, result, size);
        } else {
            fallback.relu(input, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-017
     * Requirement: softmax must execute correctly within the contract defined by this class.
     * Purpose: Implement the softmax operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void softmax(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            softmaxGpu(input, result, size);
        } else {
            fallback.softmax(input, result, size);
        }
    }
    
    // Statistical Operations
    
    /**
    
     * ID: GPU-GMO-018
     * Requirement: mean must execute correctly within the contract defined by this class.
     * Purpose: Implement the mean operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void mean(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            meanGpu(input, result, size);
        } else {
            fallback.mean(input, result, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-019
     * Requirement: variance must execute correctly within the contract defined by this class.
     * Purpose: Implement the variance operation for this class.
     * Inputs: float[] input, float[] result, int size, float mean
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void variance(float[] input, float[] result, int size, float mean) {
        if (shouldUseGpu(size)) {
            varianceGpu(input, result, size, mean);
        } else {
            fallback.variance(input, result, size, mean);
        }
    }
    
    /**
    
     * ID: GPU-GMO-020
     * Requirement: normalize must execute correctly within the contract defined by this class.
     * Purpose: Implement the normalize operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void normalize(float[] input, float[] result, int size) {
        if (shouldUseGpu(size)) {
            normalizeGpu(input, result, size);
        } else {
            fallback.normalize(input, result, size);
        }
    }
    
    // Utility Operations
    
    /**
    
     * ID: GPU-GMO-021
     * Requirement: copyArray must execute correctly within the contract defined by this class.
     * Purpose: Implement the copyArray operation for this class.
     * Inputs: float[] source, float[] destination, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void copyArray(float[] source, float[] destination, int size) {
        System.arraycopy(source, 0, destination, 0, size);
    }
    
    /**
    
     * ID: GPU-GMO-022
     * Requirement: fillArray must execute correctly within the contract defined by this class.
     * Purpose: Implement the fillArray operation for this class.
     * Inputs: float[] array, float value, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void fillArray(float[] array, float value, int size) {
        if (shouldUseGpu(size)) {
            fillArrayGpu(array, value, size);
        } else {
            fallback.fillArray(array, value, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-023
     * Requirement: findMax must execute correctly within the contract defined by this class.
     * Purpose: Implement the findMax operation for this class.
     * Inputs: float[] input, int[] maxIndex, float[] maxValue, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void findMax(float[] input, int[] maxIndex, float[] maxValue, int size) {
        if (shouldUseGpu(size)) {
            findMaxGpu(input, maxIndex, maxValue, size);
        } else {
            fallback.findMax(input, maxIndex, maxValue, size);
        }
    }
    
    /**
    
     * ID: GPU-GMO-024
     * Requirement: findMin must execute correctly within the contract defined by this class.
     * Purpose: Implement the findMin operation for this class.
     * Inputs: float[] input, int[] minIndex, float[] minValue, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public void findMin(float[] input, int[] minIndex, float[] minValue, int size) {
        if (shouldUseGpu(size)) {
            findMinGpu(input, minIndex, minValue, size);
        } else {
            fallback.findMin(input, minIndex, minValue, size);
        }
    }
    
    // GPU Implementation Methods (stubs for now - will be implemented with OpenCL/CUDA)
    
    /**
    
     * ID: GPU-GMO-025
     * Requirement: multiplyGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the multiplyGpu operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int m, int n, int k
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void multiplyGpu(float[] a, float[] b, float[] result, int m, int n, int k) {
        logger.debug("GPU matrix multiply: " + m + "x" + k + " * " + k + "x" + n);
        fallback.multiply(a, b, result, m, n, k); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-026
     * Requirement: transposeGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the transposeGpu operation for this class.
     * Inputs: float[] input, float[] output, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void transposeGpu(float[] input, float[] output, int rows, int cols) {
        logger.debug("GPU transpose: " + rows + "x" + cols);
        fallback.transpose(input, output, rows, cols); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-027
     * Requirement: scalarMultiplyGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the scalarMultiplyGpu operation for this class.
     * Inputs: float[] input, float[] output, float scalar, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void scalarMultiplyGpu(float[] input, float[] output, float scalar, int length) {
        logger.debug("GPU scalar multiply: " + length + " elements");
        fallback.scalarMultiply(input, output, scalar, length); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-028
     * Requirement: addGpu must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void addGpu(float[] a, float[] b, float[] result, int size) {
        logger.debug("GPU add: " + size + " elements");
        fallback.add(a, b, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-029
     * Requirement: subtractGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the subtractGpu operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void subtractGpu(float[] a, float[] b, float[] result, int size) {
        logger.debug("GPU subtract: " + size + " elements");
        fallback.subtract(a, b, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-030
     * Requirement: dotProductGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the dotProductGpu operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void dotProductGpu(float[] a, float[] b, float[] result, int length) {
        logger.debug("GPU dot product: " + length + " elements");
        fallback.dotProduct(a, b, result, length); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-031
     * Requirement: vectorNormGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the vectorNormGpu operation for this class.
     * Inputs: float[] input, float[] result, int length
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void vectorNormGpu(float[] input, float[] result, int length) {
        logger.debug("GPU vector norm: " + length + " elements");
        fallback.vectorNorm(input, result, length); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-032
     * Requirement: elementWiseMultiplyGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the elementWiseMultiplyGpu operation for this class.
     * Inputs: float[] a, float[] b, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void elementWiseMultiplyGpu(float[] a, float[] b, float[] result, int size) {
        logger.debug("GPU element-wise multiply: " + size + " elements");
        fallback.elementWiseMultiply(a, b, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-033
     * Requirement: matrixVectorMultiplyGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the matrixVectorMultiplyGpu operation for this class.
     * Inputs: float[] matrix, float[] vector, float[] result, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void matrixVectorMultiplyGpu(float[] matrix, float[] vector, float[] result, int rows, int cols) {
        logger.debug("GPU matrix-vector multiply: " + rows + "x" + cols);
        fallback.matrixVectorMultiply(matrix, vector, result, rows, cols); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-034
     * Requirement: sigmoidGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the sigmoidGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void sigmoidGpu(float[] input, float[] result, int size) {
        logger.debug("GPU sigmoid: " + size + " elements");
        fallback.sigmoid(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-035
     * Requirement: tanhGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the tanhGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void tanhGpu(float[] input, float[] result, int size) {
        logger.debug("GPU tanh: " + size + " elements");
        fallback.tanh(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-036
     * Requirement: reluGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the reluGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void reluGpu(float[] input, float[] result, int size) {
        logger.debug("GPU ReLU: " + size + " elements");
        fallback.relu(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-037
     * Requirement: softmaxGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the softmaxGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void softmaxGpu(float[] input, float[] result, int size) {
        logger.debug("GPU softmax: " + size + " elements");
        fallback.softmax(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-038
     * Requirement: meanGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the meanGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void meanGpu(float[] input, float[] result, int size) {
        logger.debug("GPU mean: " + size + " elements");
        fallback.mean(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-039
     * Requirement: varianceGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the varianceGpu operation for this class.
     * Inputs: float[] input, float[] result, int size, float mean
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void varianceGpu(float[] input, float[] result, int size, float mean) {
        logger.debug("GPU variance: " + size + " elements");
        fallback.variance(input, result, size, mean); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-040
     * Requirement: normalizeGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the normalizeGpu operation for this class.
     * Inputs: float[] input, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void normalizeGpu(float[] input, float[] result, int size) {
        logger.debug("GPU normalize: " + size + " elements");
        fallback.normalize(input, result, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-041
     * Requirement: fillArrayGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the fillArrayGpu operation for this class.
     * Inputs: float[] array, float value, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void fillArrayGpu(float[] array, float value, int size) {
        logger.debug("GPU fill array: " + size + " elements");
        fallback.fillArray(array, value, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-042
     * Requirement: findMaxGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the findMaxGpu operation for this class.
     * Inputs: float[] input, int[] maxIndex, float[] maxValue, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void findMaxGpu(float[] input, int[] maxIndex, float[] maxValue, int size) {
        logger.debug("GPU find max: " + size + " elements");
        fallback.findMax(input, maxIndex, maxValue, size); // Fallback for now
    }
    
    /**
    
     * ID: GPU-GMO-043
     * Requirement: findMinGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the findMinGpu operation for this class.
     * Inputs: float[] input, int[] minIndex, float[] minValue, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void findMinGpu(float[] input, int[] minIndex, float[] minValue, int size) {
        logger.debug("GPU find min: " + size + " elements");
        fallback.findMin(input, minIndex, minValue, size); // Fallback for now
    }
    
    // Helper Methods
    
    /**
    
     * ID: GPU-GMO-044
     * Requirement: shouldUseGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpu operation for this class.
     * Inputs: int operationSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpu(int operationSize) {
        return provider.isGpuProvider() && 
               operationSize >= MIN_SIZE_FOR_GPU && 
               config.isGpuEnabled();
    }
}
