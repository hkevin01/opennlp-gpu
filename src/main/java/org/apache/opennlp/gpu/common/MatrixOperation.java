package org.apache.opennlp.gpu.common;

/**

 * ID: GPU-MO-001
 * Requirement: MatrixOperation must define hardware-agnostic matrix and vector arithmetic
 *              operations callable by any compute backend implementation.
 * Purpose: Shared interface in the common package for both CPU and GPU backends;
 *          provides default no-op implementations so subinterfaces need only override
 *          the operations they specialise.
 * Rationale: Placing a MatrixOperation interface in common lets the common-package
 *            providers (CpuComputeProvider, CudaComputeProvider, etc.) implement it
 *            without importing the compute sub-package, avoiding circular dependencies.
 * Inputs: float[] or Object matrix arguments; int dimensions m, n, k.
 * Outputs: void (result written into caller-supplied output arrays) or Object references.
 * Preconditions: Input arrays non-null; dimension parameters positive and consistent.
 * Postconditions: Result array elements reflect the requested mathematical operation.
 * Assumptions: Caller manages array lifecycle; arrays are host-side float[] by default.
 * Side Effects: Writes results into caller-supplied result arrays; no I/O.
 * Failure Modes: Array index out of bounds if dimensions are inconsistent; default
 *               implementations return null or no-op.
 * Error Handling: Implementors must validate dimensions; default methods silently no-op.
 * Constraints: No thread-safety requirement at interface level; callers synchronise.
 * Verification: Implementations tested via CpuMatrixOperation unit tests for parity.
 * References: compute.MatrixOperation; Apache OpenNLP 2.5.8; ARCHITECTURE_OVERVIEW.md.
 */
public interface MatrixOperation {

    /**
     * Multiplies two matrices.
     *
     * @param matrixA the first matrix
     * @param matrixB the second matrix
     * @return the result matrix
     */
    /**
    
     * ID: GPU-MO-002
     * Requirement: multiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the multiply operation for this class.
     * Inputs: Object matrixA, Object matrixB
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default Object multiply(Object matrixA, Object matrixB) {
        // Default implementation does nothing
        return null;
    }

    /**
     * Multiplies two matrices with specific dimensions.
     *
     * @param matrixA the first matrix
     * @param matrixB the second matrix
     * @param result the result matrix
     * @param m rows in matrixA
     * @param n columns in matrixB
     * @param k columns in matrixA / rows in matrixB
     */
    /**
    
     * ID: GPU-MO-003
     * Requirement: multiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the multiply operation for this class.
     * Inputs: float[] matrixA, float[] matrixB, float[] result, int m, int n, int k
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default void multiply(float[] matrixA, float[] matrixB, float[] result, int m, int n, int k) {
        // Default implementation does nothing
    }

    /**
     * Adds two matrices.
     *
     * @param matrixA the first matrix
     * @param matrixB the second matrix
     * @return the result matrix
     */
    /**
    
     * ID: GPU-MO-004
     * Requirement: add must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: Object matrixA, Object matrixB
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default Object add(Object matrixA, Object matrixB) {
        // Default implementation does nothing
        return null;
    }

    /**
     * Adds two matrices of specific size.
     *
     * @param matrixA the first matrix
     * @param matrixB the second matrix
     * @param result the result matrix
     * @param size the size of the matrices
     */
    /**
    
     * ID: GPU-MO-005
     * Requirement: add must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: float[] matrixA, float[] matrixB, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default void add(float[] matrixA, float[] matrixB, float[] result, int size) {
        // Default implementation does nothing
    }

    /**
     * Subtracts the second matrix from the first.
     *
     * @param matrixA the first matrix
     * @param matrixB the second matrix
     * @param result the result matrix
     * @param size the size of the matrices
     */
    /**
    
     * ID: GPU-MO-006
     * Requirement: subtract must execute correctly within the contract defined by this class.
     * Purpose: Implement the subtract operation for this class.
     * Inputs: float[] matrixA, float[] matrixB, float[] result, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default void subtract(float[] matrixA, float[] matrixB, float[] result, int size) {
        // Default implementation does nothing
    }

    /**
     * Multiplies a matrix by a scalar value.
     *
     * @param matrix the input matrix
     * @param result the result matrix
     * @param scalar the scalar value
     * @param size the size of the matrix
     */
    /**
    
     * ID: GPU-MO-007
     * Requirement: scalarMultiply must execute correctly within the contract defined by this class.
     * Purpose: Implement the scalarMultiply operation for this class.
     * Inputs: float[] matrix, float[] result, float scalar, int size
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default void scalarMultiply(float[] matrix, float[] result, float scalar, int size) {
        // Default implementation does nothing
    }

    /**
     * Transposes a matrix.
     *
     * @param matrix the input matrix
     * @param result the result matrix
     * @param rows the number of rows
     * @param cols the number of columns
     */
    /**
    
     * ID: GPU-MO-008
     * Requirement: transpose must execute correctly within the contract defined by this class.
     * Purpose: Implement the transpose operation for this class.
     * Inputs: float[] matrix, float[] result, int rows, int cols
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default void transpose(float[] matrix, float[] result, int rows, int cols) {
        // Default implementation does nothing
    }

    /**
     * Checks if this operation is supported on the current device.
     *
     * @return true if the operation is supported
     */
    /**
    
     * ID: GPU-MO-009
     * Requirement: Evaluate and return the boolean result of isSupported.
     * Purpose: Return whether isSupported condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    default boolean isSupported() {
        return true;
    }
}
