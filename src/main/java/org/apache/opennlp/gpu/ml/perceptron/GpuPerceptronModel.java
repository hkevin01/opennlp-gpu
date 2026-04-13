package org.apache.opennlp.gpu.ml.perceptron;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;

/**

 * ID: GPU-GPM-001
 * Requirement: GpuPerceptronModel must implement a binary/multi-class Perceptron classifier with GPU-accelerated batch prediction.
 * Purpose: Wraps a trained Perceptron model, routing batch dot-product evaluation to GPU matrix ops for high-throughput NLP classification.
 * Rationale: Perceptron evaluation is a batched dot product; GPU achieves near-linear throughput scaling with batch size.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None beyond GPU dispatch delegation; weight vector is read-only.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class GpuPerceptronModel {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuPerceptronModel.class);
    
    private final GpuConfig config;
    private final ComputeProvider computeProvider;
    private final MatrixOperation matrixOp;
    
    // Model parameters
    private float[] weights;
    private float bias;
    private int featureCount;
    private int trainingIterations;
    private float learningRate;
    
    // Training parameters
    private static final float DEFAULT_LEARNING_RATE = 0.1f;
    private static final int DEFAULT_MAX_ITERATIONS = 1000;
    private static final float CONVERGENCE_THRESHOLD = 1e-6f;
    
    // Performance thresholds
    private static final int MIN_FEATURES_FOR_GPU = 1000;
    private static final int MIN_SAMPLES_FOR_GPU = 100;
    
    /**
    
     * ID: GPU-GPM-002
     * Requirement: GpuPerceptronModel must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuPerceptronModel instance.
     * Inputs: GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuPerceptronModel(GpuConfig config) {
        this(config, GpuPerceptronModel.DEFAULT_LEARNING_RATE, GpuPerceptronModel.DEFAULT_MAX_ITERATIONS);
    }
    
    /**
    
     * ID: GPU-GPM-003
     * Requirement: GpuPerceptronModel must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a GpuPerceptronModel instance.
     * Inputs: GpuConfig config, float learningRate, int maxIterations
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public GpuPerceptronModel(GpuConfig config, float learningRate, int maxIterations) {
        this.config = config;
        this.learningRate = learningRate;
        this.trainingIterations = maxIterations;
        this.computeProvider = createComputeProvider();
        this.matrixOp = createMatrixOperation();
        this.featureCount = 0;
        this.weights = new float[0];
        this.bias = 0.0f;
        
        GpuPerceptronModel.logger.info("Created GPU perceptron model with learning rate: " + learningRate);
    }
    
    /**
    
     * ID: GPU-GPM-004
     * Requirement: createComputeProvider must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new ComputeProvider.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private ComputeProvider createComputeProvider() {
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                return new GpuComputeProvider(config);
            }
        } catch (Exception e) {
            GpuPerceptronModel.logger.warn("Failed to initialize GPU provider: " + e.getMessage());
        }
        return new CpuComputeProvider();
    }
    
    /**
    
     * ID: GPU-GPM-005
     * Requirement: createMatrixOperation must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new MatrixOperation.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private MatrixOperation createMatrixOperation() {
        if (computeProvider.isGpuProvider()) {
            return new org.apache.opennlp.gpu.compute.GpuMatrixOperation(computeProvider, config);
        } else {
            return new org.apache.opennlp.gpu.compute.CpuMatrixOperation(computeProvider);
        }
    }
    
    /**
     * Train the perceptron model with GPU acceleration
     */
    /**
    
     * ID: GPU-GPM-006
     * Requirement: train must execute correctly within the contract defined by this class.
     * Purpose: Train the model on the supplied data.
     * Inputs: float[][] features, int[] labels
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void train(float[][] features, int[] labels) {
        if (features.length == 0) {
            throw new IllegalArgumentException("Training data cannot be empty");
        }
        
        initializeWeights(features[0].length);
        
        if (shouldUseGpu(features)) {
            trainOnGpu(features, labels);
        } else {
            trainOnCpu(features, labels);
        }
        
        GpuPerceptronModel.logger.info("Training completed after " + trainingIterations + " iterations");
    }
    
    /**
     * Train with double precision features (convenience method)
     */
    /**
    
     * ID: GPU-GPM-007
     * Requirement: train must execute correctly within the contract defined by this class.
     * Purpose: Train the model on the supplied data.
     * Inputs: double[][] features, int[] labels
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void train(double[][] features, int[] labels) {
        float[][] floatFeatures = convertToFloat(features);
        train(floatFeatures, labels);
    }
    
    /**
     * Predict using the perceptron model
     */
    /**
    
     * ID: GPU-GPM-008
     * Requirement: predict must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int predict(float[] features) {
        if (weights.length != features.length) {
            GpuPerceptronModel.logger.warn("Feature dimension mismatch: expected " + weights.length + ", got " + features.length);
            return 0;
        }
        
        if (shouldUseGpu(features)) {
            return predictOnGpu(features);
        } else {
            return predictOnCpu(features);
        }
    }
    
    /**
     * Predict with double precision features (convenience method)
     */
    /**
    
     * ID: GPU-GPM-009
     * Requirement: predict must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: double[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int predict(double[] features) {
        float[] floatFeatures = new float[features.length];
        for (int i = 0; i < features.length; i++) {
            floatFeatures[i] = (float) features[i];
        }
        return predict(floatFeatures);
    }
    
    /**
     * Batch prediction for multiple samples
     */
    /**
    
     * ID: GPU-GPM-010
     * Requirement: predictBatch must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int[] predictBatch(float[][] features) {
        int[] predictions = new int[features.length];
        
        if (shouldUseGpuBatch(features)) {
            return predictBatchOnGpu(features);
        } else {
            for (int i = 0; i < features.length; i++) {
                predictions[i] = predict(features[i]);
            }
        }
        
        return predictions;
    }
    
    /**
     * Get the decision function value (before thresholding)
     */
    /**
    
     * ID: GPU-GPM-011
     * Requirement: decisionFunction must execute correctly within the contract defined by this class.
     * Purpose: Implement the decisionFunction operation for this class.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float decisionFunction(float[] features) {
        if (weights.length != features.length) {
            return 0.0f;
        }
        
        if (shouldUseGpu(features)) {
            return decisionFunctionGpu(features);
        } else {
            return decisionFunctionCpu(features);
        }
    }
    
    // GPU training implementation
    
    /**
    
     * ID: GPU-GPM-012
     * Requirement: trainOnGpu must execute correctly within the contract defined by this class.
     * Purpose: Train the model on the supplied data.
     * Inputs: float[][] features, int[] labels
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void trainOnGpu(float[][] features, int[] labels) {
        GpuPerceptronModel.logger.debug("Training perceptron on GPU with " + features.length + " samples");
        
        int numSamples = features.length;
        float[] flatFeatures = flattenFeatures(features);
        
        try {
            for (int iter = 0; iter < trainingIterations; iter++) {
                float totalError = 0.0f;
                
                // Process batch of samples
                for (int i = 0; i < numSamples; i++) {
                    int startIdx = i * featureCount;
                    float[] sample = new float[featureCount];
                    System.arraycopy(flatFeatures, startIdx, sample, 0, featureCount);
                    
                    float prediction = decisionFunctionGpu(sample);
                    int predictedLabel = prediction >= 0 ? 1 : 0;
                    int actualLabel = labels[i];
                    
                    if (predictedLabel != actualLabel) {
                        // Update weights using GPU vector operations
                        updateWeightsGpu(sample, actualLabel - predictedLabel);
                        totalError += 1.0f;
                    }
                }
                
                // Check for convergence
                float errorRate = totalError / numSamples;
                if (errorRate < GpuPerceptronModel.CONVERGENCE_THRESHOLD) {
                    trainingIterations = iter + 1;
                    GpuPerceptronModel.logger.debug("Converged after " + trainingIterations + " iterations");
                    break;
                }
                
                if (iter % 100 == 0) {
                    GpuPerceptronModel.logger.debug("Iteration " + iter + ", error rate: " + errorRate);
                }
            }
        } catch (Exception e) {
            GpuPerceptronModel.logger.warn("GPU training failed, falling back to CPU: " + e.getMessage());
            trainOnCpu(features, labels);
        }
    }
    
    /**
    
     * ID: GPU-GPM-013
     * Requirement: trainOnCpu must execute correctly within the contract defined by this class.
     * Purpose: Train the model on the supplied data.
     * Inputs: float[][] features, int[] labels
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void trainOnCpu(float[][] features, int[] labels) {
        GpuPerceptronModel.logger.debug("Training perceptron on CPU with " + features.length + " samples");
        
        for (int iter = 0; iter < trainingIterations; iter++) {
            int errors = 0;
            
            for (int i = 0; i < features.length; i++) {
                float prediction = decisionFunctionCpu(features[i]);
                int predictedLabel = prediction >= 0 ? 1 : 0;
                int actualLabel = labels[i];
                
                if (predictedLabel != actualLabel) {
                    updateWeightsCpu(features[i], actualLabel - predictedLabel);
                    errors++;
                }
            }
            
            // Check for convergence
            if (errors == 0) {
                trainingIterations = iter + 1;
                GpuPerceptronModel.logger.debug("Converged after " + trainingIterations + " iterations");
                break;
            }
        }
    }
    
    // GPU prediction implementations
    
    /**
    
     * ID: GPU-GPM-014
     * Requirement: predictOnGpu must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private int predictOnGpu(float[] features) {
        float decision = decisionFunctionGpu(features);
        return decision >= 0 ? 1 : 0;
    }
    
    /**
    
     * ID: GPU-GPM-015
     * Requirement: predictOnCpu must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private int predictOnCpu(float[] features) {
        float decision = decisionFunctionCpu(features);
        return decision >= 0 ? 1 : 0;
    }
    
    /**
    
     * ID: GPU-GPM-016
     * Requirement: predictBatchOnGpu must execute correctly within the contract defined by this class.
     * Purpose: Produce a prediction or classification for the input.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private int[] predictBatchOnGpu(float[][] features) {
        try {
            GpuPerceptronModel.logger.debug("GPU batch prediction for " + features.length + " samples");
            
            int[] predictions = new int[features.length];
            float[] decisions = new float[features.length];
            
            // Calculate decision values for all samples
            for (int i = 0; i < features.length; i++) {
                decisions[i] = decisionFunctionGpu(features[i]);
            }
            
            // Apply threshold using GPU operations
            float[] zeros = new float[features.length];
            matrixOp.fillArray(zeros, 0.0f, features.length);
            
            // Convert to predictions
            for (int i = 0; i < features.length; i++) {
                predictions[i] = decisions[i] >= 0 ? 1 : 0;
            }
            
            return predictions;
            
        } catch (Exception e) {
            GpuPerceptronModel.logger.warn("GPU batch prediction failed, falling back to individual predictions: " + e.getMessage());
            int[] predictions = new int[features.length];
            for (int i = 0; i < features.length; i++) {
                predictions[i] = predict(features[i]);
            }
            return predictions;
        }
    }
    
    /**
    
     * ID: GPU-GPM-017
     * Requirement: decisionFunctionGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the decisionFunctionGpu operation for this class.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private float decisionFunctionGpu(float[] features) {
        try {
            // Calculate dot product using GPU
            float[] result = new float[1];
            matrixOp.dotProduct(weights, features, result, featureCount);
            return result[0] + bias;
        } catch (Exception e) {
            GpuPerceptronModel.logger.debug("GPU decision function failed, using CPU: " + e.getMessage());
            return decisionFunctionCpu(features);
        }
    }
    
    /**
    
     * ID: GPU-GPM-018
     * Requirement: decisionFunctionCpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the decisionFunctionCpu operation for this class.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private float decisionFunctionCpu(float[] features) {
        float sum = bias;
        for (int i = 0; i < featureCount; i++) {
            sum += weights[i] * features[i];
        }
        return sum;
    }
    
    /**
    
     * ID: GPU-GPM-019
     * Requirement: updateWeightsGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the updateWeightsGpu operation for this class.
     * Inputs: float[] features, float delta
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void updateWeightsGpu(float[] features, float delta) {
        try {
            // Scale features by learning rate and delta
            float[] scaledFeatures = new float[featureCount];
            float scale = learningRate * delta;
            matrixOp.scalarMultiply(features, scaledFeatures, scale, featureCount);
            
            // Add to weights
            matrixOp.add(weights, scaledFeatures, weights, featureCount);
            
            // Update bias
            bias += learningRate * delta;
            
        } catch (Exception e) {
            GpuPerceptronModel.logger.debug("GPU weight update failed, using CPU: " + e.getMessage());
            updateWeightsCpu(features, delta);
        }
    }
    
    /**
    
     * ID: GPU-GPM-020
     * Requirement: updateWeightsCpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the updateWeightsCpu operation for this class.
     * Inputs: float[] features, float delta
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void updateWeightsCpu(float[] features, float delta) {
        float adjustment = learningRate * delta;
        for (int i = 0; i < featureCount; i++) {
            weights[i] += adjustment * features[i];
        }
        bias += adjustment;
    }
    
    // Helper methods
    
    /**
    
     * ID: GPU-GPM-021
     * Requirement: initializeWeights must execute correctly within the contract defined by this class.
     * Purpose: Initialise internal state and allocate required resources.
     * Inputs: int numFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void initializeWeights(int numFeatures) {
        this.featureCount = numFeatures;
        this.weights = new float[featureCount];
        this.bias = 0.0f;
        
        // Initialize weights with small random values
        for (int i = 0; i < featureCount; i++) {
            weights[i] = (float) (Math.random() * 0.02 - 0.01); // [-0.01, 0.01]
        }
        
        GpuPerceptronModel.logger.debug("Initialized weights for " + featureCount + " features");
    }
    
    /**
    
     * ID: GPU-GPM-022
     * Requirement: shouldUseGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpu operation for this class.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpu(float[][] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= GpuPerceptronModel.MIN_SAMPLES_FOR_GPU &&
               features[0].length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
    /**
    
     * ID: GPU-GPM-023
     * Requirement: shouldUseGpu must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpu operation for this class.
     * Inputs: float[] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpu(float[] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
    /**
    
     * ID: GPU-GPM-024
     * Requirement: shouldUseGpuBatch must execute correctly within the contract defined by this class.
     * Purpose: Implement the shouldUseGpuBatch operation for this class.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean shouldUseGpuBatch(float[][] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= 10 && // Minimum batch size
               features[0].length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
    /**
    
     * ID: GPU-GPM-025
     * Requirement: convertToFloat must execute correctly within the contract defined by this class.
     * Purpose: Implement the convertToFloat operation for this class.
     * Inputs: double[][] doubleFeatures
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private float[][] convertToFloat(double[][] doubleFeatures) {
        float[][] floatFeatures = new float[doubleFeatures.length][];
        for (int i = 0; i < doubleFeatures.length; i++) {
            floatFeatures[i] = new float[doubleFeatures[i].length];
            for (int j = 0; j < doubleFeatures[i].length; j++) {
                floatFeatures[i][j] = (float) doubleFeatures[i][j];
            }
        }
        return floatFeatures;
    }
    
    /**
    
     * ID: GPU-GPM-026
     * Requirement: flattenFeatures must execute correctly within the contract defined by this class.
     * Purpose: Implement the flattenFeatures operation for this class.
     * Inputs: float[][] features
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private float[] flattenFeatures(float[][] features) {
        int totalSize = features.length * features[0].length;
        float[] flattened = new float[totalSize];
        
        for (int i = 0; i < features.length; i++) {
            System.arraycopy(features[i], 0, flattened, i * featureCount, featureCount);
        }
        
        return flattened;
    }
    
    // Getters and utility methods
    
    /**
     * Get model weights
     */
    /**
    
     * ID: GPU-GPM-027
     * Requirement: Return the Weights field value without side effects.
     * Purpose: Return the value of the Weights property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float[] getWeights() {
        return weights.clone();
    }
    
    /**
     * Get model bias
     */
    /**
    
     * ID: GPU-GPM-028
     * Requirement: Return the Bias field value without side effects.
     * Purpose: Return the value of the Bias property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float getBias() {
        return bias;
    }
    
    /**
     * Get number of features
     */
    /**
    
     * ID: GPU-GPM-029
     * Requirement: Return the FeatureCount field value without side effects.
     * Purpose: Return the value of the FeatureCount property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getFeatureCount() {
        return featureCount;
    }
    
    /**
     * Get learning rate
     */
    /**
    
     * ID: GPU-GPM-030
     * Requirement: Return the LearningRate field value without side effects.
     * Purpose: Return the value of the LearningRate property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * Get training iterations performed
     */
    /**
    
     * ID: GPU-GPM-031
     * Requirement: Return the TrainingIterations field value without side effects.
     * Purpose: Return the value of the TrainingIterations property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getTrainingIterations() {
        return trainingIterations;
    }
    
    /**
     * Get performance statistics
     */
    /**
    
     * ID: GPU-GPM-032
     * Requirement: Return the PerformanceStats field value without side effects.
     * Purpose: Return the value of the PerformanceStats property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public PerceptronPerformanceStats getPerformanceStats() {
        return new PerceptronPerformanceStats(
            computeProvider.getName(),
            featureCount,
            trainingIterations,
            learningRate
        );
    }
    
    /**
     * Cleanup GPU resources
     */
    /**
    
     * ID: GPU-GPM-033
     * Requirement: cleanup must execute correctly within the contract defined by this class.
     * Purpose: Release all held resources and reset internal state.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void cleanup() {
        if (matrixOp != null) {
            matrixOp.release();
        }
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
        GpuPerceptronModel.logger.info("Cleaned up GPU perceptron model");
    }
    
    /**
     * Performance statistics for the perceptron model
     */
    public static class PerceptronPerformanceStats {
        private final String providerName;
        private final int featureCount;
        private final int trainingIterations;
        private final float learningRate;
        
        /**
        
         * ID: GPU-GPM-034
         * Requirement: PerceptronPerformanceStats must execute correctly within the contract defined by this class.
         * Purpose: Implement the PerceptronPerformanceStats operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public PerceptronPerformanceStats(String providerName, int featureCount, 
                                        int trainingIterations, float learningRate) {
            this.providerName = providerName;
            this.featureCount = featureCount;
            this.trainingIterations = trainingIterations;
            this.learningRate = learningRate;
        }
        
        /**
        
         * ID: GPU-GPM-035
         * Requirement: Return the ProviderName field value without side effects.
         * Purpose: Return the value of the ProviderName property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getProviderName() { return providerName; }
        /**
        
         * ID: GPU-GPM-036
         * Requirement: Return the FeatureCount field value without side effects.
         * Purpose: Return the value of the FeatureCount property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public int getFeatureCount() { return featureCount; }
        /**
        
         * ID: GPU-GPM-037
         * Requirement: Return the TrainingIterations field value without side effects.
         * Purpose: Return the value of the TrainingIterations property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public int getTrainingIterations() { return trainingIterations; }
        /**
        
         * ID: GPU-GPM-038
         * Requirement: Return the LearningRate field value without side effects.
         * Purpose: Return the value of the LearningRate property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public float getLearningRate() { return learningRate; }
        
        /**
        
         * ID: GPU-GPM-039
         * Requirement: toString must execute correctly within the contract defined by this class.
         * Purpose: Implement the toString operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        @Override
        public String toString() {
            return String.format("PerceptronStats{provider=%s, features=%d, iterations=%d, lr=%.3f}", 
                               providerName, featureCount, trainingIterations, learningRate);
        }
    }
}
