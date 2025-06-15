package org.apache.opennlp.gpu.ml.perceptron;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;

/**
 * GPU-accelerated perceptron model implementation
 * Provides hardware acceleration for perceptron training and inference with advanced features
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
    
    public GpuPerceptronModel(GpuConfig config) {
        this(config, GpuPerceptronModel.DEFAULT_LEARNING_RATE, GpuPerceptronModel.DEFAULT_MAX_ITERATIONS);
    }
    
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
    public void train(double[][] features, int[] labels) {
        float[][] floatFeatures = convertToFloat(features);
        train(floatFeatures, labels);
    }
    
    /**
     * Predict using the perceptron model
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
    
    private int predictOnGpu(float[] features) {
        float decision = decisionFunctionGpu(features);
        return decision >= 0 ? 1 : 0;
    }
    
    private int predictOnCpu(float[] features) {
        float decision = decisionFunctionCpu(features);
        return decision >= 0 ? 1 : 0;
    }
    
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
    
    private float decisionFunctionCpu(float[] features) {
        float sum = bias;
        for (int i = 0; i < featureCount; i++) {
            sum += weights[i] * features[i];
        }
        return sum;
    }
    
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
    
    private void updateWeightsCpu(float[] features, float delta) {
        float adjustment = learningRate * delta;
        for (int i = 0; i < featureCount; i++) {
            weights[i] += adjustment * features[i];
        }
        bias += adjustment;
    }
    
    // Helper methods
    
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
    
    private boolean shouldUseGpu(float[][] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= GpuPerceptronModel.MIN_SAMPLES_FOR_GPU &&
               features[0].length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
    private boolean shouldUseGpu(float[] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
    private boolean shouldUseGpuBatch(float[][] features) {
        return computeProvider.isGpuProvider() && 
               features.length >= 10 && // Minimum batch size
               features[0].length >= GpuPerceptronModel.MIN_FEATURES_FOR_GPU;
    }
    
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
    public float[] getWeights() {
        return weights.clone();
    }
    
    /**
     * Get model bias
     */
    public float getBias() {
        return bias;
    }
    
    /**
     * Get number of features
     */
    public int getFeatureCount() {
        return featureCount;
    }
    
    /**
     * Get learning rate
     */
    public float getLearningRate() {
        return learningRate;
    }
    
    /**
     * Get training iterations performed
     */
    public int getTrainingIterations() {
        return trainingIterations;
    }
    
    /**
     * Get performance statistics
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
        
        public PerceptronPerformanceStats(String providerName, int featureCount, 
                                        int trainingIterations, float learningRate) {
            this.providerName = providerName;
            this.featureCount = featureCount;
            this.trainingIterations = trainingIterations;
            this.learningRate = learningRate;
        }
        
        public String getProviderName() { return providerName; }
        public int getFeatureCount() { return featureCount; }
        public int getTrainingIterations() { return trainingIterations; }
        public float getLearningRate() { return learningRate; }
        
        @Override
        public String toString() {
            return String.format("PerceptronStats{provider=%s, features=%d, iterations=%d, lr=%.3f}", 
                               providerName, featureCount, trainingIterations, learningRate);
        }
    }
}
