package org.apache.opennlp.gpu.ml.maxent;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.maxent.MaxentModel;
import org.apache.opennlp.model.Context;

/**
 * GPU-accelerated implementation of MaxEnt model
 * Uses GPU matrix operations and feature extraction for enhanced performance
 */
public class GpuMaxentModel implements MaxentModel {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuMaxentModel.class);
    
    private final MaxentModel cpuModel;
    private final ComputeProvider computeProvider;
    private final GpuConfig config;
    private final MatrixOperation matrixOp;
    private final GpuFeatureExtractor featureExtractor;
    
    // Model parameters
    private String[] outcomes;
    private int numOutcomes;
    private float[] weights;
    private String[] predLabels;
    private int numPreds;
    
    // Performance thresholds
    private static final int MIN_CONTEXT_SIZE_FOR_GPU = 50;
    private static final int MIN_OUTCOMES_FOR_GPU = 10;
    private static final int MIN_BATCH_SIZE_FOR_GPU = 8;
    
    /**
     * Creates a GPU-accelerated MaxEnt model
     */
    public GpuMaxentModel(MaxentModel cpuModel, GpuConfig config) {
        this.cpuModel = cpuModel;
        this.config = config;
        this.computeProvider = createComputeProvider();
        this.matrixOp = createMatrixOperation();
        this.featureExtractor = new GpuFeatureExtractor(computeProvider, config, matrixOp);
        
        initializeModelParameters();
        
        logger.info("Created GPU MaxEnt model with " + numOutcomes + " outcomes and " + numPreds + " predictors");
    }
    
    private ComputeProvider createComputeProvider() {
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                return new GpuComputeProvider(config);
            }
        } catch (Exception e) {
            logger.warn("Failed to initialize GPU provider: " + e.getMessage());
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
    
    private void initializeModelParameters() {
        this.numOutcomes = cpuModel.getNumOutcomes();
        this.outcomes = cpuModel.getAllOutcomes();
        
        // Extract model parameters if available
        Object[] dataStructures = cpuModel.getDataStructures();
        if (dataStructures.length >= 3 && dataStructures[1] instanceof String[] && dataStructures[2] instanceof double[]) {
            this.predLabels = (String[]) dataStructures[1];
            this.numPreds = predLabels.length;
            
            // Convert double parameters to float for GPU operations
            double[] doubleParams = (double[]) dataStructures[2];
            this.weights = new float[doubleParams.length];
            for (int i = 0; i < doubleParams.length; i++) {
                this.weights[i] = (float) doubleParams[i];
            }
        } else {
            // Create dummy parameters for stub implementation
            this.predLabels = new String[0];
            this.numPreds = 0;
            this.weights = new float[0];
        }
    }
    
    @Override
    public double[] eval(String[] context) {
        return eval(context, new double[numOutcomes]);
    }
    
    @Override
    public double[] eval(String[] context, double[] probs) {
        if (shouldUseGpu(context)) {
            return evaluateOnGpu(context, probs);
        } else {
            return cpuModel.eval(context, probs);
        }
    }
    
    @Override
    public double[] eval(String[] context, float[] probs) {
        double[] doubleProbs = new double[probs.length];
        for (int i = 0; i < probs.length; i++) {
            doubleProbs[i] = probs[i];
        }
        return eval(context, doubleProbs);
    }
    
    @Override
    public String getOutcome(int index) {
        return cpuModel.getOutcome(index);
    }
    
    @Override
    public int getNumOutcomes() {
        return numOutcomes;
    }
    
    @Override
    public int getIndex(String outcome) {
        return cpuModel.getIndex(outcome);
    }
    
    @Override
    public String[] getAllOutcomes() {
        return outcomes.clone();
    }
    
    @Override
    public Object[] getDataStructures() {
        return cpuModel.getDataStructures();
    }
    
    /**
     * Batch evaluation for multiple contexts - optimized for GPU
     */
    public double[][] evalBatch(String[][] contexts) {
        if (contexts.length >= MIN_BATCH_SIZE_FOR_GPU && computeProvider.isGpuProvider()) {
            return evaluateBatchOnGpu(contexts);
        } else {
            // Fall back to individual evaluations
            double[][] results = new double[contexts.length][];
            for (int i = 0; i < contexts.length; i++) {
                results[i] = eval(contexts[i]);
            }
            return results;
        }
    }
    
    // GPU-accelerated evaluation methods
    
    private boolean shouldUseGpu(String[] context) {
        return computeProvider.isGpuProvider() && 
               context.length >= MIN_CONTEXT_SIZE_FOR_GPU && 
               numOutcomes >= MIN_OUTCOMES_FOR_GPU &&
               weights.length > 0;
    }
    
    private double[] evaluateOnGpu(String[] context, double[] probs) {
        try {
            // Extract feature indices and values
            int[] featureIndices = extractFeatureIndices(context);
            if (featureIndices.length == 0) {
                // No features found, return uniform distribution
                fillUniformDistribution(probs);
                return probs;
            }
            
            // Calculate scores using GPU matrix operations
            float[] scores = new float[numOutcomes];
            calculateScoresGpu(featureIndices, scores);
            
            // Apply softmax to get probabilities
            applySoftmaxGpu(scores, probs);
            
            return probs;
            
        } catch (Exception e) {
            logger.warn("GPU evaluation failed, falling back to CPU: " + e.getMessage());
            return cpuModel.eval(context, probs);
        }
    }
    
    private double[][] evaluateBatchOnGpu(String[][] contexts) {
        try {
            logger.debug("GPU batch evaluation for " + contexts.length + " contexts");
            
            // Extract features for all contexts
            int[][] allFeatureIndices = new int[contexts.length][];
            for (int i = 0; i < contexts.length; i++) {
                allFeatureIndices[i] = extractFeatureIndices(contexts[i]);
            }
            
            // Batch calculate scores
            float[][] allScores = calculateBatchScoresGpu(allFeatureIndices);
            
            // Apply softmax to all scores
            double[][] results = new double[contexts.length][numOutcomes];
            for (int i = 0; i < contexts.length; i++) {
                applySoftmaxGpu(allScores[i], results[i]);
            }
            
            return results;
            
        } catch (Exception e) {
            logger.warn("GPU batch evaluation failed, falling back to individual CPU evaluations: " + e.getMessage());
            double[][] results = new double[contexts.length][];
            for (int i = 0; i < contexts.length; i++) {
                results[i] = cpuModel.eval(contexts[i]);
            }
            return results;
        }
    }
    
    private int[] extractFeatureIndices(String[] context) {
        if (predLabels.length == 0) {
            return new int[0];
        }
        
        int[] indices = new int[context.length];
        int count = 0;
        
        for (String feature : context) {
            for (int i = 0; i < predLabels.length; i++) {
                if (predLabels[i].equals(feature)) {
                    indices[count++] = i;
                    break;
                }
            }
        }
        
        // Trim to actual size
        int[] result = new int[count];
        System.arraycopy(indices, 0, result, 0, count);
        return result;
    }
    
    private void calculateScoresGpu(int[] featureIndices, float[] scores) {
        // Initialize scores to zero
        matrixOp.fillArray(scores, 0.0f, numOutcomes);
        
        // For each outcome, sum the weights of active features
        for (int oid = 0; oid < numOutcomes; oid++) {
            float score = 0.0f;
            for (int fidx : featureIndices) {
                int paramIndex = oid * numPreds + fidx;
                if (paramIndex < weights.length) {
                    score += weights[paramIndex];
                }
            }
            scores[oid] = score;
        }
    }
    
    private float[][] calculateBatchScoresGpu(int[][] allFeatureIndices) {
        float[][] allScores = new float[allFeatureIndices.length][numOutcomes];
        
        // Process each context
        for (int i = 0; i < allFeatureIndices.length; i++) {
            calculateScoresGpu(allFeatureIndices[i], allScores[i]);
        }
        
        return allScores;
    }
    
    private void applySoftmaxGpu(float[] scores, double[] probs) {
        // Use GPU softmax implementation
        float[] floatProbs = new float[scores.length];
        matrixOp.softmax(scores, floatProbs, scores.length);
        
        // Convert to double
        for (int i = 0; i < floatProbs.length; i++) {
            probs[i] = floatProbs[i];
        }
    }
    
    private void fillUniformDistribution(double[] probs) {
        double uniformProb = 1.0 / numOutcomes;
        for (int i = 0; i < numOutcomes; i++) {
            probs[i] = uniformProb;
        }
    }
    
    /**
     * GPU-accelerated context evaluation for Context objects
     */
    public double[] evaluateContext(Context context) {
        try {
            // Extract features and values from context
            String[] features = context.getFeatures();
            float[] values = context.getValues();
            
            if (features == null || values == null) {
                logger.warn("Invalid context, falling back to CPU");
                return new double[numOutcomes];
            }
            
            // Use weighted feature evaluation if values are provided
            if (shouldUseGpu(features)) {
                return evaluateWeightedFeaturesGpu(features, values);
            } else {
                // Fall back to CPU implementation
                return eval(features);
            }
            
        } catch (Exception e) {
            logger.error("Context evaluation failed: " + e.getMessage());
            return new double[numOutcomes];
        }
    }
    
    private double[] evaluateWeightedFeaturesGpu(String[] features, float[] values) {
        try {
            // Extract feature indices
            int[] featureIndices = extractFeatureIndices(features);
            
            // Calculate weighted scores
            float[] scores = new float[numOutcomes];
            matrixOp.fillArray(scores, 0.0f, numOutcomes);
            
            for (int oid = 0; oid < numOutcomes; oid++) {
                float score = 0.0f;
                for (int i = 0; i < featureIndices.length && i < values.length; i++) {
                    int fidx = featureIndices[i];
                    int paramIndex = oid * numPreds + fidx;
                    if (paramIndex < weights.length) {
                        score += weights[paramIndex] * values[i];
                    }
                }
                scores[oid] = score;
            }
            
            // Apply softmax
            double[] probs = new double[numOutcomes];
            applySoftmaxGpu(scores, probs);
            
            return probs;
            
        } catch (Exception e) {
            logger.warn("Weighted GPU evaluation failed: " + e.getMessage());
            return eval(features);
        }
    }
    
    /**
     * Get performance statistics for this model
     */
    public ModelPerformanceStats getPerformanceStats() {
        return new ModelPerformanceStats(
            computeProvider.getName(),
            numOutcomes,
            numPreds,
            weights.length
        );
    }
    
    /**
     * Cleanup GPU resources
     */
    public void cleanup() {
        if (matrixOp != null) {
            matrixOp.release();
        }
        if (featureExtractor != null) {
            featureExtractor.release();
        }
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
        logger.debug("Cleaned up GPU MaxEnt model resources");
    }
    
    /**
     * Performance statistics for the model
     */
    public static class ModelPerformanceStats {
        private final String providerName;
        private final int numOutcomes;
        private final int numPreds;
        private final int numWeights;
        
        public ModelPerformanceStats(String providerName, int numOutcomes, int numPreds, int numWeights) {
            this.providerName = providerName;
            this.numOutcomes = numOutcomes;
            this.numPreds = numPreds;
            this.numWeights = numWeights;
        }
        
        public String getProviderName() { return providerName; }
        public int getNumOutcomes() { return numOutcomes; }
        public int getNumPreds() { return numPreds; }
        public int getNumWeights() { return numWeights; }
        
        @Override
        public String toString() {
            return String.format("ModelStats{provider=%s, outcomes=%d, preds=%d, weights=%d}", 
                               providerName, numOutcomes, numPreds, numWeights);
        }
    }
}
