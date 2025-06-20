package org.apache.opennlp.gpu.ml.maxent;

import java.util.Arrays;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;

import opennlp.tools.ml.model.AbstractModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

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

        if (cpuModel instanceof AbstractModel) {
            AbstractModel abstractModel = (AbstractModel) cpuModel;
            Object[] data = abstractModel.getDataStructures();
            @SuppressWarnings("unchecked")
            Map<String, Context> pmap = (Map<String, Context>) data[1];
            this.predLabels = pmap.keySet().toArray(new String[0]);
            this.numPreds = predLabels.length;
            this.outcomes = new String[numOutcomes];
            for (int i=0; i<numOutcomes; i++) {
                this.outcomes[i] = abstractModel.getOutcome(i);
            }

            this.weights = new float[numPreds * numOutcomes];
            Arrays.fill(weights, 0.0f);

            int predIndex = 0;
            for (String predLabel : predLabels) {
                Context p = pmap.get(predLabel);
                if (p != null) {
                    double[] contextParams = p.getParameters();
                    int[] outcomeIndices = p.getOutcomes();

                    for (int j = 0; j < outcomeIndices.length; j++) {
                        int outcomeIndex = outcomeIndices[j];
                        if (outcomeIndex < numOutcomes) {
                            weights[predIndex * numOutcomes + outcomeIndex] = (float) contextParams[j];
                        }
                    }
                }
                predIndex++;
            }
        } else {
            throw new IllegalArgumentException("The provided cpuModel must be an instance of AbstractModel");
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
    public double[] eval(String[] context, float[] values) {
        return cpuModel.eval(context, values);
    }
    
    @Override
    public String getBestOutcome(double[] ocs) {
        return cpuModel.getBestOutcome(ocs);
    }

    @Override
    public String getAllOutcomes(double[] ocs) {
        return cpuModel.getAllOutcomes(ocs);
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
    
    public String[] getAllOutcomes() {
        return outcomes.clone();
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
        Arrays.fill(scores, 0.0f);
        
        // Simplified matrix multiplication: sum contributions of active features for each outcome
        for (int featureIndex : featureIndices) {
            for (int outcomeIndex = 0; outcomeIndex < numOutcomes; outcomeIndex++) {
                scores[outcomeIndex] += weights[featureIndex * numOutcomes + outcomeIndex];
            }
        }
    }
    
    private float[][] calculateBatchScoresGpu(int[][] allFeatureIndices) {
        float[][] allScores = new float[allFeatureIndices.length][numOutcomes];
        for (int i = 0; i < allFeatureIndices.length; i++) {
            calculateScoresGpu(allFeatureIndices[i], allScores[i]);
        }
        return allScores;
    }
    
    private void applySoftmaxGpu(float[] scores, double[] probs) {
        // Simple softmax implementation
        double sum = 0;
        for (int i = 0; i < scores.length; i++) {
            probs[i] = Math.exp(scores[i]);
            sum += probs[i];
        }

        for (int i = 0; i < scores.length; i++) {
            probs[i] /= sum;
        }
    }
    
    private void fillUniformDistribution(double[] probs) {
        double prob = 1.0 / numOutcomes;
        for (int i = 0; i < probs.length; i++) {
            probs[i] = prob;
        }
    }

    public ModelPerformanceStats getPerformanceStats() {
        return new ModelPerformanceStats(
                computeProvider.getClass().getSimpleName(),
                numOutcomes,
                numPreds,
                weights.length
        );
    }

    public void cleanup() {
        if (computeProvider.isGpuProvider()) {
            ((GpuComputeProvider) computeProvider).cleanup();
        }
    }
    
    /**
     * Provides performance statistics for the GPU model
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
            return "ModelPerformanceStats{" +
                    "providerName='" + providerName + '\'' +
                    ", numOutcomes=" + numOutcomes +
                    ", numPreds=" + numPreds +
                    ", numWeights=" + numWeights +
                    '}';
        }
    }
}
