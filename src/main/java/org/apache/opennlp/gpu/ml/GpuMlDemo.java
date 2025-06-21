package org.apache.opennlp.gpu.ml;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.gpu.ml.perceptron.GpuPerceptronModel;

import opennlp.tools.ml.model.MaxentModel;

/**
 * Demonstration of GPU-accelerated ML models
 * Shows how to use MaxEnt and Perceptron models with GPU acceleration
 */
public class GpuMlDemo {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuMlDemo.class);
    
    public static void main(String[] args) {
        GpuMlDemo.logger.info("Starting OpenNLP GPU ML Demo");
        
        try {
            // Demo GPU MaxEnt model
            GpuMlDemo.demoMaxentModel();
            
            // Demo GPU Perceptron model
            GpuMlDemo.demoPerceptronModel();
            
            GpuMlDemo.logger.info("GPU ML Demo completed successfully");
            
        } catch (Exception e) {
            GpuMlDemo.logger.error("Demo failed: " + e.getMessage(), e);
        }
    }
    
    private static void demoMaxentModel() {
        GpuMlDemo.logger.info("=== GPU MaxEnt Model Demo ===");
        
        // Create GPU configuration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // Create sample training data for MaxEnt model
        String[] outcomes = {"positive", "negative", "neutral"};
        String[] predLabels = {"word_good", "word_bad", "word_okay", "word_great", "word_terrible"};
        double[] parameters = GpuMlDemo.createSampleParameters(outcomes.length, predLabels.length);
        
        // Create CPU MaxEnt model using a dummy implementation
        MaxentModel cpuModel = new DummyMaxentModel(outcomes, predLabels, parameters, 1, 0.0);
        
        // Create GPU-accelerated MaxEnt model
        GpuMaxentModel gpuModel = new GpuMaxentModel(cpuModel, config);
        
        // Test single evaluation
        String[] context = {"word_good", "word_great"};
        double[] probs = gpuModel.eval(context);
        
        GpuMlDemo.logger.info("MaxEnt evaluation results:");
        for (int i = 0; i < outcomes.length; i++) {
            GpuMlDemo.logger.info("  " + outcomes[i] + ": " + String.format("%.4f", probs[i]));
        }
        
        // Test batch evaluation
        String[][] contexts = {
            {"word_good", "word_great"},
            {"word_bad", "word_terrible"},
            {"word_okay"}
        };
        
        double[][] batchProbs = gpuModel.evalBatch(contexts);
        GpuMlDemo.logger.info("Batch evaluation results:");
        for (int i = 0; i < contexts.length; i++) {
            GpuMlDemo.logger.info("  Context " + i + ": " + java.util.Arrays.toString(batchProbs[i]));
        }
        
        // Print performance stats
        GpuMlDemo.logger.info("MaxEnt Performance: " + gpuModel.getPerformanceStats());
        
        // Cleanup
        gpuModel.cleanup();
    }
    
    private static void demoPerceptronModel() {
        GpuMlDemo.logger.info("=== GPU Perceptron Model Demo ===");
        
        // Create GPU configuration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // Create GPU Perceptron model
        GpuPerceptronModel perceptron = new GpuPerceptronModel(config, 0.1f, 1000);
        
        // Generate sample training data
        int numSamples = 1000;
        int numFeatures = 2000; // Large enough for GPU acceleration
        float[][] features = GpuMlDemo.generateSampleFeatures(numSamples, numFeatures);
        int[] labels = GpuMlDemo.generateSampleLabels(numSamples, features);
        
        GpuMlDemo.logger.info("Training perceptron with " + numSamples + " samples and " + numFeatures + " features");
        
        // Train the model
        long startTime = System.currentTimeMillis();
        perceptron.train(features, labels);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        GpuMlDemo.logger.info("Training completed in " + trainingTime + "ms");
        
        // Test prediction
        float[] testSample = features[0];
        int prediction = perceptron.predict(testSample);
        float decisionValue = perceptron.decisionFunction(testSample);
        
        GpuMlDemo.logger.info("Test prediction: " + prediction + " (decision value: " + String.format("%.4f", decisionValue) + ")");
        
        // Test batch prediction
        float[][] testBatch = java.util.Arrays.copyOfRange(features, 0, 10);
        int[] batchPredictions = perceptron.predictBatch(testBatch);
        GpuMlDemo.logger.info("Batch predictions: " + java.util.Arrays.toString(batchPredictions));
        
        // Print performance stats
        GpuMlDemo.logger.info("Perceptron Performance: " + perceptron.getPerformanceStats());
        
        // Cleanup
        perceptron.cleanup();
    }
    
    // Helper methods for generating sample data
    
    private static double[] createSampleParameters(int numOutcomes, int numPreds) {
        double[] params = new double[numOutcomes * numPreds];
        for (int i = 0; i < params.length; i++) {
            params[i] = Math.random() * 2.0 - 1.0; // Random values between -1 and 1
        }
        return params;
    }
    
    private static float[][] generateSampleFeatures(int numSamples, int numFeatures) {
        float[][] features = new float[numSamples][numFeatures];
        
        for (int i = 0; i < numSamples; i++) {
            for (int j = 0; j < numFeatures; j++) {
                // Generate features with some structure for meaningful classification
                if (j < numFeatures / 2) {
                    features[i][j] = (float) (Math.random() + (i % 2)); // Class-dependent features
                } else {
                    features[i][j] = (float) (Math.random() * 0.1); // Noise features
                }
            }
        }
        
        return features;
    }
    
    private static int[] generateSampleLabels(int numSamples, float[][] features) {
        int[] labels = new int[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Simple classification rule based on feature sum
            float sum = 0;
            int relevantFeatures = features[i].length / 2;
            for (int j = 0; j < relevantFeatures; j++) {
                sum += features[i][j];
            }
            labels[i] = sum > relevantFeatures ? 1 : 0;
        }
        
        return labels;
    }
    
    // Dummy implementation of MaxentModel for demo purposes
    private static class DummyMaxentModel implements MaxentModel {
    
        private String[] outcomes;
        private String[] predLabels;
        private double[] parameters;
        private int iterations;
        private double smoothing;
    
        public DummyMaxentModel(String[] outcomes, String[] predLabels, double[] parameters, int iterations, double smoothing) {
            this.outcomes = outcomes;
            this.predLabels = predLabels;
            this.parameters = parameters;
            this.iterations = iterations;
            this.smoothing = smoothing;
        }
    
        @Override
        public double[] eval(String[] context) {
            double[] probs = new double[outcomes.length];
            // Return uniform probability distribution for demonstration
            for (int i = 0; i < probs.length; i++) {
                probs[i] = 1.0 / outcomes.length;
            }
            return probs;
        }
        
        @Override
        public double[] eval(String[] context, double[] probs) {
            // Fill the provided array with uniform probabilities
            for (int i = 0; i < probs.length; i++) {
                probs[i] = 1.0 / outcomes.length;
            }
            return probs;
        }
        
        @Override
        public double[] eval(String[] context, float[] probs) {
            // Convert float array to double array and fill with uniform probabilities
            double[] doubleProbs = new double[probs.length];
            for (int i = 0; i < probs.length; i++) {
                doubleProbs[i] = 1.0 / outcomes.length;
            }
            return doubleProbs;
        }
    
        @Override
        public String getBestOutcome(double[] outcomes) {
            if (outcomes.length == 0) return null;
            int bestIndex = 0;
            for (int i = 1; i < outcomes.length; i++) {
                if (outcomes[i] > outcomes[bestIndex]) {
                    bestIndex = i;
                }
            }
            return this.outcomes[bestIndex];
        }
        
        @Override
        public String getAllOutcomes(double[] outcomes) {
            // Return all outcomes as a space-separated string
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < this.outcomes.length; i++) {
                if (i > 0) sb.append(" ");
                sb.append(this.outcomes[i]);
            }
            return sb.toString();
        }
        
        @Override
        public String getOutcome(int index) {
            if (index >= 0 && index < outcomes.length) {
                return outcomes[index];
            }
            return null;
        }
        
        @Override
        public int getIndex(String outcome) {
            for (int i = 0; i < outcomes.length; i++) {
                if (outcomes[i].equals(outcome)) {
                    return i;
                }
            }
            return -1;
        }
    
        @Override
        public int getNumOutcomes() {
            return outcomes.length;
        }
    }
}
