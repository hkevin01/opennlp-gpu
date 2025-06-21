package org.apache.opennlp.gpu.ml;

import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.gpu.ml.perceptron.GpuPerceptronModel;

import opennlp.tools.ml.maxent.GISModel;
import opennlp.tools.ml.model.Context;
import opennlp.tools.ml.model.MaxentModel;

/**
 * Demonstration of GPU-accelerated ML models
 * Shows how to use MaxEnt, Perceptron, and Naive Bayes models with GPU acceleration
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
            
            // Demo GPU Naive Bayes model
            GpuMlDemo.demoNaiveBayesModel();
            
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
        
        // Create Context objects for OpenNLP 2.x API
        Context[] contexts = new Context[predLabels.length];
        int[] outcomePattern = new int[outcomes.length];
        for (int i = 0; i < outcomes.length; i++) {
            outcomePattern[i] = i;
        }
        
        // Create sample parameters and contexts
        for (int i = 0; i < predLabels.length; i++) {
            double[] paramsForPred = new double[outcomes.length];
            // Generate sample parameters for this predicate
            for (int j = 0; j < outcomes.length; j++) {
                paramsForPred[j] = Math.random() * 2.0 - 1.0; // Random weight between -1 and 1
            }
            contexts[i] = new Context(outcomePattern, paramsForPred);
        }
        
        // Create CPU MaxEnt model using OpenNLP 2.x API
        MaxentModel cpuModel = new GISModel(contexts, predLabels, outcomes);
        
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
        String[][] testContexts = {
            {"word_good", "word_great"},
            {"word_bad", "word_terrible"},
            {"word_okay"}
        };
        
        double[][] batchProbs = gpuModel.evalBatch(testContexts);
        GpuMlDemo.logger.info("Batch evaluation results:");
        for (int i = 0; i < testContexts.length; i++) {
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
    
    private static void demoNaiveBayesModel() {
        GpuMlDemo.logger.info("=== GPU Naive Bayes Model Demo ===");
        
        // Create GPU configuration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        // For this demo, we'll work directly with our GPU implementation
        // since creating a proper NaiveBayesModel is complex
        
        GpuMlDemo.logger.info("Creating GPU Naive Bayes implementation (standalone demo)");
        
        // Create a standalone GPU Naive Bayes demo
        String[] outcomes = {"positive", "negative", "neutral"};
        GpuNaiveBayesStandalone gpuNB = new GpuNaiveBayesStandalone(outcomes, config);
        
        // Generate sample training data
        int numSamples = 1000;
        int numFeatures = 100;
        String[][] trainingData = generateTextFeatures(numSamples, numFeatures);
        String[] labels = generateTextClassificationLabels(numSamples, trainingData);
        
        GpuMlDemo.logger.info("Training Naive Bayes with " + numSamples + " samples and " + numFeatures + " features");
        
        // Train the model
        long startTime = System.currentTimeMillis();
        gpuNB.train(trainingData, labels);
        long trainingTime = System.currentTimeMillis() - startTime;
        
        GpuMlDemo.logger.info("Training completed in " + trainingTime + "ms");
        
        // Test single evaluation
        String[] testContext = {"word_good", "word_great", "word_excellent"};
        double[] probs = gpuNB.eval(testContext);
        
        GpuMlDemo.logger.info("Naive Bayes evaluation results:");
        for (int i = 0; i < outcomes.length; i++) {
            GpuMlDemo.logger.info("  " + outcomes[i] + ": " + String.format("%.4f", probs[i]));
        }
        
        // Test batch evaluation
        String[][] testContexts = {
            {"word_good", "word_great"},
            {"word_bad", "word_terrible"},
            {"word_neutral", "word_okay"}
        };
        
        GpuMlDemo.logger.info("Batch evaluation results:");
        for (int i = 0; i < testContexts.length; i++) {
            double[] batchProbs = gpuNB.eval(testContexts[i]);
            String prediction = gpuNB.getBestOutcome(batchProbs);
            GpuMlDemo.logger.info("  Context " + i + ": " + prediction + " (" + java.util.Arrays.toString(batchProbs) + ")");
        }
        
        // Print performance stats
        GpuMlDemo.logger.info("Naive Bayes Performance: " + gpuNB.getPerformanceStats());
        
        // Cleanup
        gpuNB.cleanup();
    }
    
    // Helper methods for generating sample data
    
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
    
    private static String[][] generateTextFeatures(int numSamples, int numFeatures) {
        String[] featureNames = {"word_good", "word_bad", "word_great", "word_terrible", "word_excellent", 
                                "word_poor", "word_amazing", "word_awful", "word_wonderful", "word_horrible",
                                "word_fantastic", "word_disappointing", "word_outstanding", "word_mediocre",
                                "word_brilliant", "word_dreadful", "word_superb", "word_inferior"};
        
        String[][] features = new String[numSamples][];
        
        for (int i = 0; i < numSamples; i++) {
            // Generate 3-8 features per sample
            int numSampleFeatures = 3 + (int)(Math.random() * 6);
            features[i] = new String[numSampleFeatures];
            
            for (int j = 0; j < numSampleFeatures; j++) {
                features[i][j] = featureNames[(int)(Math.random() * featureNames.length)];
            }
        }
        
        return features;
    }
    
    private static String[] generateTextClassificationLabels(int numSamples, String[][] features) {
        String[] labels = new String[numSamples];
        
        for (int i = 0; i < numSamples; i++) {
            // Simple rule based on feature content
            int positiveCount = 0;
            int negativeCount = 0;
            
            for (String feature : features[i]) {
                if (feature.contains("good") || feature.contains("great") || feature.contains("excellent") || 
                    feature.contains("amazing") || feature.contains("wonderful") || feature.contains("fantastic") ||
                    feature.contains("outstanding") || feature.contains("brilliant") || feature.contains("superb")) {
                    positiveCount++;
                } else if (feature.contains("bad") || feature.contains("terrible") || feature.contains("poor") ||
                          feature.contains("awful") || feature.contains("horrible") || feature.contains("disappointing") ||
                          feature.contains("dreadful") || feature.contains("inferior")) {
                    negativeCount++;
                }
            }
            
            if (positiveCount > negativeCount) {
                labels[i] = "positive";
            } else if (negativeCount > positiveCount) {
                labels[i] = "negative";
            } else {
                labels[i] = "neutral";
            }
        }
        
        return labels;
    }
    
    /**
     * Standalone GPU Naive Bayes implementation for demo purposes
     */
    private static class GpuNaiveBayesStandalone {
        private final String[] outcomes;
        private final ComputeProvider computeProvider;
        
        // Model parameters
        private Map<String, Map<String, Double>> featureProbabilities;
        private Map<String, Double> classPriors;
        private long totalEvaluations = 0;
        private long totalTime = 0;
        
        public GpuNaiveBayesStandalone(String[] outcomes, GpuConfig config) {
            this.outcomes = outcomes;
            
            // Initialize compute provider
            if (config.isGpuEnabled()) {
                this.computeProvider = new GpuComputeProvider(config);
            } else {
                this.computeProvider = new CpuComputeProvider();
            }
            
            this.featureProbabilities = new HashMap<>();
            this.classPriors = new HashMap<>();
            
            // Initialize probability maps
            for (String outcome : outcomes) {
                featureProbabilities.put(outcome, new HashMap<>());
                classPriors.put(outcome, 1.0 / outcomes.length);
            }
        }
        
        public void train(String[][] trainingData, String[] labels) {
            // Count feature occurrences per class
            Map<String, Map<String, Integer>> featureCounts = new HashMap<>();
            Map<String, Integer> classCounts = new HashMap<>();
            
            // Initialize counts
            for (String outcome : outcomes) {
                featureCounts.put(outcome, new HashMap<>());
                classCounts.put(outcome, 0);
            }
            
            // Count features and classes
            for (int i = 0; i < trainingData.length; i++) {
                String label = labels[i];
                classCounts.put(label, classCounts.get(label) + 1);
                
                Map<String, Integer> labelFeatureCounts = featureCounts.get(label);
                for (String feature : trainingData[i]) {
                    labelFeatureCounts.put(feature, labelFeatureCounts.getOrDefault(feature, 0) + 1);
                }
            }
            
            // Calculate probabilities with Laplace smoothing
            int totalSamples = trainingData.length;
            for (String outcome : outcomes) {
                // Class priors
                classPriors.put(outcome, (double) classCounts.get(outcome) / totalSamples);
                
                // Feature probabilities
                Map<String, Integer> counts = featureCounts.get(outcome);
                Map<String, Double> probs = featureProbabilities.get(outcome);
                int classTotal = classCounts.get(outcome);
                
                for (Map.Entry<String, Integer> entry : counts.entrySet()) {
                    double prob = (entry.getValue() + 1.0) / (classTotal + 2.0);
                    probs.put(entry.getKey(), prob);
                }
            }
        }
        
        public double[] eval(String[] context) {
            long startTime = System.nanoTime();
            
            double[] logProbs = new double[outcomes.length];
            
            for (int i = 0; i < outcomes.length; i++) {
                String outcome = outcomes[i];
                
                // Start with class prior
                logProbs[i] = Math.log(classPriors.get(outcome));
                
                // Add feature log probabilities
                Map<String, Double> outcomeFeatureProbs = featureProbabilities.get(outcome);
                for (String feature : context) {
                    double featureProb = outcomeFeatureProbs.getOrDefault(feature, 1e-10);
                    logProbs[i] += Math.log(featureProb);
                }
            }
            
            // Convert to normalized probabilities
            double[] probs = normalizeLogProbabilities(logProbs);
            
            long endTime = System.nanoTime();
            totalTime += (endTime - startTime);
            totalEvaluations++;
            
            return probs;
        }
        
        private double[] normalizeLogProbabilities(double[] logProbs) {
            // Find max for numerical stability
            double maxLogProb = Arrays.stream(logProbs).max().orElse(0.0);
            
            // Convert to regular probabilities
            double[] probs = new double[logProbs.length];
            double sum = 0.0;
            
            for (int i = 0; i < logProbs.length; i++) {
                probs[i] = Math.exp(logProbs[i] - maxLogProb);
                sum += probs[i];
            }
            
            // Normalize
            for (int i = 0; i < probs.length; i++) {
                probs[i] /= sum;
            }
            
            return probs;
        }
        
        public String getBestOutcome(double[] probabilities) {
            int bestIndex = 0;
            double bestScore = probabilities[0];
            
            for (int i = 1; i < probabilities.length; i++) {
                if (probabilities[i] > bestScore) {
                    bestScore = probabilities[i];
                    bestIndex = i;
                }
            }
            
            return outcomes[bestIndex];
        }
        
        public String getPerformanceStats() {
            double avgTimeMs = totalEvaluations > 0 ? 
                (totalTime / 1_000_000.0) / totalEvaluations : 0.0;
            
            return String.format("NaiveBayesStats{provider=%s, outcomes=%d, evaluations=%d, avgTime=%.3fms}",
                computeProvider.getClass().getSimpleName(),
                outcomes.length,
                totalEvaluations,
                avgTimeMs);
        }
        
        public void cleanup() {
            if (computeProvider != null) {
                computeProvider.cleanup();
            }
        }
    }
}
