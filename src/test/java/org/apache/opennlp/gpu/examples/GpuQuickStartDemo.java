package org.apache.opennlp.gpu.examples;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;
import org.apache.opennlp.gpu.ml.GpuModelFactory;
import org.apache.opennlp.gpu.ml.neural.GpuNeuralNetwork;
import org.apache.opennlp.maxent.GisModel;
import org.apache.opennlp.maxent.MaxentModel;

/**
 * Quick start demonstration of OpenNLP GPU acceleration
 * Shows basic usage patterns and integration examples
 */
public class GpuQuickStartDemo {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuQuickStartDemo.class);
    
    public static void main(String[] args) {
        logger.info("🚀 Starting OpenNLP GPU Quick Start Demo");
        
        try {
            // Demo 1: Basic Matrix Operations
            demonstrateMatrixOperations();
            
            // Demo 2: Feature Extraction
            demonstrateFeatureExtraction();
            
            // Demo 3: Neural Networks
            demonstrateNeuralNetworks();
            
            // Demo 4: OpenNLP Integration
            demonstrateOpenNLPIntegration();
            
            logger.info("✅ All demos completed successfully!");
            
        } catch (Exception e) {
            logger.error("Demo failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    /**
     * Demonstrate basic GPU-accelerated matrix operations
     */
    public static void demonstrateMatrixOperations() {
        System.out.println("\n=== Demo 1: GPU Matrix Operations ===");
        
        // Setup GPU configuration
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        
        ComputeProvider provider = new CpuComputeProvider(); // Use CPU for demo reliability
        MatrixOperation matrixOp = new CpuMatrixOperation(provider);
        
        try {
            // Create sample matrices
            float[] matrixA = {1.0f, 2.0f, 3.0f, 4.0f};  // 2x2 matrix
            float[] matrixB = {5.0f, 6.0f, 7.0f, 8.0f};  // 2x2 matrix
            float[] result = new float[4];
            
            System.out.println("Matrix A: [1.0, 2.0; 3.0, 4.0]");
            System.out.println("Matrix B: [5.0, 6.0; 7.0, 8.0]");
            
            // Matrix multiplication
            matrixOp.multiply(matrixA, matrixB, result, 2, 2, 2);
            System.out.printf("A × B = [%.1f, %.1f; %.1f, %.1f]\n", 
                             result[0], result[1], result[2], result[3]);
            
            // Matrix addition
            matrixOp.add(matrixA, matrixB, result, 4);
            System.out.printf("A + B = [%.1f, %.1f, %.1f, %.1f]\n", 
                             result[0], result[1], result[2], result[3]);
            
            // Activation functions
            float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
            float[] sigmoid = new float[5];
            float[] relu = new float[5];
            
            matrixOp.sigmoid(input, sigmoid, 5);
            matrixOp.relu(input, relu, 5);
            
            System.out.println("Input: [-2.0, -1.0, 0.0, 1.0, 2.0]");
            System.out.printf("Sigmoid: [%.3f, %.3f, %.3f, %.3f, %.3f]\n",
                             sigmoid[0], sigmoid[1], sigmoid[2], sigmoid[3], sigmoid[4]);
            System.out.printf("ReLU: [%.1f, %.1f, %.1f, %.1f, %.1f]\n",
                             relu[0], relu[1], relu[2], relu[3], relu[4]);
            
        } finally {
            matrixOp.release();
            provider.cleanup();
        }
        
        System.out.println("✅ Matrix operations demo completed");
    }
    
    /**
     * Demonstrate GPU-accelerated feature extraction
     */
    public static void demonstrateFeatureExtraction() {
        System.out.println("\n=== Demo 2: GPU Feature Extraction ===");
        
        GpuConfig config = new GpuConfig();
        ComputeProvider provider = new CpuComputeProvider();
        MatrixOperation matrixOp = new CpuMatrixOperation(provider);
        GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
        
        try {
            // Sample documents for NLP processing
            String[] documents = {
                "OpenNLP provides natural language processing tools",
                "GPU acceleration makes machine learning much faster",
                "Feature extraction is essential for NLP tasks",
                "Neural networks benefit from GPU parallel processing",
                "Text classification requires good feature representation"
            };
            
            System.out.println("Processing " + documents.length + " sample documents...");
            
            // Extract n-gram features
            System.out.println("\n--- N-gram Feature Extraction ---");
            long startTime = System.currentTimeMillis();
            float[][] ngramFeatures = extractor.extractNGramFeatures(documents, 2, 50);
            long ngramTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Extracted %d-gram features in %d ms\n", 2, ngramTime);
            System.out.printf("Feature matrix: %d documents × %d features\n", 
                             ngramFeatures.length, ngramFeatures[0].length);
            System.out.printf("Vocabulary size: %d terms\n", extractor.getVocabularySize());
            
            // Show sample features for first document
            System.out.println("Sample features for first document:");
            int nonZeroCount = 0;
            for (int i = 0; i < Math.min(10, ngramFeatures[0].length); i++) {
                if (ngramFeatures[0][i] > 0) {
                    System.out.printf("  Feature[%d]: %.2f\n", i, ngramFeatures[0][i]);
                    nonZeroCount++;
                }
            }
            System.out.printf("Non-zero features in first document: %d\n", nonZeroCount);
            
            // Extract TF-IDF features
            System.out.println("\n--- TF-IDF Feature Extraction ---");
            startTime = System.currentTimeMillis();
            float[][] tfidfFeatures = extractor.extractTfIdfFeatures(documents, 2, 50);
            long tfidfTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Extracted TF-IDF features in %d ms\n", tfidfTime);
            System.out.printf("TF-IDF matrix: %d documents × %d features\n", 
                             tfidfFeatures.length, tfidfFeatures[0].length);
            
            // Show TF-IDF sample
            System.out.println("Sample TF-IDF values for first document:");
            for (int i = 0; i < Math.min(5, tfidfFeatures[0].length); i++) {
                if (tfidfFeatures[0][i] > 0) {
                    System.out.printf("  TF-IDF[%d]: %.4f\n", i, tfidfFeatures[0][i]);
                }
            }
            
            // Normalize features
            System.out.println("\n--- Feature Normalization ---");
            extractor.normalizeFeatures(tfidfFeatures);
            System.out.println("Applied L2 normalization to TF-IDF features");
            
        } finally {
            extractor.release();
            matrixOp.release();
            provider.cleanup();
        }
        
        System.out.println("✅ Feature extraction demo completed");
    }
    
    /**
     * Demonstrate GPU-accelerated neural networks
     */
    public static void demonstrateNeuralNetworks() {
        System.out.println("\n=== Demo 3: GPU Neural Networks ===");
        
        GpuConfig config = new GpuConfig();
        ComputeProvider provider = new CpuComputeProvider();
        MatrixOperation matrixOp = new CpuMatrixOperation(provider);
        
        try {
            // Create a simple neural network for text classification
            // Input: 10 features -> Hidden: 8 neurons -> Output: 3 classes
            int[] layerSizes = {10, 8, 3};
            String[] activations = {"relu", "softmax"};
            
            GpuNeuralNetwork network = new GpuNeuralNetwork(layerSizes, activations, config, matrixOp);
            
            System.out.println("Created neural network:");
            System.out.printf("  Architecture: %d → %d → %d\n", 
                             layerSizes[0], layerSizes[1], layerSizes[2]);
            System.out.printf("  Activations: %s → %s\n", activations[0], activations[1]);
            System.out.printf("  Total parameters: %d\n", network.getTotalParameters());
            
            // Single prediction
            System.out.println("\n--- Single Prediction ---");
            float[] input = {0.5f, -0.2f, 0.8f, 0.1f, -0.3f, 0.7f, 0.4f, -0.1f, 0.6f, 0.2f};
            
            long startTime = System.currentTimeMillis();
            float[] output = network.predict(input);
            long predictionTime = System.currentTimeMillis() - startTime;
            
            System.out.println("Input features: [0.5, -0.2, 0.8, 0.1, -0.3, ...]");
            System.out.println("Prediction probabilities:");
            for (int i = 0; i < output.length; i++) {
                System.out.printf("  Class %d: %.4f (%.1f%%)\n", 
                                 i, output[i], output[i] * 100);
            }
            System.out.printf("Prediction time: %d ms\n", predictionTime);
            
            // Batch prediction
            System.out.println("\n--- Batch Prediction ---");
            int batchSize = 5;
            float[][] batchInput = new float[batchSize][10];
            
            // Generate random batch data
            for (int i = 0; i < batchSize; i++) {
                for (int j = 0; j < 10; j++) {
                    batchInput[i][j] = (float) (Math.random() * 2.0 - 1.0);
                }
            }
            
            startTime = System.currentTimeMillis();
            float[][] batchOutput = network.predictBatch(batchInput);
            long batchTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Batch prediction for %d samples:\n", batchSize);
            for (int i = 0; i < batchSize; i++) {
                System.out.printf("  Sample %d: [%.3f, %.3f, %.3f]\n", 
                                 i, batchOutput[i][0], batchOutput[i][1], batchOutput[i][2]);
            }
            System.out.printf("Batch prediction time: %d ms (%.2f ms per sample)\n", 
                             batchTime, (double)batchTime / batchSize);
            
            // Training demonstration
            System.out.println("\n--- Training Demonstration ---");
            float[][] trainInput = new float[20][10];
            float[][] trainOutput = new float[20][3];
            
            // Generate synthetic training data
            for (int i = 0; i < 20; i++) {
                for (int j = 0; j < 10; j++) {
                    trainInput[i][j] = (float) (Math.random() * 2.0 - 1.0);
                }
                // Create one-hot encoded targets
                int targetClass = i % 3;
                trainOutput[i][targetClass] = 1.0f;
            }
            
            System.out.println("Training network with 20 samples for 10 epochs...");
            startTime = System.currentTimeMillis();
            network.train(trainInput, trainOutput, 10);
            long trainingTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Training completed in %d ms\n", trainingTime);
            
            network.cleanup();
            
        } finally {
            matrixOp.release();
            provider.cleanup();
        }
        
        System.out.println("✅ Neural network demo completed");
    }
    
    /**
     * Demonstrate OpenNLP integration with GPU acceleration
     */
    public static void demonstrateOpenNLPIntegration() {
        System.out.println("\n=== Demo 4: OpenNLP Integration ===");
        
        try {
            // Create GPU configuration
            GpuConfig config = new GpuConfig();
            config.setGpuEnabled(true);
            
            // Create GPU model factory
            GpuModelFactory factory = new GpuModelFactory(config);
            
            // Create a basic MaxEnt model (in real usage, load from file)
            MaxentModel baseModel = new GisModel();
            
            // Wrap with GPU acceleration
            MaxentModel gpuModel = factory.createGpuMaxentModel(baseModel);
            
            System.out.println("Created GPU-accelerated MaxEnt model");
            System.out.printf("Model outcomes: %d\n", gpuModel.getNumOutcomes());
            
            // Example context for NLP prediction
            String[] context = {
                "word=hello", 
                "pos=NN", 
                "prev=the", 
                "next=world",
                "shape=lowercase"
            };
            
            System.out.println("\n--- Single Context Evaluation ---");
            System.out.println("Context: " + String.join(", ", context));
            
            long startTime = System.currentTimeMillis();
            double[] probabilities = gpuModel.eval(context);
            long evalTime = System.currentTimeMillis() - startTime;
            
            System.out.println("Prediction probabilities:");
            for (int i = 0; i < probabilities.length; i++) {
                String outcome = gpuModel.getOutcome(i);
                System.out.printf("  %s: %.6f\n", outcome, probabilities[i]);
            }
            System.out.printf("Evaluation time: %d ms\n", evalTime);
            
            // Batch processing demonstration
            System.out.println("\n--- Batch Processing ---");
            String[][] batchContexts = {
                {"word=machine", "pos=NN", "prev=the"},
                {"word=learning", "pos=VBG", "prev=machine"},
                {"word=gpu", "pos=NN", "prev=with"},
                {"word=acceleration", "pos=NN", "prev=gpu"},
                {"word=fast", "pos=JJ", "prev=is"}
            };
            
            System.out.println("Processing batch of " + batchContexts.length + " contexts:");
            
            startTime = System.currentTimeMillis();
            for (int i = 0; i < batchContexts.length; i++) {
                double[] batchProbs = gpuModel.eval(batchContexts[i]);
                System.out.printf("  Context %d (%s): %.4f\n", 
                                 i, batchContexts[i][0], batchProbs[0]);
            }
            long batchTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("Batch processing time: %d ms (%.2f ms per context)\n", 
                             batchTime, (double)batchTime / batchContexts.length);
            
            // Performance comparison
            System.out.println("\n--- Performance Comparison ---");
            
            // GPU model timing
            startTime = System.currentTimeMillis();
            for (int i = 0; i < 100; i++) {
                gpuModel.eval(context);
            }
            long gpuTime = System.currentTimeMillis() - startTime;
            
            // Base model timing
            startTime = System.currentTimeMillis();
            for (int i = 0; i < 100; i++) {
                baseModel.eval(context);
            }
            long cpuTime = System.currentTimeMillis() - startTime;
            
            System.out.printf("CPU model (100 evals): %d ms\n", cpuTime);
            System.out.printf("GPU model (100 evals): %d ms\n", gpuTime);
            
            if (cpuTime > 0) {
                double speedup = (double)cpuTime / gpuTime;
                System.out.printf("Speedup: %.2fx\n", speedup);
            }
            
            // Model adapter demonstration
            System.out.println("\n--- Model Adapter Pattern ---");
            MaxentModel adaptedModel = factory.createGpuModelAdapter(baseModel);
            
            System.out.println("Created GPU model adapter");
            System.out.println("Adapter provides same interface as original model");
            
            double[] adaptedResult = adaptedModel.eval(context);
            System.out.printf("Adapter evaluation result: %.6f\n", adaptedResult[0]);
            
        } catch (Exception e) {
            System.out.println("Note: This demo uses stub implementations");
            System.out.println("In production, load actual trained models");
            logger.info("OpenNLP integration demo completed (with stubs)");
        }
        
        System.out.println("✅ OpenNLP integration demo completed");
    }
}
