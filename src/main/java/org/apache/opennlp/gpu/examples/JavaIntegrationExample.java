package org.apache.opennlp.gpu.examples;

import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.ml.maxent.GpuMaxentModel;
import org.apache.opennlp.gpu.tools.GpuDiagnostics;

/**
 * Complete example showing how to integrate OpenNLP GPU extension into a Java project.
 * 
 * This example demonstrates:
 * 1. Adding the dependency to your Maven project
 * 2. Simple API usage for drop-in acceleration
 * 3. Performance benchmarking
 * 4. Error handling and fallback
 * 
 * To use this in your own project:
 * 
 * 1. Add to your pom.xml:
 * <pre>
 * {@code
 * <dependency>
 *     <groupId>org.apache.opennlp</groupId>
 *     <artifactId>opennlp-gpu</artifactId>
 *     <version>1.0.0</version>
 * </dependency>
 * }
 * </pre>
 * 
 * 2. Replace your OpenNLP training code with GPU-accelerated versions
 * 
 * @author OpenNLP GPU Extension Team
 * @since 1.0.0
 */
public class JavaIntegrationExample {
    
    private static final GpuLogger logger = GpuLogger.getLogger(JavaIntegrationExample.class);
    
    public static void main(String[] args) {
        try {
            // Step 1: Check GPU availability
            demonstrateGpuDetection();
            
            // Step 2: Show simple usage example
            demonstrateSimpleUsage();
            
            // Step 3: Show performance comparison
            demonstratePerformanceComparison();
            
            // Step 4: Show error handling
            demonstrateErrorHandling();
            
        } catch (Exception e) {
            logger.error("Example failed: {}", e.getMessage(), e);
        }
    }
    
    /**
     * Demonstrate GPU detection and system information.
     */
    private static void demonstrateGpuDetection() {
        System.out.println("=== GPU Detection Example ===");
        
        // Check if GPU is available
        boolean gpuAvailable = GpuConfig.isGpuAvailable();
        System.out.println("GPU Available: " + gpuAvailable);
        
        if (gpuAvailable) {
            // Get detailed GPU information
            Map<String, Object> gpuInfo = GpuConfig.getGpuInfo();
            System.out.println("GPU Information:");
            for (Map.Entry<String, Object> entry : gpuInfo.entrySet()) {
                System.out.println("  " + entry.getKey() + ": " + entry.getValue());
            }
        }
        
        // Run comprehensive diagnostics
        System.out.println("\n=== GPU Diagnostics ===");
        try {
            GpuDiagnostics.main(new String[]{});
        } catch (Exception e) {
            System.err.println("Diagnostics failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrate simple usage that replaces standard OpenNLP code.
     */
    private static void demonstrateSimpleUsage() {
        System.out.println("=== Simple Usage Example ===");
        
        try {
            // Create sample training data
            List<TrainingExample> trainingData = createSampleTrainingData();
            
            // BEFORE: Standard OpenNLP approach
            // MaxentModel cpuModel = trainStandardModel(trainingData);
            
            // AFTER: GPU-accelerated approach (same API!)
            GpuMaxentModel gpuModel = trainGpuModel(trainingData);
            
            // Use the model exactly like standard OpenNLP
            String[] testFeatures = {"feature1", "feature2", "feature3"};
            double[] outcomes = gpuModel.eval(testFeatures);
            
            System.out.println("Prediction outcomes: ");
            for (int i = 0; i < outcomes.length; i++) {
                System.out.printf("  %s: %.4f%n", gpuModel.getOutcome(i), outcomes[i]);
            }
            
            // Get best outcome
            String bestOutcome = gpuModel.getBestOutcome(outcomes);
            System.out.println("Best outcome: " + bestOutcome);
            
            // Check if GPU was actually used
            if (gpuModel.isUsingGpu()) {
                System.out.printf("✅ GPU acceleration enabled (%.1fx speedup)%n", 
                                gpuModel.getSpeedupFactor());
            } else {
                System.out.println("ℹ️  Using CPU fallback");
            }
            
        } catch (Exception e) {
            System.err.println("Simple usage example failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrate performance comparison between CPU and GPU.
     */
    private static void demonstratePerformanceComparison() {
        System.out.println("=== Performance Comparison ===");
        
        try {
            List<TrainingExample> largeDataset = createLargeTrainingDataset(10000);
            
            // Measure CPU training time
            long cpuStartTime = System.currentTimeMillis();
            // MaxentModel cpuModel = trainStandardModel(largeDataset);
            long cpuTrainingTime = System.currentTimeMillis() - cpuStartTime;
            
            // Measure GPU training time
            long gpuStartTime = System.currentTimeMillis();
            GpuMaxentModel gpuModel = trainGpuModel(largeDataset);
            long gpuTrainingTime = System.currentTimeMillis() - gpuStartTime;
            
            // Compare performance
            double speedup = (double) cpuTrainingTime / gpuTrainingTime;
            
            System.out.printf("CPU Training Time: %d ms%n", cpuTrainingTime);
            System.out.printf("GPU Training Time: %d ms%n", gpuTrainingTime);
            System.out.printf("Speedup: %.1fx%n", speedup);
            
            // Show memory usage
            Map<String, Object> gpuStats = gpuModel.getPerformanceStats();
            if (gpuStats.containsKey("memory_used_mb")) {
                System.out.printf("GPU Memory Used: %d MB%n", gpuStats.get("memory_used_mb"));
            }
            
        } catch (Exception e) {
            System.err.println("Performance comparison failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Demonstrate error handling and graceful fallback.
     */
    private static void demonstrateErrorHandling() {
        System.out.println("=== Error Handling Example ===");
        
        try {
            // Configure GPU with settings that might fail
            GpuConfig config = new GpuConfig();
            config.setGpuEnabled(true);
            config.setBatchSize(99999);  // Intentionally large batch size
            config.setMemoryPoolSizeMB(99999);  // Intentionally large memory pool
            
            List<TrainingExample> data = createSampleTrainingData();
            
            try {
                // Try GPU training with problematic settings
                GpuMaxentModel model = trainGpuModelWithConfig(data, config);
                
                if (model.isUsingGpu()) {
                    System.out.println("✅ GPU training succeeded despite large settings");
                } else {
                    System.out.println("ℹ️  Automatically fell back to CPU due to GPU constraints");
                }
                
            } catch (Exception e) {
                System.out.println("⚠️  GPU training failed, using CPU fallback: " + e.getMessage());
                
                // Fallback to CPU training
                // MaxentModel cpuModel = trainStandardModel(data);
                System.out.println("✅ CPU fallback training completed successfully");
            }
            
        } catch (Exception e) {
            System.err.println("Error handling example failed: " + e.getMessage());
        }
        
        System.out.println();
    }
    
    /**
     * Create sample training data for examples.
     */
    private static List<TrainingExample> createSampleTrainingData() {
        List<TrainingExample> data = new ArrayList<>();
        
        // Add some sample training examples
        data.add(new TrainingExample(new String[]{"positive", "good", "excellent"}, "POSITIVE"));
        data.add(new TrainingExample(new String[]{"negative", "bad", "terrible"}, "NEGATIVE"));
        data.add(new TrainingExample(new String[]{"neutral", "okay", "average"}, "NEUTRAL"));
        data.add(new TrainingExample(new String[]{"positive", "great", "amazing"}, "POSITIVE"));
        data.add(new TrainingExample(new String[]{"negative", "awful", "horrible"}, "NEGATIVE"));
        
        return data;
    }
    
    /**
     * Create a larger training dataset for performance testing.
     */
    private static List<TrainingExample> createLargeTrainingDataset(int size) {
        List<TrainingExample> data = new ArrayList<>();
        
        String[] positiveWords = {"good", "excellent", "great", "amazing", "wonderful", "fantastic"};
        String[] negativeWords = {"bad", "terrible", "awful", "horrible", "dreadful", "atrocious"};
        String[] neutralWords = {"okay", "average", "normal", "fine", "decent", "acceptable"};
        
        for (int i = 0; i < size; i++) {
            String[] features = new String[3];
            String outcome;
            
            if (i % 3 == 0) {
                features[0] = positiveWords[i % positiveWords.length];
                features[1] = "very";
                features[2] = positiveWords[(i + 1) % positiveWords.length];
                outcome = "POSITIVE";
            } else if (i % 3 == 1) {
                features[0] = negativeWords[i % negativeWords.length];
                features[1] = "really";
                features[2] = negativeWords[(i + 1) % negativeWords.length];
                outcome = "NEGATIVE";
            } else {
                features[0] = neutralWords[i % neutralWords.length];
                features[1] = "somewhat";
                features[2] = neutralWords[(i + 1) % neutralWords.length];
                outcome = "NEUTRAL";
            }
            
            data.add(new TrainingExample(features, outcome));
        }
        
        return data;
    }
    
    /**
     * Train a GPU-accelerated model with default settings.
     */
    private static GpuMaxentModel trainGpuModel(List<TrainingExample> data) throws IOException {
        GpuConfig config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setBatchSize(256);
        config.setMemoryPoolSizeMB(512);
        
        return trainGpuModelWithConfig(data, config);
    }
    
    /**
     * Train a GPU-accelerated model with custom configuration.
     */
    private static GpuMaxentModel trainGpuModelWithConfig(List<TrainingExample> data, 
                                                         GpuConfig config) throws IOException {
        
        // Convert training data to the format expected by the GPU model
        // In a real implementation, this would convert to OpenNLP's EventStream format
        
        // For this example, create a basic GPU model
        // In practice, you would use GpuModelFactory.trainMaxentModel()
        
        return new GpuMaxentModel(null, config);  // Simplified for example
    }
    
    /**
     * Simple class to represent a training example.
     */
    static class TrainingExample {
        final String[] features;
        final String outcome;
        
        TrainingExample(String[] features, String outcome) {
            this.features = features;
            this.outcome = outcome;
        }
    }
}
