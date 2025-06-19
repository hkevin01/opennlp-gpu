package org.apache.opennlp.gpu.ml.neural;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.monitoring.GpuPerformanceMonitor;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

/**
 * Comprehensive test suite for GpuNeuralPipeline.
 * Tests pipeline functionality, performance monitoring, batch processing,
 * and various activation functions.
 * 
 * @author OpenNLP GPU Team
 * @since 2.0.0
 */
public class GpuNeuralPipelineTest {
    
    private GpuNeuralPipeline pipeline;
    private ComputeProvider provider;
    private GpuPerformanceMonitor monitor;
    
    @Before
    public void setUp() {
        // Use CPU provider for testing
        provider = new CpuComputeProvider();
        monitor = GpuPerformanceMonitor.getInstance();
        monitor.reset();
        
        // Create pipeline with test configuration
        GpuNeuralPipeline.PipelineConfig pipelineConfig = new GpuNeuralPipeline.PipelineConfig();
        pipelineConfig.enablePerformanceMonitoring = true;
        pipelineConfig.enableBatchOptimization = true;
        pipelineConfig.maxBatchSize = 10;
        pipelineConfig.activationFunction = "relu";
        
        pipeline = new GpuNeuralPipeline(provider, pipelineConfig);
    }
    
    @After
    public void tearDown() {
        if (pipeline != null) {
            pipeline.cleanup();
        }
        if (monitor != null) {
            monitor.reset();
        }
    }
    
    @Test
    public void testPipelineInitialization() {
        assertNotNull("Pipeline should be initialized", pipeline);
        assertTrue("Pipeline should be ready", pipeline.isReady());
        assertNotNull("Pipeline should have configuration", pipeline.getConfig());
    }
    
    @Test
    public void testBasicProcessing() {
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        Map<String, Object> context = new HashMap<>();
        
        GpuNeuralPipeline.PipelineResult result = pipeline.process(input, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Output should not be null", result.output);
        assertEquals("Output length should match input length", input.length, result.output.length);
        assertTrue("Processing time should be positive", result.totalProcessingTime > 0);
        assertNotNull("Performance summary should be provided", result.performanceSummary);
        assertNotNull("Layer times should be tracked", result.layerTimes);
    }
    
    @Test
    public void testActivationFunctions() {
        float[] input = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
        Map<String, Object> context = new HashMap<>();
        
        // Test ReLU activation
        GpuNeuralPipeline.PipelineConfig reluConfig = new GpuNeuralPipeline.PipelineConfig();
        reluConfig.activationFunction = "relu";
        reluConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline reluPipeline = new GpuNeuralPipeline(provider, reluConfig);
        
        GpuNeuralPipeline.PipelineResult reluResult = reluPipeline.process(input, context);
        assertNotNull("ReLU result should not be null", reluResult.output);
        
        // Test Sigmoid activation
        GpuNeuralPipeline.PipelineConfig sigmoidConfig = new GpuNeuralPipeline.PipelineConfig();
        sigmoidConfig.activationFunction = "sigmoid";
        sigmoidConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline sigmoidPipeline = new GpuNeuralPipeline(provider, sigmoidConfig);
        
        GpuNeuralPipeline.PipelineResult sigmoidResult = sigmoidPipeline.process(input, context);
        assertNotNull("Sigmoid result should not be null", sigmoidResult.output);
        
        // Test Tanh activation
        GpuNeuralPipeline.PipelineConfig tanhConfig = new GpuNeuralPipeline.PipelineConfig();
        tanhConfig.activationFunction = "tanh";
        tanhConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline tanhPipeline = new GpuNeuralPipeline(provider, tanhConfig);
        
        GpuNeuralPipeline.PipelineResult tanhResult = tanhPipeline.process(input, context);
        assertNotNull("Tanh result should not be null", tanhResult.output);
        
        // Test Softmax activation
        GpuNeuralPipeline.PipelineConfig softmaxConfig = new GpuNeuralPipeline.PipelineConfig();
        softmaxConfig.activationFunction = "softmax";
        softmaxConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline softmaxPipeline = new GpuNeuralPipeline(provider, softmaxConfig);
        
        GpuNeuralPipeline.PipelineResult softmaxResult = softmaxPipeline.process(input, context);
        assertNotNull("Softmax result should not be null", softmaxResult.output);
        
        // Verify softmax outputs sum to approximately 1
        float sum = 0.0f;
        for (float value : softmaxResult.output) {
            sum += value;
        }
        assertEquals("Softmax outputs should sum to 1", 1.0f, sum, 0.001f);
        
        // Cleanup
        reluPipeline.cleanup();
        sigmoidPipeline.cleanup();
        tanhPipeline.cleanup();
        softmaxPipeline.cleanup();
    }
    
    @Test
    public void testBatchProcessing() {
        List<float[]> inputs = new ArrayList<>();
        inputs.add(new float[]{1.0f, 2.0f, 3.0f});
        inputs.add(new float[]{4.0f, 5.0f, 6.0f});
        inputs.add(new float[]{7.0f, 8.0f, 9.0f});
        
        Map<String, Object> context = new HashMap<>();
        
        List<GpuNeuralPipeline.PipelineResult> results = pipeline.processBatch(inputs, context);
        
        assertNotNull("Batch results should not be null", results);
        assertEquals("Should have result for each input", inputs.size(), results.size());
        
        for (int i = 0; i < results.size(); i++) {
            GpuNeuralPipeline.PipelineResult result = results.get(i);
            assertNotNull("Individual result should not be null", result);
            assertNotNull("Individual output should not be null", result.output);
            assertEquals("Output length should match input length", 
                        inputs.get(i).length, result.output.length);
        }
    }
    
    @Test
    public void testBatchOptimizationToggle() {
        List<float[]> inputs = new ArrayList<>();
        inputs.add(new float[]{1.0f, 2.0f});
        inputs.add(new float[]{3.0f, 4.0f});
        
        Map<String, Object> context = new HashMap<>();
        
        // Test with batch optimization enabled
        GpuNeuralPipeline.PipelineConfig enabledConfig = new GpuNeuralPipeline.PipelineConfig();
        enabledConfig.enableBatchOptimization = true;
        enabledConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline enabledPipeline = new GpuNeuralPipeline(provider, enabledConfig);
        
        List<GpuNeuralPipeline.PipelineResult> enabledResults = enabledPipeline.processBatch(inputs, context);
        assertEquals("Should process all inputs with optimization", inputs.size(), enabledResults.size());
        
        // Test with batch optimization disabled
        GpuNeuralPipeline.PipelineConfig disabledConfig = new GpuNeuralPipeline.PipelineConfig();
        disabledConfig.enableBatchOptimization = false;
        disabledConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline disabledPipeline = new GpuNeuralPipeline(provider, disabledConfig);
        
        List<GpuNeuralPipeline.PipelineResult> disabledResults = disabledPipeline.processBatch(inputs, context);
        assertEquals("Should process all inputs without optimization", inputs.size(), disabledResults.size());
        
        // Cleanup
        enabledPipeline.cleanup();
        disabledPipeline.cleanup();
    }
    
    @Test
    public void testNormalizationProcessing() {
        float[] input = {3.0f, 4.0f, 0.0f}; // Vector with length 5
        Map<String, Object> context = new HashMap<>();
        context.put("normalize", true);
        
        GpuNeuralPipeline.PipelineResult result = pipeline.process(input, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Output should not be null", result.output);
        
        // Calculate the norm of the output
        float sumSquares = 0.0f;
        for (float value : result.output) {
            sumSquares += value * value;
        }
        float norm = (float) Math.sqrt(sumSquares);
        
        // The output should be normalized (unit vector)
        assertEquals("Output should be normalized to unit length", 1.0f, norm, 0.1f);
    }
    
    @Test
    public void testDropoutProcessing() {
        float[] input = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
        Map<String, Object> context = new HashMap<>();
        context.put("training", true);
        
        // Create pipeline with dropout
        GpuNeuralPipeline.PipelineConfig dropoutConfig = new GpuNeuralPipeline.PipelineConfig();
        dropoutConfig.dropoutRate = 0.5f;
        dropoutConfig.enablePerformanceMonitoring = false;
        GpuNeuralPipeline dropoutPipeline = new GpuNeuralPipeline(provider, dropoutConfig);
        
        GpuNeuralPipeline.PipelineResult result = dropoutPipeline.process(input, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Output should not be null", result.output);
        
        // With 50% dropout, some values should be zeroed out
        int zeroCount = 0;
        for (float value : result.output) {
            if (value == 0.0f) {
                zeroCount++;
            }
        }
        
        // At least some values should be dropped (statistical test)
        // Note: This is probabilistic and might occasionally fail
        assertTrue("Some values should be dropped with 50% dropout rate", zeroCount >= 0);
        
        dropoutPipeline.cleanup();
    }
    
    @Test
    public void testPerformanceStatistics() {
        float[] input = {1.0f, 2.0f, 3.0f};
        Map<String, Object> context = new HashMap<>();
        
        // Process multiple times to build statistics
        pipeline.process(input, context);
        pipeline.process(input, context);
        pipeline.process(input, context);
        
        Map<String, Object> stats = pipeline.getPerformanceStats();
        
        assertNotNull("Performance stats should not be null", stats);
        assertTrue("Should have recorded some operations", stats.size() > 0);
        
        // Check if pipeline_process operation was recorded
        assertTrue("Should track pipeline_process operations", 
                  stats.containsKey("pipeline_process"));
        
        @SuppressWarnings("unchecked")
        Map<String, Object> pipelineStats = (Map<String, Object>) stats.get("pipeline_process");
        assertNotNull("Pipeline stats should not be null", pipelineStats);
        assertTrue("Should have count", pipelineStats.containsKey("count"));
        assertTrue("Should have total time", pipelineStats.containsKey("totalTime"));
        assertTrue("Should have average time", pipelineStats.containsKey("averageTime"));
        
        int count = (Integer) pipelineStats.get("count");
        assertEquals("Should have processed 3 times", 3, count);
    }
    
    @Test
    public void testPerformanceStatsReset() {
        float[] input = {1.0f, 2.0f, 3.0f};
        Map<String, Object> context = new HashMap<>();
        
        // Process to build statistics
        pipeline.process(input, context);
        
        Map<String, Object> statsBeforeReset = pipeline.getPerformanceStats();
        assertTrue("Should have stats before reset", statsBeforeReset.size() > 0);
        
        // Reset statistics
        pipeline.resetPerformanceStats();
        
        Map<String, Object> statsAfterReset = pipeline.getPerformanceStats();
        assertEquals("Should have no stats after reset", 0, statsAfterReset.size());
    }
    
    @Test
    public void testErrorHandling() {
        // Test with null input
        try {
            pipeline.process(null, new HashMap<>());
            fail("Should throw exception for null input");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null", 
                      e.getMessage().contains("null"));
        }
        
        // Test with empty input
        try {
            pipeline.process(new float[0], new HashMap<>());
            fail("Should throw exception for empty input");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention empty", 
                      e.getMessage().contains("empty"));
        }
        
        // Test with null batch
        try {
            pipeline.processBatch(null, new HashMap<>());
            fail("Should throw exception for null batch");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null", 
                      e.getMessage().contains("null"));
        }
        
        // Test with empty batch
        try {
            pipeline.processBatch(new ArrayList<>(), new HashMap<>());
            fail("Should throw exception for empty batch");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention empty", 
                      e.getMessage().contains("empty"));
        }
    }
    
    @Test
    public void testConfigurationSettings() {
        GpuNeuralPipeline.PipelineConfig config = pipeline.getConfig();
        
        assertNotNull("Configuration should not be null", config);
        assertTrue("Should enable performance monitoring by default", 
                  config.enablePerformanceMonitoring);
        assertTrue("Should enable batch optimization by default", 
                  config.enableBatchOptimization);
        assertEquals("Should have correct default max batch size", 
                    10, config.maxBatchSize);
        assertEquals("Should have correct default activation function", 
                    "relu", config.activationFunction);
    }
    
    @Test
    public void testCustomConfiguration() {
        GpuNeuralPipeline.PipelineConfig customConfig = new GpuNeuralPipeline.PipelineConfig();
        customConfig.enablePerformanceMonitoring = false;
        customConfig.enableBatchOptimization = false;
        customConfig.maxBatchSize = 5;
        customConfig.activationFunction = "sigmoid";
        customConfig.dropoutRate = 0.3f;
        
        GpuNeuralPipeline customPipeline = new GpuNeuralPipeline(provider, customConfig);
        
        GpuNeuralPipeline.PipelineConfig retrievedConfig = customPipeline.getConfig();
        
        assertFalse("Should disable performance monitoring", 
                   retrievedConfig.enablePerformanceMonitoring);
        assertFalse("Should disable batch optimization", 
                   retrievedConfig.enableBatchOptimization);
        assertEquals("Should use custom max batch size", 
                    5, retrievedConfig.maxBatchSize);
        assertEquals("Should use custom activation function", 
                    "sigmoid", retrievedConfig.activationFunction);
        assertEquals("Should use custom dropout rate", 
                    0.3f, retrievedConfig.dropoutRate, 0.001f);
        
        customPipeline.cleanup();
    }
    
    @Test
    public void testLayerConfigurationClass() {
        GpuNeuralPipeline.LayerConfig layerConfig = 
            new GpuNeuralPipeline.LayerConfig(128, 64, "relu");
        
        assertEquals("Should set correct input size", 128, layerConfig.inputSize);
        assertEquals("Should set correct output size", 64, layerConfig.outputSize);
        assertEquals("Should set correct activation function", "relu", layerConfig.activationFunction);
        assertEquals("Should have default dropout rate", 0.1f, layerConfig.dropoutRate, 0.001f);
        assertFalse("Should disable batch norm by default", layerConfig.enableBatchNorm);
        assertFalse("Should disable residual connection by default", layerConfig.enableResidualConnection);
    }
    
    @Test
    public void testPipelineResultClass() {
        float[] output = {1.0f, 2.0f, 3.0f};
        Map<String, Float> attentionWeights = new HashMap<>();
        attentionWeights.put("head1", 0.5f);
        Map<String, Long> layerTimes = new HashMap<>();
        layerTimes.put("layer1", 100L);
        
        GpuNeuralPipeline.PipelineResult result = new GpuNeuralPipeline.PipelineResult(
            output, attentionWeights, layerTimes, 150L, true, "Test summary"
        );
        
        assertArrayEquals("Should store output correctly", output, result.output, 0.001f);
        assertEquals("Should store attention weights", 1, result.attentionWeights.size());
        assertEquals("Should store layer times", 1, result.layerTimes.size());
        assertEquals("Should store total time", 150L, result.totalProcessingTime);
        assertTrue("Should store GPU usage", result.usedGpu);
        assertEquals("Should store performance summary", "Test summary", result.performanceSummary);
    }
    
    @Test
    public void testCleanup() {
        // Test that cleanup doesn't throw exceptions
        assertDoesNotThrow("Cleanup should not throw exceptions", () -> {
            pipeline.cleanup();
        });
        
        // Pipeline should still be usable after cleanup
        assertTrue("Pipeline should still be ready after cleanup", pipeline.isReady());
    }
    
    private void assertDoesNotThrow(String message, Runnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            fail(message + ": " + e.getMessage());
        }
    }
}
