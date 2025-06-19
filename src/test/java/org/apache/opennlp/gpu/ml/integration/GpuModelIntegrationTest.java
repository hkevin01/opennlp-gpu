package org.apache.opennlp.gpu.ml.integration;

import static org.junit.Assert.assertEquals;
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
 * Integration tests for GpuModelIntegration system.
 * Tests the complete integration of neural pipelines, feature extraction,
 * performance monitoring, and model ensemble capabilities.
 * 
 * @author OpenNLP GPU Team
 * @since 2.0.0
 */
public class GpuModelIntegrationTest {
    
    private GpuModelIntegration integration;
    private ComputeProvider provider;
    private GpuPerformanceMonitor monitor;
    
    @Before
    public void setUp() {
        // Use CPU provider for testing
        provider = new CpuComputeProvider();
        monitor = GpuPerformanceMonitor.getInstance();
        monitor.reset();
        
        // Create integration system with test configuration
        GpuModelIntegration.IntegrationConfig config = new GpuModelIntegration.IntegrationConfig();
        config.enableNeuralEnhancement = true;
        config.enableEnsembleProcessing = true;
        config.enablePerformanceOptimization = true;
        config.maxConcurrentModels = 3;
        config.featureVectorSize = 128;
        
        integration = new GpuModelIntegration(provider, config);
    }
    
    @After
    public void tearDown() {
        if (integration != null) {
            integration.cleanup();
        }
        if (monitor != null) {
            monitor.reset();
        }
    }
    
    @Test
    public void testIntegrationInitialization() {
        assertNotNull("Integration should be initialized", integration);
        assertNotNull("Integration should have configuration", integration.getConfig());
        
        GpuModelIntegration.IntegrationConfig config = integration.getConfig();
        assertTrue("Should enable neural enhancement", config.enableNeuralEnhancement);
        assertTrue("Should enable ensemble processing", config.enableEnsembleProcessing);
        assertTrue("Should enable performance optimization", config.enablePerformanceOptimization);
        assertEquals("Should have correct max concurrent models", 3, config.maxConcurrentModels);
        assertEquals("Should have correct feature vector size", 128, config.featureVectorSize);
    }
    
    @Test
    public void testModelRegistration() {
        // Register test models
        integration.registerModel("neural_model", "neural_network", 1.0f);
        integration.registerModel("traditional_model", "maxent", 0.8f);
        integration.registerModel("ensemble_model", "ensemble", 0.9f);
        
        Map<String, Object> stats = integration.getIntegrationStatistics();
        
        assertNotNull("Statistics should not be null", stats);
        assertEquals("Should have 3 registered models", 3, stats.get("registeredModels"));
        assertEquals("Should have 3 active models", 3, stats.get("activeModels"));
        
        Map<String, Float> weights = integration.getModelWeights();
        assertEquals("Should have 3 model weights", 3, weights.size());
        assertEquals("Neural model weight should be 1.0", 1.0f, weights.get("neural_model"), 0.001f);
        assertEquals("Traditional model weight should be 0.8", 0.8f, weights.get("traditional_model"), 0.001f);
        assertEquals("Ensemble model weight should be 0.9", 0.9f, weights.get("ensemble_model"), 0.001f);
    }
    
    @Test
    public void testBasicIntegratedProcessing() {
        // Register a test model
        integration.registerModel("test_model", "neural", 1.0f);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
        String[] textTokens = {"hello", "world", "test", "neural", "pipeline"};
        Map<String, Object> context = new HashMap<>();
        
        GpuModelIntegration.IntegrationResult result = 
            integration.processIntegrated(input, textTokens, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Primary output should not be null", result.primaryOutput);
        assertTrue("Should have some model outputs", result.modelOutputs.size() > 0);
        assertTrue("Should have processing time", result.totalProcessingTime > 0);
        assertNotNull("Should have performance summary", result.performanceSummary);
        assertNotNull("Should have diagnostics", result.diagnostics);
    }
    
    @Test
    public void testFeatureExtraction() {
        float[] input = {1.0f, 2.0f, 3.0f};
        String[] textTokens = {"feature", "extraction", "test"};
        Map<String, Object> context = new HashMap<>();
        
        GpuModelIntegration.IntegrationResult result = 
            integration.processIntegrated(input, textTokens, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Primary output should not be null", result.primaryOutput);
        
        // The output should be larger than input if text features were extracted and combined
        assertTrue("Output should include text features", 
                  result.primaryOutput.length >= input.length);
    }
    
    @Test
    public void testEnsembleProcessing() {
        // Register multiple models to enable ensemble
        integration.registerModel("model1", "neural", 1.0f);
        integration.registerModel("model2", "traditional", 0.8f);
        integration.registerModel("model3", "hybrid", 0.9f);
        
        float[] input = {2.0f, 3.0f, 4.0f};
        String[] textTokens = {"ensemble", "processing", "test"};
        Map<String, Object> context = new HashMap<>();
        
        GpuModelIntegration.IntegrationResult result = 
            integration.processIntegrated(input, textTokens, context);
        
        assertNotNull("Result should not be null", result);
        assertNotNull("Should have ensemble output", result.ensembleOutput);
        assertTrue("Should have multiple model outputs", result.modelOutputs.size() > 1);
        assertTrue("Should have model confidences", result.modelConfidences.size() > 0);
        
        // Check that diagnostics include model count
        assertTrue("Diagnostics should include total models used", 
                  result.diagnostics.containsKey("total_models_used"));
    }
    
    @Test
    public void testBatchProcessing() {
        // Register a test model
        integration.registerModel("batch_model", "neural", 1.0f);
        
        List<float[]> inputs = new ArrayList<>();
        inputs.add(new float[]{1.0f, 2.0f, 3.0f});
        inputs.add(new float[]{4.0f, 5.0f, 6.0f});
        inputs.add(new float[]{7.0f, 8.0f, 9.0f});
        
        List<String[]> textTokensList = new ArrayList<>();
        textTokensList.add(new String[]{"batch", "test", "one"});
        textTokensList.add(new String[]{"batch", "test", "two"});
        textTokensList.add(new String[]{"batch", "test", "three"});
        
        Map<String, Object> context = new HashMap<>();
        
        List<GpuModelIntegration.IntegrationResult> results = 
            integration.processBatch(inputs, textTokensList, context);
        
        assertNotNull("Batch results should not be null", results);
        assertEquals("Should have result for each input", inputs.size(), results.size());
        
        for (int i = 0; i < results.size(); i++) {
            GpuModelIntegration.IntegrationResult result = results.get(i);
            assertNotNull("Individual result should not be null", result);
            assertNotNull("Individual output should not be null", result.primaryOutput);
            assertTrue("Individual processing time should be positive", result.totalProcessingTime > 0);
        }
    }
    
    @Test
    public void testModelWeightUpdates() {
        integration.registerModel("weight_test_model", "neural", 0.5f);
        
        // Test initial weight
        Map<String, Float> initialWeights = integration.getModelWeights();
        assertEquals("Initial weight should be 0.5", 0.5f, 
                    initialWeights.get("weight_test_model"), 0.001f);
        
        // Update weight
        integration.updateModelWeight("weight_test_model", 0.8f);
        
        Map<String, Float> updatedWeights = integration.getModelWeights();
        assertEquals("Updated weight should be 0.8", 0.8f, 
                    updatedWeights.get("weight_test_model"), 0.001f);
        
        // Test weight bounds
        integration.updateModelWeight("weight_test_model", -0.5f); // Should be clamped to 0
        Map<String, Float> clampedWeights = integration.getModelWeights();
        assertEquals("Negative weight should be clamped to 0", 0.0f, 
                    clampedWeights.get("weight_test_model"), 0.001f);
        
        integration.updateModelWeight("weight_test_model", 3.0f); // Should be clamped to 2
        Map<String, Float> maxClampedWeights = integration.getModelWeights();
        assertEquals("High weight should be clamped to 2", 2.0f, 
                    maxClampedWeights.get("weight_test_model"), 0.001f);
    }
    
    @Test
    public void testModelActivation() {
        integration.registerModel("activation_test_model", "neural", 1.0f);
        
        // Check initial state
        Map<String, Object> initialStats = integration.getIntegrationStatistics();
        assertEquals("Should have 1 active model initially", 1, initialStats.get("activeModels"));
        
        // Deactivate model
        integration.setModelActive("activation_test_model", false);
        
        Map<String, Object> deactivatedStats = integration.getIntegrationStatistics();
        assertEquals("Should have 0 active models after deactivation", 0, deactivatedStats.get("activeModels"));
        
        // Reactivate model
        integration.setModelActive("activation_test_model", true);
        
        Map<String, Object> reactivatedStats = integration.getIntegrationStatistics();
        assertEquals("Should have 1 active model after reactivation", 1, reactivatedStats.get("activeModels"));
    }
    
    @Test
    public void testPerformanceMonitoringIntegration() {
        integration.registerModel("performance_test_model", "neural", 1.0f);
        
        float[] input = {1.0f, 2.0f, 3.0f, 4.0f};
        String[] textTokens = {"performance", "monitoring", "integration"};
        Map<String, Object> context = new HashMap<>();
        
        // Process multiple times to generate performance data
        for (int i = 0; i < 3; i++) {
            integration.processIntegrated(input, textTokens, context);
        }
        
        // Check that performance monitoring captured the operations
        GpuPerformanceMonitor.PerformanceSummary performanceSummary = monitor.getPerformanceSummary();
        assertNotNull("Performance summary should not be null", performanceSummary);
        assertTrue("Should have recorded some operations", performanceSummary.totalOperations > 0);
    }
    
    @Test
    public void testErrorHandling() {
        // Test with null input
        try {
            integration.processIntegrated(null, null, new HashMap<>());
            fail("Should throw exception for null input");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null or empty", 
                      e.getMessage().toLowerCase().contains("null") || 
                      e.getMessage().toLowerCase().contains("empty"));
        }
        
        // Test with empty input
        try {
            integration.processIntegrated(new float[0], null, new HashMap<>());
            fail("Should throw exception for empty input");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null or empty", 
                      e.getMessage().toLowerCase().contains("null") || 
                      e.getMessage().toLowerCase().contains("empty"));
        }
        
        // Test with null batch
        try {
            integration.processBatch(null, null, new HashMap<>());
            fail("Should throw exception for null batch");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null or empty", 
                      e.getMessage().toLowerCase().contains("null") || 
                      e.getMessage().toLowerCase().contains("empty"));
        }
        
        // Test with empty batch
        try {
            integration.processBatch(new ArrayList<>(), new ArrayList<>(), new HashMap<>());
            fail("Should throw exception for empty batch");
        } catch (IllegalArgumentException e) {
            assertTrue("Exception message should mention null or empty", 
                      e.getMessage().toLowerCase().contains("null") || 
                      e.getMessage().toLowerCase().contains("empty"));
        }
    }
    
    @Test
    public void testStatisticsAndDiagnostics() {
        integration.registerModel("stats_model", "neural", 1.0f);
        
        float[] input = {1.0f, 2.0f, 3.0f};
        String[] textTokens = {"statistics", "test"};
        Map<String, Object> context = new HashMap<>();
        
        GpuModelIntegration.IntegrationResult result = 
            integration.processIntegrated(input, textTokens, context);
        
        // Check statistics
        Map<String, Object> stats = integration.getIntegrationStatistics();
        assertNotNull("Statistics should not be null", stats);
        assertTrue("Should have registered models", (Integer) stats.get("registeredModels") > 0);
        assertTrue("Should have active models", (Integer) stats.get("activeModels") > 0);
        assertTrue("Should have model statistics", stats.containsKey("modelStatistics"));
        
        // Check diagnostics in result
        assertNotNull("Diagnostics should not be null", result.diagnostics);
        assertTrue("Should include GPU acceleration info", 
                  result.diagnostics.containsKey("gpu_acceleration_used"));
        assertTrue("Should include total models used", 
                  result.diagnostics.containsKey("total_models_used"));
    }
    
    @Test
    public void testConfigurationOptions() {
        // Test with neural enhancement disabled
        GpuModelIntegration.IntegrationConfig config = new GpuModelIntegration.IntegrationConfig();
        config.enableNeuralEnhancement = false;
        config.enableEnsembleProcessing = false;
        
        GpuModelIntegration testIntegration = new GpuModelIntegration(provider, config);
        testIntegration.registerModel("config_test_model", "traditional", 1.0f);
        
        float[] input = {1.0f, 2.0f, 3.0f};
        String[] textTokens = {"config", "test"};
        Map<String, Object> context = new HashMap<>();
        
        GpuModelIntegration.IntegrationResult result = 
            testIntegration.processIntegrated(input, textTokens, context);
        
        assertNotNull("Result should not be null even with neural enhancement disabled", result);
        assertNotNull("Should still have primary output", result.primaryOutput);
        
        testIntegration.cleanup();
    }
    
    @Test
    public void testCleanup() {
        integration.registerModel("cleanup_test_model", "neural", 1.0f);
        
        // Process some data
        float[] input = {1.0f, 2.0f, 3.0f};
        integration.processIntegrated(input, null, new HashMap<>());
        
        // Verify that cleanup doesn't throw exceptions
        assertDoesNotThrow("Cleanup should not throw exceptions", () -> {
            integration.cleanup();
        });
        
        // After cleanup, statistics should be cleared
        Map<String, Object> stats = integration.getIntegrationStatistics();
        assertEquals("Should have no registered models after cleanup", 0, stats.get("registeredModels"));
        assertEquals("Should have no active models after cleanup", 0, stats.get("activeModels"));
    }
    
    private void assertDoesNotThrow(String message, Runnable runnable) {
        try {
            runnable.run();
        } catch (Exception e) {
            fail(message + ": " + e.getMessage());
        }
    }
}
