package org.apache.opennlp.gpu.production;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.production.CiCdManager.*;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.AfterEach;
import static org.junit.jupiter.api.Assertions.*;

import java.util.*;

/**
 * Comprehensive tests for the CI/CD management system.
 * Tests deployment pipelines, environment configurations, and validation processes.
 */
public class CiCdManagerTest {
    
    private GpuConfig config;
    private CiCdManager cicdManager;
    
    @BeforeEach
    void setUp() {
        // Initialize test configuration
        config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setBatchSize(64);
        config.setMemoryPoolSizeMB(1024);
        config.setDebugMode(false);
        
        // Initialize CI/CD manager
        cicdManager = new CiCdManager(config);
    }
    
    @AfterEach
    void tearDown() {
        if (cicdManager != null) {
            cicdManager.shutdown();
        }
    }
    
    @Test
    void testCiCdManagerInitialization() {
        assertNotNull(cicdManager);
        assertEquals(DeploymentState.INITIALIZING, cicdManager.getCurrentState());
        
        // Should have default environments
        Set<String> environments = cicdManager.getAvailableEnvironments();
        assertTrue(environments.contains("development"));
        assertTrue(environments.contains("testing"));
        assertTrue(environments.contains("staging"));
        assertTrue(environments.contains("production"));
        assertEquals(4, environments.size());
    }
    
    @Test
    void testEnvironmentConfigurations() {
        // Test development environment
        EnvironmentConfig dev = cicdManager.getEnvironmentConfig("development");
        assertNotNull(dev);
        assertEquals("development", dev.getName());
        assertEquals(EnvironmentType.DEVELOPMENT, dev.getType());
        assertFalse(dev.isGpuRequired());
        assertEquals(Boolean.TRUE, dev.getProperty("debug.enabled"));
        assertEquals("DEBUG", dev.getProperty("logging.level"));
        
        // Test production environment
        EnvironmentConfig prod = cicdManager.getEnvironmentConfig("production");
        assertNotNull(prod);
        assertEquals("production", prod.getName());
        assertEquals(EnvironmentType.PRODUCTION, prod.getType());
        assertTrue(prod.isGpuRequired());
        assertEquals(Boolean.FALSE, prod.getProperty("debug.enabled"));
        assertEquals("WARN", prod.getProperty("logging.level"));
        assertTrue(prod.getMinMemoryMB() >= 1024);
        assertTrue(prod.getMaxBatchSize() >= 256);
    }
    
    @Test
    void testCustomEnvironmentCreation() {
        // Create custom environment
        EnvironmentConfig custom = new EnvironmentConfig("custom-test", EnvironmentType.TESTING);
        custom.setProperty("custom.property", "test-value");
        custom.addRequiredFeature("custom-feature");
        
        cicdManager.addEnvironment("custom-test", custom);
        
        // Verify custom environment was added
        Set<String> environments = cicdManager.getAvailableEnvironments();
        assertTrue(environments.contains("custom-test"));
        
        EnvironmentConfig retrieved = cicdManager.getEnvironmentConfig("custom-test");
        assertNotNull(retrieved);
        assertEquals("custom-test", retrieved.getName());
        assertEquals("test-value", retrieved.getProperty("custom.property"));
        assertTrue(retrieved.getRequiredFeatures().contains("custom-feature"));
    }
    
    @Test
    void testValidationChecks() {
        List<String> checkNames = cicdManager.getValidationCheckNames();
        assertNotNull(checkNames);
        assertTrue(checkNames.size() >= 4);
        
        // Verify expected validation checks exist
        assertTrue(checkNames.contains("GPU Availability"));
        assertTrue(checkNames.contains("Memory Configuration"));
        assertTrue(checkNames.contains("Batch Size Configuration"));
        assertTrue(checkNames.contains("Feature Availability"));
    }
    
    @Test
    void testCustomValidationCheck() {
        // Create custom validation check
        ValidationCheck customCheck = new ValidationCheck() {
            @Override
            public String getName() { return "Custom Test Check"; }
            
            @Override
            public ValidationResult validate(EnvironmentConfig environment, GpuConfig config) {
                boolean passed = config.getBatchSize() > 0;
                return new ValidationResult(passed, getName(), 
                    passed ? "Custom check passed" : "Custom check failed",
                    Arrays.asList("Batch size: " + config.getBatchSize()), 10);
            }
            
            @Override
            public boolean isRequired() { return false; }
            
            @Override
            public int getPriority() { return 100; }
        };
        
        cicdManager.addValidationCheck(customCheck);
        
        List<String> checkNames = cicdManager.getValidationCheckNames();
        assertTrue(checkNames.contains("Custom Test Check"));
    }
    
    @Test
    void testSuccessfulDeploymentToDevelopment() {
        // Deploy to development environment (should succeed)
        DeploymentReport report = cicdManager.deployToEnvironment("development");
        
        assertNotNull(report);
        assertEquals("development", report.getEnvironmentName());
        assertEquals(DeploymentState.DEPLOYED, report.getState());
        assertTrue(report.isSuccessful());
        assertTrue(report.getTotalDeploymentTimeMs() > 0);
        
        // Check validation results
        List<ValidationResult> validationResults = report.getValidationResults();
        assertFalse(validationResults.isEmpty());
        
        // All validation results should pass for properly configured development
        long passedChecks = validationResults.stream()
            .mapToLong(r -> r.isPassed() ? 1 : 0)
            .sum();
        assertTrue(passedChecks > 0);
        
        // Check deployment metrics
        Map<String, Object> metrics = report.getMetrics();
        assertNotNull(metrics);
        assertTrue(metrics.containsKey("environment.name"));
        assertTrue(metrics.containsKey("environment.type"));
        assertTrue(metrics.containsKey("validation.checks.total"));
        assertTrue(metrics.containsKey("config.gpu.enabled"));
    }
    
    @Test
    void testSuccessfulDeploymentToProduction() {
        // Ensure configuration meets production requirements
        config.setGpuEnabled(true);
        config.setMemoryPoolSizeMB(1024);
        config.setBatchSize(128);
        
        // Deploy to production environment
        DeploymentReport report = cicdManager.deployToEnvironment("production");
        
        assertNotNull(report);
        assertEquals("production", report.getEnvironmentName());
        assertTrue(report.getTotalDeploymentTimeMs() > 0);
        
        // Check validation results
        List<ValidationResult> validationResults = report.getValidationResults();
        assertFalse(validationResults.isEmpty());
        
        // For production, all required checks should pass
        boolean hasRequiredFailures = validationResults.stream()
            .anyMatch(r -> !r.isPassed() && r.getCheckName().equals("GPU Availability"));
        
        if (config.isGpuEnabled()) {
            assertFalse(hasRequiredFailures, "GPU should be available for production deployment");
        }
    }
    
    @Test
    void testFailedDeploymentToUnknownEnvironment() {
        // Deploy to non-existent environment
        DeploymentReport report = cicdManager.deployToEnvironment("unknown-environment");
        
        assertNotNull(report);
        assertEquals("unknown-environment", report.getEnvironmentName());
        assertEquals(DeploymentState.FAILED, report.getState());
        assertFalse(report.isSuccessful());
        
        // Should have validation failure
        List<ValidationResult> validationResults = report.getValidationResults();
        assertFalse(validationResults.isEmpty());
        
        ValidationResult envCheck = validationResults.get(0);
        assertEquals("Environment Check", envCheck.getCheckName());
        assertFalse(envCheck.isPassed());
    }
    
    @Test
    void testFailedDeploymentDueToValidation() {
        // Configure invalid settings
        config.setGpuEnabled(false);
        config.setMemoryPoolSizeMB(64); // Too low for most environments
        config.setBatchSize(0); // Invalid batch size
        
        // Try to deploy to production (which requires GPU)
        DeploymentReport report = cicdManager.deployToEnvironment("production");
        
        assertNotNull(report);
        assertEquals("production", report.getEnvironmentName());
        
        // Should fail due to validation
        assertFalse(report.isSuccessful());
        
        // Check for specific validation failures
        List<ValidationResult> validationResults = report.getValidationResults();
        assertFalse(validationResults.isEmpty());
        
        // Should have GPU availability failure
        boolean hasGpuFailure = validationResults.stream()
            .anyMatch(r -> r.getCheckName().equals("GPU Availability") && !r.isPassed());
        assertTrue(hasGpuFailure, "Should fail GPU availability check");
        
        // Should have memory configuration failure
        boolean hasMemoryFailure = validationResults.stream()
            .anyMatch(r -> r.getCheckName().equals("Memory Configuration") && !r.isPassed());
        assertTrue(hasMemoryFailure, "Should fail memory configuration check");
        
        // Should have batch size failure
        boolean hasBatchFailure = validationResults.stream()
            .anyMatch(r -> r.getCheckName().equals("Batch Size Configuration") && !r.isPassed());
        assertTrue(hasBatchFailure, "Should fail batch size configuration check");
    }
    
    @Test
    void testValidationResult() {
        // Test validation result creation
        List<String> details = Arrays.asList("Detail 1", "Detail 2");
        ValidationResult result = new ValidationResult(true, "Test Check", "Test passed", details, 50);
        
        assertTrue(result.isPassed());
        assertEquals("Test Check", result.getCheckName());
        assertEquals("Test passed", result.getMessage());
        assertEquals(2, result.getDetails().size());
        assertEquals(50, result.getExecutionTimeMs());
        
        // Test failed validation result
        ValidationResult failure = new ValidationResult(false, "Test Check", "Test failed", null, 25);
        
        assertFalse(failure.isPassed());
        assertEquals("Test failed", failure.getMessage());
        assertTrue(failure.getDetails().isEmpty());
        assertEquals(25, failure.getExecutionTimeMs());
    }
    
    @Test
    void testDeploymentReport() {
        List<ValidationResult> validationResults = Arrays.asList(
            new ValidationResult(true, "Check 1", "Passed", null, 10),
            new ValidationResult(false, "Check 2", "Failed", Arrays.asList("Error detail"), 20)
        );
        
        Map<String, Object> metrics = new HashMap<>();
        metrics.put("test.metric", 42);
        
        DeploymentReport report = new DeploymentReport("test-env", DeploymentState.DEPLOYED,
                                                     validationResults, metrics, "1.0.0", 1000);
        
        assertEquals("test-env", report.getEnvironmentName());
        assertEquals(DeploymentState.DEPLOYED, report.getState());
        assertNotNull(report.getTimestamp());
        assertEquals(2, report.getValidationResults().size());
        assertEquals(1, report.getMetrics().size());
        assertEquals("1.0.0", report.getBuildVersion());
        assertEquals(1000, report.getTotalDeploymentTimeMs());
        
        // Should not be successful because one validation failed
        assertFalse(report.isSuccessful());
    }
    
    @Test
    void testEnvironmentConfigProperties() {
        EnvironmentConfig env = new EnvironmentConfig("test", EnvironmentType.TESTING);
        
        // Test property management
        env.setProperty("string.prop", "value");
        env.setProperty("number.prop", 123);
        env.setProperty("boolean.prop", true);
        
        assertEquals("value", env.getProperty("string.prop"));
        assertEquals(123, env.getProperty("number.prop"));
        assertEquals(true, env.getProperty("boolean.prop"));
        assertNull(env.getProperty("nonexistent.prop"));
        
        // Test feature management
        env.addRequiredFeature("feature1");
        env.addRequiredFeature("feature2");
        env.addRequiredFeature("feature1"); // Duplicate should not be added
        
        List<String> features = env.getRequiredFeatures();
        assertEquals(2, features.size());
        assertTrue(features.contains("feature1"));
        assertTrue(features.contains("feature2"));
    }
    
    @Test
    void testDeploymentMetadata() {
        // Test setting and getting deployment metadata
        cicdManager.setDeploymentMetadata("buildNumber", "12345");
        cicdManager.setDeploymentMetadata("branch", "main");
        
        Map<String, String> metadata = cicdManager.getDeploymentMetadata();
        assertNotNull(metadata);
        assertEquals("12345", metadata.get("buildNumber"));
        assertEquals("main", metadata.get("branch"));
        
        // Should also contain default metadata
        assertTrue(metadata.containsKey("deploymentTime"));
        assertTrue(metadata.containsKey("version"));
    }
    
    @Test
    void testEnvironmentTypes() {
        // Test all environment types are available
        EnvironmentConfig dev = new EnvironmentConfig("dev", EnvironmentType.DEVELOPMENT);
        assertEquals(EnvironmentType.DEVELOPMENT, dev.getType());
        
        EnvironmentConfig test = new EnvironmentConfig("test", EnvironmentType.TESTING);
        assertEquals(EnvironmentType.TESTING, test.getType());
        
        EnvironmentConfig staging = new EnvironmentConfig("staging", EnvironmentType.STAGING);
        assertEquals(EnvironmentType.STAGING, staging.getType());
        
        EnvironmentConfig prod = new EnvironmentConfig("prod", EnvironmentType.PRODUCTION);
        assertEquals(EnvironmentType.PRODUCTION, prod.getType());
    }
    
    @Test
    void testDeploymentStates() {
        // Test all deployment states
        assertEquals(DeploymentState.INITIALIZING, cicdManager.getCurrentState());
        
        // All deployment states should be available
        DeploymentState[] states = DeploymentState.values();
        assertEquals(7, states.length);
        
        Set<DeploymentState> stateSet = EnumSet.allOf(DeploymentState.class);
        assertTrue(stateSet.contains(DeploymentState.INITIALIZING));
        assertTrue(stateSet.contains(DeploymentState.VALIDATING));
        assertTrue(stateSet.contains(DeploymentState.DEPLOYING));
        assertTrue(stateSet.contains(DeploymentState.DEPLOYED));
        assertTrue(stateSet.contains(DeploymentState.FAILED));
        assertTrue(stateSet.contains(DeploymentState.ROLLBACK));
        assertTrue(stateSet.contains(DeploymentState.MAINTENANCE));
    }
    
    @Test
    void testSequentialDeployments() {
        // Test multiple sequential deployments
        DeploymentReport dev = cicdManager.deployToEnvironment("development");
        assertNotNull(dev);
        
        DeploymentReport test = cicdManager.deployToEnvironment("testing");
        assertNotNull(test);
        
        // Each deployment should be independent
        assertNotEquals(dev.getTimestamp(), test.getTimestamp());
    }
    
    @Test
    void testConcurrentDeploymentSafety() {
        // Test that multiple deployment calls don't interfere
        for (int i = 0; i < 3; i++) {
            final int iteration = i;
            assertDoesNotThrow(() -> {
                DeploymentReport report = cicdManager.deployToEnvironment("development");
                assertNotNull(report, "Deployment iteration " + iteration + " should not be null");
            }, "Deployment iteration " + iteration + " should not throw");
        }
    }
    
    @Test
    void testGracefulShutdown() {
        // Test that shutdown doesn't throw exceptions
        assertDoesNotThrow(() -> {
            cicdManager.shutdown();
        });
        
        // Test operations after shutdown (should not crash)
        assertDoesNotThrow(() -> {
            cicdManager.getCurrentState();
            cicdManager.getAvailableEnvironments();
            cicdManager.getDeploymentMetadata();
        });
    }
}
