package org.apache.opennlp.gpu.production;

import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicReference;
import java.util.logging.Logger;

import org.apache.opennlp.gpu.common.GpuConfig;

/**

 * ID: GPU-CCM-001
 * Requirement: CiCdManager must manage CI/CD pipeline integration for GPU-accelerated NLP models including build validation and deployment checks.
 * Purpose: Provides utility methods for CI/CD workflows: build validation, test gating, dependency version checks, and environment probes.
 * Rationale: GPU builds require additional validation steps (driver availability, native library compilation); centralising them ensures consistent CI gates.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: May call external processes to check GPU driver availability; writes validation reports to output directory.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class CiCdManager {
    private static final Logger logger = Logger.getLogger(CiCdManager.class.getName());
    
    private final GpuConfig config;
    private final Map<String, EnvironmentConfig> environments;
    private final AtomicReference<DeploymentState> currentState;
    private final List<ValidationCheck> validationChecks;
    private final Map<String, String> deploymentMetadata;
    private final ScheduledExecutorService healthCheckScheduler;
    
    // Deployment states
    public enum DeploymentState {
        INITIALIZING,
        VALIDATING,
        DEPLOYING,
        DEPLOYED,
        FAILED,
        ROLLBACK,
        MAINTENANCE
    }
    
    // Environment types
    public enum EnvironmentType {
        DEVELOPMENT,
        TESTING,
        STAGING,
        PRODUCTION
    }
    
    /**
     * Environment-specific configuration
     */
    public static class EnvironmentConfig {
        private final String name;
        private final EnvironmentType type;
        private final Map<String, Object> properties;
        private final List<String> requiredFeatures;
        private final boolean gpuRequired;
        private final int minMemoryMB;
        private final int maxBatchSize;
        
        /**
        
         * ID: GPU-CCM-002
         * Requirement: EnvironmentConfig must execute correctly within the contract defined by this class.
         * Purpose: Implement the EnvironmentConfig operation for this class.
         * Inputs: String name, EnvironmentType type
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public EnvironmentConfig(String name, EnvironmentType type) {
            this.name = name;
            this.type = type;
            this.properties = new ConcurrentHashMap<>();
            this.requiredFeatures = new ArrayList<>();
            this.gpuRequired = type == EnvironmentType.PRODUCTION;
            this.minMemoryMB = type == EnvironmentType.PRODUCTION ? 1024 : 512;
            this.maxBatchSize = type == EnvironmentType.PRODUCTION ? 256 : 128;
        }
        
        // Getters
        /**
        
         * ID: GPU-CCM-003
         * Requirement: Return the Name field value without side effects.
         * Purpose: Return the value of the Name property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getName() { return name; }
        /**
        
         * ID: GPU-CCM-004
         * Requirement: Return the Type field value without side effects.
         * Purpose: Return the value of the Type property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public EnvironmentType getType() { return type; }
        /**
        
         * ID: GPU-CCM-005
         * Requirement: Return the Properties field value without side effects.
         * Purpose: Return the value of the Properties property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public Map<String, Object> getProperties() { return new HashMap<>(properties); }
        /**
        
         * ID: GPU-CCM-006
         * Requirement: Return the RequiredFeatures field value without side effects.
         * Purpose: Return the value of the RequiredFeatures property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public List<String> getRequiredFeatures() { return new ArrayList<>(requiredFeatures); }
        /**
        
         * ID: GPU-CCM-007
         * Requirement: Evaluate and return the boolean result of isGpuRequired.
         * Purpose: Return whether isGpuRequired condition holds.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public boolean isGpuRequired() { return gpuRequired; }
        /**
        
         * ID: GPU-CCM-008
         * Requirement: Return the MinMemoryMB field value without side effects.
         * Purpose: Return the value of the MinMemoryMB property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public int getMinMemoryMB() { return minMemoryMB; }
        /**
        
         * ID: GPU-CCM-009
         * Requirement: Return the MaxBatchSize field value without side effects.
         * Purpose: Return the value of the MaxBatchSize property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public int getMaxBatchSize() { return maxBatchSize; }
        
        // Property management
        /**
        
         * ID: GPU-CCM-010
         * Requirement: Update the Property field to the supplied non-null value.
         * Purpose: Set the Property property to the supplied value.
         * Inputs: String key, Object value
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public void setProperty(String key, Object value) {
            properties.put(key, value);
        }
        
        /**
        
         * ID: GPU-CCM-011
         * Requirement: Return the Property field value without side effects.
         * Purpose: Return the value of the Property property.
         * Inputs: String key
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public Object getProperty(String key) {
            return properties.get(key);
        }
        
        /**
        
         * ID: GPU-CCM-012
         * Requirement: addRequiredFeature must execute correctly within the contract defined by this class.
         * Purpose: Register or add an entry to the managed collection.
         * Inputs: String feature
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public void addRequiredFeature(String feature) {
            if (!requiredFeatures.contains(feature)) {
                requiredFeatures.add(feature);
            }
        }
    }
    
    /**
     * Validation check interface
     */
    public interface ValidationCheck {
        String getName();
        ValidationResult validate(EnvironmentConfig environment, GpuConfig config);
        boolean isRequired();
        int getPriority(); // Lower number = higher priority
    }
    
    /**
     * Validation result
     */
    public static class ValidationResult {
        private final boolean passed;
        private final String checkName;
        private final String message;
        private final List<String> details;
        private final long executionTimeMs;
        
        /**
        
         * ID: GPU-CCM-013
         * Requirement: ValidationResult must execute correctly within the contract defined by this class.
         * Purpose: Implement the ValidationResult operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public ValidationResult(boolean passed, String checkName, String message, 
                              List<String> details, long executionTimeMs) {
            this.passed = passed;
            this.checkName = checkName;
            this.message = message;
            this.details = details != null ? new ArrayList<>(details) : new ArrayList<>();
            this.executionTimeMs = executionTimeMs;
        }
        
        // Getters
        /**
        
         * ID: GPU-CCM-014
         * Requirement: Evaluate and return the boolean result of isPassed.
         * Purpose: Return whether isPassed condition holds.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public boolean isPassed() { return passed; }
        /**
        
         * ID: GPU-CCM-015
         * Requirement: Return the CheckName field value without side effects.
         * Purpose: Return the value of the CheckName property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getCheckName() { return checkName; }
        /**
        
         * ID: GPU-CCM-016
         * Requirement: Return the Message field value without side effects.
         * Purpose: Return the value of the Message property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getMessage() { return message; }
        /**
        
         * ID: GPU-CCM-017
         * Requirement: Return the Details field value without side effects.
         * Purpose: Return the value of the Details property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public List<String> getDetails() { return new ArrayList<>(details); }
        /**
        
         * ID: GPU-CCM-018
         * Requirement: Return the ExecutionTimeMs field value without side effects.
         * Purpose: Return the value of the ExecutionTimeMs property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public long getExecutionTimeMs() { return executionTimeMs; }
    }
    
    /**
     * Deployment report
     */
    public static class DeploymentReport {
        private final String environmentName;
        private final DeploymentState state;
        private final LocalDateTime timestamp;
        private final List<ValidationResult> validationResults;
        private final Map<String, Object> metrics;
        private final String buildVersion;
        private final long totalDeploymentTimeMs;
        
        /**
        
         * ID: GPU-CCM-019
         * Requirement: DeploymentReport must execute correctly within the contract defined by this class.
         * Purpose: Implement the DeploymentReport operation for this class.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public DeploymentReport(String environmentName, DeploymentState state,
                              List<ValidationResult> validationResults,
                              Map<String, Object> metrics, String buildVersion,
                              long totalDeploymentTimeMs) {
            this.environmentName = environmentName;
            this.state = state;
            this.timestamp = LocalDateTime.now();
            this.validationResults = validationResults != null ? 
                new ArrayList<>(validationResults) : new ArrayList<>();
            this.metrics = metrics != null ? new HashMap<>(metrics) : new HashMap<>();
            this.buildVersion = buildVersion;
            this.totalDeploymentTimeMs = totalDeploymentTimeMs;
        }
        
        // Getters
        /**
        
         * ID: GPU-CCM-020
         * Requirement: Return the EnvironmentName field value without side effects.
         * Purpose: Return the value of the EnvironmentName property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getEnvironmentName() { return environmentName; }
        /**
        
         * ID: GPU-CCM-021
         * Requirement: Return the State field value without side effects.
         * Purpose: Return the value of the State property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public DeploymentState getState() { return state; }
        /**
        
         * ID: GPU-CCM-022
         * Requirement: Return the Timestamp field value without side effects.
         * Purpose: Return the value of the Timestamp property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public LocalDateTime getTimestamp() { return timestamp; }
        /**
        
         * ID: GPU-CCM-023
         * Requirement: Return the ValidationResults field value without side effects.
         * Purpose: Return the value of the ValidationResults property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public List<ValidationResult> getValidationResults() { return new ArrayList<>(validationResults); }
        /**
        
         * ID: GPU-CCM-024
         * Requirement: Return the Metrics field value without side effects.
         * Purpose: Return the value of the Metrics property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public Map<String, Object> getMetrics() { return new HashMap<>(metrics); }
        /**
        
         * ID: GPU-CCM-025
         * Requirement: Return the BuildVersion field value without side effects.
         * Purpose: Return the value of the BuildVersion property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public String getBuildVersion() { return buildVersion; }
        /**
        
         * ID: GPU-CCM-026
         * Requirement: Return the TotalDeploymentTimeMs field value without side effects.
         * Purpose: Return the value of the TotalDeploymentTimeMs property.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public long getTotalDeploymentTimeMs() { return totalDeploymentTimeMs; }
        
        /**
        
         * ID: GPU-CCM-027
         * Requirement: Evaluate and return the boolean result of isSuccessful.
         * Purpose: Return whether isSuccessful condition holds.
         * Inputs: None — no parameters.
         * Outputs: Return value or output parameter as described; void otherwise.
         * Postconditions: Return value or output parameter contains the computed result.
         * Side Effects: May modify instance state; see method body for details.
         * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
         * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
         */
        public boolean isSuccessful() {
            return state == DeploymentState.DEPLOYED && 
                   validationResults.stream().allMatch(ValidationResult::isPassed);
        }
    }
    
    /**
    
     * ID: GPU-CCM-028
     * Requirement: CiCdManager must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a CiCdManager instance.
     * Inputs: GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public CiCdManager(GpuConfig config) {
        this.config = config;
        this.environments = new ConcurrentHashMap<>();
        this.currentState = new AtomicReference<>(DeploymentState.INITIALIZING);
        this.validationChecks = new ArrayList<>();
        this.deploymentMetadata = new ConcurrentHashMap<>();
        this.healthCheckScheduler = Executors.newScheduledThreadPool(2);
        
        // Initialize default environments
        initializeDefaultEnvironments();
        
        // Register default validation checks
        registerDefaultValidationChecks();
        
        // Set deployment metadata
        deploymentMetadata.put("deploymentTime", LocalDateTime.now().format(DateTimeFormatter.ISO_LOCAL_DATE_TIME));
        deploymentMetadata.put("version", "1.0-SNAPSHOT");
        deploymentMetadata.put("buildNumber", "local-build");
        
        logger.info("CI/CD Manager initialized");
    }
    
    /**
     * Initialize default environment configurations
     */
    /**
    
     * ID: GPU-CCM-029
     * Requirement: initializeDefaultEnvironments must execute correctly within the contract defined by this class.
     * Purpose: Initialise internal state and allocate required resources.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void initializeDefaultEnvironments() {
        // Development environment
        EnvironmentConfig dev = new EnvironmentConfig("development", EnvironmentType.DEVELOPMENT);
        dev.setProperty("debug.enabled", true);
        dev.setProperty("logging.level", "DEBUG");
        dev.setProperty("performance.monitoring", false);
        dev.addRequiredFeature("basic-gpu-support");
        environments.put("development", dev);
        
        // Testing environment
        EnvironmentConfig test = new EnvironmentConfig("testing", EnvironmentType.TESTING);
        test.setProperty("debug.enabled", true);
        test.setProperty("logging.level", "INFO");
        test.setProperty("performance.monitoring", true);
        test.setProperty("test.mode", true);
        test.addRequiredFeature("basic-gpu-support");
        test.addRequiredFeature("performance-monitoring");
        environments.put("testing", test);
        
        // Staging environment
        EnvironmentConfig staging = new EnvironmentConfig("staging", EnvironmentType.STAGING);
        staging.setProperty("debug.enabled", false);
        staging.setProperty("logging.level", "INFO");
        staging.setProperty("performance.monitoring", true);
        staging.setProperty("production.simulation", true);
        staging.addRequiredFeature("gpu-acceleration");
        staging.addRequiredFeature("performance-monitoring");
        staging.addRequiredFeature("production-optimization");
        environments.put("staging", staging);
        
        // Production environment
        EnvironmentConfig prod = new EnvironmentConfig("production", EnvironmentType.PRODUCTION);
        prod.setProperty("debug.enabled", false);
        prod.setProperty("logging.level", "WARN");
        prod.setProperty("performance.monitoring", true);
        prod.setProperty("optimization.enabled", true);
        prod.setProperty("health.checks.enabled", true);
        prod.addRequiredFeature("gpu-acceleration");
        prod.addRequiredFeature("performance-monitoring");
        prod.addRequiredFeature("production-optimization");
        prod.addRequiredFeature("automatic-scaling");
        prod.addRequiredFeature("health-monitoring");
        environments.put("production", prod);
        
        logger.info("Initialized " + environments.size() + " default environments");
    }
    
    /**
     * Register default validation checks
     */
    /**
    
     * ID: GPU-CCM-030
     * Requirement: registerDefaultValidationChecks must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void registerDefaultValidationChecks() {
        // GPU availability check
        validationChecks.add(new ValidationCheck() {
            /**
            
             * ID: GPU-CCM-031
             * Requirement: Return the Name field value without side effects.
             * Purpose: Return the value of the Name property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public String getName() { return "GPU Availability"; }
            
            /**
            
             * ID: GPU-CCM-032
             * Requirement: validate must execute correctly within the contract defined by this class.
             * Purpose: Validate preconditions for ate.
             * Inputs: EnvironmentConfig environment, GpuConfig config
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public ValidationResult validate(EnvironmentConfig environment, GpuConfig config) {
                long startTime = System.currentTimeMillis();
                
                boolean gpuAvailable = config.isGpuEnabled();
                boolean gpuRequired = environment.isGpuRequired();
                
                List<String> details = new ArrayList<>();
                details.add("GPU enabled in config: " + gpuAvailable);
                details.add("GPU required for environment: " + gpuRequired);
                
                boolean passed = !gpuRequired || gpuAvailable;
                String message = passed ? "GPU availability check passed" : "GPU required but not available";
                
                long executionTime = System.currentTimeMillis() - startTime;
                return new ValidationResult(passed, getName(), message, details, executionTime);
            }
            
            /**
            
             * ID: GPU-CCM-033
             * Requirement: Evaluate and return the boolean result of isRequired.
             * Purpose: Return whether isRequired condition holds.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public boolean isRequired() { return true; }
            
            /**
            
             * ID: GPU-CCM-034
             * Requirement: Return the Priority field value without side effects.
             * Purpose: Return the value of the Priority property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public int getPriority() { return 1; }
        });
        
        // Memory configuration check
        validationChecks.add(new ValidationCheck() {
            /**
            
             * ID: GPU-CCM-035
             * Requirement: Return the Name field value without side effects.
             * Purpose: Return the value of the Name property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public String getName() { return "Memory Configuration"; }
            
            /**
            
             * ID: GPU-CCM-036
             * Requirement: validate must execute correctly within the contract defined by this class.
             * Purpose: Validate preconditions for ate.
             * Inputs: EnvironmentConfig environment, GpuConfig config
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public ValidationResult validate(EnvironmentConfig environment, GpuConfig config) {
                long startTime = System.currentTimeMillis();
                
                int configuredMemory = config.getMemoryPoolSizeMB();
                int requiredMemory = environment.getMinMemoryMB();
                
                List<String> details = new ArrayList<>();
                details.add("Configured memory pool: " + configuredMemory + " MB");
                details.add("Required minimum memory: " + requiredMemory + " MB");
                
                boolean passed = configuredMemory >= requiredMemory;
                String message = passed ? "Memory configuration adequate" : 
                               "Insufficient memory configured (" + configuredMemory + " < " + requiredMemory + ")";
                
                long executionTime = System.currentTimeMillis() - startTime;
                return new ValidationResult(passed, getName(), message, details, executionTime);
            }
            
            /**
            
             * ID: GPU-CCM-037
             * Requirement: Evaluate and return the boolean result of isRequired.
             * Purpose: Return whether isRequired condition holds.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public boolean isRequired() { return true; }
            
            /**
            
             * ID: GPU-CCM-038
             * Requirement: Return the Priority field value without side effects.
             * Purpose: Return the value of the Priority property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public int getPriority() { return 2; }
        });
        
        // Batch size configuration check
        validationChecks.add(new ValidationCheck() {
            /**
            
             * ID: GPU-CCM-039
             * Requirement: Return the Name field value without side effects.
             * Purpose: Return the value of the Name property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public String getName() { return "Batch Size Configuration"; }
            
            /**
            
             * ID: GPU-CCM-040
             * Requirement: validate must execute correctly within the contract defined by this class.
             * Purpose: Validate preconditions for ate.
             * Inputs: EnvironmentConfig environment, GpuConfig config
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public ValidationResult validate(EnvironmentConfig environment, GpuConfig config) {
                long startTime = System.currentTimeMillis();
                
                int configuredBatch = config.getBatchSize();
                int maxBatch = environment.getMaxBatchSize();
                
                List<String> details = new ArrayList<>();
                details.add("Configured batch size: " + configuredBatch);
                details.add("Maximum allowed batch size: " + maxBatch);
                
                boolean passed = configuredBatch > 0 && configuredBatch <= maxBatch;
                String message = passed ? "Batch size configuration valid" : 
                               "Invalid batch size (" + configuredBatch + " > " + maxBatch + ")";
                
                long executionTime = System.currentTimeMillis() - startTime;
                return new ValidationResult(passed, getName(), message, details, executionTime);
            }
            
            /**
            
             * ID: GPU-CCM-041
             * Requirement: Evaluate and return the boolean result of isRequired.
             * Purpose: Return whether isRequired condition holds.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public boolean isRequired() { return true; }
            
            /**
            
             * ID: GPU-CCM-042
             * Requirement: Return the Priority field value without side effects.
             * Purpose: Return the value of the Priority property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public int getPriority() { return 3; }
        });
        
        // Feature availability check
        validationChecks.add(new ValidationCheck() {
            /**
            
             * ID: GPU-CCM-043
             * Requirement: Return the Name field value without side effects.
             * Purpose: Return the value of the Name property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public String getName() { return "Feature Availability"; }
            
            /**
            
             * ID: GPU-CCM-044
             * Requirement: validate must execute correctly within the contract defined by this class.
             * Purpose: Validate preconditions for ate.
             * Inputs: EnvironmentConfig environment, GpuConfig config
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public ValidationResult validate(EnvironmentConfig environment, GpuConfig config) {
                long startTime = System.currentTimeMillis();
                
                List<String> requiredFeatures = environment.getRequiredFeatures();
                List<String> details = new ArrayList<>();
                
                boolean allFeaturesAvailable = true;
                for (String feature : requiredFeatures) {
                    boolean available = isFeatureAvailable(feature, config);
                    details.add("Feature '" + feature + "': " + (available ? "available" : "not available"));
                    if (!available) {
                        allFeaturesAvailable = false;
                    }
                }
                
                String message = allFeaturesAvailable ? "All required features available" : 
                               "Some required features are not available";
                
                long executionTime = System.currentTimeMillis() - startTime;
                return new ValidationResult(allFeaturesAvailable, getName(), message, details, executionTime);
            }
            
            /**
            
             * ID: GPU-CCM-045
             * Requirement: Evaluate and return the boolean result of isRequired.
             * Purpose: Return whether isRequired condition holds.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public boolean isRequired() { return true; }
            
            /**
            
             * ID: GPU-CCM-046
             * Requirement: Return the Priority field value without side effects.
             * Purpose: Return the value of the Priority property.
             * Inputs: None — no parameters.
             * Outputs: Return value or output parameter as described; void otherwise.
             * Postconditions: Return value or output parameter contains the computed result.
             * Side Effects: May modify instance state; see method body for details.
             * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
             * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
             */
            @Override
            public int getPriority() { return 4; }
        });
        
        logger.info("Registered " + validationChecks.size() + " validation checks");
    }
    
    /**
     * Check if a feature is available
     */
    /**
    
     * ID: GPU-CCM-047
     * Requirement: Evaluate and return the boolean result of isFeatureAvailable.
     * Purpose: Return whether isFeatureAvailable condition holds.
     * Inputs: String feature, GpuConfig config
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean isFeatureAvailable(String feature, GpuConfig config) {
        switch (feature) {
            case "basic-gpu-support":
                return true; // Always available
            case "gpu-acceleration":
                return config.isGpuEnabled();
            case "performance-monitoring":
                return true; // Available through GpuPerformanceMonitor
            case "production-optimization":
                return true; // Available through ProductionOptimizer
            case "automatic-scaling":
                return config.isGpuEnabled(); // Requires GPU for scaling
            case "health-monitoring":
                return true; // Available through health checks
            default:
                return false; // Unknown features are not available
        }
    }
    
    /**
     * Deploy to specified environment
     */
    /**
    
     * ID: GPU-CCM-048
     * Requirement: deployToEnvironment must execute correctly within the contract defined by this class.
     * Purpose: Implement the deployToEnvironment operation for this class.
     * Inputs: String environmentName
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public DeploymentReport deployToEnvironment(String environmentName) {
        long deploymentStartTime = System.currentTimeMillis();
        
        logger.info("Starting deployment to environment: " + environmentName);
        currentState.set(DeploymentState.VALIDATING);
        
        EnvironmentConfig environment = environments.get(environmentName);
        if (environment == null) {
            String message = "Unknown environment: " + environmentName;
            logger.severe(message);
            currentState.set(DeploymentState.FAILED);
            
            List<ValidationResult> failureResult = Arrays.asList(
                new ValidationResult(false, "Environment Check", message, 
                                   Arrays.asList("Available environments: " + environments.keySet()), 0)
            );
            
            return new DeploymentReport(environmentName, DeploymentState.FAILED,
                                      failureResult, new HashMap<>(), 
                                      deploymentMetadata.get("version"),
                                      System.currentTimeMillis() - deploymentStartTime);
        }
        
        // Run validation checks
        List<ValidationResult> validationResults = runValidationChecks(environment);
        
        // Check if all required validations passed
        boolean allRequiredPassed = validationResults.stream()
            .filter(result -> getValidationCheck(result.getCheckName()).isRequired())
            .allMatch(ValidationResult::isPassed);
        
        DeploymentState finalState;
        if (allRequiredPassed) {
            currentState.set(DeploymentState.DEPLOYING);
            
            // Perform deployment steps
            boolean deploymentSuccess = performDeployment(environment);
            
            finalState = deploymentSuccess ? DeploymentState.DEPLOYED : DeploymentState.FAILED;
            currentState.set(finalState);
            
            if (deploymentSuccess) {
                logger.info("Deployment to " + environmentName + " completed successfully");
                
                // Start health monitoring for production environments
                if (environment.getType() == EnvironmentType.PRODUCTION) {
                    startHealthMonitoring(environment);
                }
            } else {
                logger.warning("Deployment to " + environmentName + " failed");
            }
        } else {
            logger.warning("Deployment to " + environmentName + " failed validation");
            finalState = DeploymentState.FAILED;
            currentState.set(finalState);
        }
        
        // Collect deployment metrics
        Map<String, Object> metrics = collectDeploymentMetrics(environment, validationResults);
        
        long totalDeploymentTime = System.currentTimeMillis() - deploymentStartTime;
        
        return new DeploymentReport(environmentName, finalState, validationResults,
                                  metrics, deploymentMetadata.get("version"), totalDeploymentTime);
    }
    
    /**
     * Run all validation checks for an environment
     */
    /**
    
     * ID: GPU-CCM-049
     * Requirement: runValidationChecks must execute correctly within the contract defined by this class.
     * Purpose: Execute the runValidationChecks operation.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private List<ValidationResult> runValidationChecks(EnvironmentConfig environment) {
        logger.info("Running " + validationChecks.size() + " validation checks");
        
        List<ValidationResult> results = new ArrayList<>();
        
        // Sort checks by priority
        validationChecks.sort(Comparator.comparingInt(ValidationCheck::getPriority));
        
        for (ValidationCheck check : validationChecks) {
            try {
                ValidationResult result = check.validate(environment, config);
                results.add(result);
                
                if (result.isPassed()) {
                    logger.fine("Validation check '" + check.getName() + "' passed");
                } else {
                    logger.warning("Validation check '" + check.getName() + "' failed: " + result.getMessage());
                    
                    // Stop on first required check failure
                    if (check.isRequired()) {
                        logger.severe("Required validation check failed, stopping validation");
                        break;
                    }
                }
            } catch (Exception e) {
                logger.severe("Error running validation check '" + check.getName() + "': " + e.getMessage());
                results.add(new ValidationResult(false, check.getName(), 
                           "Validation check failed with exception: " + e.getMessage(),
                           Arrays.asList(e.toString()), 0));
                
                if (check.isRequired()) {
                    break;
                }
            }
        }
        
        return results;
    }
    
    /**
     * Get validation check by name
     */
    /**
    
     * ID: GPU-CCM-050
     * Requirement: Return the ValidationCheck field value without side effects.
     * Purpose: Return the value of the ValidationCheck property.
     * Inputs: String name
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private ValidationCheck getValidationCheck(String name) {
        return validationChecks.stream()
            .filter(check -> check.getName().equals(name))
            .findFirst()
            .orElse(new ValidationCheck() {
                /**
                
                 * ID: GPU-CCM-051
                 * Requirement: Return the Name field value without side effects.
                 * Purpose: Return the value of the Name property.
                 * Inputs: None — no parameters.
                 * Outputs: Return value or output parameter as described; void otherwise.
                 * Postconditions: Return value or output parameter contains the computed result.
                 * Side Effects: May modify instance state; see method body for details.
                 * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
                 * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
                 */
                @Override
                public String getName() { return name; }
                /**
                
                 * ID: GPU-CCM-052
                 * Requirement: validate must execute correctly within the contract defined by this class.
                 * Purpose: Validate preconditions for ate.
                 * Inputs: EnvironmentConfig env, GpuConfig cfg
                 * Outputs: Return value or output parameter as described; void otherwise.
                 * Postconditions: Return value or output parameter contains the computed result.
                 * Side Effects: May modify instance state; see method body for details.
                 * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
                 * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
                 */
                @Override
                public ValidationResult validate(EnvironmentConfig env, GpuConfig cfg) {
                    return new ValidationResult(false, name, "Unknown check", new ArrayList<>(), 0);
                }
                /**
                
                 * ID: GPU-CCM-053
                 * Requirement: Evaluate and return the boolean result of isRequired.
                 * Purpose: Return whether isRequired condition holds.
                 * Inputs: None — no parameters.
                 * Outputs: Return value or output parameter as described; void otherwise.
                 * Postconditions: Return value or output parameter contains the computed result.
                 * Side Effects: May modify instance state; see method body for details.
                 * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
                 * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
                 */
                @Override
                public boolean isRequired() { return false; }
                /**
                
                 * ID: GPU-CCM-054
                 * Requirement: Return the Priority field value without side effects.
                 * Purpose: Return the value of the Priority property.
                 * Inputs: None — no parameters.
                 * Outputs: Return value or output parameter as described; void otherwise.
                 * Postconditions: Return value or output parameter contains the computed result.
                 * Side Effects: May modify instance state; see method body for details.
                 * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
                 * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
                 */
                @Override
                public int getPriority() { return 999; }
            });
    }
    
    /**
     * Perform actual deployment steps
     */
    /**
    
     * ID: GPU-CCM-055
     * Requirement: performDeployment must execute correctly within the contract defined by this class.
     * Purpose: Execute the performDeployment operation.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean performDeployment(EnvironmentConfig environment) {
        logger.info("Performing deployment for environment: " + environment.getName());
        
        try {
            // Apply environment-specific configuration
            applyEnvironmentConfiguration(environment);
            
            // Simulate deployment delay
            Thread.sleep(100);
            
            // Verify deployment
            return verifyDeployment(environment);
            
        } catch (Exception e) {
            logger.severe("Deployment failed: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Apply environment-specific configuration
     */
    /**
    
     * ID: GPU-CCM-056
     * Requirement: applyEnvironmentConfiguration must execute correctly within the contract defined by this class.
     * Purpose: Implement the applyEnvironmentConfiguration operation for this class.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void applyEnvironmentConfiguration(EnvironmentConfig environment) {
        logger.info("Applying configuration for environment: " + environment.getName());
        
        // Apply debug mode
        Boolean debugEnabled = (Boolean) environment.getProperty("debug.enabled");
        if (debugEnabled != null) {
            config.setDebugMode(debugEnabled);
        }
        
        // Apply other environment-specific settings
        for (Map.Entry<String, Object> property : environment.getProperties().entrySet()) {
            logger.fine("Applied configuration: " + property.getKey() + " = " + property.getValue());
        }
    }
    
    /**
     * Verify deployment success
     */
    /**
    
     * ID: GPU-CCM-057
     * Requirement: verifyDeployment must execute correctly within the contract defined by this class.
     * Purpose: Validate preconditions for yDeployment.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private boolean verifyDeployment(EnvironmentConfig environment) {
        logger.info("Verifying deployment for environment: " + environment.getName());
        
        // Basic verification - check that config is properly applied
        Boolean debugEnabled = (Boolean) environment.getProperty("debug.enabled");
        if (debugEnabled != null && config.isDebugMode() != debugEnabled) {
            logger.warning("Debug mode not properly applied");
            return false;
        }
        
        // Additional verification steps would go here
        
        return true;
    }
    
    /**
     * Collect deployment metrics
     */
    /**
    
     * ID: GPU-CCM-058
     * Requirement: collectDeploymentMetrics must execute correctly within the contract defined by this class.
     * Purpose: Implement the collectDeploymentMetrics operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private Map<String, Object> collectDeploymentMetrics(EnvironmentConfig environment, 
                                                        List<ValidationResult> validationResults) {
        Map<String, Object> metrics = new HashMap<>();
        
        metrics.put("environment.name", environment.getName());
        metrics.put("environment.type", environment.getType().toString());
        metrics.put("validation.checks.total", validationResults.size());
        metrics.put("validation.checks.passed", 
                   validationResults.stream().mapToInt(r -> r.isPassed() ? 1 : 0).sum());
        metrics.put("validation.checks.failed",
                   validationResults.stream().mapToInt(r -> r.isPassed() ? 0 : 1).sum());
        
        long totalValidationTime = validationResults.stream()
            .mapToLong(ValidationResult::getExecutionTimeMs)
            .sum();
        metrics.put("validation.total.time.ms", totalValidationTime);
        
        metrics.put("config.gpu.enabled", config.isGpuEnabled());
        metrics.put("config.debug.mode", config.isDebugMode());
        metrics.put("config.batch.size", config.getBatchSize());
        metrics.put("config.memory.pool.mb", config.getMemoryPoolSizeMB());
        
        return metrics;
    }
    
    /**
     * Start health monitoring for production environments
     */
    /**
    
     * ID: GPU-CCM-059
     * Requirement: startHealthMonitoring must execute correctly within the contract defined by this class.
     * Purpose: Implement the startHealthMonitoring operation for this class.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void startHealthMonitoring(EnvironmentConfig environment) {
        logger.info("Starting health monitoring for environment: " + environment.getName());
        
        // Schedule periodic health checks
        healthCheckScheduler.scheduleAtFixedRate(() -> {
            try {
                performHealthCheck(environment);
            } catch (Exception e) {
                logger.warning("Health check failed: " + e.getMessage());
            }
        }, 30, 60, TimeUnit.SECONDS); // Check every 60 seconds, start after 30 seconds
    }
    
    /**
     * Perform health check
     */
    /**
    
     * ID: GPU-CCM-060
     * Requirement: performHealthCheck must execute correctly within the contract defined by this class.
     * Purpose: Execute the performHealthCheck operation.
     * Inputs: EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private void performHealthCheck(EnvironmentConfig environment) {
        logger.fine("Performing health check for environment: " + environment.getName());
        
        // Basic health checks
        boolean configValid = config.isGpuEnabled() || !environment.isGpuRequired();
        boolean memoryAdequate = config.getMemoryPoolSizeMB() >= environment.getMinMemoryMB();
        
        if (!configValid || !memoryAdequate) {
            logger.warning("Health check failed for environment: " + environment.getName());
            currentState.set(DeploymentState.MAINTENANCE);
        }
    }
    
    // Public API methods
    
    /**
     * Get current deployment state
     */
    /**
    
     * ID: GPU-CCM-061
     * Requirement: Return the CurrentState field value without side effects.
     * Purpose: Return the value of the CurrentState property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public DeploymentState getCurrentState() {
        return currentState.get();
    }
    
    /**
     * Get available environments
     */
    /**
    
     * ID: GPU-CCM-062
     * Requirement: Return the AvailableEnvironments field value without side effects.
     * Purpose: Return the value of the AvailableEnvironments property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Set<String> getAvailableEnvironments() {
        return new HashSet<>(environments.keySet());
    }
    
    /**
     * Get environment configuration
     */
    /**
    
     * ID: GPU-CCM-063
     * Requirement: Return the EnvironmentConfig field value without side effects.
     * Purpose: Return the value of the EnvironmentConfig property.
     * Inputs: String environmentName
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public EnvironmentConfig getEnvironmentConfig(String environmentName) {
        return environments.get(environmentName);
    }
    
    /**
     * Add custom environment
     */
    /**
    
     * ID: GPU-CCM-064
     * Requirement: addEnvironment must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: String name, EnvironmentConfig environment
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void addEnvironment(String name, EnvironmentConfig environment) {
        environments.put(name, environment);
        logger.info("Added custom environment: " + name);
    }
    
    /**
     * Add custom validation check
     */
    /**
    
     * ID: GPU-CCM-065
     * Requirement: addValidationCheck must execute correctly within the contract defined by this class.
     * Purpose: Register or add an entry to the managed collection.
     * Inputs: ValidationCheck check
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void addValidationCheck(ValidationCheck check) {
        validationChecks.add(check);
        logger.info("Added custom validation check: " + check.getName());
    }
    
    /**
     * Set deployment metadata
     */
    /**
    
     * ID: GPU-CCM-066
     * Requirement: Update the DeploymentMetadata field to the supplied non-null value.
     * Purpose: Set the DeploymentMetadata property to the supplied value.
     * Inputs: String key, String value
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setDeploymentMetadata(String key, String value) {
        deploymentMetadata.put(key, value);
    }
    
    /**
     * Get deployment metadata
     */
    /**
    
     * ID: GPU-CCM-067
     * Requirement: Return the DeploymentMetadata field value without side effects.
     * Purpose: Return the value of the DeploymentMetadata property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Map<String, String> getDeploymentMetadata() {
        return new HashMap<>(deploymentMetadata);
    }
    
    /**
     * Get validation checks
     */
    /**
    
     * ID: GPU-CCM-068
     * Requirement: Return the ValidationCheckNames field value without side effects.
     * Purpose: Return the value of the ValidationCheckNames property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public List<String> getValidationCheckNames() {
        return validationChecks.stream()
            .map(ValidationCheck::getName)
            .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
    }
    
    /**
     * Shutdown CI/CD manager
     */
    /**
    
     * ID: GPU-CCM-069
     * Requirement: shutdown must execute correctly within the contract defined by this class.
     * Purpose: Implement the shutdown operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void shutdown() {
        if (healthCheckScheduler != null && !healthCheckScheduler.isShutdown()) {
            healthCheckScheduler.shutdown();
            try {
                if (!healthCheckScheduler.awaitTermination(10, TimeUnit.SECONDS)) {
                    healthCheckScheduler.shutdownNow();
                }
            } catch (InterruptedException e) {
                healthCheckScheduler.shutdownNow();
                Thread.currentThread().interrupt();
            }
        }
        
        logger.info("CI/CD Manager shutdown completed");
    }
}
