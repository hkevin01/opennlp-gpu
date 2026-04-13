package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**

 * ID: GPU-CC-001
 * Requirement: ComputeConfiguration must expose all configurable parameters for compute provider selection and behaviour.
 * Purpose: Centralises GPU/CPU compute tuning parameters (provider preference, benchmark caching, problem-size thresholds) into a single value object.
 * Rationale: Separating configuration from behaviour simplifies testing and hot-reload of compute parameters without rebuilding providers.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: None beyond storing configuration state in memory.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class ComputeConfiguration {

    // The preferred provider type, null for automatic selection
    private ComputeProvider.Type preferredProviderType = null;

    // Problem size threshold below which CPU is preferred
    private int smallProblemThreshold = 1000;

    // Whether to perform automatic benchmarking
    private boolean autoBenchmark = true;

    // How long (in ms) benchmark results are considered valid
    private long benchmarkCacheTimeMs = 3600000; // 1 hour

    // Provider-specific options
    private final Map<String, String> providerOptions = new HashMap<String, String>();

    /**
     * Default constructor.
     */
    /**
    
     * ID: GPU-CC-002
     * Requirement: ComputeConfiguration must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a ComputeConfiguration instance.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public ComputeConfiguration() {
        // Default constructor with no parameters
    }

    /**
     * Creates a configuration from a Properties object.
     *
     * @param properties the properties to load
     */
    /**
    
     * ID: GPU-CC-003
     * Requirement: ComputeConfiguration must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a ComputeConfiguration instance.
     * Inputs: Properties properties
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public ComputeConfiguration(Properties properties) {
        // Load provider type
        String providerTypeStr = properties.getProperty("compute.provider");
        if (providerTypeStr != null && !providerTypeStr.isEmpty()) {
            try {
                preferredProviderType = ComputeProvider.Type.valueOf(providerTypeStr.toUpperCase());
            } catch (IllegalArgumentException e) {
                // Invalid provider type, ignore
            }
        }

        // Load small problem threshold
        String thresholdStr = properties.getProperty("compute.smallProblemThreshold");
        if (thresholdStr != null && !thresholdStr.isEmpty()) {
            try {
                smallProblemThreshold = Integer.parseInt(thresholdStr);
            } catch (NumberFormatException e) {
                // Invalid threshold, ignore
            }
        }

        // Load auto benchmark flag
        String autoBenchmarkStr = properties.getProperty("compute.autoBenchmark");
        if (autoBenchmarkStr != null && !autoBenchmarkStr.isEmpty()) {
            autoBenchmark = Boolean.parseBoolean(autoBenchmarkStr);
        }

        // Load benchmark cache time
        String cacheTimeStr = properties.getProperty("compute.benchmarkCacheTimeMs");
        if (cacheTimeStr != null && !cacheTimeStr.isEmpty()) {
            try {
                benchmarkCacheTimeMs = Long.parseLong(cacheTimeStr);
            } catch (NumberFormatException e) {
                // Invalid cache time, ignore
            }
        }

        // Load provider-specific options
        for (String key : properties.stringPropertyNames()) {
            if (key.startsWith("provider.")) {
                providerOptions.put(key, properties.getProperty(key));
            }
        }
    }

    /**
     * Get a provider-specific option.
     *
     * @param key the option key
     * @return the option value, or null if not set
     */
    /**
    
     * ID: GPU-CC-004
     * Requirement: Return the ProviderOption field value without side effects.
     * Purpose: Return the value of the ProviderOption property.
     * Inputs: String key
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public String getProviderOption(String key) {
        return providerOptions.get("provider." + key);
    }

    /**
     * Set a provider-specific option.
     *
     * @param key the option key
     * @param value the option value
     */
    /**
    
     * ID: GPU-CC-005
     * Requirement: Update the ProviderOption field to the supplied non-null value.
     * Purpose: Set the ProviderOption property to the supplied value.
     * Inputs: String key, String value
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setProviderOption(String key, String value) {
        providerOptions.put("provider." + key, value);
    }

    /**
     * Get all provider-specific options.
     *
     * @return a map of option keys to values
     */
    /**
    
     * ID: GPU-CC-006
     * Requirement: Return the AllProviderOptions field value without side effects.
     * Purpose: Return the value of the AllProviderOptions property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Map<String, String> getAllProviderOptions() {
        return new HashMap<>(providerOptions);
    }

    public ComputeProvider.Type getPreferredProviderType() {
        return preferredProviderType;
    }

    /**
    
     * ID: GPU-CC-007
     * Requirement: Update the PreferredProviderType field to the supplied non-null value.
     * Purpose: Set the PreferredProviderType property to the supplied value.
     * Inputs: ComputeProvider.Type preferredProviderType
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setPreferredProviderType(ComputeProvider.Type preferredProviderType) {
        this.preferredProviderType = preferredProviderType;
    }

    /**
    
     * ID: GPU-CC-008
     * Requirement: Return the SmallProblemThreshold field value without side effects.
     * Purpose: Return the value of the SmallProblemThreshold property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getSmallProblemThreshold() {
        return smallProblemThreshold;
    }

    /**
    
     * ID: GPU-CC-009
     * Requirement: Update the SmallProblemThreshold field to the supplied non-null value.
     * Purpose: Set the SmallProblemThreshold property to the supplied value.
     * Inputs: int smallProblemThreshold
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setSmallProblemThreshold(int smallProblemThreshold) {
        this.smallProblemThreshold = smallProblemThreshold;
    }

    /**
    
     * ID: GPU-CC-010
     * Requirement: Evaluate and return the boolean result of isAutoBenchmark.
     * Purpose: Return whether isAutoBenchmark condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public boolean isAutoBenchmark() {
        return autoBenchmark;
    }

    /**
    
     * ID: GPU-CC-011
     * Requirement: Update the AutoBenchmark field to the supplied non-null value.
     * Purpose: Set the AutoBenchmark property to the supplied value.
     * Inputs: boolean autoBenchmark
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setAutoBenchmark(boolean autoBenchmark) {
        this.autoBenchmark = autoBenchmark;
    }

    /**
    
     * ID: GPU-CC-012
     * Requirement: Return the BenchmarkCacheTimeMs field value without side effects.
     * Purpose: Return the value of the BenchmarkCacheTimeMs property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public long getBenchmarkCacheTimeMs() {
        return benchmarkCacheTimeMs;
    }

    /**
    
     * ID: GPU-CC-013
     * Requirement: Update the BenchmarkCacheTimeMs field to the supplied non-null value.
     * Purpose: Set the BenchmarkCacheTimeMs property to the supplied value.
     * Inputs: long benchmarkCacheTimeMs
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setBenchmarkCacheTimeMs(long benchmarkCacheTimeMs) {
        this.benchmarkCacheTimeMs = benchmarkCacheTimeMs;
    }

    /**
    
     * ID: GPU-CC-014
     * Requirement: Return the ProviderOptions field value without side effects.
     * Purpose: Return the value of the ProviderOptions property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public Map<String, String> getProviderOptions() {
        return providerOptions;
    }

    // Implement equals, hashCode, and toString methods

    /**
    
     * ID: GPU-CC-015
     * Requirement: equals must execute correctly within the contract defined by this class.
     * Purpose: Implement the equals operation for this class.
     * Inputs: Object o
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        ComputeConfiguration that = (ComputeConfiguration) o;
        return smallProblemThreshold == that.smallProblemThreshold &&
               autoBenchmark == that.autoBenchmark &&
               benchmarkCacheTimeMs == that.benchmarkCacheTimeMs &&
               Objects.equals(preferredProviderType, that.preferredProviderType) &&
               Objects.equals(providerOptions, that.providerOptions);
    }

    /**
    
     * ID: GPU-CC-016
     * Requirement: Evaluate and return the boolean result of hashCode.
     * Purpose: Return whether hashCode condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public int hashCode() {
        return Objects.hash(preferredProviderType, smallProblemThreshold, autoBenchmark,
                          benchmarkCacheTimeMs, providerOptions);
    }

    /**
    
     * ID: GPU-CC-017
     * Requirement: toString must execute correctly within the contract defined by this class.
     * Purpose: Implement the toString operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    @Override
    public String toString() {
        return "ComputeConfiguration{" +
               "preferredProviderType=" + preferredProviderType +
               ", smallProblemThreshold=" + smallProblemThreshold +
               ", autoBenchmark=" + autoBenchmark +
               ", benchmarkCacheTimeMs=" + benchmarkCacheTimeMs +
               ", providerOptions=" + providerOptions +
               '}';
    }

    /**
    
     * ID: GPU-CC-018
     * Requirement: Update the PreferredProvider field to the supplied non-null value.
     * Purpose: Set the PreferredProvider property to the supplied value.
     * Inputs: ComputeProvider.Type type
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setPreferredProvider(ComputeProvider.Type type) {
        this.preferredProviderType = type;
    }

    public ComputeProvider.Type getPreferredProvider() {
        return this.preferredProviderType;
    }

    /**
    
     * ID: GPU-CC-019
     * Requirement: createProvider must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new Provider.
     * Inputs: ComputeProvider.Type type
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public ComputeProvider createProvider(ComputeProvider.Type type) {
        return ComputeProviderFactory.createProvider(type);
    }
}
