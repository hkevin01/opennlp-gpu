package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Properties;

/**
 * Configuration for compute providers, allowing fine-tuning of provider selection and behavior.
 */
// Remove Lombok annotations and implement manually
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
    public ComputeConfiguration() {
        // Default constructor with no parameters
    }
    
    /**
     * Creates a configuration from a Properties object.
     *
     * @param properties the properties to load
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
    public String getProviderOption(String key) {
        return providerOptions.get("provider." + key);
    }
    
    /**
     * Set a provider-specific option.
     *
     * @param key the option key
     * @param value the option value
     */
    public void setProviderOption(String key, String value) {
        providerOptions.put("provider." + key, value);
    }
    
    /**
     * Get all provider-specific options.
     *
     * @return a map of option keys to values
     */
    public Map<String, String> getAllProviderOptions() {
        return new HashMap<>(providerOptions);
    }
    
    public ComputeProvider.Type getPreferredProviderType() {
        return preferredProviderType;
    }
    
    public void setPreferredProviderType(ComputeProvider.Type preferredProviderType) {
        this.preferredProviderType = preferredProviderType;
    }
    
    public int getSmallProblemThreshold() {
        return smallProblemThreshold;
    }
    
    public void setSmallProblemThreshold(int smallProblemThreshold) {
        this.smallProblemThreshold = smallProblemThreshold;
    }
    
    public boolean isAutoBenchmark() {
        return autoBenchmark;
    }
    
    public void setAutoBenchmark(boolean autoBenchmark) {
        this.autoBenchmark = autoBenchmark;
    }
    
    public long getBenchmarkCacheTimeMs() {
        return benchmarkCacheTimeMs;
    }
    
    public void setBenchmarkCacheTimeMs(long benchmarkCacheTimeMs) {
        this.benchmarkCacheTimeMs = benchmarkCacheTimeMs;
    }
    
    public Map<String, String> getProviderOptions() {
        return providerOptions;
    }
    
    // Implement equals, hashCode, and toString methods
    
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
    
    @Override
    public int hashCode() {
        return Objects.hash(preferredProviderType, smallProblemThreshold, autoBenchmark, 
                          benchmarkCacheTimeMs, providerOptions);
    }
    
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
}
