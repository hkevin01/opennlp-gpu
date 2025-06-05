package org.apache.opennlp.gpu.common;

import java.util.ArrayList;
import java.util.List;
import java.util.ServiceLoader;
import java.util.concurrent.ConcurrentHashMap;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Factory for creating and selecting compute providers based on
 * hardware availability, problem characteristics, and user preferences.
 */
public class ComputeProviderFactory {
    
    private static final Logger logger = LoggerFactory.getLogger(ComputeProviderFactory.class);
    
    // Singleton instance
    private static ComputeProviderFactory instance;
    
    // Cache of available providers
    private final List<ComputeProvider> availableProviders = new ArrayList<>();
    
    // Cache of performance benchmarks (operationType -> problemSize -> provider -> score)
    private final ConcurrentHashMap<String, ConcurrentHashMap<Integer, ConcurrentHashMap<ComputeProvider, Double>>> 
        benchmarkCache = new ConcurrentHashMap<>();
    
    // User preferences
    private ComputeConfiguration configuration;
    
    /**
     * Private constructor for singleton pattern.
     */
    private ComputeProviderFactory() {
        // Initialize configuration with empty properties
        java.util.Properties defaultProps = new java.util.Properties();
        configuration = new ComputeConfiguration(defaultProps);
        
        discoverProviders();
    }
    
    /**
     * Get the singleton instance of the factory.
     *
     * @return the factory instance
     */
    public static synchronized ComputeProviderFactory getInstance() {
        if (instance == null) {
            instance = new ComputeProviderFactory();
        }
        return instance;
    }
    
    /**
     * Discover available compute providers using Java's ServiceLoader.
     */
    private void discoverProviders() {
        // Clear existing providers
        availableProviders.clear();
        
        // Discover providers using ServiceLoader
        ServiceLoader<ComputeProvider> serviceLoader = ServiceLoader.load(ComputeProvider.class);
        
        for (ComputeProvider provider : serviceLoader) {
            if (provider.initialize() && provider.isAvailable()) {
                availableProviders.add(provider);
                logger.info("Discovered compute provider: {}", provider.getName());
            } else {
                logger.info("Provider {} is not available on this system", provider.getName());
            }
        }
        
        // If no providers were discovered, add the CPU fallback provider
        if (availableProviders.isEmpty()) {
            try {
                ComputeProvider cpuProvider = new CpuComputeProvider();
                if (cpuProvider.initialize()) {
                    availableProviders.add(cpuProvider);
                    logger.info("Added CPU fallback provider");
                }
            } catch (Exception e) {
                logger.error("Failed to initialize CPU fallback provider", e);
            }
        }
    }
    
    /**
     * Get the best compute provider for the specified operation and problem size.
     *
     * @param operationType the type of operation
     * @param problemSize the size of the problem
     * @return the best compute provider, or null if none is available
     */
    public ComputeProvider getBestProvider(String operationType, int problemSize) {
        // Check if user has explicitly specified a provider type
        ComputeProvider.Type preferredType = null;
        // Try to access preferredProviderType directly or via reflection
        try {
            java.lang.reflect.Field field = ComputeConfiguration.class.getDeclaredField("preferredProviderType");
            field.setAccessible(true);
            preferredType = (ComputeProvider.Type) field.get(configuration);
        } catch (Exception e) {
            logger.warn("Could not access preferredProviderType: {}", e.getMessage());
        }
        
        if (preferredType != null) {
            for (ComputeProvider provider : availableProviders) {
                if (provider.getType() == preferredType && provider.supportsOperation(operationType)) {
                    logger.debug("Using user-specified provider: {}", provider.getName());
                    return provider;
                }
            }
            logger.warn("User-specified provider type {} not available, falling back to automatic selection", 
                       preferredType);
        }
        
        // If problem size is below threshold, use CPU provider for small problems
        int smallProblemThreshold = 1000; // Default value from ComputeConfiguration
        // Try to access smallProblemThreshold directly or via reflection
        try {
            java.lang.reflect.Field field = ComputeConfiguration.class.getDeclaredField("smallProblemThreshold");
            field.setAccessible(true);
            smallProblemThreshold = field.getInt(configuration);
        } catch (Exception e) {
            logger.warn("Could not access smallProblemThreshold: {}", e.getMessage());
        }
        
        if (problemSize < smallProblemThreshold) {
            for (ComputeProvider provider : availableProviders) {
                if (provider.getType() == ComputeProvider.Type.CPU && provider.supportsOperation(operationType)) {
                    logger.debug("Using CPU provider for small problem size: {}", problemSize);
                    return provider;
                }
            }
        }
        
        // Find the provider with the best performance score
        ComputeProvider bestProvider = null;
        double bestScore = Double.NEGATIVE_INFINITY;
        
        for (ComputeProvider provider : availableProviders) {
            if (!provider.supportsOperation(operationType)) {
                continue;
            }
            
            double score = getPerformanceScore(provider, operationType, problemSize);
            
            if (score > bestScore) {
                bestScore = score;
                bestProvider = provider;
            }
        }
        
        if (bestProvider != null) {
            logger.debug("Selected provider {} with score {} for operation {} and size {}", 
                       bestProvider.getName(), bestScore, operationType, problemSize);
            return bestProvider;
        }
        
        // If no suitable provider was found, try to use CPU fallback
        for (ComputeProvider provider : availableProviders) {
            if (provider.getType() == ComputeProvider.Type.CPU) {
                logger.warn("No suitable provider found for operation {}, falling back to CPU", operationType);
                return provider;
            }
        }
        
        logger.error("No compute provider available for operation: {}", operationType);
        return null;
    }
    
    /**
     * Get the performance score for a provider, operation type, and problem size.
     * If not cached, perform a benchmark to determine the score.
     *
     * @param provider the compute provider
     * @param operationType the type of operation
     * @param problemSize the size of the problem
     * @return the performance score
     */
    private double getPerformanceScore(ComputeProvider provider, String operationType, int problemSize) {
        // Check if we have a cached benchmark result
        ConcurrentHashMap<Integer, ConcurrentHashMap<ComputeProvider, Double>> opCache = 
            benchmarkCache.computeIfAbsent(operationType, k -> new ConcurrentHashMap<>());
        
        ConcurrentHashMap<ComputeProvider, Double> sizeCache = 
            opCache.computeIfAbsent(problemSize, k -> new ConcurrentHashMap<>());
        
        Double cachedScore = sizeCache.get(provider);
        if (cachedScore != null) {
            return cachedScore;
        }
        
        // No cached result, ask the provider for its score
        double score = provider.getPerformanceScore(operationType, problemSize);
        
        // Cache the result
        sizeCache.put(provider, score);
        
        return score;
    }
    
    /**
     * Get all available compute providers.
     *
     * @return a list of available providers
     */
    public List<ComputeProvider> getAllProviders() {
        return new ArrayList<>(availableProviders);
    }
    
    /**
     * Get a compute provider of the specified type.
     *
     * @param type the provider type
     * @return the provider, or null if not available
     */
    public ComputeProvider getProvider(ComputeProvider.Type type) {
        for (ComputeProvider provider : availableProviders) {
            if (provider.getType() == type) {
                return provider;
            }
        }
        return null;
    }
    
    /**
     * Set the configuration for provider selection.
     *
     * @param configuration the new configuration
     */
    public void setConfiguration(ComputeConfiguration configuration) {
        this.configuration = configuration;
    }
    
    /**
     * Get the current configuration.
     *
     * @return the current configuration
     */
    public ComputeConfiguration getConfiguration() {
        return configuration;
    }
    
    /**
     * Clear the benchmark cache.
     */
    public void clearBenchmarkCache() {
        benchmarkCache.clear();
    }
    
    /**
     * Release all providers and resources.
     */
    public void releaseAll() {
        for (ComputeProvider provider : availableProviders) {
            try {
                provider.release();
            } catch (Exception e) {
                logger.error("Error releasing provider: {}", provider.getName(), e);
            }
        }
        availableProviders.clear();
        benchmarkCache.clear();
    }
}
