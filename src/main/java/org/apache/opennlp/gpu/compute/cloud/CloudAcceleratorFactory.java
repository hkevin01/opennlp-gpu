/*
 * Copyright 2025 OpenNLP GPU Extension Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This project is a third-party GPU acceleration extension for Apache OpenNLP.
 * It is not officially endorsed or maintained by the Apache Software Foundation.
 */
package org.apache.opennlp.gpu.compute.cloud;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;

/**
 * Factory for creating and discovering cloud accelerator compute providers.
 *
 * This factory automatically detects available cloud accelerators and provides
 * a unified interface for accessing AWS Inferentia, Google TPU, and traditional
 * GPU compute providers.
 *
 * @author OpenNLP GPU Extension Contributors
 * @since 1.1.0
 */
public final class CloudAcceleratorFactory {

    private static final GpuLogger logger = GpuLogger.getLogger(CloudAcceleratorFactory.class);

    // Available provider instances
    private static volatile List<ComputeProvider> availableProviders = null;
    private static volatile ComputeProvider bestProvider = null;

    private CloudAcceleratorFactory() {
        // Utility class
    }

    /**
     * Get the best available compute provider based on performance characteristics
     * @return The highest-performance available compute provider
     */
    public static ComputeProvider getBestProvider() {
        if (bestProvider == null) {
            bestProvider = detectBestProvider();
        }
        return bestProvider;
    }

    /**
     * Get all available compute providers
     * @return List of all detected compute providers
     */
    public static List<ComputeProvider> getAvailableProviders() {
        if (availableProviders == null) {
            availableProviders = detectAvailableProviders();
        }
        return new ArrayList<>(availableProviders);
    }

    /**
     * Create a specific provider by type
     * @param providerType The type of provider to create
     * @return The requested compute provider, or null if not available
     */
    public static ComputeProvider createProvider(String providerType) {
        switch (providerType.toLowerCase()) {
            case "inferentia":
            case "aws_inferentia":
                return new InferentiaComputeProvider();
            case "tpu":
            case "google_tpu":
                return new TpuComputeProvider();
            case "gpu":
                return new GpuComputeProvider(new org.apache.opennlp.gpu.common.GpuConfig());
            case "cpu":
                return new CpuComputeProvider();
            default:
                logger.warn("Unknown provider type: " + providerType);
                return null;
        }
    }

    /**
     * Check if cloud accelerators are available
     * @return true if AWS Inferentia or Google TPU are detected
     */
    public static boolean hasCloudAccelerators() {
        List<ComputeProvider> providers = getAvailableProviders();
        for (ComputeProvider provider : providers) {
            if (provider instanceof InferentiaComputeProvider ||
                provider instanceof TpuComputeProvider) {
                return true;
            }
        }
        return false;
    }

    /**
     * Get cloud accelerator capabilities summary
     * @return Map containing cloud accelerator information
     */
    public static Map<String, Object> getCloudCapabilities() {
        Map<String, Object> capabilities = new java.util.HashMap<>();

        InferentiaComputeProvider inferentia = new InferentiaComputeProvider();
        TpuComputeProvider tpu = new TpuComputeProvider();

        capabilities.put("aws_inferentia_available", inferentia.isAvailable());
        capabilities.put("google_tpu_available", tpu.isAvailable());
        capabilities.put("cloud_accelerators_detected", hasCloudAccelerators());

        if (inferentia.isAvailable()) {
            capabilities.put("inferentia_info", inferentia.getDeviceProperties());
        }

        if (tpu.isAvailable()) {
            capabilities.put("tpu_info", tpu.getDeviceProperties());
        }

        return capabilities;
    }

    /**
     * Reset factory state (useful for testing)
     */
    public static void reset() {
        availableProviders = null;
        bestProvider = null;
    }

    /**
     * Detect the best available compute provider
     */
    private static ComputeProvider detectBestProvider() {
        List<ComputeProvider> providers = detectAvailableProviders();

        // Priority order: TPU > Inferentia > GPU > CPU
        for (ComputeProvider provider : providers) {
            if (provider instanceof TpuComputeProvider && provider.isAvailable()) {
                logger.info("Selected Google TPU as best compute provider");
                return provider;
            }
        }

        for (ComputeProvider provider : providers) {
            if (provider instanceof InferentiaComputeProvider && provider.isAvailable()) {
                logger.info("Selected AWS Inferentia as best compute provider");
                return provider;
            }
        }

        for (ComputeProvider provider : providers) {
            if (provider instanceof GpuComputeProvider && provider.isAvailable()) {
                logger.info("Selected GPU as best compute provider");
                return provider;
            }
        }

        // Fallback to CPU
        logger.info("No GPU/accelerator found, using CPU compute provider");
        return new CpuComputeProvider();
    }

    /**
     * Detect all available compute providers
     */
    private static List<ComputeProvider> detectAvailableProviders() {
        List<ComputeProvider> providers = new ArrayList<>();

        // Cloud accelerators
        providers.add(new InferentiaComputeProvider());
        providers.add(new TpuComputeProvider());

        // Traditional GPU provider
        providers.add(new GpuComputeProvider(new org.apache.opennlp.gpu.common.GpuConfig()));

        // CPU fallback
        providers.add(new CpuComputeProvider());

        // Filter to only available providers
        List<ComputeProvider> available = new ArrayList<>();
        for (ComputeProvider provider : providers) {
            if (provider.isAvailable()) {
                available.add(provider);
                logger.debug("Detected available provider: " + provider.getName());
            }
        }

        return available;
    }
}
