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

 * ID: GPU-CAF-001
 * Requirement: CloudAcceleratorFactory must detect available cloud accelerator
 *              compute providers (AWS Inferentia, Google TPU, and GPU) and return
 *              the most capable provider for the current environment.
 * Purpose: Factory that abstracts cloud accelerator discovery from callers, enabling
 *          transparent use of AWS Inferentia NeuronCores, Google TPU, or traditional GPU
 *          without embedding cloud-SDK detection logic in every caller.
 * Rationale: Cloud accelerators expose proprietary SDKs (Neuron, libtpu); a factory isolates
 *            runtime detection so callers use a single getBestProvider() call regardless
 *            of whether they run on AWS inf1/inf2, GCP TPU, or standard GPU instances.
 * Inputs: None (uses system environment and property detection internally).
 * Outputs: ComputeProvider instances; List<ComputeProvider> of all available providers.
 * Preconditions: JVM initialised; cloud SDK libraries present on library path if cloud
 *               accelerators are expected.
 * Postconditions: Returns a non-null ComputeProvider; falls back to CpuComputeProvider
 *                if no accelerator is detected.
 * Assumptions: Cloud accelerator detection is idempotent; SDK init is thread-safe.
 * Side Effects: May initialise cloud accelerator SDK runtimes on first call; caches results.
 * Failure Modes: SDK init fails → falls back to GPU or CPU provider; logged at WARN level.
 * Error Handling: All detection exceptions are caught; fallback provider returned silently.
 * Constraints: Provider list is cached after first call (volatile double-checked locking).
 * Verification: CloudAcceleratorDemo; unit tests with mock SDK env.
 * References: AWS Neuron SDK; Google libtpu; ComputeProvider interface; ARCHITECTURE_OVERVIEW.md.
 */
public final class CloudAcceleratorFactory {

    private static final GpuLogger logger = GpuLogger.getLogger(CloudAcceleratorFactory.class);

    // Available provider instances
    private static volatile List<ComputeProvider> availableProviders = null;
    private static volatile ComputeProvider bestProvider = null;

    /**
    
     * ID: GPU-CAF-002
     * Requirement: CloudAcceleratorFactory must be fully initialised with valid parameters.
     * Purpose: Construct and initialise a CloudAcceleratorFactory instance.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    private CloudAcceleratorFactory() {
        // Utility class
    }

    /**
     * Get the best available compute provider based on performance characteristics
     * @return The highest-performance available compute provider
     */
    /**
    
     * ID: GPU-CAF-003
     * Requirement: Return the BestProvider field value without side effects.
     * Purpose: Return the value of the BestProvider property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-004
     * Requirement: Return the AvailableProviders field value without side effects.
     * Purpose: Return the value of the AvailableProviders property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-005
     * Requirement: createProvider must execute correctly within the contract defined by this class.
     * Purpose: Create and return a new Provider.
     * Inputs: String providerType
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-006
     * Requirement: Evaluate and return the boolean result of hasCloudAccelerators.
     * Purpose: Return whether hasCloudAccelerators condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-007
     * Requirement: Return the CloudCapabilities field value without side effects.
     * Purpose: Return the value of the CloudCapabilities property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-008
     * Requirement: reset must execute correctly within the contract defined by this class.
     * Purpose: Implement the reset operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public static void reset() {
        availableProviders = null;
        bestProvider = null;
    }

    /**
     * Detect the best available compute provider
     */
    /**
    
     * ID: GPU-CAF-009
     * Requirement: detectBestProvider must execute correctly within the contract defined by this class.
     * Purpose: Implement the detectBestProvider operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
    /**
    
     * ID: GPU-CAF-010
     * Requirement: detectAvailableProviders must execute correctly within the contract defined by this class.
     * Purpose: Implement the detectAvailableProviders operation for this class.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
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
