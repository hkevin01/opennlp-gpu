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
package org.apache.opennlp.gpu.test.cloud;

import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.compute.cloud.CloudAcceleratorFactory;
import org.apache.opennlp.gpu.compute.cloud.InferentiaComputeProvider;
import org.apache.opennlp.gpu.compute.cloud.TpuComputeProvider;

/**
 * Test and demonstration class for cloud accelerator functionality.
 *
 * This class demonstrates how to use the cloud accelerator providers
 * and validates their functionality in the OpenNLP GPU Extension.
 *
 * @author OpenNLP GPU Extension Contributors
 * @since 1.1.0
 */
public class CloudAcceleratorDemo {

    public static void main(String[] args) {
        System.out.println("🚀 OpenNLP GPU Extension - Cloud Accelerator Demo");
        System.out.println("=================================================");

        // Test AWS Inferentia
        testInferentiaProvider();

        // Test Google TPU
        testTpuProvider();

        // Test Cloud Factory
        testCloudFactory();

        // Performance comparison
        performanceComparison();

        System.out.println("\n✅ Cloud accelerator demo completed successfully!");
    }

    /**
     * Test AWS Inferentia compute provider
     */
    private static void testInferentiaProvider() {
        System.out.println("\n--- AWS Inferentia Provider Test ---");

        InferentiaComputeProvider inferentia = new InferentiaComputeProvider();

        System.out.println("Provider Name: " + inferentia.getName());
        System.out.println("Available: " + inferentia.isAvailable());
        System.out.println("Device Type: " + inferentia.getType());
        System.out.println("Memory (MB): " + inferentia.getMaxMemoryMB());
        System.out.println("Device Info: " + inferentia.getDeviceInfo());

        // Test device properties
        Map<String, Object> properties = inferentia.getDeviceProperties();
        System.out.println("Device Properties:");
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }

        // Test operations
        testMatrixOperations(inferentia, "AWS Inferentia");
    }

    /**
     * Test Google TPU compute provider
     */
    private static void testTpuProvider() {
        System.out.println("\n--- Google TPU Provider Test ---");

        TpuComputeProvider tpu = new TpuComputeProvider();

        System.out.println("Provider Name: " + tpu.getName());
        System.out.println("Available: " + tpu.isAvailable());
        System.out.println("Device Type: " + tpu.getType());
        System.out.println("Memory (MB): " + tpu.getMaxMemoryMB());
        System.out.println("Device Info: " + tpu.getDeviceInfo());

        // Test device properties
        Map<String, Object> properties = tpu.getDeviceProperties();
        System.out.println("Device Properties:");
        for (Map.Entry<String, Object> entry : properties.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }

        // Test operations
        testMatrixOperations(tpu, "Google TPU");
    }

    /**
     * Test cloud accelerator factory
     */
    private static void testCloudFactory() {
        System.out.println("\n--- Cloud Accelerator Factory Test ---");

        // Get best provider
        ComputeProvider bestProvider = CloudAcceleratorFactory.getBestProvider();
        System.out.println("Best Provider: " + bestProvider.getName());

        // Get all available providers
        List<ComputeProvider> providers = CloudAcceleratorFactory.getAvailableProviders();
        System.out.println("Available Providers (" + providers.size() + "):");
        for (ComputeProvider provider : providers) {
            System.out.println("  - " + provider.getName() + " (Available: " + provider.isAvailable() + ")");
        }

        // Check cloud accelerators
        boolean hasCloudAccelerators = CloudAcceleratorFactory.hasCloudAccelerators();
        System.out.println("Has Cloud Accelerators: " + hasCloudAccelerators);

        // Get cloud capabilities
        Map<String, Object> capabilities = CloudAcceleratorFactory.getCloudCapabilities();
        System.out.println("Cloud Capabilities:");
        for (Map.Entry<String, Object> entry : capabilities.entrySet()) {
            System.out.println("  " + entry.getKey() + ": " + entry.getValue());
        }
    }

    /**
     * Test matrix operations on a compute provider
     */
    private static void testMatrixOperations(ComputeProvider provider, String providerName) {
        System.out.println("\nTesting " + providerName + " operations...");

        try {
            // Initialize provider
            provider.initialize();

            // Test matrix multiplication
            float[] a = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] b = {5.0f, 6.0f, 7.0f, 8.0f};
            float[] result = new float[4];

            long startTime = System.nanoTime();
            provider.matrixMultiply(a, b, result, 2, 2, 2);
            long endTime = System.nanoTime();

            System.out.println("  Matrix Multiplication: " + (endTime - startTime) / 1000000.0 + " ms");

            // Test matrix addition
            startTime = System.nanoTime();
            provider.matrixAdd(a, b, result, 4);
            endTime = System.nanoTime();

            System.out.println("  Matrix Addition: " + (endTime - startTime) / 1000000.0 + " ms");

            // Test feature extraction
            String[] text = {"hello", "world", "test"};
            float[] features = new float[100];

            startTime = System.nanoTime();
            provider.extractFeatures(text, features);
            endTime = System.nanoTime();

            System.out.println("  Feature Extraction: " + (endTime - startTime) / 1000000.0 + " ms");

            // Cleanup
            provider.cleanup();

        } catch (Exception e) {
            System.out.println("  Error testing " + providerName + ": " + e.getMessage());
        }
    }

    /**
     * Performance comparison between providers
     */
    private static void performanceComparison() {
        System.out.println("\n--- Performance Comparison ---");

        List<ComputeProvider> providers = CloudAcceleratorFactory.getAvailableProviders();

        // Test data
        float[] a = new float[1000];
        float[] b = new float[1000];
        float[] result = new float[1000];

        // Initialize test data
        for (int i = 0; i < 1000; i++) {
            a[i] = (float) Math.random();
            b[i] = (float) Math.random();
        }

        System.out.println("Matrix operation performance (1000 elements):");

        for (ComputeProvider provider : providers) {
            if (provider.isAvailable()) {
                try {
                    provider.initialize();

                    // Warm up
                    for (int i = 0; i < 10; i++) {
                        provider.matrixAdd(a, b, result, 1000);
                    }

                    // Benchmark
                    long startTime = System.nanoTime();
                    for (int i = 0; i < 100; i++) {
                        provider.matrixAdd(a, b, result, 1000);
                    }
                    long endTime = System.nanoTime();

                    double avgTime = (endTime - startTime) / 1000000.0 / 100.0;
                    System.out.println("  " + provider.getName() + ": " + String.format("%.3f", avgTime) + " ms");

                    provider.cleanup();

                } catch (Exception e) {
                    System.out.println("  " + provider.getName() + ": Error - " + e.getMessage());
                }
            }
        }
    }
}
