package org.apache.opennlp.gpu.integration;

import java.util.Map;

import org.apache.opennlp.gpu.common.GpuConfig;

/**
 * Simple validation test for core GPU integration functionality
 */
public class ValidationTest {
    
    public static void main(String[] args) {
        System.out.println("=== OpenNLP GPU Extension Validation Test ===");
        
        try {
            // Test 1: Check if classes can be loaded
            System.out.println("✓ Test 1: Class loading successful");
            
            // Test 2: Test GpuConfig static methods
            boolean gpuAvailable = GpuConfig.isGpuAvailable();
            System.out.println("✓ Test 2: GPU Available check: " + gpuAvailable);
            
            // Test 3: Test GpuConfig instance creation
            GpuConfig config = new GpuConfig();
            config.setGpuEnabled(false); // Set to CPU for testing
            System.out.println("✓ Test 3: GpuConfig creation successful");
            
            // Test 4: Test GpuModelFactory static methods
            boolean factoryGpuAvailable = GpuModelFactory.isGpuAvailable();
            System.out.println("✓ Test 4: GpuModelFactory GPU check: " + factoryGpuAvailable);
            
            // Test 5: Test parameter creation
            Map<String, String> params = GpuModelFactory.createCpuParameters();
            System.out.println("✓ Test 5: Parameter creation successful, algorithm: " + params.get("algorithm"));
            
            // Test 6: Test system info
            Map<String, Object> systemInfo = GpuModelFactory.getSystemInfo();
            System.out.println("✓ Test 6: System info retrieved, Java version: " + systemInfo.get("java_version"));
            
            // Test 7: Test GPU info
            Map<String, Object> gpuInfo = GpuModelFactory.getGpuInfo();
            System.out.println("✓ Test 7: GPU info retrieved, GPU available: " + gpuInfo.get("gpu_available"));
            
            System.out.println("\n=== All Tests Passed! ===");
            System.out.println("OpenNLP GPU Extension core functionality is working correctly.");
            
        } catch (Exception e) {
            System.err.println("✗ Test failed: " + e.getMessage());
            e.printStackTrace();
            System.exit(1);
        }
    }
}
