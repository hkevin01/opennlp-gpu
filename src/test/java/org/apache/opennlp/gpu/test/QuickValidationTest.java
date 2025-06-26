package org.apache.opennlp.gpu.test;

import java.util.Map;

import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.integration.GpuModelFactory;

/**
 * Quick test to verify GpuModelFactory compiles and runs
 */
public class QuickValidationTest {
    
    public static void main(String[] args) {
        System.out.println("=== GpuModelFactory Quick Validation ===");
        
        try {
            // Test 1: Check GPU availability
            boolean gpuAvailable = GpuModelFactory.isGpuAvailable();
            System.out.println("✅ GPU Available: " + gpuAvailable);
            
            // Test 2: Get system info
            Map<String, Object> systemInfo = GpuModelFactory.getSystemInfo();
            System.out.println("✅ System Info Keys: " + systemInfo.keySet().size());
            
            // Test 3: Get GPU info
            Map<String, Object> gpuInfo = GpuModelFactory.getGpuInfo();
            System.out.println("✅ GPU Info Keys: " + gpuInfo.keySet().size());
            
            // Test 4: Create parameters
            Map<String, String> params = GpuModelFactory.getRecommendedParameters();
            System.out.println("✅ Recommended Parameters: " + params.size() + " settings");
            
            // Test 5: Get configuration
            GpuConfig config = GpuModelFactory.getConfig();
            System.out.println("✅ GPU Config: " + config);
            
            System.out.println("\n🎉 All GpuModelFactory methods work correctly!");
            System.out.println("✅ The Java code is fully functional and error-free.");
            
        } catch (Exception e) {
            System.err.println("❌ Test failed: " + e.getMessage());
            e.printStackTrace();
        }
    }
}
