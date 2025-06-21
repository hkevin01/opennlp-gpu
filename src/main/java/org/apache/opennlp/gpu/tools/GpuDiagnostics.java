package org.apache.opennlp.gpu.tools;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * GPU Diagnostics Tool for OpenNLP GPU Acceleration
 * 
 * Comprehensive tool to detect and validate GPU drivers, SDKs, and runtime environments
 * required for GPU acceleration to work properly.
 */
public class GpuDiagnostics {
    
    private static final GpuLogger logger = GpuLogger.getLogger(GpuDiagnostics.class);
    
    public static void main(String[] args) {
        System.out.println("🔍 OpenNLP GPU Acceleration - Hardware Diagnostics");
        System.out.println("==================================================");
        
        GpuDiagnostics diagnostics = new GpuDiagnostics();
        DiagnosticReport report = diagnostics.runComprehensiveDiagnostics();
        
        report.printReport();
        
        if (report.isGpuReady()) {
            System.out.println("\n🎉 GPU acceleration is ready to use!");
            System.exit(0);
        } else {
            System.out.println("\n⚠️ GPU acceleration setup incomplete.");
            System.out.println("See recommendations above to fix issues.");
            System.exit(1);
        }
    }
    
    public DiagnosticReport runComprehensiveDiagnostics() {
        DiagnosticReport report = new DiagnosticReport();
        
        // System information
        gatherSystemInfo(report);
        
        // Java environment
        checkJavaEnvironment(report);
        
        // GPU Hardware Detection
        detectGpuHardware(report);
        
        // Driver Detection
        checkNvidiaDrivers(report);
        checkAmdDrivers(report);
        checkIntelDrivers(report);
        
        // Runtime Detection
        checkCudaRuntime(report);
        checkRocmRuntime(report);
        checkOpenClRuntime(report);
        
        // OpenNLP GPU Integration
        checkOpenNlpGpuIntegration(report);
        
        // Performance Test
        runBasicPerformanceTest(report);
        
        return report;
    }
    
    private void gatherSystemInfo(DiagnosticReport report) {
        report.addSection("System Information");
        report.addInfo("OS", System.getProperty("os.name") + " " + System.getProperty("os.version"));
        report.addInfo("Architecture", System.getProperty("os.arch"));
        report.addInfo("Java Version", System.getProperty("java.version"));
        report.addInfo("Java Vendor", System.getProperty("java.vendor"));
        report.addInfo("Java Home", System.getProperty("java.home"));
        report.addInfo("Available Processors", String.valueOf(Runtime.getRuntime().availableProcessors()));
        
        long maxMemory = Runtime.getRuntime().maxMemory();
        report.addInfo("Max JVM Memory", String.format("%.1f GB", maxMemory / (1024.0 * 1024.0 * 1024.0)));
    }
    
    private void checkJavaEnvironment(DiagnosticReport report) {
        report.addSection("Java Environment");
        
        String javaVersion = System.getProperty("java.version");
        String majorVersion = javaVersion.split("\\.")[0];
        int major = Integer.parseInt(majorVersion);
        
        if (major >= 17) {
            report.addSuccess("Java Version", "Java " + major + " ✅ Compatible");
        } else if (major >= 11) {
            report.addWarning("Java Version", "Java " + major + " ⚠️ Works but Java 17+ recommended");
        } else {
            report.addError("Java Version", "Java " + major + " ❌ Too old, need Java 11+");
            report.addRecommendation("Install Java 17+: sudo apt install openjdk-17-jdk");
        }
        
        String javaHome = System.getProperty("java.home");
        if (javaHome != null && new File(javaHome).exists()) {
            report.addSuccess("JAVA_HOME", "✅ Set and valid: " + javaHome);
        } else {
            report.addError("JAVA_HOME", "❌ Not set or invalid");
            report.addRecommendation("Set JAVA_HOME: export JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64");
        }
    }
    
    private void detectGpuHardware(DiagnosticReport report) {
        report.addSection("GPU Hardware Detection");
        
        // Use lspci to detect GPUs on Linux
        if (System.getProperty("os.name").toLowerCase().contains("linux")) {
            detectLinuxGpuHardware(report);
        } else if (System.getProperty("os.name").toLowerCase().contains("windows")) {
            detectWindowsGpuHardware(report);
        } else if (System.getProperty("os.name").toLowerCase().contains("mac")) {
            detectMacGpuHardware(report);
        }
    }
    
    private void detectLinuxGpuHardware(DiagnosticReport report) {
        try {
            Process process = Runtime.getRuntime().exec(new String[]{"/usr/bin/bash", "-c", "lspci | grep -i vga"});
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            
            List<String> gpus = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                gpus.add(line);
            }
            
            if (gpus.isEmpty()) {
                // Try alternative method
                process = Runtime.getRuntime().exec(new String[]{"/usr/bin/bash", "-c", "lspci | grep -i display"});
                reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                while ((line = reader.readLine()) != null) {
                    gpus.add(line);
                }
            }
            
            if (!gpus.isEmpty()) {
                for (String gpu : gpus) {
                    if (gpu.toLowerCase().contains("nvidia")) {
                        report.addSuccess("NVIDIA GPU", "✅ Detected: " + gpu.trim());
                        report.setHasNvidiaGpu(true);
                    } else if (gpu.toLowerCase().contains("amd") || gpu.toLowerCase().contains("radeon")) {
                        report.addSuccess("AMD GPU", "✅ Detected: " + gpu.trim());
                        report.setHasAmdGpu(true);
                    } else if (gpu.toLowerCase().contains("intel")) {
                        report.addSuccess("Intel GPU", "✅ Detected: " + gpu.trim());
                        report.setHasIntelGpu(true);
                    } else {
                        report.addInfo("Other GPU", "ℹ️ Detected: " + gpu.trim());
                    }
                }
            } else {
                report.addError("GPU Hardware", "❌ No GPU detected");
                report.addRecommendation("Ensure GPU hardware is properly installed and recognized by the system");
            }
            
        } catch (Exception e) {
            report.addError("GPU Detection", "❌ Failed to detect GPU: " + e.getMessage());
        }
    }
    
    private void detectWindowsGpuHardware(DiagnosticReport report) {
        try {
            Process process = Runtime.getRuntime().exec("wmic path win32_VideoController get name");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            
            List<String> gpus = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                line = line.trim();
                if (!line.isEmpty() && !line.equals("Name")) {
                    gpus.add(line);
                }
            }
            
            if (!gpus.isEmpty()) {
                for (String gpu : gpus) {
                    if (gpu.toLowerCase().contains("nvidia")) {
                        report.addSuccess("NVIDIA GPU", "✅ Detected: " + gpu);
                        report.setHasNvidiaGpu(true);
                    } else if (gpu.toLowerCase().contains("amd") || gpu.toLowerCase().contains("radeon")) {
                        report.addSuccess("AMD GPU", "✅ Detected: " + gpu);
                        report.setHasAmdGpu(true);
                    } else if (gpu.toLowerCase().contains("intel")) {
                        report.addSuccess("Intel GPU", "✅ Detected: " + gpu);
                        report.setHasIntelGpu(true);
                    } else {
                        report.addInfo("Other GPU", "ℹ️ Detected: " + gpu);
                    }
                }
            } else {
                report.addError("GPU Hardware", "❌ No GPU detected");
            }
            
        } catch (Exception e) {
            report.addError("GPU Detection", "❌ Failed to detect GPU: " + e.getMessage());
        }
    }
    
    private void detectMacGpuHardware(DiagnosticReport report) {
        try {
            Process process = Runtime.getRuntime().exec("system_profiler SPDisplaysDataType");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            
            StringBuilder output = new StringBuilder();
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }
            
            String outputStr = output.toString().toLowerCase();
            if (outputStr.contains("amd") || outputStr.contains("radeon")) {
                report.addSuccess("AMD GPU", "✅ Detected via system_profiler");
                report.setHasAmdGpu(true);
            }
            if (outputStr.contains("intel")) {
                report.addSuccess("Intel GPU", "✅ Detected via system_profiler");
                report.setHasIntelGpu(true);
            }
            if (outputStr.contains("apple") && outputStr.contains("gpu")) {
                report.addSuccess("Apple Silicon GPU", "✅ Detected via system_profiler");
                report.setHasAppleGpu(true);
            }
            
        } catch (Exception e) {
            report.addError("GPU Detection", "❌ Failed to detect GPU: " + e.getMessage());
        }
    }
    
    private void checkNvidiaDrivers(DiagnosticReport report) {
        if (!report.hasNvidiaGpu()) {
            return;
        }
        
        report.addSection("NVIDIA Drivers");
        
        // Check nvidia-smi
        try {
            Process process = Runtime.getRuntime().exec("nvidia-smi --version");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line = reader.readLine();
            
            if (line != null) {
                report.addSuccess("NVIDIA Driver", "✅ Installed: " + line.trim());
                
                // Get detailed GPU info
                process = Runtime.getRuntime().exec("nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader");
                reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                while ((line = reader.readLine()) != null) {
                    String[] parts = line.split(", ");
                    if (parts.length >= 3) {
                        report.addInfo("GPU Model", parts[0]);
                        report.addInfo("GPU Memory", parts[1]);
                        report.addInfo("Driver Version", parts[2]);
                    }
                }
            } else {
                report.addError("NVIDIA Driver", "❌ nvidia-smi not found");
                report.addRecommendation("Install NVIDIA drivers: sudo apt install nvidia-driver-535");
            }
            
        } catch (Exception e) {
            report.addError("NVIDIA Driver", "❌ Not installed or not accessible");
            report.addRecommendation("Install NVIDIA drivers: sudo apt install nvidia-driver-535");
        }
    }
    
    private void checkAmdDrivers(DiagnosticReport report) {
        if (!report.hasAmdGpu()) {
            return;
        }
        
        report.addSection("AMD Drivers");
        
        // Check rocm-smi
        try {
            Process process = Runtime.getRuntime().exec("rocm-smi --showproductname");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line = reader.readLine();
            
            if (line != null && !line.contains("command not found")) {
                report.addSuccess("AMD ROCm Driver", "✅ Installed and working");
                
                // Get GPU details
                process = Runtime.getRuntime().exec("rocm-smi --showmeminfo --showuse");
                reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                StringBuilder info = new StringBuilder();
                while ((line = reader.readLine()) != null) {
                    info.append(line).append("\n");
                }
                report.addInfo("ROCm Info", info.toString().trim());
                
            } else {
                report.addError("AMD ROCm Driver", "❌ rocm-smi not found");
                report.addRecommendation("Install ROCm: sudo apt install rocm-dev rocm-libs");
            }
            
        } catch (Exception e) {
            // Try alternative detection
            try {
                Process process = Runtime.getRuntime().exec("ls /opt/rocm");
                if (process.waitFor() == 0) {
                    report.addWarning("AMD ROCm", "⚠️ ROCm directory found but rocm-smi not working");
                    report.addRecommendation("Check ROCm installation: /opt/rocm should contain bin, lib directories");
                } else {
                    report.addError("AMD ROCm Driver", "❌ Not installed");
                    report.addRecommendation("Install ROCm: wget -qO - https://repo.radeon.com/rocm/rocm.gpg.key | sudo apt-key add -");
                    report.addRecommendation("sudo apt update && sudo apt install rocm-dev rocm-libs");
                }
            } catch (Exception e2) {
                report.addError("AMD ROCm Driver", "❌ Not detected: " + e2.getMessage());
            }
        }
    }
    
    private void checkIntelDrivers(DiagnosticReport report) {
        if (!report.hasIntelGpu()) {
            return;
        }
        
        report.addSection("Intel GPU Drivers");
        
        // Check for Intel GPU compute runtime
        try {
            Process process = Runtime.getRuntime().exec("ls /usr/lib/x86_64-linux-gnu/intel-opencl");
            if (process.waitFor() == 0) {
                report.addSuccess("Intel OpenCL", "✅ Intel OpenCL runtime detected");
            } else {
                report.addWarning("Intel OpenCL", "⚠️ Intel OpenCL runtime not found");
                report.addRecommendation("Install Intel OpenCL: sudo apt install intel-opencl-icd");
            }
        } catch (Exception e) {
            report.addWarning("Intel GPU", "⚠️ Intel GPU driver check inconclusive");
            report.addRecommendation("Install Intel drivers: sudo apt install intel-opencl-icd");
        }
    }
    
    private void checkCudaRuntime(DiagnosticReport report) {
        if (!report.hasNvidiaGpu()) {
            return;
        }
        
        report.addSection("CUDA Runtime");
        
        try {
            Process process = Runtime.getRuntime().exec("nvcc --version");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            String line;
            boolean cudaFound = false;
            
            while ((line = reader.readLine()) != null) {
                if (line.contains("release")) {
                    report.addSuccess("CUDA Toolkit", "✅ Installed: " + line.trim());
                    cudaFound = true;
                    break;
                }
            }
            
            if (!cudaFound) {
                report.addWarning("CUDA Toolkit", "⚠️ nvcc not found - CUDA toolkit may not be installed");
                report.addRecommendation("Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit");
            }
            
        } catch (Exception e) {
            report.addWarning("CUDA Toolkit", "⚠️ CUDA toolkit not detected");
            report.addRecommendation("Install CUDA toolkit: sudo apt install nvidia-cuda-toolkit");
        }
        
        // Check CUDA libraries
        String[] cudaLibs = {"/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"};
        boolean libsFound = false;
        
        for (String libPath : cudaLibs) {
            if (new File(libPath + "/libcuda.so").exists() || 
                new File(libPath + "/libcuda.so.1").exists()) {
                report.addSuccess("CUDA Libraries", "✅ Found in " + libPath);
                libsFound = true;
                break;
            }
        }
        
        if (!libsFound) {
            report.addWarning("CUDA Libraries", "⚠️ CUDA libraries not found in standard locations");
        }
    }
    
    private void checkRocmRuntime(DiagnosticReport report) {
        if (!report.hasAmdGpu()) {
            return;
        }
        
        report.addSection("ROCm Runtime");
        
        // Check ROCm installation
        File rocmDir = new File("/opt/rocm");
        if (rocmDir.exists()) {
            report.addSuccess("ROCm Installation", "✅ Found at /opt/rocm");
            
            // Check specific components
            File[] components = {
                new File("/opt/rocm/bin"),
                new File("/opt/rocm/lib"),
                new File("/opt/rocm/include"),
                new File("/opt/rocm/opencl")
            };
            
            for (File component : components) {
                if (component.exists()) {
                    report.addInfo("ROCm " + component.getName(), "✅ Present");
                } else {
                    report.addWarning("ROCm " + component.getName(), "⚠️ Missing");
                }
            }
            
        } else {
            report.addError("ROCm Installation", "❌ Not found at /opt/rocm");
            report.addRecommendation("Install ROCm: sudo apt install rocm-dev rocm-libs");
        }
        
        // Check ROCm environment
        String rocmPath = System.getenv("ROCM_PATH");
        if (rocmPath != null) {
            report.addSuccess("ROCM_PATH", "✅ Set to: " + rocmPath);
        } else {
            report.addWarning("ROCM_PATH", "⚠️ Environment variable not set");
            report.addRecommendation("Set ROCM_PATH: export ROCM_PATH=/opt/rocm");
        }
    }
    
    private void checkOpenClRuntime(DiagnosticReport report) {
        report.addSection("OpenCL Runtime");
        
        try {
            Process process = Runtime.getRuntime().exec("clinfo");
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            
            StringBuilder output = new StringBuilder();
            String line;
            int platformCount = 0;
            int deviceCount = 0;
            
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
                if (line.contains("Number of platforms")) {
                    try {
                        platformCount = Integer.parseInt(line.replaceAll("[^0-9]", ""));
                    } catch (NumberFormatException ignored) {}
                }
                if (line.contains("Number of devices")) {
                    try {
                        deviceCount += Integer.parseInt(line.replaceAll("[^0-9]", ""));
                    } catch (NumberFormatException ignored) {}
                }
            }
            
            if (platformCount > 0) {
                report.addSuccess("OpenCL Runtime", "✅ Available (" + platformCount + " platforms, " + deviceCount + " devices)");
                report.addInfo("OpenCL Details", output.toString().trim());
            } else {
                report.addError("OpenCL Runtime", "❌ No OpenCL platforms found");
                report.addRecommendation("Install OpenCL: sudo apt install opencl-headers ocl-icd-opencl-dev");
            }
            
        } catch (Exception e) {
            report.addError("OpenCL Runtime", "❌ clinfo command not found");
            report.addRecommendation("Install OpenCL tools: sudo apt install clinfo opencl-headers");
        }
    }
    
    private void checkOpenNlpGpuIntegration(DiagnosticReport report) {
        report.addSection("OpenNLP GPU Integration");
        
        try {
            // Test if our GPU classes can be loaded
            Class.forName("org.apache.opennlp.gpu.common.GpuConfig");
            report.addSuccess("GPU Classes", "✅ OpenNLP GPU classes available");
            
            // Test basic GPU configuration
            org.apache.opennlp.gpu.common.GpuConfig config = new org.apache.opennlp.gpu.common.GpuConfig();
            report.addSuccess("GPU Configuration", "✅ GPU configuration can be created");
            
        } catch (Exception e) {
            report.addError("GPU Integration", "❌ OpenNLP GPU classes not available: " + e.getMessage());
        }
    }
    
    private void runBasicPerformanceTest(DiagnosticReport report) {
        report.addSection("Basic Performance Test");
        
        try {
            // Run a simple matrix operation test
            long startTime = System.currentTimeMillis();
            
            // Simple computation test
            double result = 0;
            for (int i = 0; i < 1000000; i++) {
                result += Math.sin(i * 0.001);
            }
            
            long duration = System.currentTimeMillis() - startTime;
            report.addSuccess("CPU Performance", "✅ Basic test completed in " + duration + "ms");
            
            // TODO: Add actual GPU performance test when GPU is available
            
        } catch (Exception e) {
            report.addWarning("Performance Test", "⚠️ Performance test failed: " + e.getMessage());
        }
    }
    
    /**
     * Diagnostic Report Container
     */
    public static class DiagnosticReport {
        private final Map<String, List<String>> sections = new HashMap<>();
        private final List<String> recommendations = new ArrayList<>();
        private boolean hasNvidiaGpu = false;
        private boolean hasAmdGpu = false;
        private boolean hasIntelGpu = false;
        private boolean hasAppleGpu = false;
        private String currentSection = "";
        
        public void addSection(String section) {
            this.currentSection = section;
            sections.putIfAbsent(section, new ArrayList<>());
        }
        
        public void addSuccess(String item, String message) {
            sections.get(currentSection).add("✅ " + item + ": " + message);
        }
        
        public void addWarning(String item, String message) {
            sections.get(currentSection).add("⚠️ " + item + ": " + message);
        }
        
        public void addError(String item, String message) {
            sections.get(currentSection).add("❌ " + item + ": " + message);
        }
        
        public void addInfo(String item, String message) {
            sections.get(currentSection).add("ℹ️ " + item + ": " + message);
        }
        
        public void addRecommendation(String recommendation) {
            recommendations.add(recommendation);
        }
        
        // Getters and setters for GPU types
        public boolean hasNvidiaGpu() { return hasNvidiaGpu; }
        public void setHasNvidiaGpu(boolean hasNvidiaGpu) { this.hasNvidiaGpu = hasNvidiaGpu; }
        
        public boolean hasAmdGpu() { return hasAmdGpu; }
        public void setHasAmdGpu(boolean hasAmdGpu) { this.hasAmdGpu = hasAmdGpu; }
        
        public boolean hasIntelGpu() { return hasIntelGpu; }
        public void setHasIntelGpu(boolean hasIntelGpu) { this.hasIntelGpu = hasIntelGpu; }
        
        public boolean hasAppleGpu() { return hasAppleGpu; }
        public void setHasAppleGpu(boolean hasAppleGpu) { this.hasAppleGpu = hasAppleGpu; }
        
        public boolean isGpuReady() {
            // Check if we have GPU hardware and OpenCL runtime is working
            boolean hasGpu = hasNvidiaGpu || hasAmdGpu || hasIntelGpu || hasAppleGpu;
            
            // Check if OpenCL runtime is available (this is the most important check)
            boolean openclAvailable = sections.getOrDefault("OpenCL Runtime", new ArrayList<>()).stream()
                .anyMatch(line -> line.contains("✅ Available"));
            
            // Check for critical errors (GPU detection and OpenCL runtime)
            boolean hasCriticalErrors = sections.getOrDefault("GPU Hardware Detection", new ArrayList<>()).stream()
                .anyMatch(line -> line.startsWith("❌")) ||
                sections.getOrDefault("OpenCL Runtime", new ArrayList<>()).stream()
                .anyMatch(line -> line.startsWith("❌"));
            
            return hasGpu && openclAvailable && !hasCriticalErrors;
        }
        
        public void printReport() {
            System.out.println("\n=================================================================================");
            System.out.println("🔍 GPU DIAGNOSTICS REPORT");
            System.out.println("=================================================================================");
            
            for (Map.Entry<String, List<String>> entry : sections.entrySet()) {
                System.out.println("\n📋 " + entry.getKey());
                System.out.println("--------------------------------------------------");
                for (String line : entry.getValue()) {
                    System.out.println("  " + line);
                }
            }
            
            if (!recommendations.isEmpty()) {
                System.out.println("\n💡 RECOMMENDATIONS");
                System.out.println("--------------------------------------------------");
                for (String rec : recommendations) {
                    System.out.println("  • " + rec);
                }
            }
            
            System.out.println("\n=================================================================================");
            if (isGpuReady()) {
                System.out.println("🎉 STATUS: GPU acceleration is ready!");
            } else {
                System.out.println("⚠️ STATUS: GPU acceleration setup incomplete");
            }
            System.out.println("=================================================================================");
        }
    }
}
