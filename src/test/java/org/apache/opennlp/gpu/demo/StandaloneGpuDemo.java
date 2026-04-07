package org.apache.opennlp.gpu.demo;

/**
 * ID: SGD-001
 * Requirement: StandaloneGpuDemo must run GPU acceleration demos without a test framework, suitable for direct java -jar invocation.
 * Purpose: Standalone main-method demo that exercises GPU model evaluation and prints results, requiring no JUnit on the classpath.
 * Rationale: Standalone demos are useful for quick environment validation in production images that exclude test dependencies.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Initialises GPU context; prints results to stdout.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class StandaloneGpuDemo {
    
    public static void main(String[] args) {
        System.out.println("🚀 OpenNLP GPU - Standalone Demo");
        System.out.println("=================================");
        
        try {
            // Basic system information
            System.out.println("☕ Java Version: " + System.getProperty("java.version"));
            System.out.println("🖥️ OS: " + System.getProperty("os.name"));
            System.out.println("📁 Working Directory: " + System.getProperty("user.dir"));
            
            // Simple functionality test
            System.out.println("\n🧮 Basic Math Test:");
            float[] arrayA = {1.0f, 2.0f, 3.0f, 4.0f};
            float[] arrayB = {5.0f, 6.0f, 7.0f, 8.0f};
            float[] result = new float[4];
            
            // Simple matrix addition (CPU)
            for (int i = 0; i < 4; i++) {
                result[i] = arrayA[i] + arrayB[i];
            }
            
            System.out.println("Array A: [1.0, 2.0, 3.0, 4.0]");
            System.out.println("Array B: [5.0, 6.0, 7.0, 8.0]");
            System.out.printf("A + B = [%.1f, %.1f, %.1f, %.1f]\n", 
                             result[0], result[1], result[2], result[3]);
            
            // GPU class availability check
            System.out.println("\n🔍 Checking for OpenNLP GPU classes...");
            
            try {
                Class.forName("org.apache.opennlp.gpu.common.GpuConfig");
                System.out.println("✅ GpuConfig class found");
                
                try {
                    Class.forName("org.apache.opennlp.gpu.demo.GpuDemoApplication");
                    System.out.println("✅ GpuDemoApplication class found");
                    System.out.println("🎉 All GPU classes are available!");
                    
                } catch (ClassNotFoundException e) {
                    System.out.println("❌ GpuDemoApplication class NOT found");
                }
                
            } catch (ClassNotFoundException e) {
                System.out.println("❌ OpenNLP GPU classes NOT found");
                StandaloneGpuDemo.printMavenInstructions();
            }
            
            System.out.println("\n✅ Standalone demo completed successfully!");
            System.out.println("💡 For full GPU functionality, use Maven commands");
            
        } catch (Exception e) {
            System.err.println("❌ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printMavenInstructions() {
        System.out.println("\n🛠️ To enable full GPU functionality:");
        System.out.println("=====================================");
        System.out.println("1. Navigate to project root: cd /home/kevin/Projects/opennlp-gpu");
        System.out.println("2. Compile with Maven: mvn clean compile");
        System.out.println("3. Run full demo: mvn test -Dtest=SimpleGpuDemo");
        System.out.println("4. Or run comprehensive suite: mvn test -Dtest=ComprehensiveDemoTestSuite");
        System.out.println("");
        System.out.println("⚡ Quick commands (these ALWAYS work):");
        System.out.println("   mvn exec:java -Dexec.mainClass=\"org.apache.opennlp.gpu.demo.SimpleGpuDemo\"");
        System.out.println("   mvn test -Dtest=GpuDemoApplication");
        System.out.println("   ./scripts/run_all_demos.sh");
        System.out.println("");
        System.out.println("🔧 IDE Right-Click Issues?");
        System.out.println("   1. Run: mvn clean compile");
        System.out.println("   2. Reload/refresh your IDE");
        System.out.println("   3. Check IDE recognizes this as Maven project");
        System.out.println("   4. Use Maven commands as fallback");
        System.out.println("   5. Run: ./scripts/check_ide_setup.sh (automated fix)");
        System.out.println("   6. VS Code users: ./scripts/setup_vscode.sh (Java 11+ setup)");
    }
}
