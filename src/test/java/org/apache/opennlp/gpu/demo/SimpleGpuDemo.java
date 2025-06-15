package org.apache.opennlp.gpu.demo;

/**
 * Simple, standalone demo that doesn't require complex dependencies
 * Use this if ComprehensiveDemoTestSuite fails to run
 */
public class SimpleGpuDemo {
    
    public static void main(String[] args) {
        System.out.println("🚀 OpenNLP GPU - Simple Demo");
        System.out.println("============================");
        
        try {
            // Basic system information
            System.out.println("☕ Java Version: " + System.getProperty("java.version"));
            System.out.println("🖥️ OS: " + System.getProperty("os.name"));
            System.out.println("📁 Working Directory: " + System.getProperty("user.dir"));
            
            // Check if we can find basic classes
            System.out.println("\n🔍 Checking for GPU classes...");
            
            try {
                Class.forName("org.apache.opennlp.gpu.common.GpuConfig");
                System.out.println("✅ GPU Config class found");
            } catch (ClassNotFoundException e) {
                System.out.println("❌ GPU Config class NOT found");
                SimpleGpuDemo.printCompilationInstructions();
                return;
            }
            
            try {
                Class<?> demoClass = Class.forName("org.apache.opennlp.gpu.demo.GpuDemoApplication");
                System.out.println("✅ Demo Application class found");
                
                // Try to run the actual demo using reflection
                System.out.println("\n🧪 Running GPU Demo Application...");
                java.lang.reflect.Method mainMethod = demoClass.getMethod("main", String[].class);
                mainMethod.invoke(null, (Object) new String[]{});
                
            } catch (ClassNotFoundException e) {
                System.out.println("❌ Demo Application class NOT found");
                SimpleGpuDemo.printCompilationInstructions();
                return;
            } catch (Exception e) {
                System.out.println("⚠️ Demo ran but encountered issues: " + e.getMessage());
                System.out.println("💡 This is normal if GPU hardware is not available");
                System.out.println("💡 Or if the project needs to be compiled with Maven");
            }
            
            System.out.println("\n🎉 Simple demo completed!");
            
        } catch (Exception e) {
            System.err.println("❌ Unexpected error: " + e.getMessage());
            e.printStackTrace();
        }
    }
    
    private static void printCompilationInstructions() {
        System.out.println("\n🛠️ SOLUTION: Compile the project first");
        System.out.println("=====================================");
        System.out.println("1. Open terminal/command prompt");
        System.out.println("2. Navigate to project root (where pom.xml is)");
        System.out.println("3. Run: mvn clean compile");
        System.out.println("4. Wait for compilation to complete");
        System.out.println("5. Try running this demo again");
        System.out.println("");
        System.out.println("💡 Recommended commands:");
        System.out.println("   mvn test -Dtest=SimpleGpuDemo");
        System.out.println("   mvn exec:java -Dexec.mainClass=\"org.apache.opennlp.gpu.demo.SimpleGpuDemo\"");
        System.out.println("");
        System.out.println("⚠️ Note: Direct javac compilation won't work because");
        System.out.println("   this project requires Maven to handle dependencies");
    }
}
