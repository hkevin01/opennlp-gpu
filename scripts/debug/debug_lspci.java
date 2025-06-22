import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.List;

public class debug_lspci {
    public static void main(String[] args) {
        System.out.println("Testing lspci command execution...");
        
        try {
            Process process = Runtime.getRuntime().exec(new String[]{"/usr/bin/bash", "-c", "lspci | grep -i vga"});
            
            // Check for errors
            BufferedReader errorReader = new BufferedReader(new InputStreamReader(process.getErrorStream()));
            String errorLine;
            while ((errorLine = errorReader.readLine()) != null) {
                System.err.println("Error: " + errorLine);
            }
            
            BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
            
            List<String> gpus = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                gpus.add(line);
                System.out.println("Found GPU: " + line);
            }
            
            System.out.println("Total GPUs found: " + gpus.size());
            
            // Wait for process to complete
            int exitCode = process.waitFor();
            System.out.println("Process exit code: " + exitCode);
            
            if (gpus.isEmpty()) {
                System.out.println("No GPUs found with 'lspci | grep -i vga'");
                
                // Try alternative
                process = Runtime.getRuntime().exec(new String[]{"/usr/bin/bash", "-c", "lspci | grep -i display"});
                reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
                while ((line = reader.readLine()) != null) {
                    gpus.add(line);
                    System.out.println("Found GPU (display): " + line);
                }
                System.out.println("Total GPUs found (including display): " + gpus.size());
            }
            
        } catch (Exception e) {
            System.err.println("Error: " + e.getMessage());
            e.printStackTrace();
        }
    }
} 