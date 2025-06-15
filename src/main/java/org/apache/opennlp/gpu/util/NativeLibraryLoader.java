package org.apache.opennlp.gpu.util;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

/**
 * Utility class for loading native libraries
 * Java 8 compatible implementation
 */
public class NativeLibraryLoader {
    
    private static final String TEMP_DIR = System.getProperty("java.io.tmpdir");
    
    /**
     * Load a native library from resources
     */
    public static void loadLibrary(String libraryName) {
        try {
            // Try system library first
            System.loadLibrary(libraryName);
        } catch (UnsatisfiedLinkError e1) {
            try {
                // Try loading from resources
                loadLibraryFromResource(libraryName);
            } catch (Exception e2) {
                System.err.println("Failed to load native library: " + libraryName);
                System.err.println("System load error: " + e1.getMessage());
                System.err.println("Resource load error: " + e2.getMessage());
                throw new RuntimeException("Could not load native library: " + libraryName, e2);
            }
        }
    }
    
    private static void loadLibraryFromResource(String libraryName) throws IOException {
        String osName = System.getProperty("os.name").toLowerCase();
        String architecture = System.getProperty("os.arch");
        
        String libPath = getLibraryPath(libraryName, osName, architecture);
        
        try (InputStream is = NativeLibraryLoader.class.getResourceAsStream(libPath)) {
            if (is == null) {
                throw new IOException("Native library not found in resources: " + libPath);
            }
            
            // Create temporary file
            File tempFile = File.createTempFile("native_", getLibraryExtension(osName));
            tempFile.deleteOnExit();
            
            // Copy library to temporary file
            try (FileOutputStream fos = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = is.read(buffer)) != -1) {
                    fos.write(buffer, 0, bytesRead);
                }
            }
            
            // Load the library
            System.load(tempFile.getAbsolutePath());
        }
    }
    
    private static String getLibraryPath(String libraryName, String osName, String architecture) {
        StringBuilder path = new StringBuilder("/natives/");
        
        if (osName.contains("windows")) {
            path.append("windows/");
        } else if (osName.contains("linux")) {
            path.append("linux/");
        } else if (osName.contains("mac")) {
            path.append("macos/");
        }
        
        path.append(architecture).append("/");
        path.append(libraryName);
        path.append(getLibraryExtension(osName));
        
        return path.toString();
    }
    
    private static String getLibraryExtension(String osName) {
        if (osName.contains("windows")) {
            return ".dll";
        } else if (osName.contains("mac")) {
            return ".dylib";
        } else {
            return ".so";
        }
    }
}
