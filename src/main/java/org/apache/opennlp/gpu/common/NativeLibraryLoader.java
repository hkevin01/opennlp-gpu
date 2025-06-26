package org.apache.opennlp.gpu.common;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.concurrent.atomic.AtomicBoolean;

/**
 * Handles automatic loading of native GPU libraries from JAR resources.
 * 
 * This class automatically detects the platform and extracts the appropriate
 * native library from the JAR file to a temporary location for loading.
 * 
 * Features:
 * - Automatic platform detection (Windows, Linux, macOS)
 * - Architecture detection (x86_64, ARM64)
 * - Safe temporary file extraction
 * - Cleanup on JVM shutdown
 * - Fallback handling for missing libraries
 * 
 * @author OpenNLP GPU Extension Team
 * @since 1.0.0
 */
public class NativeLibraryLoader {
    
    private static final AtomicBoolean libraryLoaded = new AtomicBoolean(false);
    private static final String LIBRARY_NAME = "opennlp_gpu";
    
    // Platform detection
    private static final String OS_NAME = System.getProperty("os.name").toLowerCase();
    private static final String OS_ARCH = System.getProperty("os.arch").toLowerCase();
    
    // Platform-specific library names and paths
    private static final String WINDOWS_LIB = LIBRARY_NAME + ".dll";
    private static final String LINUX_LIB = "lib" + LIBRARY_NAME + ".so";
    private static final String MACOS_LIB = "lib" + LIBRARY_NAME + ".dylib";
    
    static {
        // Automatically load library when class is first accessed
        loadNativeLibrary();
    }
    
    /**
     * Load the native GPU library.
     * 
     * This method is automatically called when the class is first loaded.
     * It's safe to call multiple times - subsequent calls are ignored.
     * 
     * @return true if library was loaded successfully, false otherwise
     */
    public static boolean loadNativeLibrary() {
        if (libraryLoaded.get()) {
            return true;
        }
        
        synchronized (NativeLibraryLoader.class) {
            if (libraryLoaded.get()) {
                return true;
            }
            
            try {
                // Try to load from system library path first
                if (loadSystemLibrary()) {
                    libraryLoaded.set(true);
                    System.out.println("✅ Loaded native GPU library from system path");
                    return true;
                }
                
                // Extract and load from JAR resources
                if (loadFromResources()) {
                    libraryLoaded.set(true);
                    System.out.println("✅ Loaded native GPU library from JAR resources");
                    return true;
                }
                
                System.out.println("⚠️  Failed to load native GPU library - GPU acceleration will not be available");
                return false;
                
            } catch (Exception e) {
                System.err.println("❌ Error loading native GPU library: " + e.getMessage());
                return false;
            }
        }
    }
    
    /**
     * Try to load the library from the system library path.
     * 
     * @return true if loaded successfully
     */
    private static boolean loadSystemLibrary() {
        try {
            System.loadLibrary(LIBRARY_NAME);
            return true;
        } catch (UnsatisfiedLinkError e) {
            // Library not found in system path - this is expected
            return false;
        }
    }
    
    /**
     * Extract the library from JAR resources and load it.
     * 
     * @return true if loaded successfully
     */
    private static boolean loadFromResources() {
        try {
            String libraryPath = getNativeLibraryPath();
            if (libraryPath == null) {
                System.err.println("❌ Unsupported platform: " + OS_NAME + " " + OS_ARCH);
                return false;
            }
            
            // Extract library to temporary file
            File tempLibrary = extractLibraryFromJar(libraryPath);
            if (tempLibrary == null) {
                return false;
            }
            
            // Load the extracted library
            System.load(tempLibrary.getAbsolutePath());
            
            // Clean up on shutdown
            tempLibrary.deleteOnExit();
            
            return true;
            
        } catch (Exception e) {
            System.err.println("❌ Failed to load library from resources: " + e.getMessage());
            return false;
        }
    }
    
    /**
     * Determine the native library path within the JAR based on platform.
     * 
     * @return Library path or null if platform not supported
     */
    private static String getNativeLibraryPath() {
        String platform = getPlatformName();
        String architecture = getArchitectureName();
        String libraryName = getLibraryFileName();
        
        if (platform == null || architecture == null || libraryName == null) {
            return null;
        }
        
        return "/native/" + platform + "/" + architecture + "/" + libraryName;
    }
    
    /**
     * Get the platform name for library path.
     * 
     * @return Platform name or null if unsupported
     */
    private static String getPlatformName() {
        if (OS_NAME.contains("windows")) {
            return "windows";
        } else if (OS_NAME.contains("linux")) {
            return "linux";
        } else if (OS_NAME.contains("mac") || OS_NAME.contains("darwin")) {
            return "macos";
        }
        return null;
    }
    
    /**
     * Get the architecture name for library path.
     * 
     * @return Architecture name or null if unsupported
     */
    private static String getArchitectureName() {
        if (OS_ARCH.contains("amd64") || OS_ARCH.contains("x86_64")) {
            return "x86_64";
        } else if (OS_ARCH.contains("aarch64") || OS_ARCH.contains("arm64")) {
            return "arm64";
        }
        return null;
    }
    
    /**
     * Get the library file name for the current platform.
     * 
     * @return Library file name or null if unsupported
     */
    private static String getLibraryFileName() {
        if (OS_NAME.contains("windows")) {
            return WINDOWS_LIB;
        } else if (OS_NAME.contains("linux")) {
            return LINUX_LIB;
        } else if (OS_NAME.contains("mac") || OS_NAME.contains("darwin")) {
            return MACOS_LIB;
        }
        return null;
    }
    
    /**
     * Extract the native library from JAR to a temporary file.
     * 
     * @param resourcePath Path to the library within the JAR
     * @return Temporary file containing the library, or null if extraction failed
     */
    private static File extractLibraryFromJar(String resourcePath) {
        try (InputStream inputStream = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            
            if (inputStream == null) {
                System.err.println("❌ Native library not found in JAR: " + resourcePath);
                return null;
            }
            
            // Create temporary file
            String libraryName = getLibraryFileName();
            String prefix = libraryName.substring(0, libraryName.lastIndexOf('.'));
            String suffix = libraryName.substring(libraryName.lastIndexOf('.'));
            
            Path tempFile = Files.createTempFile(prefix, suffix);
            
            // Copy library to temporary file
            Files.copy(inputStream, tempFile, StandardCopyOption.REPLACE_EXISTING);
            
            // Make executable on Unix systems
            File tempLibrary = tempFile.toFile();
            if (!OS_NAME.contains("windows")) {
                tempLibrary.setExecutable(true);
            }
            
            System.out.println("📦 Extracted native library to: " + tempFile);
            return tempLibrary;
            
        } catch (IOException e) {
            System.err.println("❌ Failed to extract native library: " + e.getMessage());
            return null;
        }
    }
    
    /**
     * Check if the native library has been loaded.
     * 
     * @return true if library is loaded and available
     */
    public static boolean isNativeLibraryLoaded() {
        return libraryLoaded.get();
    }
    
    /**
     * Get information about the native library loading status.
     * 
     * @return Status information string
     */
    public static String getLoadingStatus() {
        if (libraryLoaded.get()) {
            return "Native GPU library loaded successfully";
        } else {
            return "Native GPU library not loaded - platform: " + OS_NAME + " " + OS_ARCH;
        }
    }
    
    /**
     * Force reload of the native library (for testing purposes).
     * 
     * This method should not be used in production code.
     * 
     * @return true if reload was successful
     */
    public static boolean forceReload() {
        libraryLoaded.set(false);
        return loadNativeLibrary();
    }
}
