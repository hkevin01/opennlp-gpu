package org.apache.opennlp.gpu.util;

import lombok.AccessLevel;
import lombok.NoArgsConstructor;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

/**
 * Utility class for loading native libraries.
 * Handles both loading from system library path and extracting from JAR.
 */
@NoArgsConstructor(access = AccessLevel.PRIVATE)
public final class NativeLibraryLoader {
    
    // Add logger declaration
    private static final Logger logger = LoggerFactory.getLogger(NativeLibraryLoader.class);
    
    /**
     * Attempt to load a native library.
     * First tries to load from system library path, then from JAR.
     *
     * @param libraryName the name of the library without platform-specific prefixes/suffixes
     * @return true if library was loaded successfully
     */
    public static boolean loadLibrary(String libraryName) {
        try {
            // Try to load directly first
            System.loadLibrary(libraryName);
            logger.info("Successfully loaded library '{}' from system library path", libraryName);
            return true;
        } catch (UnsatisfiedLinkError e) {
            logger.debug("Could not load library '{}' from system library path: {}", libraryName, e.getMessage());
            
            // Try to load from JAR
            return loadLibraryFromJar(libraryName);
        }
    }
    
    /**
     * Load a library from the JAR file.
     *
     * @param libraryName the name of the library without platform-specific prefixes/suffixes
     * @return true if library was loaded successfully
     */
    private static boolean loadLibraryFromJar(String libraryName) {
        String fullLibraryName = System.mapLibraryName(libraryName);
        String resourcePath = "/natives/" + fullLibraryName;
        
        try (InputStream in = NativeLibraryLoader.class.getResourceAsStream(resourcePath)) {
            if (in == null) {
                logger.error("Could not find native library '{}' in JAR at {}", libraryName, resourcePath);
                return false;
            }
            
            // Create temporary directory if it doesn't exist
            Path tempDir = createTempDirectory();
            File tempFile = new File(tempDir.toFile(), fullLibraryName);
            
            // Extract library to temporary file
            try (FileOutputStream out = new FileOutputStream(tempFile)) {
                byte[] buffer = new byte[8192];
                int bytesRead;
                while ((bytesRead = in.read(buffer)) != -1) {
                    out.write(buffer, 0, bytesRead);
                }
            }
            
            // Load the library
            System.load(tempFile.getAbsolutePath());
            logger.info("Successfully loaded library '{}' from JAR", libraryName);
            
            // Mark for deletion on exit
            tempFile.deleteOnExit();
            return true;
        } catch (IOException | UnsatisfiedLinkError e) {
            logger.error("Failed to load native library '{}' from JAR: {}", libraryName, e.getMessage());
            return false;
        }
    }
    
    /**
     * Create a temporary directory for extracting native libraries.
     *
     * @return the path to the temporary directory
     * @throws IOException if the directory could not be created
     */
    private static Path createTempDirectory() throws IOException {
        String tempDirName = "opennlp-gpu-native-" + System.nanoTime();
        Path tempDir = Paths.get(System.getProperty("java.io.tmpdir"), tempDirName);
        
        if (Files.notExists(tempDir)) {
            Files.createDirectory(tempDir);
            tempDir.toFile().deleteOnExit();
        }
        
        return tempDir;
    }
}
