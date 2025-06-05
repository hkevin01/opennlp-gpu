package org.apache.opennlp.gpu.common;

import org.jocl.*;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.HashMap;
import java.util.Map;

/**
 * OpenCL-based implementation of the ComputeProvider interface using JOCL.
 */
public class OpenClComputeProvider implements ComputeProvider {
    
    private static final Logger logger = LoggerFactory.getLogger(OpenClComputeProvider.class);
    
    // OpenCL context and command queue
    private cl_context context;
    private cl_command_queue commandQueue;
    private cl_device_id deviceId;
    
    // Resource manager for OpenCL operations
    private OpenClResourceManager resourceManager;
    
    // Device information
    private String deviceName;
    private int computeUnits;
    private long globalMemSize;
    
    // Performance benchmark cache
    private final Map<String, Map<Integer, Double>> benchmarkCache = new HashMap<>();
    
    // Supported operations
    private final Map<String, Boolean> supportedOperations = new HashMap<>();
    
    /**
     * Creates a new OpenCL compute provider.
     */
    public OpenClComputeProvider() {
        // Default constructor
    }
    
    @Override
    public boolean initialize() {
        logger.info("Initializing OpenCL compute provider");
        
        try {
            // Enable exceptions
            CL.setExceptionsEnabled(true);
            
            // Get platform
            int[] numPlatformsArray = new int[1];
            CL.clGetPlatformIDs(0, null, numPlatformsArray);
            int numPlatforms = numPlatformsArray[0];
            
            if (numPlatforms == 0) {
                logger.error("No OpenCL platforms found");
                return false;
            }
            
            cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
            CL.clGetPlatformIDs(platforms.length, platforms, null);
            
            // Find a GPU device
            for (cl_platform_id platform : platforms) {
                int[] numDevicesArray = new int[1];
                CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 0, null, numDevicesArray);
                int numDevices = numDevicesArray[0];
                
                if (numDevices > 0) {
                    cl_device_id[] devices = new cl_device_id[numDevices];
                    CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, numDevices, devices, null);
                    
                    // Use the first GPU device
                    deviceId = devices[0];
                    
                    // Get device info
                    deviceName = getString(deviceId, CL.CL_DEVICE_NAME);
                    computeUnits = getInt(deviceId, CL.CL_DEVICE_MAX_COMPUTE_UNITS);
                    globalMemSize = getLong(deviceId, CL.CL_DEVICE_GLOBAL_MEM_SIZE);
                    
                    // Create context and command queue
                    cl_context_properties contextProperties = new cl_context_properties();
                    contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
                    
                    int[] errorCode = new int[1];
                    context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{deviceId}, 
                                                null, null, errorCode);
                    
                    if (errorCode[0] != CL.CL_SUCCESS) {
                        logger.error("Failed to create OpenCL context: {}", errorCode[0]);
                        return false;
                    }
                    
                    commandQueue = CL.clCreateCommandQueue(context, deviceId, 0, errorCode);
                    
                    if (errorCode[0] != CL.CL_SUCCESS) {
                        logger.error("Failed to create command queue: {}", errorCode[0]);
                        CL.clReleaseContext(context);
                        return false;
                    }
                    
                    // Create resource manager
                    resourceManager = new OpenClResourceManager(context, commandQueue);
                    
                    // Initialize supported operations
                    initializeSupportedOperations();
                    
                    logger.info("Initialized OpenCL compute provider using device: {}", deviceName);
                    return true;
                }
            }
            
            logger.error("No GPU devices found");
            return false;
            
        } catch (Exception e) {
            logger.error("Error initializing OpenCL compute provider", e);
            return false;
        }
    }
    
    /**
     * Initialize the map of supported operations.
     */
    private void initializeSupportedOperations() {
        // Basic operations that should be supported by all OpenCL devices
        supportedOperations.put("matrixMultiply", true);
        supportedOperations.put("matrixAdd", true);
        supportedOperations.put("matrixSubtract", true);
        supportedOperations.put("vectorAdd", true);
        
        // Check if double precision is supported
        String extensions = getString(deviceId, CL.CL_DEVICE_EXTENSIONS);
        boolean supportsDouble = extensions.contains("cl_khr_fp64");
        supportedOperations.put("doubleOperations", supportsDouble);
        
        // Check for specific OpenCL version features
        int majorVersion = getOpenClVersion() / 10;
        int minorVersion = getOpenClVersion() % 10;
        
        boolean supportsOpencl12 = (majorVersion > 1 || (majorVersion == 1 && minorVersion >= 2));
        supportedOperations.put("opencl12Features", supportsOpencl12);
        
        boolean supportsOpencl20 = (majorVersion >= 2);
        supportedOperations.put("opencl20Features", supportsOpencl20);
    }
    
    /**
     * Get the OpenCL version as an integer (e.g., 12 for OpenCL 1.2).
     */
    private int getOpenClVersion() {
        String versionString = getString(deviceId, CL.CL_DEVICE_VERSION);
        // Parse "OpenCL X.Y" format
        String[] parts = versionString.split(" ")[1].split("\\.");
        return Integer.parseInt(parts[0]) * 10 + Integer.parseInt(parts[1]);
    }
    
    @Override
    public Type getType() {
        return Type.OPENCL;
    }
    
    @Override
    public boolean isAvailable() {
        return context != null && commandQueue != null;
    }
    
    @Override
    public String getName() {
        return "OpenCL Provider (" + deviceName + ")";
    }
    
    @Override
    public int getComputeCapability() {
        // Use OpenCL version and compute units to determine capability
        int openclVersion = getOpenClVersion();
        
        // Base capability on OpenCL version
        int capability = openclVersion;
        
        // Adjust based on compute units
        if (computeUnits >= 32) {
            capability += 5;
        } else if (computeUnits >= 16) {
            capability += 3;
        } else if (computeUnits >= 8) {
            capability += 1;
        }
        
        return capability;
    }
    
    @Override
    public double getPerformanceScore(String operationType, int problemSize) {
        // Check if we have a cached score
        if (benchmarkCache.containsKey(operationType) &&
            benchmarkCache.get(operationType).containsKey(problemSize)) {
            return benchmarkCache.get(operationType).get(problemSize);
        }
        
        // Check if the operation is supported
        if (!supportsOperation(operationType)) {
            return 0.0; // Not supported, score of 0
        }
        
        // Perform a benchmark
        double score = performBenchmark(operationType, problemSize);
        
        // Cache the result
        benchmarkCache.computeIfAbsent(operationType, k -> new HashMap<>())
                     .put(problemSize, score);
        
        return score;
    }
    
    /**
     * Perform a benchmark for the specified operation type and problem size.
     *
     * @param operationType the type of operation
     * @param problemSize the size of the problem
     * @return a performance score
     */
    private double performBenchmark(String operationType, int problemSize) {
        // Here we would implement actual benchmarking logic for different operations
        // For simplicity, we'll just return a score based on the device capabilities
        
        // Base score on compute units and memory size
        double baseScore = computeUnits * 10.0;
        
        // Adjust for global memory size (in GB)
        double memSizeGB = globalMemSize / (1024.0 * 1024.0 * 1024.0);
        baseScore *= (1.0 + memSizeGB / 4.0);
        
        // GPU is better for larger problems
        if (problemSize > 1000) {
            baseScore *= (1.0 + Math.log10(problemSize / 1000.0));
        }
        
        logger.debug("OpenCL benchmark for {} with size {}: score {}", 
                    operationType, problemSize, baseScore);
        
        return baseScore;
    }
    
    @Override
    public ResourceManager getResourceManager() {
        return resourceManager;
    }
    
    @Override
    public boolean supportsOperation(String operationType) {
        Boolean supported = supportedOperations.get(operationType);
        return supported != null && supported;
    }
    
    @Override
    public void release() {
        if (resourceManager != null) {
            resourceManager.releaseAll();
        }
        
        if (commandQueue != null) {
            CL.clReleaseCommandQueue(commandQueue);
            commandQueue = null;
        }
        
        if (context != null) {
            CL.clReleaseContext(context);
            context = null;
        }
        
        benchmarkCache.clear();
        supportedOperations.clear();
        
        logger.info("Released OpenCL compute provider resources");
    }
    
    /**
     * Get a string property from an OpenCL device.
     *
     * @param deviceId the device ID
     * @param paramName the parameter name
     * @return the string value
     */
    private String getString(cl_device_id deviceId, int paramName) {
        long[] size = new long[1];
        CL.clGetDeviceInfo(deviceId, paramName, 0, null, size);
        byte[] buffer = new byte[(int)size[0]];
        CL.clGetDeviceInfo(deviceId, paramName, buffer.length, Pointer.to(buffer), null);
        return new String(buffer, 0, buffer.length-1);
    }
    
    /**
     * Get an integer property from an OpenCL device.
     *
     * @param deviceId the device ID
     * @param paramName the parameter name
     * @return the integer value
     */
    private int getInt(cl_device_id deviceId, int paramName) {
        int[] values = new int[1];
        CL.clGetDeviceInfo(deviceId, paramName, Sizeof.cl_int, Pointer.to(values), null);
        return values[0];
    }
    
    /**
     * Get a long property from an OpenCL device.
     *
     * @param deviceId the device ID
     * @param paramName the parameter name
     * @return the long value
     */
    private long getLong(cl_device_id deviceId, int paramName) {
        long[] values = new long[1];
        CL.clGetDeviceInfo(deviceId, paramName, Sizeof.cl_long, Pointer.to(values), null);
        return values[0];
    }
    
    /**
     * OpenCL-specific implementation of the ResourceManager interface.
     */
    private static class OpenClResourceManager implements ResourceManager {
        
        private final cl_context context;
        private final cl_command_queue commandQueue;
        
        // Cache for compiled programs and kernels
        private final Map<String, cl_program> programCache = new HashMap<>();
        private final Map<String, cl_kernel> kernelCache = new HashMap<>();
        
        // Cache for data
        private final Map<String, Object> dataCache = new HashMap<>();
        
        /**
         * Creates a new OpenCL resource manager.
         *
         * @param context the OpenCL context
         * @param commandQueue the OpenCL command queue
         */
        public OpenClResourceManager(cl_context context, cl_command_queue commandQueue) {
            this.context = context;
            this.commandQueue = commandQueue;
        }
        
        @Override
        public Object allocateBuffer(long size, String type) {
            int flags = CL.CL_MEM_READ_WRITE;
            
            try {
                int[] errorCode = new int[1];
                cl_mem buffer = CL.clCreateBuffer(context, flags, size, null, errorCode);
                
                if (errorCode[0] != CL.CL_SUCCESS) {
                    throw new RuntimeException("Failed to allocate OpenCL buffer: " + errorCode[0]);
                }
                
                return buffer;
            } catch (Exception e) {
                throw new RuntimeException("Error allocating OpenCL buffer", e);
            }
        }
        
        @Override
        public void releaseBuffer(Object buffer) {
            if (buffer instanceof cl_mem) {
                CL.clReleaseMemObject((cl_mem)buffer);
            }
        }
        
        @Override
        public Object getOrCreateKernel(String kernelName, String kernelSource) {
            // Check if we already have this kernel
            cl_kernel kernel = kernelCache.get(kernelName);
            if (kernel != null) {
                return kernel;
            }
            
            try {
                // Create and compile the program if needed
                String programName = kernelName + "_program";
                cl_program program = programCache.get(programName);
                
                if (program == null) {
                    // Create the program
                    program = CL.clCreateProgramWithSource(context, 1,
                            new String[]{kernelSource}, null, null);
                    
                    // Build the program
                    int err = CL.clBuildProgram(program, 0, null, null, null, null);
                    if (err != CL.CL_SUCCESS) {
                        // Get build log on error
                        long[] logSize = new long[1];
                        CL.clGetProgramBuildInfo(program, null, CL.CL_PROGRAM_BUILD_LOG,
                                0, null, logSize);
                        
                        byte[] logData = new byte[(int)logSize[0]];
                        CL.clGetProgramBuildInfo(program, null, CL.CL_PROGRAM_BUILD_LOG,
                                logData.length, Pointer.to(logData), null);
                        
                        String log = new String(logData);
                        throw new RuntimeException("Error building OpenCL program: " + log);
                    }
                    
                    // Cache the program
                    programCache.put(programName, program);
                }
                
                // Create the kernel
                kernel = CL.clCreateKernel(program, kernelName, null);
                
                // Cache the kernel
                kernelCache.put(kernelName, kernel);
                
                return kernel;
                
            } catch (Exception e) {
                throw new RuntimeException("Error creating OpenCL kernel", e);
            }
        }
        
        @Override
        public void cacheData(String key, Object data) {
            dataCache.put(key, data);
        }
        
        @Override
        public Object getCachedData(String key) {
            return dataCache.get(key);
        }
        
        @Override
        public void clearCache() {
            // Release kernels
            for (cl_kernel kernel : kernelCache.values()) {
                CL.clReleaseKernel(kernel);
            }
            kernelCache.clear();
            
            // Release programs
            for (cl_program program : programCache.values()) {
                CL.clReleaseProgram(program);
            }
            programCache.clear();
            
            // Clear data cache
            dataCache.clear();
        }
        
        @Override
        public Map<String, Object> getStatistics() {
            Map<String, Object> stats = new HashMap<>();
            stats.put("programCacheSize", programCache.size());
            stats.put("kernelCacheSize", kernelCache.size());
            stats.put("dataCacheSize", dataCache.size());
            return stats;
        }
        
        @Override
        public void releaseAll() {
            clearCache();
        }
    }
}
