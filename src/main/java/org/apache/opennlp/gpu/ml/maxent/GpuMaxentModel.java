package org.apache.opennlp.gpu.ml.maxent;

import org.apache.opennlp.gpu.common.GpuDevice;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.apache.opennlp.gpu.kernels.MatrixOps;
import org.jocl.*;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU-accelerated implementation of the Maximum Entropy model.
 * This class provides GPU-accelerated methods for inference using
 * pre-trained MaxEnt models.
 */
public class GpuMaxentModel {
    private static final Logger logger = LoggerFactory.getLogger(GpuMaxentModel.class);
    
    private final cl_context context;
    private final cl_command_queue commandQueue;
    private final MatrixOps matrixOps;
    
    private float[] weights; // Model weights
    private int numOutcomes; // Number of outcomes
    private int numFeatures; // Number of features
    
    /**
     * Creates a new GPU-accelerated MaxEnt model.
     *
     * @param device the GPU device to use
     * @param weights the model weights matrix
     * @param numOutcomes the number of outcomes
     * @param numFeatures the number of features
     */
    public GpuMaxentModel(GpuDevice device, float[] weights, int numOutcomes, int numFeatures) {
        try {
            // Create OpenCL context and command queue
            cl_context_properties contextProperties = new cl_context_properties();
            // Replace this line that uses getInfo() which doesn't exist
            // contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, device.getDeviceId().getInfo());
            
            // Instead use the helper method getOpenCLDeviceName or call the proper CL API
            long platformId = getPlatformId(device.getDeviceId());
            contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platformId);
            
            int[] errorCode = new int[1];
            context = CL.clCreateContext(contextProperties, 1, new cl_device_id[]{device.getDeviceId()}, 
                                        null, null, errorCode);
            
            if (errorCode[0] != CL.CL_SUCCESS) {
                throw new RuntimeException("Failed to create OpenCL context: " + errorCode[0]);
            }
            
            commandQueue = CL.clCreateCommandQueue(context, device.getDeviceId(), 0, errorCode);
            
            if (errorCode[0] != CL.CL_SUCCESS) {
                throw new RuntimeException("Failed to create command queue: " + errorCode[0]);
            }
            
            // Initialize matrix operations
            matrixOps = new MatrixOps(context, commandQueue);
            
            // Store model parameters
            this.weights = weights;
            this.numOutcomes = numOutcomes;
            this.numFeatures = numFeatures;
            
            logger.info("Initialized GPU MaxEnt model with {} outcomes and {} features", 
                       numOutcomes, numFeatures);
            
        } catch (Exception e) {
            logger.error("Error initializing GPU MaxEnt model", e);
            throw new RuntimeException("Error initializing GPU MaxEnt model", e);
        }
    }
    
    /**
     * Evaluates a context using the model, returning outcome probabilities.
     *
     * @param context the array of active feature indices
     * @return an array of probabilities, one for each outcome
     */
    public float[] eval(int[] context) {
        return eval(context, new float[numOutcomes]);
    }
    
    /**
     * Evaluates a context using the model, storing results in the provided array.
     *
     * @param context the array of active feature indices
     * @param probabilities the array to store outcome probabilities
     * @return the array of probabilities, one for each outcome
     */
    public float[] eval(int[] context, float[] probabilities) {
        if (probabilities.length != numOutcomes) {
            throw new IllegalArgumentException(
                "The probabilities array should have room for " + numOutcomes + " outcomes");
        }
        
        try {
            // Create dense feature vector from sparse context
            float[] featureVector = new float[numFeatures];
            for (int contextIndex : context) {
                if (contextIndex < numFeatures) {
                    featureVector[contextIndex] = 1.0f;
                }
            }
            
            // Compute scores: scores = featureVector * weights
            float[] scores = new float[numOutcomes];
            matrixOps.matrixMultiply(featureVector, weights, scores, 1, numOutcomes, numFeatures);
            
            // Apply softmax to get probabilities
            softmax(scores, probabilities);
            
            return probabilities;
        } catch (Exception e) {
            logger.error("Error evaluating context", e);
            throw new RuntimeException("Error evaluating context", e);
        }
    }
    
    /**
     * Applies the softmax function to convert scores to probabilities.
     *
     * @param scores the input scores
     * @param probabilities the output probabilities
     */
    private void softmax(float[] scores, float[] probabilities) {
        // Find max score for numerical stability
        float maxScore = Float.NEGATIVE_INFINITY;
        for (float score : scores) {
            if (score > maxScore) {
                maxScore = score;
            }
        }
        
        // Compute exp(score - maxScore) and sum
        float sum = 0.0f;
        for (int i = 0; i < scores.length; i++) {
            probabilities[i] = (float) Math.exp(scores[i] - maxScore);
            sum += probabilities[i];
        }
        
        // Normalize
        for (int i = 0; i < probabilities.length; i++) {
            probabilities[i] /= sum;
        }
    }
    
    /**
     * Batch evaluates multiple contexts, which is more efficient on the GPU.
     *
     * @param contexts a list of context arrays (active feature indices)
     * @return a 2D array of probabilities, one row per context
     */
    public float[][] evalBatch(List<int[]> contexts) {
        int batchSize = contexts.size();
        float[][] results = new float[batchSize][];
        
        // TODO: Implement batch evaluation with GPU acceleration
        // This would involve:
        // 1. Creating a batch feature matrix
        // 2. Performing a single matrix multiplication
        // 3. Applying softmax to each row of the result
        
        // For now, process each context individually
        for (int i = 0; i < batchSize; i++) {
            results[i] = eval(contexts.get(i));
        }
        
        return results;
    }
    
    /**
     * Releases all resources associated with this model.
     */
    public void release() {
        try {
            // Release matrix operations
            matrixOps.release();
            
            // Release OpenCL resources
            CL.clReleaseCommandQueue(commandQueue);
            CL.clReleaseContext(context);
            
            logger.info("Released GPU MaxEnt model resources");
        } catch (Exception e) {
            logger.error("Error releasing GPU MaxEnt model resources", e);
        }
    }
    
    /**
     * Get device information.
     *
     * @param device the OpenCL device
     * @return the device information string
     */
    public static String getDeviceInfo(cl_device_id device) {
        if (device == null) {
            return "Device not available";
        }
        
        // Instead of using getInfo() which doesn't exist, build device info manually
        StringBuilder info = new StringBuilder();
        info.append("Device ID: ").append(device.toString());
        
        return info.toString();
    }
    
    private String getOpenCLDeviceName(cl_device_id device) {
        // Obtain the length of the device name
        long[] size = new long[1];
        CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 0, null, size);

        // Allocate buffer for the name
        byte[] buffer = new byte[(int)size[0]];
        CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);

        // Convert to String, excluding the null terminator
        return new String(buffer, 0, buffer.length-1);
    }

    // Example of how you might use it if line 44 was trying to get device info:
    public void someMethodThatUsesDeviceInfo(cl_device_id openClDevice) {
        // ... other code ...
        // String deviceInfo = openClDevice.getInfo(); // THIS IS THE ERRONEOUS LINE
        // Replace with:
        String deviceName = getOpenCLDeviceName(openClDevice);
        logger.info("Using OpenCL device: {}", deviceName);
        // ... or for other parameters, use CL.clGetDeviceInfo with the appropriate param name ...
        // ... other code ...
    }
    
    // Add helper method to get platform ID
    private long getPlatformId(cl_device_id deviceId) {
        int[] numPlatforms = new int[1];
        CL.clGetPlatformIDs(0, null, numPlatforms);
        
        cl_platform_id[] platforms = new cl_platform_id[numPlatforms[0]];
        CL.clGetPlatformIDs(platforms.length, platforms, null);
        
        for (cl_platform_id platform : platforms) {
            int[] numDevices = new int[1];
            CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevices);
            
            cl_device_id[] devices = new cl_device_id[numDevices[0]];
            CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, devices.length, devices, null);
            
            for (cl_device_id device : devices) {
                if (device.equals(deviceId)) {
                    long platformId = platform.getNativePointer();
                    return platformId; // Return the long value directly, don't wrap in Pointer
                }
            }
        }
        
        logger.warn("Could not find platform for device - using default");
        long platformId = platforms[0].getNativePointer();
        return platformId; // Return the long value directly, don't wrap in Pointer
    }
}
