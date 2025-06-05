package org.apache.opennlp.gpu.common;

import java.util.Collections;
import java.util.List;
import java.util.ArrayList;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.jocl.*;

/**
 * Represents a GPU device that can be used for computation.
 * This class wraps OpenCL device information and provides utility methods.
 */
public class GpuDevice {
  
  private static final Logger logger = LoggerFactory.getLogger(GpuDevice.class);
  
  private final cl_device_id deviceId;
  private final String name;
  private final String vendor;
  private final long globalMemSize;
  private final long maxMemAllocSize;
  private final int computeUnits;
  private final int maxWorkGroupSize;
  
  /**
   * Creates a new GpuDevice instance.
   * 
   * @param deviceId the OpenCL device ID
   */
  public GpuDevice(cl_device_id deviceId) {
    this.deviceId = deviceId;
    
    // Get device information
    this.name = getDeviceInfoString(CL.CL_DEVICE_NAME);
    this.vendor = getDeviceInfoString(CL.CL_DEVICE_VENDOR);
    this.globalMemSize = getDeviceInfoLong(CL.CL_DEVICE_GLOBAL_MEM_SIZE);
    this.maxMemAllocSize = getDeviceInfoLong(CL.CL_DEVICE_MAX_MEM_ALLOC_SIZE);
    this.computeUnits = getDeviceInfoInt(CL.CL_DEVICE_MAX_COMPUTE_UNITS);
    this.maxWorkGroupSize = getDeviceInfoInt(CL.CL_DEVICE_MAX_WORK_GROUP_SIZE);
    
    logger.info("Initialized GPU device: {}", name);
  }
  
  /**
   * Get the OpenCL device ID.
   * 
   * @return the device ID
   */
  public cl_device_id getDeviceId() {
    return deviceId;
  }
  
  /**
   * Get the device name.
   * 
   * @return the device name
   */
  public String getName() {
    return name;
  }
  
  /**
   * Get the device vendor.
   * 
   * @return the device vendor
   */
  public String getVendor() {
    return vendor;
  }
  
  /**
   * Get the global memory size in bytes.
   * 
   * @return the global memory size
   */
  public long getGlobalMemSize() {
    return globalMemSize;
  }
  
  /**
   * Get the maximum memory allocation size in bytes.
   * 
   * @return the maximum memory allocation size
   */
  public long getMaxMemAllocSize() {
    return maxMemAllocSize;
  }
  
  /**
   * Get the number of compute units.
   * 
   * @return the number of compute units
   */
  public int getComputeUnits() {
    return computeUnits;
  }
  
  /**
   * Get the maximum work group size.
   * 
   * @return the maximum work group size
   */
  public int getMaxWorkGroupSize() {
    return maxWorkGroupSize;
  }
  
  /**
   * Get a list of all available GPU devices.
   * 
   * @return a list of available GPU devices
   */
  public static List<GpuDevice> getAvailableDevices() {
    try {
      // Initialize OpenCL
      CL.setExceptionsEnabled(true);
      
      // Get platforms
      int[] numPlatformsArray = new int[1];
      CL.clGetPlatformIDs(0, null, numPlatformsArray);
      int numPlatforms = numPlatformsArray[0];
      
      if (numPlatforms == 0) {
        logger.warn("No OpenCL platforms found");
        return Collections.emptyList();
      }
      
      // Get platform IDs
      cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
      CL.clGetPlatformIDs(platforms.length, platforms, null);
      
      List<GpuDevice> devices = new ArrayList<>();
      
      // For each platform, get devices
      for (cl_platform_id platform : platforms) {
        int[] numDevicesArray = new int[1];
        CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, 0, null, numDevicesArray);
        int numDevices = numDevicesArray[0];
        
        if (numDevices > 0) {
          cl_device_id[] deviceIds = new cl_device_id[numDevices];
          CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_GPU, numDevices, deviceIds, null);
          
          for (cl_device_id deviceId : deviceIds) {
            devices.add(new GpuDevice(deviceId));
          }
        }
      }
      
      logger.info("Found {} GPU devices", devices.size());
      return devices;
    } catch (Exception e) {
      logger.error("Error getting GPU devices", e);
      return Collections.emptyList();
    }
  }
  
  private String getDeviceInfoString(int paramName) {
    long[] size = new long[1];
    CL.clGetDeviceInfo(deviceId, paramName, 0, null, size);
    byte[] buffer = new byte[(int)size[0]];
    CL.clGetDeviceInfo(deviceId, paramName, buffer.length, Pointer.to(buffer), null);
    return new String(buffer, 0, buffer.length-1);
  }
  
  private long getDeviceInfoLong(int paramName) {
    long[] values = new long[1];
    CL.clGetDeviceInfo(deviceId, paramName, Sizeof.cl_long, Pointer.to(values), null);
    return values[0];
  }
  
  private int getDeviceInfoInt(int paramName) {
    int[] values = new int[1];
    CL.clGetDeviceInfo(deviceId, paramName, Sizeof.cl_int, Pointer.to(values), null);
    return values[0];
  }
  
  @Override
  public String toString() {
    return String.format("GpuDevice[name=%s, vendor=%s, computeUnits=%d, globalMem=%.2f GB]", 
        name, vendor, computeUnits, globalMemSize / (1024.0 * 1024.0 * 1024.0));
  }
}
