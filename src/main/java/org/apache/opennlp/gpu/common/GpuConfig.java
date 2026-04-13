/*
 * Copyright 2025 OpenNLP GPU Extension Contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * This project is a third-party GPU acceleration extension for Apache OpenNLP.
 * It is not officially endorsed or maintained by the Apache Software Foundation.
 */
package org.apache.opennlp.gpu.common;

import java.util.HashMap;
import java.util.Map;

/**

 * ID: GPU-GC-001
 * Requirement: Encapsulate all runtime parameters that govern GPU compute
 *   provider selection, memory allocation, and batch processing behavior.
 * Purpose: Provides a single, well-typed configuration object that is passed
 *   to ComputeProvider.initialize(), removing scattered System.getProperty()
 *   calls from hot paths.
 * Rationale: Centralizing configuration simplifies tuning for production
 *   deployments, facilitates testing via constructor injection, and provides
 *   a documented interface for each tunable parameter.
 * Inputs: All fields are set via plain setters. Defaults are chosen for safe
 *   operation on a machine with no GPU.
 * Outputs: Read via getters; used by ComputeProvider implementations during
 *   initialization.
 * Preconditions: None — safe to construct without arguments.
 * Postconditions: All getters return valid values immediately after construction.
 * Assumptions: Memory values are in megabytes. batchSize ≥ 1.
 * Side Effects: None. This class is a pure value object.
 * Failure Modes: Negative or zero memoryPoolSizeMB, batchSize, or
 *   maxMemoryUsageMB values throw IllegalArgumentException at set time.
 * Constraints: maxMemoryUsageMB must be ≥ memoryPoolSizeMB for correct operation.
 * Verification: Validated by BasicValidationTest; used in all integration tests.
 * References: OpenNLP model evaluation pipeline; CUDA memory management best practices.
 */
public class GpuConfig {

    /** Whether GPU acceleration is requested. Defaults to false (safe CPU-only mode). */
    private boolean gpuEnabled = false;

    /** When true, providers emit verbose diagnostic output. */
    private boolean debugMode = false;

    /**
     * Size of the pre-allocated GPU memory pool in MB.
     * A pool avoids runtime cudaMalloc overhead during inference.
     * Default: 256 MB.
     */
    private int memoryPoolSizeMB = 256;

    /**
     * Number of samples processed per GPU kernel launch.
     * Higher values improve GPU utilization; lower values reduce latency.
     * Default: 32.
     */
    private int batchSize = 32;

    /**
     * Hard upper bound on total GPU memory this provider may consume, in MB.
     * Providers must refuse to allocate beyond this limit.
     * Default: 1024 MB (1 GB).
     */
    private int maxMemoryUsageMB = 1024;

    /**

     * ID: GPU-GC-002
     * Requirement: Construct a GpuConfig with all defaults suitable for a
     *   CPU-only or test environment.
     */
    public GpuConfig() {
        // All fields initialized to safe defaults above.
    }

    // ---- Getters and Setters ----

    /** Returns whether GPU acceleration is requested. */
    /**
    
     * ID: GPU-GC-005
     * Requirement: Evaluate and return the boolean result of isGpuEnabled.
     * Purpose: Return whether isGpuEnabled condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public boolean isGpuEnabled() { return gpuEnabled; }

    /** Enables or disables GPU acceleration. */
    /**
    
     * ID: GPU-GC-006
     * Requirement: Update the GpuEnabled field to the supplied non-null value.
     * Purpose: Set the GpuEnabled property to the supplied value.
     * Inputs: boolean enabled
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setGpuEnabled(boolean enabled) { this.gpuEnabled = enabled; }

    /** Returns whether verbose diagnostic mode is active. */
    /**
    
     * ID: GPU-GC-007
     * Requirement: Evaluate and return the boolean result of isDebugMode.
     * Purpose: Return whether isDebugMode condition holds.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public boolean isDebugMode() { return debugMode; }

    /** Enables or disables verbose diagnostic output. */
    /**
    
     * ID: GPU-GC-008
     * Requirement: Update the DebugMode field to the supplied non-null value.
     * Purpose: Set the DebugMode property to the supplied value.
     * Inputs: boolean debugMode
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setDebugMode(boolean debugMode) { this.debugMode = debugMode; }

    /** Returns the GPU memory pool size in MB. */
    /**
    
     * ID: GPU-GC-009
     * Requirement: Return the MemoryPoolSizeMB field value without side effects.
     * Purpose: Return the value of the MemoryPoolSizeMB property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getMemoryPoolSizeMB() { return memoryPoolSizeMB; }

    /**
     * Sets the GPU memory pool size in MB.
     * @param memoryPoolSizeMB must be positive
     * @throws IllegalArgumentException if memoryPoolSizeMB is not positive
     */
    /**
    
     * ID: GPU-GC-010
     * Requirement: Update the MemoryPoolSizeMB field to the supplied non-null value.
     * Purpose: Set the MemoryPoolSizeMB property to the supplied value.
     * Inputs: int memoryPoolSizeMB
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setMemoryPoolSizeMB(int memoryPoolSizeMB) {
        if (memoryPoolSizeMB <= 0) {
            throw new IllegalArgumentException("memoryPoolSizeMB must be > 0, got: " + memoryPoolSizeMB);
        }
        this.memoryPoolSizeMB = memoryPoolSizeMB;
    }

    /** Returns the inference batch size. */
    /**
    
     * ID: GPU-GC-011
     * Requirement: Return the BatchSize field value without side effects.
     * Purpose: Return the value of the BatchSize property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getBatchSize() { return batchSize; }

    /**
     * Sets the inference batch size.
     * @param batchSize must be ≥ 1
     * @throws IllegalArgumentException if batchSize is less than 1
     */
    /**
    
     * ID: GPU-GC-012
     * Requirement: Update the BatchSize field to the supplied non-null value.
     * Purpose: Set the BatchSize property to the supplied value.
     * Inputs: int batchSize
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setBatchSize(int batchSize) {
        if (batchSize < 1) {
            throw new IllegalArgumentException("batchSize must be >= 1, got: " + batchSize);
        }
        this.batchSize = batchSize;
    }

    /** Returns the maximum GPU memory usage limit in MB. */
    /**
    
     * ID: GPU-GC-013
     * Requirement: Return the MaxMemoryUsageMB field value without side effects.
     * Purpose: Return the value of the MaxMemoryUsageMB property.
     * Inputs: None — no parameters.
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public int getMaxMemoryUsageMB() { return maxMemoryUsageMB; }

    /**
     * Sets the maximum GPU memory usage limit in MB.
     * @param maxMemoryUsageMB must be positive
     * @throws IllegalArgumentException if maxMemoryUsageMB is not positive
     */
    /**
    
     * ID: GPU-GC-014
     * Requirement: Update the MaxMemoryUsageMB field to the supplied non-null value.
     * Purpose: Set the MaxMemoryUsageMB property to the supplied value.
     * Inputs: int maxMemoryUsageMB
     * Outputs: Return value or output parameter as described; void otherwise.
     * Postconditions: Return value or output parameter contains the computed result.
     * Side Effects: May modify instance state; see method body for details.
     * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
     * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
     */
    public void setMaxMemoryUsageMB(int maxMemoryUsageMB) {
        if (maxMemoryUsageMB <= 0) {
            throw new IllegalArgumentException("maxMemoryUsageMB must be > 0, got: " + maxMemoryUsageMB);
        }
        this.maxMemoryUsageMB = maxMemoryUsageMB;
    }

    // ---- Static Utility Methods ----

    /**

     * ID: GPU-GC-003
     * Requirement: Detect whether GPU acceleration is available on this host
     *   by checking the system property {@code gpu.available}.
     * Purpose: Provides a lightweight guard used by model factories before
     *   attempting GPU provider initialization.
     * Inputs: System property "gpu.available" (String "true"/"false").
     * Outputs: true only when "true".equals(System.getProperty("gpu.available")).
     * Side Effects: None. Reads one system property.
     * Failure Modes: SecurityException from System.getProperty() is caught and
     *   treated as unavailable.
     * References: Java Security Manager property access.
     */
    public static boolean isGpuAvailable() {
        try {
            String gpuProperty = System.getProperty("gpu.available", "false");
            return "true".equals(gpuProperty);
        } catch (SecurityException e) {
            return false;
        }
    }

    /**

     * ID: GPU-GC-004
     * Requirement: Collect GPU hardware metadata for diagnostic and logging use.
     * Purpose: Used by GpuDiagnostics to populate health-check reports without
     *   requiring a live GPU context.
     * Outputs: Map with keys: available, vendor, device, driver_version,
     *   memory_total, memory_free. Values are Strings or Boolean.
     * Side Effects: Reads up to 6 system properties.
     */
    public static Map<String, Object> getGpuInfo() {
        Map<String, Object> info = new HashMap<>();
        info.put("available",       isGpuAvailable());
        info.put("vendor",          System.getProperty("gpu.vendor",         "Unknown"));
        info.put("device",          System.getProperty("gpu.device",         "Unknown"));
        info.put("driver_version",  System.getProperty("gpu.driver",         "Unknown"));
        info.put("memory_total",    System.getProperty("gpu.memory.total",   "0"));
        info.put("memory_free",     System.getProperty("gpu.memory.free",    "0"));
        return info;
    }
}
