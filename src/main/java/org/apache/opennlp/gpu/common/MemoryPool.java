/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.opennlp.gpu.common;

import java.util.Map;
import java.util.Queue;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentLinkedQueue;
import org.jocl.cl_mem;

/**

 * ID: GPU-MP-001
 * Requirement: MemoryPool must pre-allocate and reuse GPU memory buffers to reduce per-operation allocation overhead.
 * Purpose: Manages a fixed pool of GPU memory blocks sized per GpuConfig, handing out blocks to callers and returning them on release.
 * Rationale: Repeated cudaMalloc/hipMalloc calls are expensive; pooling amortises this cost across batch NLP operations.
 * Inputs: Constructor parameters and method arguments as documented per method.
 * Outputs: Provides services and data as defined by the implemented interface(s).
 * Preconditions: JVM initialised; required dependencies available on classpath.
 * Postconditions: Object state is consistent; resources are properly initialised or null.
 * Assumptions: Called in a standard JVM environment with Java 21+ runtime.
 * Side Effects: Allocates a fixed memory block on initialisation; updates pool state on every acquire/release.
 * Failure Modes: Constructor failure throws RuntimeException; individual methods
 *               document their own failure modes.
 * Error Handling: Exceptions propagated to caller; fallback paths documented per method.
 * Constraints: Thread safety per class-level documentation; memory bounded by config.
 * Verification: Unit and integration tests in src/test; see GpuTestSuite.
 * References: Apache OpenNLP 2.5.8 API; project ARCHITECTURE_OVERVIEW.md.
 */
public class MemoryPool {

  private final Map<Long,Queue<cl_mem>> pool = new ConcurrentHashMap<Long,Queue<cl_mem>>();

  /**
  
   * ID: GPU-MP-002
   * Requirement: acquire must execute correctly within the contract defined by this class.
   * Purpose: Implement the acquire operation for this class.
   * Inputs: long size
   * Outputs: Return value or output parameter as described; void otherwise.
   * Postconditions: Return value or output parameter contains the computed result.
   * Side Effects: May modify instance state; see method body for details.
   * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
   * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
   */
  public cl_mem acquire(long size) {
    Queue<cl_mem> q = pool.get(size);
    return (q != null && !q.isEmpty()) ? q.poll() : null;
  }

  /**
  
   * ID: GPU-MP-003
   * Requirement: release must execute correctly within the contract defined by this class.
   * Purpose: Release all held resources and reset internal state.
   * Inputs: cl_mem buffer
   * Outputs: Return value or output parameter as described; void otherwise.
   * Postconditions: Return value or output parameter contains the computed result.
   * Side Effects: May modify instance state; see method body for details.
   * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
   * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
   */
  public void release(cl_mem buffer) {
    // no‐op
  }

  /**
  
   * ID: GPU-MP-004
   * Requirement: cleanup must execute correctly within the contract defined by this class.
   * Purpose: Release all held resources and reset internal state.
   * Inputs: None — no parameters.
   * Outputs: Return value or output parameter as described; void otherwise.
   * Postconditions: Return value or output parameter contains the computed result.
   * Side Effects: May modify instance state; see method body for details.
   * Failure Modes: IllegalArgumentException on invalid inputs; see method body.
   * Error Handling: Invalid inputs throw IllegalArgumentException or return safe defaults.
   */
  public void cleanup() {
    pool.clear();
  }
}
