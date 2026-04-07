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
 */
package org.apache.opennlp.gpu.test;

import static org.junit.jupiter.api.Assertions.*;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.Test;

/**
 * Core functionality tests — always execute without GPU hardware.
 * Covers CpuComputeProvider, GpuConfig, and OperationFactory / DummyMatrixOperation.
 */
@DisplayName("Core Functionality Tests (CPU path)")
public class CoreFunctionalityTest {

    private static final float TOLERANCE = 1e-5f;

    private CpuComputeProvider provider;
    private GpuConfig config;
    private MatrixOperation ops;

    @BeforeEach
    void setUp() {
        config   = new GpuConfig();
        provider = new CpuComputeProvider();
        provider.initialize(config);
        ops      = OperationFactory.createMatrixOperation();
    }

    // ── GpuConfig ─────────────────────────────────────────────────────────────

    @Test
    @DisplayName("GpuConfig: default values are sane")
    void testGpuConfigDefaults() {
        GpuConfig cfg = new GpuConfig();
        assertFalse(cfg.isGpuEnabled(),           "GPU must default to disabled");
        assertFalse(cfg.isDebugMode(),             "debug must default to false");
        assertTrue(cfg.getMemoryPoolSizeMB() > 0,  "memoryPoolSizeMB must be positive");
        assertTrue(cfg.getBatchSize() >= 1,         "batchSize must be >= 1");
        assertTrue(cfg.getMaxMemoryUsageMB() > 0,   "maxMemoryUsageMB must be positive");
    }

    @Test
    @DisplayName("GpuConfig: setters round-trip correctly")
    void testGpuConfigSetters() {
        config.setGpuEnabled(true);
        config.setDebugMode(true);
        config.setMemoryPoolSizeMB(512);
        config.setBatchSize(64);
        config.setMaxMemoryUsageMB(2048);

        assertTrue(config.isGpuEnabled());
        assertTrue(config.isDebugMode());
        assertEquals(512,  config.getMemoryPoolSizeMB());
        assertEquals(64,   config.getBatchSize());
        assertEquals(2048, config.getMaxMemoryUsageMB());
    }

    @Test
    @DisplayName("GpuConfig: invalid memoryPoolSizeMB throws IllegalArgumentException")
    void testGpuConfigMemoryPoolValidation() {
        assertThrows(IllegalArgumentException.class, () -> config.setMemoryPoolSizeMB(0));
        assertThrows(IllegalArgumentException.class, () -> config.setMemoryPoolSizeMB(-1));
    }

    @Test
    @DisplayName("GpuConfig: invalid batchSize throws IllegalArgumentException")
    void testGpuConfigBatchSizeValidation() {
        assertThrows(IllegalArgumentException.class, () -> config.setBatchSize(0));
        assertThrows(IllegalArgumentException.class, () -> config.setBatchSize(-5));
    }

    @Test
    @DisplayName("GpuConfig: invalid maxMemoryUsageMB throws IllegalArgumentException")
    void testGpuConfigMaxMemoryValidation() {
        assertThrows(IllegalArgumentException.class, () -> config.setMaxMemoryUsageMB(0));
        assertThrows(IllegalArgumentException.class, () -> config.setMaxMemoryUsageMB(-64));
    }

    @Test
    @DisplayName("GpuConfig.isGpuAvailable() completes without exception")
    void testGpuConfigStaticIsGpuAvailable() {
        boolean result = GpuConfig.isGpuAvailable();
        // Value depends on environment; just verify no exception
        assertTrue(result == Boolean.TRUE || result == Boolean.FALSE);
    }

    @Test
    @DisplayName("GpuConfig.getGpuInfo() returns non-empty map with required keys")
    void testGpuConfigGetGpuInfo() {
        var info = GpuConfig.getGpuInfo();
        assertNotNull(info);
        assertFalse(info.isEmpty(),              "GPU info map must not be empty");
        assertTrue(info.containsKey("available"), "must contain 'available' key");
        assertTrue(info.containsKey("vendor"),    "must contain 'vendor' key");
    }

    // ── CpuComputeProvider ────────────────────────────────────────────────────

    @Test
    @DisplayName("CpuComputeProvider: is always available")
    void testCpuProviderIsAvailable() {
        assertTrue(provider.isAvailable(), "CPU provider must always be available");
    }

    @Test
    @DisplayName("CpuComputeProvider: type is CPU")
    void testCpuProviderType() {
        assertEquals(ComputeProvider.Type.CPU, provider.getType());
    }

    @Test
    @DisplayName("CpuComputeProvider: isGpuProvider returns false")
    void testCpuProviderIsNotGpu() {
        assertFalse(provider.isGpuProvider());
    }

    @Test
    @DisplayName("CpuComputeProvider: getName is non-null and non-blank")
    void testCpuProviderName() {
        assertNotNull(provider.getName());
        assertFalse(provider.getName().isBlank());
    }

    @Test
    @DisplayName("CpuComputeProvider: memory stats are non-negative")
    void testCpuProviderMemoryStats() {
        assertTrue(provider.getMaxMemoryMB() > 0,        "max memory must be > 0");
        assertTrue(provider.getCurrentMemoryUsageMB() >= 0, "current usage must be >= 0");
    }

    @Test
    @DisplayName("CpuComputeProvider: cleanup is idempotent (no exception)")
    void testCpuProviderCleanupIdempotent() {
        provider.cleanup();
        provider.cleanup();
    }

    // ── OperationFactory / DummyMatrixOperation ───────────────────────────────

    @Test
    @DisplayName("OperationFactory: createMatrixOperation() returns non-null")
    void testOperationFactoryReturnsNonNull() {
        assertNotNull(ops);
        assertNotNull(OperationFactory.createMatrixOperation(provider));
    }

    @Test
    @DisplayName("DummyMatrixOperation: 2x2 matrix multiply is correct")
    void testMatrixMultiply2x2() {
        // A = [[1,2],[3,4]]  B = [[5,6],[7,8]]  => C = [[19,22],[43,50]]
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {5f, 6f, 7f, 8f};
        float[] c = new float[4];

        ops.multiply(a, b, c, 2, 2, 2);

        assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, c, TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: 1x1 multiply is correct")
    void testMatrixMultiply1x1() {
        float[] a = {3f}, b = {4f}, result = new float[1];
        ops.multiply(a, b, result, 1, 1, 1);
        assertEquals(12f, result[0], TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: element-wise add is correct")
    void testElementWiseAdd() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {5f, 6f, 7f, 8f};
        float[] r = new float[4];

        ops.add(a, b, r, 4);

        assertArrayEquals(new float[]{6f, 8f, 10f, 12f}, r, TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: element-wise subtract is correct")
    void testElementWiseSubtract() {
        float[] a = {10f, 8f, 6f, 4f};
        float[] b = {1f,  2f, 3f, 4f};
        float[] r = new float[4];

        ops.subtract(a, b, r, 4);

        assertArrayEquals(new float[]{9f, 6f, 3f, 0f}, r, TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: scalar multiply is correct")
    void testScalarMultiply() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] r = new float[4];

        ops.scalarMultiply(a, r, 2.5f, 4);

        assertArrayEquals(new float[]{2.5f, 5f, 7.5f, 10f}, r, TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: transpose of 2x3 equals 3x2")
    void testTranspose2x3() {
        // Input 2x3: [[1,2,3],[4,5,6]]
        float[] input    = {1f, 2f, 3f, 4f, 5f, 6f};
        float[] output   = new float[6];
        // Expected 3x2: [[1,4],[2,5],[3,6]]
        float[] expected = {1f, 4f, 2f, 5f, 3f, 6f};

        ops.transpose(input, output, 2, 3);

        assertArrayEquals(expected, output, TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: dot product of identical unit vector is 1.0")
    void testDotProduct() {
        float[] a = {1f, 0f, 0f};
        float[] b = {1f, 0f, 0f};
        float[] r = new float[1];

        ops.dotProduct(a, b, r, 3);

        assertEquals(1f, r[0], TOLERANCE);
    }

    @Test
    @DisplayName("DummyMatrixOperation: L2 norm of [3,4] is 5")
    void testVectorNorm() {
        float[] v = {3f, 4f};
        float[] r = new float[1];

        ops.vectorNorm(v, r, 2);

        assertEquals(5f, r[0], TOLERANCE);
    }

    // ── CpuComputeProvider matrix ops ─────────────────────────────────────────

    @Test
    @DisplayName("CpuComputeProvider: matrixMultiply 2x2 is correct")
    void testCpuMatrixMultiply() {
        float[] a = {1f, 2f, 3f, 4f};
        float[] b = {5f, 6f, 7f, 8f};
        float[] c = new float[4];

        provider.matrixMultiply(a, b, c, 2, 2, 2);

        assertArrayEquals(new float[]{19f, 22f, 43f, 50f}, c, TOLERANCE);
    }

    @Test
    @DisplayName("CpuComputeProvider: matrixAdd is correct")
    void testCpuMatrixAdd() {
        float[] a = {1f, 2f}, b = {3f, 4f}, r = new float[2];
        provider.matrixAdd(a, b, r, 2);
        assertArrayEquals(new float[]{4f, 6f}, r, TOLERANCE);
    }

    @Test
    @DisplayName("CpuComputeProvider: matrixTranspose 3x2 is correct")
    void testCpuMatrixTranspose() {
        // 3x2: [[1,2],[3,4],[5,6]]  => 2x3: [[1,3,5],[2,4,6]]
        float[] in  = {1f, 2f, 3f, 4f, 5f, 6f};
        float[] out = new float[6];

        provider.matrixTranspose(in, out, 3, 2);

        assertArrayEquals(new float[]{1f, 3f, 5f, 2f, 4f, 6f}, out, TOLERANCE);
    }
}
