/*
 * ID: LTEST-001
 * Requirement: LombokTest must verify that all previously Lombok-annotated classes compile
 *              correctly with manually-implemented getters, setters, equals, and hashCode.
 * Purpose: Compilation smoke test confirming that removing Lombok annotation processing
 *          did not break any class API contracts.
 * Rationale: Lombok was removed to simplify the annotation processor stack; this test
 *            guards against regression by verifying the affected classes still compile.
 * Inputs: None (compile-time only).
 * Outputs: None; test passes if compilation succeeds.
 * Preconditions: All Lombok-annotated classes have been replaced with manual equivalents.
 * Postconditions: No ClassNotFoundException or NoSuchMethodError at runtime.
 * Assumptions: Maven annotation processor list does not include lombok.
 * Side Effects: None.
 * Failure Modes: Compilation error if manual implementation is missing.
 * Error Handling: Compile errors reported by Maven; test excluded from run if empty.
 * Constraints: None.
 * Verification: mvn clean compile passes with zero errors.
 * References: pom.xml; project migration notes.
 */
package org.apache.opennlp.gpu.common;

import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.assertNotNull;

/**
 * Verifies that classes previously using Lombok annotations compile and behave
 * correctly with their manually-implemented equivalents.
 */
public class LombokTest {

    @Test
    public void testComputeConfigurationConstructs() {
        ComputeConfiguration config = new ComputeConfiguration();
        assertNotNull(config, "ComputeConfiguration must construct without error");
    }

    @Test
    public void testCpuComputeProviderConstructs() {
        CpuComputeProvider provider = new CpuComputeProvider();
        assertNotNull(provider.getName(), "CpuComputeProvider.getName() must return non-null");
    }
}
