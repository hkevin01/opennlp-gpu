package org.apache.opennlp.gpu.compute;

/**
 * Configuration for matrix operations.
 */
public class MatrixOperationConfig {
    // Example properties - replace with actual config needed
    private int preferredWorkGroupSizeMultiple;
    private boolean usePinnedMemory;

    public MatrixOperationConfig() {
        // Default constructor
    }

    // Example getter/setter - add actual methods
    public int getPreferredWorkGroupSizeMultiple() {
        return preferredWorkGroupSizeMultiple;
    }

    public void setPreferredWorkGroupSizeMultiple(int preferredWorkGroupSizeMultiple) {
        this.preferredWorkGroupSizeMultiple = preferredWorkGroupSizeMultiple;
    }

    public boolean shouldUsePinnedMemory() {
        return usePinnedMemory;
    }

    public void setUsePinnedMemory(boolean usePinnedMemory) {
        this.usePinnedMemory = usePinnedMemory;
    }
}
