package org.apache.opennlp.model;

/**
 * Stub implementation of OpenNLP Context class
 * This is a temporary placeholder until proper OpenNLP integration
 */
public class Context {
    private int[] outcomes;
    private float[] parameters;
    private float[] values;
    private String[] features;
    
    public Context(int[] outcomes, float[] parameters) {
        this.outcomes = outcomes;
        this.parameters = parameters;
        this.values = new float[outcomes.length];
        this.features = new String[outcomes.length];
    }
    
    public int[] getOutcomes() {
        return outcomes;
    }
    
    public float[] getParameters() {
        return parameters;
    }
    
    // Add missing methods that are being called
    public float[] getValues() {
        return values;
    }
    
    public String[] getFeatures() {
        return features;
    }
    
    // Alternative method name that might be used
    public String[] getFeature() {
        return features;
    }
}
