package org.apache.opennlp.maxent;

import org.apache.opennlp.model.Context;

/**
 * Stub implementation of OpenNLP MaxentModel interface
 * This is a temporary placeholder until proper OpenNLP integration
 */
public interface MaxentModel {
    
    /**
     * Evaluates a context and returns the probability distribution over outcomes.
     */
    double[] eval(String[] context);
    
    /**
     * Evaluates a context and returns the probability distribution over outcomes.
     */
    double[] eval(String[] context, double[] probs);
    
    /**
     * Evaluates a context and returns the probability distribution over outcomes.
     */
    double[] eval(String[] context, float[] probs);
    
    /**
     * Returns the outcome associated with the index.
     */
    String getOutcome(int index);
    
    /**
     * Returns the number of outcomes for this model.
     */
    int getNumOutcomes();
    
    /**
     * Gets the index associated with the String name of the given outcome.
     */
    int getIndex(String outcome);
    
    /**
     * Returns all outcome names.
     */
    String[] getAllOutcomes();
    
    /**
     * Returns the data structures relevant to storing the model.
     */
    Object[] getDataStructures();
}
