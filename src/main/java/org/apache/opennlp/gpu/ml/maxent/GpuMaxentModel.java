package org.apache.opennlp.gpu.ml.maxent;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.ComputeProviderFactory;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.compute.OperationFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GPU-accelerated implementation of the Maximum Entropy model.
 * This class extends the standard OpenNLP MaxEnt model with GPU-accelerated
 * computation for probability distributions and evaluations.
 */
public class GpuMaxentModel {
    
    private static final Logger logger = LoggerFactory.getLogger(GpuMaxentModel.class);
    
    private final ComputeProvider provider;
    private final MatrixOperation matrixOp;
    private final int numOutcomes;
    private final int numFeatures;
    private final Context[] contexts;
    private final float[] parameters;
    
    /**
     * Creates a new GPU-accelerated MaxEnt model.
     * 
     * @param provider The compute provider to use for GPU operations
     * @param numOutcomes The number of outcomes
     * @param numFeatures The number of features
     * @param parameters The model parameters
     * @param contexts The model contexts
     */
    public GpuMaxentModel(ComputeProvider provider, int numOutcomes, int numFeatures, 
                          float[] parameters, Context[] contexts) {
        this.provider = provider;
        this.matrixOp = OperationFactory.createMatrixOperation(provider);
        this.numOutcomes = numOutcomes;
        this.numFeatures = numFeatures;
        this.parameters = parameters;
        this.contexts = contexts;
        
        logger.info("Initialized GPU-accelerated MaxEnt model with {} outcomes and {} features",
                numOutcomes, numFeatures);
    }
    
    /**
     * Creates a new GPU-accelerated MaxEnt model using the default compute provider.
     */
    public GpuMaxentModel(int numOutcomes, int numFeatures, float[] parameters, Context[] contexts) {
        this(ComputeProviderFactory.getDefaultProvider(), numOutcomes, numFeatures, parameters, contexts);
    }
    
    /**
     * Evaluates a context using the GPU-accelerated model.
     * 
     * @param context The features present in the context
     * @param prior The prior distribution
     * @param outcomesVal The outcome values array to be populated
     */
    public void eval(int[] context, float[] prior, float[] outcomesVal) {
        // Initialize output array
        for (int i = 0; i < numOutcomes; i++) {
            outcomesVal[i] = prior[i];
        }
        
        // Create feature vector on GPU
        float[] featureVector = new float[numFeatures];
        for (int i : context) {
            if (i < numFeatures) {
                featureVector[i] = 1.0f;
            }
        }
        
        // GPU-accelerated matrix multiplication for calculating activations
        // This replaces the standard loop-based approach with GPU matrix operations
        float[] activations = new float[numOutcomes];
        float[] modelWeights = extractModelWeights();
        
        // Execute matrix multiplication on GPU: activations = featureVector * modelWeights
        matrixOp.multiply(featureVector, modelWeights, activations, 1, numOutcomes, numFeatures);
        
        // Apply exponential and normalization (using CPU for now)
        // Future improvement: implement on GPU as well
        float sum = 0.0f;
        for (int i = 0; i < numOutcomes; i++) {
            outcomesVal[i] = (float) Math.exp(activations[i]);
            sum += outcomesVal[i];
        }
        
        // Normalize
        if (sum != 0.0f) {
            for (int i = 0; i < numOutcomes; i++) {
                outcomesVal[i] /= sum;
            }
        }
    }
    
    /**
     * Extract model weights into a matrix format suitable for GPU operations.
     */
    private float[] extractModelWeights() {
        float[] modelWeights = new float[numFeatures * numOutcomes];
        
        // Populate the weight matrix from contexts and parameters
        for (Context context : contexts) {
            int[] features = context.getOutcomes();
            float[] values = context.getValues();
            
            for (int i = 0; i < features.length; i++) {
                int outcomeId = features[i];
                modelWeights[context.getFeature() * numOutcomes + outcomeId] = values[i];
            }
        }
        
        return modelWeights;
    }
    
    /**
     * Release GPU resources when model is no longer needed.
     */
    public void release() {
        if (matrixOp != null) {
            matrixOp.release();
        }
        logger.info("Released GPU resources for MaxEnt model");
    }
}
