package org.apache.opennlp.gpu.ml.neural.advanced;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.common.GpuLogger;
import org.apache.opennlp.gpu.compute.CpuComputeProvider;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.GpuComputeProvider;
import org.apache.opennlp.gpu.compute.GpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;

/**
 * Advanced GPU-accelerated neural network with modern features
 * Includes batch normalization, dropout, advanced optimizers, and regularization
 */
public class AdvancedGpuNeuralNetwork {
    
    private static final GpuLogger logger = GpuLogger.getLogger(AdvancedGpuNeuralNetwork.class);
    
    private final ComputeProvider computeProvider;
    private final MatrixOperation matrixOp;
    private final GpuConfig config;
    
    // Network architecture
    private final int[] layerSizes;
    private final String[] activationFunctions;
    private final int numLayers;
    
    // Network parameters
    private float[][][] weights; // [layer][input][output]
    private float[][] biases;    // [layer][neuron]
    private float[][] activations; // [layer][neuron]
    private float[][] zValues;   // pre-activation values
    
    // Batch normalization parameters
    private float[][] bnGamma;   // scaling parameters
    private float[][] bnBeta;    // shift parameters
    private float[][] bnMean;    // running mean
    private float[][] bnVar;     // running variance
    private boolean useBatchNorm;
    
    // Dropout parameters
    private float[][] dropoutMasks;
    private float dropoutRate;
    private boolean useDropout;
    
    // Optimizer parameters
    private OptimizerType optimizerType;
    private float learningRate;
    private float momentum;
    private float beta1, beta2; // for Adam optimizer
    private float weightDecay;   // L2 regularization
    
    // Optimizer state
    private float[][][] velocityWeights; // for momentum/Adam
    private float[][] velocityBiases;
    private float[][][] momentWeights;   // for Adam
    private float[][] momentBiases;
    private int adamStep;
    
    // Training configuration
    private int batchSize;
    private boolean isTraining;
    
    public enum OptimizerType {
        SGD, MOMENTUM, ADAM, RMSPROP
    }
    
    public enum LossFunction {
        MEAN_SQUARED_ERROR, CROSS_ENTROPY, BINARY_CROSS_ENTROPY
    }
    
    /**
     * Constructor for advanced neural network
     */
    public AdvancedGpuNeuralNetwork(int[] layerSizes, String[] activationFunctions, 
                                   GpuConfig config, NetworkConfiguration netConfig) {
        this.layerSizes = layerSizes.clone();
        this.activationFunctions = activationFunctions.clone();
        this.numLayers = layerSizes.length;
        this.config = config;
        this.batchSize = netConfig.batchSize;
        this.useBatchNorm = netConfig.useBatchNorm;
        this.useDropout = netConfig.useDropout;
        this.dropoutRate = netConfig.dropoutRate;
        this.optimizerType = netConfig.optimizerType;
        this.learningRate = netConfig.learningRate;
        this.momentum = netConfig.momentum;
        this.beta1 = netConfig.beta1;
        this.beta2 = netConfig.beta2;
        this.weightDecay = netConfig.weightDecay;
        this.isTraining = false;
        this.adamStep = 0;
        
        this.computeProvider = createComputeProvider();
        this.matrixOp = createMatrixOperation();
        
        initializeNetwork();
        initializeOptimizer();
        
        logger.info("Advanced neural network created with " + getTotalParameters() + " parameters");
        logger.info("Features: BatchNorm=" + useBatchNorm + ", Dropout=" + useDropout + 
                   ", Optimizer=" + optimizerType);
    }
    
    private ComputeProvider createComputeProvider() {
        try {
            if (config.isGpuEnabled() && GpuComputeProvider.isGpuAvailable()) {
                return new GpuComputeProvider(config);
            }
        } catch (Exception e) {
            logger.warn("Failed to initialize GPU provider: " + e.getMessage());
        }
        return new CpuComputeProvider();
    }
    
    private MatrixOperation createMatrixOperation() {
        if (computeProvider.isGpuProvider()) {
            return new GpuMatrixOperation(computeProvider, config);
        } else {
            return new CpuMatrixOperation(computeProvider);
        }
    }
    
    private void initializeNetwork() {
        // Initialize weights and biases
        weights = new float[numLayers - 1][][];
        biases = new float[numLayers - 1][];
        activations = new float[numLayers][];
        zValues = new float[numLayers - 1][];
        
        // Initialize batch normalization parameters
        if (useBatchNorm) {
            bnGamma = new float[numLayers - 1][];
            bnBeta = new float[numLayers - 1][];
            bnMean = new float[numLayers - 1][];
            bnVar = new float[numLayers - 1][];
        }
        
        // Initialize dropout masks
        if (useDropout) {
            dropoutMasks = new float[numLayers - 1][];
        }
        
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            
            // Xavier/He initialization based on activation function
            float scale = getInitializationScale(activationFunctions[layer], inputSize, outputSize);
            
            // Initialize weights
            weights[layer] = new float[inputSize][outputSize];
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[layer][i][j] = (float) (Math.random() * 2 * scale - scale);
                }
            }
            
            // Initialize biases
            biases[layer] = new float[outputSize];
            matrixOp.fillArray(biases[layer], 0.0f, outputSize);
            
            // Initialize batch normalization
            if (useBatchNorm) {
                bnGamma[layer] = new float[outputSize];
                bnBeta[layer] = new float[outputSize];
                bnMean[layer] = new float[outputSize];
                bnVar[layer] = new float[outputSize];
                
                matrixOp.fillArray(bnGamma[layer], 1.0f, outputSize);
                matrixOp.fillArray(bnBeta[layer], 0.0f, outputSize);
                matrixOp.fillArray(bnMean[layer], 0.0f, outputSize);
                matrixOp.fillArray(bnVar[layer], 1.0f, outputSize);
            }
            
            // Initialize dropout masks
            if (useDropout) {
                dropoutMasks[layer] = new float[outputSize];
            }
            
            // Initialize z-values buffer
            zValues[layer] = new float[outputSize];
        }
        
        // Initialize activation buffers
        for (int layer = 0; layer < numLayers; layer++) {
            activations[layer] = new float[layerSizes[layer]];
        }
    }
    
    private void initializeOptimizer() {
        if (optimizerType == OptimizerType.MOMENTUM || optimizerType == OptimizerType.ADAM) {
            velocityWeights = new float[numLayers - 1][][];
            velocityBiases = new float[numLayers - 1][];
            
            for (int layer = 0; layer < numLayers - 1; layer++) {
                int inputSize = layerSizes[layer];
                int outputSize = layerSizes[layer + 1];
                
                velocityWeights[layer] = new float[inputSize][outputSize];
                velocityBiases[layer] = new float[outputSize];
                
                // Initialize to zero
                for (int i = 0; i < inputSize; i++) {
                    matrixOp.fillArray(velocityWeights[layer][i], 0.0f, outputSize);
                }
                matrixOp.fillArray(velocityBiases[layer], 0.0f, outputSize);
            }
        }
        
        if (optimizerType == OptimizerType.ADAM) {
            momentWeights = new float[numLayers - 1][][];
            momentBiases = new float[numLayers - 1][];
            
            for (int layer = 0; layer < numLayers - 1; layer++) {
                int inputSize = layerSizes[layer];
                int outputSize = layerSizes[layer + 1];
                
                momentWeights[layer] = new float[inputSize][outputSize];
                momentBiases[layer] = new float[outputSize];
                
                // Initialize to zero
                for (int i = 0; i < inputSize; i++) {
                    matrixOp.fillArray(momentWeights[layer][i], 0.0f, outputSize);
                }
                matrixOp.fillArray(momentBiases[layer], 0.0f, outputSize);
            }
        }
    }
    
    private float getInitializationScale(String activation, int inputSize, int outputSize) {
        switch (activation.toLowerCase()) {
            case "relu":
            case "leaky_relu":
                // He initialization
                return (float) Math.sqrt(2.0 / inputSize);
            case "sigmoid":
            case "tanh":
                // Xavier initialization
                return (float) Math.sqrt(6.0 / (inputSize + outputSize));
            default:
                // Default Xavier
                return (float) Math.sqrt(2.0 / (inputSize + outputSize));
        }
    }
    
    /**
     * Forward propagation with advanced features
     */
    public float[] forward(float[] input) {
        if (input.length != layerSizes[0]) {
            throw new IllegalArgumentException("Input size mismatch: expected " + layerSizes[0] + 
                                             ", got " + input.length);
        }
        
        // Copy input to first layer
        matrixOp.copyArray(input, activations[0], input.length);
        
        // Forward pass through hidden layers
        for (int layer = 0; layer < numLayers - 1; layer++) {
            forwardLayer(layer);
        }
        
        // Return output
        float[] output = new float[layerSizes[numLayers - 1]];
        matrixOp.copyArray(activations[numLayers - 1], output, output.length);
        return output;
    }
    
    private void forwardLayer(int layer) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];
        
        // Linear transformation: z = Wx + b
        matrixOp.fillArray(zValues[layer], 0.0f, outputSize);
        
        // Matrix multiplication
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                zValues[layer][j] += activations[layer][i] * weights[layer][i][j];
            }
        }
        
        // Add bias
        matrixOp.add(zValues[layer], biases[layer], zValues[layer], outputSize);
        
        // Batch normalization (if enabled and training)
        if (useBatchNorm && isTraining) {
            applyBatchNormalization(layer);
        }
        
        // Apply activation function
        applyActivation(zValues[layer], activations[layer + 1], outputSize, 
                       activationFunctions[layer]);
        
        // Apply dropout (if enabled and training)
        if (useDropout && isTraining && layer < numLayers - 2) { // Not on output layer
            applyDropout(layer);
        }
    }
    
    private void applyBatchNormalization(int layer) {
        int size = layerSizes[layer + 1];
        
        // For simplicity, we'll use a basic batch norm implementation
        // In practice, this would be computed across the batch dimension
        
        // Normalize: (x - mean) / sqrt(var + epsilon)
        float epsilon = 1e-8f;
        
        for (int i = 0; i < size; i++) {
            float normalized = (zValues[layer][i] - bnMean[layer][i]) / 
                              (float) Math.sqrt(bnVar[layer][i] + epsilon);
            
            // Scale and shift: gamma * normalized + beta
            zValues[layer][i] = bnGamma[layer][i] * normalized + bnBeta[layer][i];
        }
    }
    
    private void applyDropout(int layer) {
        int size = layerSizes[layer + 1];
        float scale = 1.0f / (1.0f - dropoutRate);
        
        for (int i = 0; i < size; i++) {
            if (Math.random() < dropoutRate) {
                dropoutMasks[layer][i] = 0.0f;
                activations[layer + 1][i] = 0.0f;
            } else {
                dropoutMasks[layer][i] = scale;
                activations[layer + 1][i] *= scale;
            }
        }
    }
    
    private void applyActivation(float[] input, float[] output, int size, String activationType) {
        switch (activationType.toLowerCase()) {
            case "sigmoid":
                matrixOp.sigmoid(input, output, size);
                break;
            case "tanh":
                matrixOp.tanh(input, output, size);
                break;
            case "relu":
                matrixOp.relu(input, output, size);
                break;
            case "leaky_relu":
                applyLeakyRelu(input, output, size, 0.01f);
                break;
            case "softmax":
                matrixOp.softmax(input, output, size);
                break;
            case "linear":
            default:
                matrixOp.copyArray(input, output, size);
                break;
        }
    }
    
    private void applyLeakyRelu(float[] input, float[] output, int size, float alpha) {
        for (int i = 0; i < size; i++) {
            output[i] = input[i] > 0 ? input[i] : alpha * input[i];
        }
    }
    
    /**
     * Train the network using backpropagation with advanced optimizers
     */
    public void train(float[][] inputs, float[][] targets, int epochs, LossFunction lossFunction) {
        if (inputs.length != targets.length) {
            throw new IllegalArgumentException("Input and target batch sizes don't match");
        }
        
        isTraining = true;
        int numSamples = inputs.length;
        
        logger.info("Training advanced neural network for " + epochs + " epochs with " + 
                   numSamples + " samples");
        
        for (int epoch = 0; epoch < epochs; epoch++) {
            float totalLoss = 0.0f;
            
            // Shuffle training data
            int[] indices = shuffleIndices(numSamples);
            
            // Process mini-batches
            for (int batchStart = 0; batchStart < numSamples; batchStart += batchSize) {
                int currentBatchSize = Math.min(batchSize, numSamples - batchStart);
                
                // Extract mini-batch
                float[][] batchInputs = new float[currentBatchSize][];
                float[][] batchTargets = new float[currentBatchSize][];
                
                for (int i = 0; i < currentBatchSize; i++) {
                    batchInputs[i] = inputs[indices[batchStart + i]];
                    batchTargets[i] = targets[indices[batchStart + i]];
                }
                
                // Train on mini-batch
                float batchLoss = trainBatch(batchInputs, batchTargets, lossFunction);
                totalLoss += batchLoss;
            }
            
            // Log progress
            if (epoch % 10 == 0 || epoch == epochs - 1) {
                float avgLoss = totalLoss / (numSamples / batchSize);
                logger.debug("Epoch " + epoch + "/" + epochs + ", Loss: " + 
                           String.format("%.6f", avgLoss));
            }
        }
        
        isTraining = false;
        logger.info("Training completed");
    }
    
    private float trainBatch(float[][] inputs, float[][] targets, LossFunction lossFunction) {
        float batchLoss = 0.0f;
        int batchSize = inputs.length;
        
        // Accumulate gradients
        float[][][] weightGradients = initializeWeightGradients();
        float[][] biasGradients = initializeBiasGradients();
        
        for (int sample = 0; sample < batchSize; sample++) {
            // Forward pass
            float[] prediction = forward(inputs[sample]);
            
            // Calculate loss
            float sampleLoss = calculateLoss(prediction, targets[sample], lossFunction);
            batchLoss += sampleLoss;
            
            // Backward pass
            backpropagate(targets[sample], weightGradients, biasGradients, lossFunction);
        }
        
        // Average gradients
        averageGradients(weightGradients, biasGradients, batchSize);
        
        // Update weights using optimizer
        updateWeights(weightGradients, biasGradients);
        
        return batchLoss / batchSize;
    }
    
    private void backpropagate(float[] target, float[][][] weightGradients, 
                              float[][] biasGradients, LossFunction lossFunction) {
        int outputLayer = numLayers - 1;
        int outputSize = layerSizes[outputLayer];
        
        // Calculate output layer delta
        float[] delta = new float[outputSize];
        calculateOutputDelta(activations[outputLayer], target, delta, lossFunction);
        
        // Backward pass through all layers
        for (int layer = numLayers - 2; layer >= 0; layer--) {
            // Calculate gradients for current layer
            calculateLayerGradients(layer, delta, weightGradients[layer], biasGradients[layer]);
            
            // Calculate delta for previous layer (if not input layer)
            if (layer > 0) {
                float[] prevDelta = new float[layerSizes[layer]];
                calculateHiddenDelta(layer, delta, prevDelta);
                delta = prevDelta;
            }
        }
    }
    
    private void calculateOutputDelta(float[] output, float[] target, float[] delta, 
                                     LossFunction lossFunction) {
        int size = output.length;
        
        switch (lossFunction) {
            case MEAN_SQUARED_ERROR:
                for (int i = 0; i < size; i++) {
                    delta[i] = 2.0f * (output[i] - target[i]) / size;
                }
                break;
            case CROSS_ENTROPY:
            case BINARY_CROSS_ENTROPY:
                for (int i = 0; i < size; i++) {
                    delta[i] = output[i] - target[i];
                }
                break;
            default:
                throw new IllegalArgumentException("Unsupported loss function: " + lossFunction);
        }
    }
    
    private void calculateLayerGradients(int layer, float[] delta, float[][] weightGrads, 
                                        float[] biasGrads) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];
        
        // Weight gradients
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                weightGrads[i][j] += activations[layer][i] * delta[j];
            }
        }
        
        // Bias gradients
        for (int j = 0; j < outputSize; j++) {
            biasGrads[j] += delta[j];
        }
        
        // Add L2 regularization to weight gradients
        if (weightDecay > 0) {
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weightGrads[i][j] += weightDecay * weights[layer][i][j];
                }
            }
        }
    }
    
    private void calculateHiddenDelta(int layer, float[] nextDelta, float[] delta) {
        int inputSize = layerSizes[layer];
        int outputSize = layerSizes[layer + 1];
        
        // Initialize delta
        matrixOp.fillArray(delta, 0.0f, inputSize);
        
        // Backpropagate through weights
        for (int i = 0; i < inputSize; i++) {
            for (int j = 0; j < outputSize; j++) {
                delta[i] += weights[layer][i][j] * nextDelta[j];
            }
        }
        
        // Apply derivative of activation function
        applyActivationDerivative(delta, activations[layer], inputSize, 
                                 layer > 0 ? activationFunctions[layer - 1] : "linear");
        
        // Apply dropout mask if applicable
        if (useDropout && layer > 0) {
            for (int i = 0; i < inputSize; i++) {
                delta[i] *= dropoutMasks[layer - 1][i];
            }
        }
    }
    
    private void applyActivationDerivative(float[] delta, float[] activation, int size, 
                                          String activationType) {
        switch (activationType.toLowerCase()) {
            case "sigmoid":
                for (int i = 0; i < size; i++) {
                    delta[i] *= activation[i] * (1.0f - activation[i]);
                }
                break;
            case "tanh":
                for (int i = 0; i < size; i++) {
                    delta[i] *= (1.0f - activation[i] * activation[i]);
                }
                break;
            case "relu":
                for (int i = 0; i < size; i++) {
                    delta[i] *= activation[i] > 0 ? 1.0f : 0.0f;
                }
                break;
            case "leaky_relu":
                for (int i = 0; i < size; i++) {
                    delta[i] *= activation[i] > 0 ? 1.0f : 0.01f;
                }
                break;
            case "linear":
            default:
                // Linear derivative is 1, no change needed
                break;
        }
    }
    
    private void updateWeights(float[][][] weightGradients, float[][] biasGradients) {
        switch (optimizerType) {
            case SGD:
                updateWeightsSGD(weightGradients, biasGradients);
                break;
            case MOMENTUM:
                updateWeightsMomentum(weightGradients, biasGradients);
                break;
            case ADAM:
                updateWeightsAdam(weightGradients, biasGradients);
                break;
            default:
                throw new IllegalArgumentException("Unsupported optimizer: " + optimizerType);
        }
    }
    
    private void updateWeightsSGD(float[][][] weightGradients, float[][] biasGradients) {
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            
            // Update weights
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weights[layer][i][j] -= learningRate * weightGradients[layer][i][j];
                }
            }
            
            // Update biases
            for (int j = 0; j < outputSize; j++) {
                biases[layer][j] -= learningRate * biasGradients[layer][j];
            }
        }
    }
    
    private void updateWeightsMomentum(float[][][] weightGradients, float[][] biasGradients) {
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            
            // Update weight velocities and weights
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    velocityWeights[layer][i][j] = momentum * velocityWeights[layer][i][j] + 
                                                  learningRate * weightGradients[layer][i][j];
                    weights[layer][i][j] -= velocityWeights[layer][i][j];
                }
            }
            
            // Update bias velocities and biases
            for (int j = 0; j < outputSize; j++) {
                velocityBiases[layer][j] = momentum * velocityBiases[layer][j] + 
                                         learningRate * biasGradients[layer][j];
                biases[layer][j] -= velocityBiases[layer][j];
            }
        }
    }
    
    private void updateWeightsAdam(float[][][] weightGradients, float[][] biasGradients) {
        adamStep++;
        float lr = learningRate * (float) Math.sqrt(1 - Math.pow(beta2, adamStep)) / 
                   (1 - (float) Math.pow(beta1, adamStep));
        
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            
            // Update weight moments and weights
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    float grad = weightGradients[layer][i][j];
                    
                    // Update first moment
                    velocityWeights[layer][i][j] = beta1 * velocityWeights[layer][i][j] + 
                                                  (1 - beta1) * grad;
                    
                    // Update second moment
                    momentWeights[layer][i][j] = beta2 * momentWeights[layer][i][j] + 
                                               (1 - beta2) * grad * grad;
                    
                    // Update weights
                    float update = lr * velocityWeights[layer][i][j] / 
                                  (float) (Math.sqrt(momentWeights[layer][i][j]) + 1e-8);
                    weights[layer][i][j] -= update;
                }
            }
            
            // Update bias moments and biases
            for (int j = 0; j < outputSize; j++) {
                float grad = biasGradients[layer][j];
                
                velocityBiases[layer][j] = beta1 * velocityBiases[layer][j] + (1 - beta1) * grad;
                momentBiases[layer][j] = beta2 * momentBiases[layer][j] + (1 - beta2) * grad * grad;
                
                float update = lr * velocityBiases[layer][j] / 
                              (float) (Math.sqrt(momentBiases[layer][j]) + 1e-8);
                biases[layer][j] -= update;
            }
        }
    }
    
    private float calculateLoss(float[] prediction, float[] target, LossFunction lossFunction) {
        int size = prediction.length;
        float loss = 0.0f;
        
        switch (lossFunction) {
            case MEAN_SQUARED_ERROR:
                for (int i = 0; i < size; i++) {
                    float diff = prediction[i] - target[i];
                    loss += diff * diff;
                }
                return loss / size;
                
            case CROSS_ENTROPY:
                for (int i = 0; i < size; i++) {
                    loss -= target[i] * (float) Math.log(Math.max(prediction[i], 1e-15));
                }
                return loss;
                
            case BINARY_CROSS_ENTROPY:
                for (int i = 0; i < size; i++) {
                    loss -= target[i] * Math.log(Math.max(prediction[i], 1e-15)) + 
                           (1 - target[i]) * Math.log(Math.max(1 - prediction[i], 1e-15));
                }
                return loss / size;
                
            default:
                throw new IllegalArgumentException("Unsupported loss function: " + lossFunction);
        }
    }
    
    // Helper methods
    
    private float[][][] initializeWeightGradients() {
        float[][][] gradients = new float[numLayers - 1][][];
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            gradients[layer] = new float[inputSize][outputSize];
        }
        return gradients;
    }
    
    private float[][] initializeBiasGradients() {
        float[][] gradients = new float[numLayers - 1][];
        for (int layer = 0; layer < numLayers - 1; layer++) {
            gradients[layer] = new float[layerSizes[layer + 1]];
        }
        return gradients;
    }
    
    private void averageGradients(float[][][] weightGradients, float[][] biasGradients, 
                                 int batchSize) {
        for (int layer = 0; layer < numLayers - 1; layer++) {
            int inputSize = layerSizes[layer];
            int outputSize = layerSizes[layer + 1];
            
            for (int i = 0; i < inputSize; i++) {
                for (int j = 0; j < outputSize; j++) {
                    weightGradients[layer][i][j] /= batchSize;
                }
            }
            
            for (int j = 0; j < outputSize; j++) {
                biasGradients[layer][j] /= batchSize;
            }
        }
    }
    
    private int[] shuffleIndices(int size) {
        int[] indices = new int[size];
        for (int i = 0; i < size; i++) {
            indices[i] = i;
        }
        
        // Fisher-Yates shuffle
        for (int i = size - 1; i > 0; i--) {
            int j = (int) (Math.random() * (i + 1));
            int temp = indices[i];
            indices[i] = indices[j];
            indices[j] = temp;
        }
        
        return indices;
    }
    
    public int getTotalParameters() {
        int total = 0;
        for (int layer = 0; layer < numLayers - 1; layer++) {
            total += layerSizes[layer] * layerSizes[layer + 1]; // weights
            total += layerSizes[layer + 1]; // biases
            
            if (useBatchNorm) {
                total += 2 * layerSizes[layer + 1]; // gamma and beta
            }
        }
        return total;
    }
    
    public void setTrainingMode(boolean training) {
        this.isTraining = training;
    }
    
    public boolean isTraining() {
        return isTraining;
    }
    
    public void cleanup() {
        if (matrixOp != null) {
            matrixOp.release();
        }
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
        logger.debug("Cleaned up advanced neural network resources");
    }
    
    /**
     * Configuration class for network hyperparameters
     */
    public static class NetworkConfiguration {
        public int batchSize = 32;
        public boolean useBatchNorm = false;
        public boolean useDropout = false;
        public float dropoutRate = 0.5f;
        public OptimizerType optimizerType = OptimizerType.ADAM;
        public float learningRate = 0.001f;
        public float momentum = 0.9f;
        public float beta1 = 0.9f;
        public float beta2 = 0.999f;
        public float weightDecay = 0.0f;
        
        public NetworkConfiguration() {}
        
        public NetworkConfiguration setBatchSize(int batchSize) {
            this.batchSize = batchSize;
            return this;
        }
        
        public NetworkConfiguration useBatchNorm(boolean use) {
            this.useBatchNorm = use;
            return this;
        }
        
        public NetworkConfiguration useDropout(boolean use, float rate) {
            this.useDropout = use;
            this.dropoutRate = rate;
            return this;
        }
        
        public NetworkConfiguration setOptimizer(OptimizerType type, float lr) {
            this.optimizerType = type;
            this.learningRate = lr;
            return this;
        }
        
        public NetworkConfiguration setRegularization(float weightDecay) {
            this.weightDecay = weightDecay;
            return this;
        }
    }
}
