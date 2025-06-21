package org.apache.opennlp.gpu.test;

/**
 * Test to check what ML algorithms are available in OpenNLP 2.5.4
 */
public class MLAlgorithmChecker {
    public static void main(String[] args) {
        System.out.println("=== OpenNLP 2.5.4 ML Algorithms Check ===");
        
        // Check MaxEnt
        try {
            Class<?> maxentClass = Class.forName("opennlp.tools.ml.maxent.GISModel");
            System.out.println("✅ MaxEnt (GISModel): Available - " + maxentClass.getName());
        } catch (ClassNotFoundException e) {
            System.out.println("❌ MaxEnt (GISModel): Not available");
        }
        
        // Check Perceptron
        try {
            Class<?> perceptronClass = Class.forName("opennlp.tools.ml.perceptron.PerceptronModel");
            System.out.println("✅ Perceptron: Available - " + perceptronClass.getName());
        } catch (ClassNotFoundException e) {
            System.out.println("❌ Perceptron: Not available");
        }
        
        // Check Naive Bayes
        try {
            Class<?> naiveBayesClass = Class.forName("opennlp.tools.ml.naivebayes.NaiveBayesModel");
            System.out.println("✅ Naive Bayes: Available - " + naiveBayesClass.getName());
        } catch (ClassNotFoundException e) {
            System.out.println("❌ Naive Bayes: Not available");
        }
        
        // Check general ML model interface
        try {
            Class<?> maxentModelClass = Class.forName("opennlp.tools.ml.model.MaxentModel");
            System.out.println("✅ MaxentModel Interface: Available - " + maxentModelClass.getName());
            
            // Check available methods
            java.lang.reflect.Method[] methods = maxentModelClass.getMethods();
            System.out.println("Available methods in MaxentModel:");
            for (java.lang.reflect.Method method : methods) {
                if (method.getDeclaringClass().equals(maxentModelClass)) {
                    System.out.println("  - " + method.getName() + "()");
                }
            }
        } catch (ClassNotFoundException e) {
            System.out.println("❌ MaxentModel Interface: Not available");
        }
        
        // Check if context-based models are available
        try {
            Class<?> contextClass = Class.forName("opennlp.tools.ml.model.Context");
            System.out.println("✅ Context class: Available - " + contextClass.getName());
        } catch (ClassNotFoundException e) {
            System.out.println("❌ Context class: Not available");
        }
        
        System.out.println("\n=== Summary ===");
        System.out.println("OpenNLP 2.5.4 primarily uses MaxEnt models for most NLP tasks.");
        System.out.println("The MaxentModel interface is the main abstraction for ML models.");
    }
}
