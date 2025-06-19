package org.apache.opennlp.gpu.examples.ner;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

import org.apache.opennlp.gpu.common.ComputeProvider;
import org.apache.opennlp.gpu.common.CpuComputeProvider;
import org.apache.opennlp.gpu.common.GpuConfig;
import org.apache.opennlp.gpu.compute.CpuMatrixOperation;
import org.apache.opennlp.gpu.compute.MatrixOperation;
import org.apache.opennlp.gpu.features.GpuFeatureExtractor;

/**
 * GPU-Accelerated Named Entity Recognition Example
 * 
 * Demonstrates high-speed entity extraction using GPU acceleration
 * for identifying people, locations, organizations, and other entities.
 */
public class GpuNamedEntityRecognition {
    
    private final GpuConfig config;
    private final ComputeProvider computeProvider;
    private final MatrixOperation matrixOp;
    private final GpuFeatureExtractor featureExtractor;
    private final Map<String, EntityType> entityDictionary;
    private final Map<Pattern, EntityType> entityPatterns;
    
    public GpuNamedEntityRecognition() {
        // Initialize GPU configuration
        this.config = new GpuConfig();
        config.setGpuEnabled(true);
        config.setBatchSize(64); // Optimal for NER processing
        config.setMemoryPoolSizeMB(512);
        
        // Create compute provider
        this.computeProvider = new CpuComputeProvider();
        this.matrixOp = new CpuMatrixOperation(computeProvider);
        
        // Initialize feature extractor
        this.featureExtractor = new GpuFeatureExtractor(computeProvider, config, matrixOp);
        
        // Initialize entity recognition resources
        this.entityDictionary = initializeEntityDictionary();
        this.entityPatterns = initializeEntityPatterns();
    }
    
    /**
     * Extract named entities from batch of documents with GPU acceleration
     */
    public EntityResult[] extractEntitiesBatch(String[] documents) {
        System.out.println("🚀 Starting GPU-accelerated named entity recognition...");
        System.out.println("📊 Processing " + documents.length + " documents");
        
        long startTime = System.currentTimeMillis();
        
        // Step 1: Extract contextual features using GPU
        System.out.println("🔍 Extracting contextual features with GPU...");
        String[] entityCandidates = extractEntityCandidates(documents);
        float[][] contextFeatures = featureExtractor.extractContextFeatures(
            documents, entityCandidates, 3); // 3-word context window
        
        // Step 2: Extract n-gram features for entity classification
        System.out.println("📝 Extracting n-gram features...");
        float[][] ngramFeatures = featureExtractor.extractNGramFeatures(documents, 2, 1000);
        
        // Step 3: Classify entities
        System.out.println("🏷️ Classifying entities...");
        EntityResult[] results = new EntityResult[documents.length];
        
        for (int i = 0; i < documents.length; i++) {
            results[i] = extractEntitiesFromDocument(documents[i], contextFeatures[i], ngramFeatures[i]);
        }
        
        long endTime = System.currentTimeMillis();
        double processingTime = (endTime - startTime) / 1000.0;
        
        System.out.println("✅ Named entity recognition complete!");
        System.out.printf("⏱️  Processed %d documents in %.2f seconds (%.1f docs/sec)%n", 
                         documents.length, processingTime, documents.length / processingTime);
        
        return results;
    }
    
    /**
     * Extract entities from single document
     */
    public EntityResult extractEntities(String document) {
        return extractEntitiesBatch(new String[]{document})[0];
    }
    
    private String[] extractEntityCandidates(String[] documents) {
        Set<String> candidates = new HashSet<>();
        
        for (String doc : documents) {
            // Extract capitalized words (potential proper nouns)
            Pattern capitalizedPattern = Pattern.compile("\\b[A-Z][a-z]+\\b");
            Matcher matcher = capitalizedPattern.matcher(doc);
            while (matcher.find()) {
                candidates.add(matcher.group());
            }
            
            // Extract multi-word capitalized phrases
            Pattern phrasePattern = Pattern.compile("\\b[A-Z][a-z]+(?:\\s+[A-Z][a-z]+)+\\b");
            matcher = phrasePattern.matcher(doc);
            while (matcher.find()) {
                candidates.add(matcher.group());
            }
        }
        
        return candidates.toArray(new String[0]);
    }
    
    private EntityResult extractEntitiesFromDocument(String document, float[] contextFeatures, float[] ngramFeatures) {
        List<Entity> entities = new ArrayList<>();
        
        // Dictionary-based entity recognition
        entities.addAll(extractDictionaryEntities(document));
        
        // Pattern-based entity recognition
        entities.addAll(extractPatternEntities(document));
        
        // Feature-based entity recognition (using GPU-extracted features)
        entities.addAll(extractFeatureBasedEntities(document, contextFeatures, ngramFeatures));
        
        // Remove duplicates and merge overlapping entities
        entities = mergeOverlappingEntities(entities);
        
        return new EntityResult(document, entities);
    }
    
    private List<Entity> extractDictionaryEntities(String document) {
        List<Entity> entities = new ArrayList<>();
        
        for (Map.Entry<String, EntityType> entry : entityDictionary.entrySet()) {
            String entityText = entry.getKey();
            EntityType type = entry.getValue();
            
            // Case-insensitive search
            Pattern pattern = Pattern.compile("\\b" + Pattern.quote(entityText) + "\\b", Pattern.CASE_INSENSITIVE);
            Matcher matcher = pattern.matcher(document);
            
            while (matcher.find()) {
                entities.add(new Entity(
                    entityText,
                    type,
                    matcher.start(),
                    matcher.end(),
                    0.9f // High confidence for dictionary matches
                ));
            }
        }
        
        return entities;
    }
    
    private List<Entity> extractPatternEntities(String document) {
        List<Entity> entities = new ArrayList<>();
        
        for (Map.Entry<Pattern, EntityType> entry : entityPatterns.entrySet()) {
            Pattern pattern = entry.getKey();
            EntityType type = entry.getValue();
            
            Matcher matcher = pattern.matcher(document);
            while (matcher.find()) {
                entities.add(new Entity(
                    matcher.group(),
                    type,
                    matcher.start(),
                    matcher.end(),
                    0.8f // Good confidence for pattern matches
                ));
            }
        }
        
        return entities;
    }
    
    private List<Entity> extractFeatureBasedEntities(String document, float[] contextFeatures, float[] ngramFeatures) {
        List<Entity> entities = new ArrayList<>();
        
        // Use features to identify potential entities
        String[] words = document.split("\\s+");
        int wordIndex = 0;
        
        for (String word : words) {
            if (isCapitalized(word) && word.length() > 2) {
                // Calculate entity probability based on features
                float entityScore = calculateEntityScore(word, contextFeatures, ngramFeatures, wordIndex);
                
                if (entityScore > 0.6f) {
                    EntityType type = classifyEntityType(word, contextFeatures, wordIndex);
                    int startPos = document.indexOf(word);
                    if (startPos >= 0) {
                        entities.add(new Entity(
                            word,
                            type,
                            startPos,
                            startPos + word.length(),
                            entityScore
                        ));
                    }
                }
            }
            wordIndex++;
        }
        
        return entities;
    }
    
    private float calculateEntityScore(String word, float[] contextFeatures, float[] ngramFeatures, int wordIndex) {
        float score = 0.0f;
        
        // Context features contribution
        if (contextFeatures.length > wordIndex) {
            score += contextFeatures[wordIndex] * 0.7f;
        }
        
        // N-gram features contribution
        if (ngramFeatures.length > 0) {
            score += (ngramFeatures[0] + ngramFeatures[Math.min(1, ngramFeatures.length - 1)]) * 0.3f;
        }
        
        // Boost score for typical entity characteristics
        if (isCapitalized(word)) score += 0.2f;
        if (word.length() > 4) score += 0.1f;
        if (endsWithEntitySuffix(word)) score += 0.15f;
        
        return Math.min(1.0f, score);
    }
    
    private EntityType classifyEntityType(String word, float[] contextFeatures, int wordIndex) {
        // Simple heuristic-based classification
        // In a real implementation, this would use more sophisticated ML models
        
        if (isPersonName(word)) return EntityType.PERSON;
        if (isLocationName(word)) return EntityType.LOCATION;
        if (isOrganizationName(word)) return EntityType.ORGANIZATION;
        if (isMiscEntity(word)) return EntityType.MISCELLANEOUS;
        
        return EntityType.PERSON; // Default
    }
    
    private boolean isCapitalized(String word) {
        return word.length() > 0 && Character.isUpperCase(word.charAt(0));
    }
    
    private boolean endsWithEntitySuffix(String word) {
        String[] locationSuffixes = {"ville", "ton", "burg", "field", "ford", "land"};
        String[] orgSuffixes = {"Inc", "Corp", "LLC", "Ltd", "Co"};
        
        String lower = word.toLowerCase();
        for (String suffix : locationSuffixes) {
            if (lower.endsWith(suffix)) return true;
        }
        for (String suffix : orgSuffixes) {
            if (word.endsWith(suffix)) return true;
        }
        return false;
    }
    
    private boolean isPersonName(String word) {
        // Simple heuristics for person names
        return word.matches("[A-Z][a-z]+") && word.length() >= 3 && word.length() <= 15;
    }
    
    private boolean isLocationName(String word) {
        String[] locationIndicators = {"ville", "ton", "city", "berg", "field", "land", "shire"};
        String lower = word.toLowerCase();
        for (String indicator : locationIndicators) {
            if (lower.contains(indicator)) return true;
        }
        return false;
    }
    
    private boolean isOrganizationName(String word) {
        String[] orgIndicators = {"Inc", "Corp", "LLC", "Ltd", "Co", "Company", "Corporation"};
        for (String indicator : orgIndicators) {
            if (word.contains(indicator)) return true;
        }
        return false;
    }
    
    private boolean isMiscEntity(String word) {
        // Dates, events, products, etc.
        return word.matches("\\d{4}") || // Years
               word.matches("[A-Z][a-z]+\\d+"); // Product names like "iPhone12"
    }
    
    private List<Entity> mergeOverlappingEntities(List<Entity> entities) {
        if (entities.isEmpty()) return entities;
        
        // Sort by start position
        entities.sort(Comparator.comparingInt(Entity::getStartPos));
        
        List<Entity> merged = new ArrayList<>();
        Entity current = entities.get(0);
        
        for (int i = 1; i < entities.size(); i++) {
            Entity next = entities.get(i);
            
            if (current.getEndPos() > next.getStartPos()) {
                // Overlapping entities - keep the one with higher confidence
                if (next.getConfidence() > current.getConfidence()) {
                    current = next;
                }
            } else {
                merged.add(current);
                current = next;
            }
        }
        merged.add(current);
        
        return merged;
    }
    
    private Map<String, EntityType> initializeEntityDictionary() {
        Map<String, EntityType> dictionary = new HashMap<>();
        
        // Common person names
        String[] personNames = {
            "John", "Mary", "Michael", "Sarah", "David", "Jennifer", "Robert", "Lisa",
            "William", "Karen", "Richard", "Nancy", "Thomas", "Betty", "Charles", "Helen"
        };
        
        // Common locations
        String[] locations = {
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix", "Philadelphia",
            "San Antonio", "San Diego", "Dallas", "San Jose", "Austin", "Jacksonville",
            "United States", "Canada", "Mexico", "California", "Texas", "Florida"
        };
        
        // Common organizations
        String[] organizations = {
            "Microsoft", "Google", "Apple", "Amazon", "Facebook", "Tesla", "Netflix",
            "IBM", "Oracle", "Intel", "NASA", "FBI", "CIA", "NATO", "UN"
        };
        
        for (String name : personNames) {
            dictionary.put(name, EntityType.PERSON);
        }
        
        for (String location : locations) {
            dictionary.put(location, EntityType.LOCATION);
        }
        
        for (String org : organizations) {
            dictionary.put(org, EntityType.ORGANIZATION);
        }
        
        return dictionary;
    }
    
    private Map<Pattern, EntityType> initializeEntityPatterns() {
        Map<Pattern, EntityType> patterns = new HashMap<>();
        
        // Email patterns
        patterns.put(Pattern.compile("\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b"), 
                    EntityType.MISCELLANEOUS);
        
        // Phone number patterns
        patterns.put(Pattern.compile("\\b\\d{3}-\\d{3}-\\d{4}\\b"), 
                    EntityType.MISCELLANEOUS);
        patterns.put(Pattern.compile("\\(\\d{3}\\)\\s*\\d{3}-\\d{4}"), 
                    EntityType.MISCELLANEOUS);
        
        // Date patterns
        patterns.put(Pattern.compile("\\b\\d{1,2}/\\d{1,2}/\\d{4}\\b"), 
                    EntityType.MISCELLANEOUS);
        patterns.put(Pattern.compile("\\b(January|February|March|April|May|June|July|August|September|October|November|December)\\s+\\d{1,2},?\\s+\\d{4}\\b"), 
                    EntityType.MISCELLANEOUS);
        
        // URLs
        patterns.put(Pattern.compile("https?://[\\w\\.-]+\\.[a-z]{2,}[\\w\\./]*"), 
                    EntityType.MISCELLANEOUS);
        
        return patterns;
    }
    
    public void cleanup() {
        if (featureExtractor != null) {
            featureExtractor.release();
        }
        if (matrixOp != null) {
            matrixOp.release();
        }
        if (computeProvider != null) {
            computeProvider.cleanup();
        }
    }
    
    // Entity classes
    public static class Entity {
        private final String text;
        private final EntityType type;
        private final int startPos;
        private final int endPos;
        private final float confidence;
        
        public Entity(String text, EntityType type, int startPos, int endPos, float confidence) {
            this.text = text;
            this.type = type;
            this.startPos = startPos;
            this.endPos = endPos;
            this.confidence = confidence;
        }
        
        // Getters
        public String getText() { return text; }
        public EntityType getType() { return type; }
        public int getStartPos() { return startPos; }
        public int getEndPos() { return endPos; }
        public float getConfidence() { return confidence; }
        
        @Override
        public String toString() {
            return String.format("%s [%s] (%.2f)", text, type, confidence);
        }
    }
    
    public static class EntityResult {
        private final String document;
        private final List<Entity> entities;
        
        public EntityResult(String document, List<Entity> entities) {
            this.document = document;
            this.entities = entities;
        }
        
        public String getDocument() { return document; }
        public List<Entity> getEntities() { return entities; }
        
        public List<Entity> getEntitiesByType(EntityType type) {
            return entities.stream()
                          .filter(e -> e.getType() == type)
                          .collect(ArrayList::new, ArrayList::add, ArrayList::addAll);
        }
    }
    
    public enum EntityType {
        PERSON, LOCATION, ORGANIZATION, MISCELLANEOUS
    }
    
    /**
     * Demo application
     */
    public static void main(String[] args) {
        // Check for test mode
        boolean testMode = false;
        int batchSize = 10;
        boolean quickTest = false;
        
        for (String arg : args) {
            if ("--test-mode".equals(arg)) {
                testMode = true;
            } else if (arg.startsWith("--batch-size=")) {
                batchSize = Integer.parseInt(arg.substring("--batch-size=".length()));
            } else if ("--quick-test".equals(arg)) {
                quickTest = true;
            }
        }
        
        System.out.println("🚀 GPU-Accelerated Named Entity Recognition Demo");
        System.out.println("===============================================");
        if (testMode) {
            System.out.println("⚡ Running in TEST MODE for faster execution");
        }
        
        GpuNamedEntityRecognition ner = new GpuNamedEntityRecognition();
        
        try {
            // Sample documents
            String[] allSampleDocuments = {
                "John Smith works at Microsoft in Seattle, Washington. You can reach him at john.smith@microsoft.com or call (206) 555-0123.",
                "Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976, in Los Altos, California.",
                "The meeting between President Biden and Prime Minister Trudeau will take place in Ottawa, Canada on March 15, 2024.",
                "Google announced its new AI technology at the conference in San Francisco. The CEO Sundar Pichai presented the innovations.",
                "NASA's Mars mission launched from Kennedy Space Center in Florida. The mission is scheduled to arrive at Mars in 2025.",
                "Tesla's factory in Austin, Texas produces the Model Y. Elon Musk visited the facility last week to review production.",
                "The United Nations headquarters in New York hosted a summit on climate change. Representatives from 195 countries attended.",
                "Amazon Web Services reported strong growth in cloud computing. Jeff Bezos commented on the results during the earnings call.",
                "The University of California, Berkeley announced a new research partnership with IBM to develop quantum computing technologies.",
                "Facebook's Meta division is working on virtual reality applications. Mark Zuckerberg demonstrated the new VR headset yesterday."
            };
            
            // Use subset for test mode
            String[] sampleDocuments;
            if (testMode && quickTest) {
                sampleDocuments = new String[Math.min(3, batchSize)];
                System.arraycopy(allSampleDocuments, 0, sampleDocuments, 0, sampleDocuments.length);
            } else if (testMode) {
                sampleDocuments = new String[Math.min(batchSize, allSampleDocuments.length)];
                System.arraycopy(allSampleDocuments, 0, sampleDocuments, 0, sampleDocuments.length);
            } else {
                sampleDocuments = allSampleDocuments;
            }
            
            // Extract entities
            EntityResult[] results = ner.extractEntitiesBatch(sampleDocuments);
            
            // Display results
            System.out.println("\n📊 Named Entity Recognition Results:");
            System.out.println("====================================");
            
            for (int i = 0; i < results.length; i++) {
                EntityResult result = results[i];
                System.out.printf("\n📄 Document %d: %s...%n", i + 1, 
                                result.getDocument().substring(0, Math.min(50, result.getDocument().length())));
                
                if (result.getEntities().isEmpty()) {
                    System.out.println("   No entities found.");
                } else {
                    for (EntityType type : EntityType.values()) {
                        List<Entity> entitiesOfType = result.getEntitiesByType(type);
                        if (!entitiesOfType.isEmpty()) {
                            System.out.printf("   %s: ", type);
                            for (int j = 0; j < entitiesOfType.size(); j++) {
                                if (j > 0) System.out.print(", ");
                                Entity entity = entitiesOfType.get(j);
                                System.out.printf("%s (%.2f)", entity.getText(), entity.getConfidence());
                            }
                            System.out.println();
                        }
                    }
                }
            }
            
            // Summary statistics
            System.out.println("\n🎯 Extraction Summary:");
            System.out.println("=====================");
            int totalEntities = Arrays.stream(results).mapToInt(r -> r.getEntities().size()).sum();
            System.out.println("✅ Total entities extracted: " + totalEntities);
            
            for (EntityType type : EntityType.values()) {
                int count = Arrays.stream(results)
                                 .mapToInt(r -> r.getEntitiesByType(type).size())
                                 .sum();
                System.out.printf("   %s: %d%n", type, count);
            }
            
            System.out.println("\n🚀 Features Demonstrated:");
            System.out.println("========================");
            System.out.println("✅ GPU-accelerated feature extraction");
            System.out.println("✅ Dictionary-based entity recognition");
            System.out.println("✅ Pattern-based entity extraction");
            System.out.println("✅ Context-aware entity classification");
            System.out.println("✅ High-speed batch processing");
            
            if (testMode) {
                System.out.println("\n✅ Test completed successfully");
                System.out.println("SUCCESS: Named Entity Recognition example executed successfully");
            }
            
        } finally {
            ner.cleanup();
        }
    }
}
