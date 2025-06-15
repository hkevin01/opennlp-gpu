package org.apache.opennlp.gpu.util;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * Utility class for loading OpenNLP test data and models
 * Provides easy access to real NLP datasets for testing GPU acceleration
 */
public class TestDataLoader {
    
    private static final GpuLogger logger = GpuLogger.getLogger(TestDataLoader.class);
    
    // OpenNLP test data URLs
    private static final String OPENNLP_BASE = "https://raw.githubusercontent.com/apache/opennlp/main/opennlp-tools/src/test/resources/opennlp/tools/";
    
    private static final String[] TEST_DATA_URLS = {
        OPENNLP_BASE + "sentdetect/Sentences.txt",
        OPENNLP_BASE + "tokenize/token.train",
        OPENNLP_BASE + "postag/AnnotatedSentences.txt",
        OPENNLP_BASE + "namefind/AnnotatedSentencesWithNames.txt"
    };
    
    private static final Path CACHE_DIR = Paths.get("target/test-data-cache");
    
    /**
     * Load sentences for sentence detection testing
     */
    public static List<String> loadSentences() {
        return loadDataFromUrl(OPENNLP_BASE + "sentdetect/Sentences.txt", "sentences");
    }
    
    /**
     * Load tokenization test data
     */
    public static List<String> loadTokenizationData() {
        return loadDataFromUrl(OPENNLP_BASE + "tokenize/token.train", "tokenization");
    }
    
    /**
     * Load POS tagging test data
     */
    public static List<String> loadPosTaggingData() {
        return loadDataFromUrl(OPENNLP_BASE + "postag/AnnotatedSentences.txt", "pos-tagging");
    }
    
    /**
     * Load named entity recognition test data
     */
    public static List<String> loadNerData() {
        return loadDataFromUrl(OPENNLP_BASE + "namefind/AnnotatedSentencesWithNames.txt", "ner");
    }
    
    /**
     * Load any dataset by name
     */
    public static List<String> loadDataset(String datasetName) {
        switch (datasetName.toLowerCase()) {
            case "sentences":
                return loadSentences();
            case "tokenization":
                return loadTokenizationData();
            case "pos-tagging":
            case "pos":
                return loadPosTaggingData();
            case "ner":
            case "namefind":
                return loadNerData();
            default:
                logger.warn("Unknown dataset: " + datasetName);
                return generateFallbackData(datasetName);
        }
    }
    
    /**
     * Load large dataset for performance testing
     */
    public static List<String> loadLargeDataset(int size) {
        logger.info("Loading large dataset with " + size + " documents");
        
        List<String> data = new ArrayList<String>();
        
        // Try to load from multiple sources
        data.addAll(loadSentences());
        data.addAll(loadTokenizationData());
        data.addAll(loadPosTaggingData());
        
        // If we need more data, generate synthetic data
        while (data.size() < size) {
            data.addAll(generateSyntheticNLPData(Math.min(100, size - data.size())));
        }
        
        // Return requested size
        return data.subList(0, Math.min(size, data.size()));
    }
    
    /**
     * Create performance test datasets of varying sizes
     */
    public static List<List<String>> createPerformanceTestSets() {
        List<List<String>> testSets = new ArrayList<List<String>>();
        
        int[] sizes = {10, 50, 100, 500, 1000, 2000};
        
        for (int size : sizes) {
            List<String> testSet = loadLargeDataset(size);
            testSets.add(testSet);
            logger.info("Created test set with " + testSet.size() + " documents");
        }
        
        return testSets;
    }
    
    private static List<String> loadDataFromUrl(String url, String datasetType) {
        List<String> data = new ArrayList<String>();
        
        try {
            // Try to load from cache first
            Path cacheFile = CACHE_DIR.resolve(datasetType + ".txt");
            if (Files.exists(cacheFile)) {
                logger.info("Loading " + datasetType + " from cache");
                return Files.readAllLines(cacheFile);
            }
            
            // Download from URL
            logger.info("Downloading " + datasetType + " test data from: " + url);
            
            URL dataUrl = new URL(url);
            try (BufferedReader reader = new BufferedReader(new InputStreamReader(dataUrl.openStream()))) {
                String line;
                while ((line = reader.readLine()) != null) {
                    line = line.trim();
                    if (!line.isEmpty() && !line.startsWith("#")) {
                        // Clean the line for our purposes
                        String cleanLine = cleanTextLine(line);
                        if (!cleanLine.isEmpty()) {
                            data.add(cleanLine);
                        }
                    }
                }
            }
            
            // Cache the data
            Files.createDirectories(CACHE_DIR);
            Files.write(cacheFile, data);
            
            logger.info("Downloaded and cached " + data.size() + " " + datasetType + " entries");
            
        } catch (IOException e) {
            logger.warn("Failed to download " + datasetType + " data: " + e.getMessage());
            logger.info("Generating fallback data for " + datasetType);
            data = generateFallbackData(datasetType);
        }
        
        return data;
    }
    
    private static String cleanTextLine(String line) {
        // Remove OpenNLP training annotations but keep the text
        // Handle different annotation formats
        
        // For POS tagging data: "word_POS word_POS" -> "word word"
        line = line.replaceAll("_[A-Z$]+", "");
        
        // For NER data: "<START:PERSON> John <END> Smith" -> "John Smith"
        line = line.replaceAll("<START:[^>]+>", "");
        line = line.replaceAll("<END>", "");
        
        // For tokenization data, might have special markers
        line = line.replaceAll("<SPLIT>", " ");
        
        // Clean up extra whitespace
        line = line.replaceAll("\\s+", " ").trim();
        
        return line;
    }
    
    private static List<String> generateFallbackData(String datasetType) {
        logger.info("Generating fallback " + datasetType + " data");
        
        List<String> data = new ArrayList<String>();
        
        switch (datasetType) {
            case "sentences":
                data.addAll(generateSampleSentences());
                break;
            case "tokenization":
                data.addAll(generateTokenizationExamples());
                break;
            case "pos-tagging":
                data.addAll(generatePosTaggingExamples());
                break;
            case "ner":
                data.addAll(generateNerExamples());
                break;
            default:
                data.addAll(generateGenericNLPData());
        }
        
        return data;
    }
    
    private static List<String> generateSampleSentences() {
        List<String> sentences = new ArrayList<String>();
        
        String[] templates = {
            "The quick brown fox jumps over the lazy dog.",
            "Natural language processing enables computers to understand human language.",
            "Machine learning algorithms can be accelerated using GPU computing.",
            "OpenNLP provides a robust toolkit for processing natural language text.",
            "Feature extraction is a crucial step in text classification tasks."
        };
        
        for (String template : templates) {
            sentences.add(template);
        }
        
        // Generate variations
        String[] subjects = {"Scientists", "Researchers", "Engineers", "Developers"};
        String[] verbs = {"discovered", "developed", "implemented", "optimized"};
        String[] objects = {"algorithms", "systems", "models", "frameworks"};
        
        for (int i = 0; i < 20; i++) {
            String sentence = String.format("%s %s new %s for natural language processing.",
                subjects[i % subjects.length],
                verbs[i % verbs.length], 
                objects[i % objects.length]);
            sentences.add(sentence);
        }
        
        return sentences;
    }
    
    private static List<String> generateTokenizationExamples() {
        List<String> examples = new ArrayList<String>();
        
        examples.add("Hello, world! How are you today?");
        examples.add("Dr. Smith went to the U.S.A. yesterday.");
        examples.add("The temperature was 32.5°C at 3:30 p.m.");
        examples.add("Visit our website at http://example.com for more info.");
        examples.add("Email us at support@company.com or call 555-123-4567.");
        
        return examples;
    }
    
    private static List<String> generatePosTaggingExamples() {
        List<String> examples = new ArrayList<String>();
        
        examples.add("The cat sat on the mat");
        examples.add("John quickly ran to the store");
        examples.add("She is reading a very interesting book");
        examples.add("They will arrive tomorrow morning");
        examples.add("The algorithm efficiently processes large datasets");
        
        return examples;
    }
    
    private static List<String> generateNerExamples() {
        List<String> examples = new ArrayList<String>();
        
        examples.add("John Smith works at Microsoft in Seattle");
        examples.add("The meeting is scheduled for Monday in New York");
        examples.add("Apple Inc. was founded by Steve Jobs in California");
        examples.add("Google announced new features at the conference in San Francisco");
        examples.add("Amazon Web Services provides cloud computing from Virginia");
        
        return examples;
    }
    
    private static List<String> generateGenericNLPData() {
        List<String> data = new ArrayList<String>();
        
        data.addAll(generateSampleSentences());
        data.addAll(generateTokenizationExamples());
        data.addAll(generatePosTaggingExamples());
        data.addAll(generateNerExamples());
        
        return data;
    }
    
    private static List<String> generateSyntheticNLPData(int count) {
        List<String> data = new ArrayList<String>();
        
        String[] templates = {
            "The %s %s %s the %s %s in the %s.",
            "Machine learning %s can %s %s patterns in %s data.",
            "Natural language processing %s %s to understand %s text.",
            "GPU acceleration %s %s performance for %s computations.",
            "Deep learning %s %s %s representations from %s data."
        };
        
        String[] adjectives = {"advanced", "efficient", "powerful", "intelligent", "sophisticated"};
        String[] nouns = {"algorithm", "system", "model", "framework", "technology"};
        String[] verbs = {"processes", "analyzes", "transforms", "optimizes", "accelerates"};
        String[] contexts = {"large", "complex", "structured", "unstructured", "multilingual"};
        
        for (int i = 0; i < count; i++) {
            String template = templates[i % templates.length];
            String sentence = String.format(template,
                adjectives[i % adjectives.length],
                nouns[i % nouns.length],
                verbs[i % verbs.length],
                contexts[i % contexts.length],
                nouns[(i + 1) % nouns.length],
                contexts[(i + 1) % contexts.length]
            );
            data.add(sentence);
        }
        
        return data;
    }
    
    /**
     * Cleanup cached test data
     */
    public static void clearCache() {
        try {
            if (Files.exists(CACHE_DIR)) {
                Files.walk(CACHE_DIR)
                     .filter(Files::isRegularFile)
                     .forEach(file -> {
                         try {
                             Files.delete(file);
                         } catch (IOException e) {
                             logger.warn("Failed to delete cache file: " + file);
                         }
                     });
                Files.deleteIfExists(CACHE_DIR);
                logger.info("Cleared test data cache");
            }
        } catch (IOException e) {
            logger.warn("Failed to clear cache: " + e.getMessage());
        }
    }
}
