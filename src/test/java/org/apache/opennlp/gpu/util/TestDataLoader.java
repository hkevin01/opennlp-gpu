package org.apache.opennlp.gpu.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.opennlp.gpu.common.GpuLogger;

/**
 * Utility class for loading and generating test data for GPU acceleration tests
 * Provides both real OpenNLP test data and synthetic data generation
 */
public class TestDataLoader {
    
    private static final GpuLogger logger = GpuLogger.getLogger(TestDataLoader.class);
    
    private static final Path TEST_DATA_DIR = Paths.get("target/test-data");
    private static final String CACHE_FILE = "test-data-cache.txt";
    
    // Ensure test data directory exists
    static {
        try {
            Files.createDirectories(TEST_DATA_DIR);
        } catch (IOException e) {
            logger.warn("Could not create test data directory: " + e.getMessage());
        }
    }
    
    /**
     * Load a large dataset for stress testing
     */
    public static List<String> loadLargeDataset(int size) {
        logger.info("Loading large dataset with {} documents", size);
        
        List<String> dataset = new ArrayList<>();
        
        // Try to load cached data first
        List<String> cachedData = loadCachedData(size);
        if (cachedData != null && cachedData.size() >= size) {
            return cachedData.subList(0, size);
        }
        
        // Generate new data
        dataset.addAll(generateRealisticNLPText(size));
        
        // Cache the generated data
        cacheData(dataset, size);
        
        return dataset;
    }
    
    /**
     * Create performance test sets with graduated dataset sizes
     */
    public static List<List<String>> createPerformanceTestSets() {
        int[] sizes = {10, 50, 100, 500, 1000, 2000};
        List<List<String>> testSets = new ArrayList<>();
        
        for (int size : sizes) {
            testSets.add(loadLargeDataset(size));
        }
        
        return testSets;
    }
    
    /**
     * Generate realistic NLP text with controlled characteristics
     */
    public static List<String> generateRealisticNLPText(int count) {
        List<String> texts = new ArrayList<>();
        
        // Text templates for different domains
        String[] newsTemplates = {
            "Breaking news: %s %s in %s today, affecting thousands of %s.",
            "Scientists at %s University discovered that %s can %s %s significantly.",
            "The %s government announced new %s policies to %s %s issues.",
            "Economic experts predict %s %s will %s by %s percent this year.",
            "Technology companies are investing heavily in %s %s development."
        };
        
        String[] academicTemplates = {
            "Recent studies in %s %s demonstrate that %s %s can improve %s.",
            "The %s methodology shows %s results in %s %s applications.",
            "Researchers analyzed %s %s data to understand %s %s patterns.",
            "This %s %s framework enables %s %s optimization techniques.",
            "Experimental %s %s validation confirms %s %s effectiveness."
        };
        
        String[] technicalTemplates = {
            "The %s %s algorithm processes %s %s data efficiently.",
            "GPU acceleration enables %s %s computation with %s performance.",
            "Machine learning models require %s %s feature extraction.",
            "Neural networks with %s %s architecture achieve %s accuracy.",
            "Parallel processing improves %s %s throughput significantly."
        };
        
        // Vocabulary sets
        String[] nouns = {
            "system", "algorithm", "network", "model", "data", "process", "method", "approach",
            "framework", "architecture", "implementation", "optimization", "analysis", "research",
            "technology", "application", "solution", "platform", "infrastructure", "computation"
        };
        
        String[] adjectives = {
            "advanced", "efficient", "robust", "scalable", "innovative", "comprehensive",
            "sophisticated", "optimized", "intelligent", "adaptive", "dynamic", "flexible",
            "powerful", "reliable", "effective", "modern", "cutting-edge", "state-of-the-art"
        };
        
        String[] verbs = {
            "improve", "optimize", "enhance", "accelerate", "process", "analyze", "implement",
            "develop", "create", "design", "execute", "perform", "compute", "calculate",
            "generate", "transform", "extract", "classify", "predict", "evaluate"
        };
        
        String[] technical = {
            "machine learning", "artificial intelligence", "deep learning", "neural network",
            "natural language", "computer vision", "data science", "big data", "cloud computing",
            "GPU acceleration", "parallel processing", "distributed computing", "high performance"
        };
        
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < count; i++) {
            String template;
            
            // Select template based on document type distribution
            int templateType = i % 3;
            switch (templateType) {
                case 0:
                    template = newsTemplates[i % newsTemplates.length];
                    break;
                case 1:
                    template = academicTemplates[i % academicTemplates.length];
                    break;
                default:
                    template = technicalTemplates[i % technicalTemplates.length];
                    break;
            }
            
            // Fill template with random vocabulary
            String text = String.format(template,
                adjectives[random.nextInt(adjectives.length)],
                nouns[random.nextInt(nouns.length)],
                adjectives[random.nextInt(adjectives.length)],
                nouns[random.nextInt(nouns.length)],
                verbs[random.nextInt(verbs.length)]
            );
            
            // Add some technical terms for variety
            if (random.nextInt(3) == 0) {
                text += " This involves " + technical[random.nextInt(technical.length)] + 
                       " techniques for " + adjectives[random.nextInt(adjectives.length)] + 
                       " " + nouns[random.nextInt(nouns.length)] + " processing.";
            }
            
            texts.add(text);
        }
        
        return texts;
    }
    
    /**
     * Generate test documents for specific NLP tasks
     */
    public static List<String> generateTaskSpecificText(String task, int count) {
        switch (task.toLowerCase()) {
            case "sentiment":
                return generateSentimentText(count);
            case "classification":
                return generateClassificationText(count);
            case "ner":
                return generateNERText(count);
            default:
                return generateRealisticNLPText(count);
        }
    }
    
    private static List<String> generateSentimentText(int count) {
        List<String> texts = new ArrayList<>();
        String[] positive = {"excellent", "amazing", "wonderful", "fantastic", "great"};
        String[] negative = {"terrible", "awful", "horrible", "disappointing", "poor"};
        String[] neutral = {"average", "standard", "typical", "normal", "regular"};
        
        for (int i = 0; i < count; i++) {
            String sentiment = i % 3 == 0 ? positive[i % positive.length] :
                              i % 3 == 1 ? negative[i % negative.length] :
                              neutral[i % neutral.length];
            
            texts.add("This product is " + sentiment + " and I would " + 
                     (i % 2 == 0 ? "recommend" : "not recommend") + " it to others.");
        }
        
        return texts;
    }
    
    private static List<String> generateClassificationText(int count) {
        List<String> texts = new ArrayList<>();
        String[] categories = {"technology", "sports", "politics", "entertainment", "science"};
        
        for (int i = 0; i < count; i++) {
            String category = categories[i % categories.length];
            texts.add("This is a " + category + " article about recent developments in " +
                     category + " that will interest " + category + " enthusiasts.");
        }
        
        return texts;
    }
    
    private static List<String> generateNERText(int count) {
        List<String> texts = new ArrayList<>();
        String[] names = {"John Smith", "Sarah Johnson", "Michael Brown", "Emily Davis"};
        String[] locations = {"New York", "London", "Tokyo", "Sydney", "Berlin"};
        String[] organizations = {"Google", "Microsoft", "Amazon", "Apple", "Tesla"};
        
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        for (int i = 0; i < count; i++) {
            String name = names[random.nextInt(names.length)];
            String location = locations[random.nextInt(locations.length)];
            String org = organizations[random.nextInt(organizations.length)];
            
            texts.add(name + " works at " + org + " in " + location + 
                     " and specializes in machine learning research.");
        }
        
        return texts;
    }
    
    /**
     * Load cached test data if available
     */
    private static List<String> loadCachedData(int size) {
        try {
            Path cacheFile = TEST_DATA_DIR.resolve(CACHE_FILE + "." + size);
            if (Files.exists(cacheFile)) {
                List<String> lines = Files.readAllLines(cacheFile);
                logger.debug("Loaded {} cached documents from {}", lines.size(), cacheFile);
                return lines;
            }
        } catch (IOException e) {
            logger.debug("Could not load cached data: " + e.getMessage());
        }
        
        return null;
    }
    
    /**
     * Cache generated test data for future use
     */
    private static void cacheData(List<String> data, int size) {
        try {
            Path cacheFile = TEST_DATA_DIR.resolve(CACHE_FILE + "." + size);
            Files.write(cacheFile, data);
            logger.debug("Cached {} documents to {}", data.size(), cacheFile);
        } catch (IOException e) {
            logger.debug("Could not cache data: " + e.getMessage());
        }
    }
    
    /**
     * Generate checksum for test data validation
     */
    public static String generateChecksum(List<String> data) {
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            for (String text : data) {
                md.update(text.getBytes());
            }
            byte[] hash = md.digest();
            StringBuilder hexString = new StringBuilder();
            for (byte b : hash) {
                String hex = Integer.toHexString(0xff & b);
                if (hex.length() == 1) {
                    hexString.append('0');
                }
                hexString.append(hex);
            }
            return hexString.toString().substring(0, 16); // First 16 chars
        } catch (NoSuchAlgorithmException e) {
            return "checksum-unavailable";
        }
    }
    
    /**
     * Validate test data quality
     */
    public static boolean validateDataQuality(List<String> data) {
        if (data == null || data.isEmpty()) {
            return false;
        }
        
        // Check for minimum text length
        int minLength = 10;
        int shortTexts = 0;
        int totalLength = 0;
        
        for (String text : data) {
            if (text == null || text.trim().length() < minLength) {
                shortTexts++;
            }
            totalLength += text != null ? text.length() : 0;
        }
        
        double avgLength = (double) totalLength / data.size();
        double shortTextRatio = (double) shortTexts / data.size();
        
        logger.debug("Data quality - Avg length: {}, Short texts: {}%", 
                    avgLength, shortTextRatio * 100);
        
        // Quality criteria
        return avgLength >= 20 && shortTextRatio < 0.1; // < 10% short texts
    }
    
    /**
     * Create test data with specific characteristics for performance testing
     */
    public static List<String> createPerformanceTestData(int count, int avgLength) {
        List<String> data = new ArrayList<>();
        ThreadLocalRandom random = ThreadLocalRandom.current();
        
        String[] words = {
            "the", "quick", "brown", "fox", "jumps", "over", "lazy", "dog", "and", "runs",
            "through", "forest", "with", "great", "speed", "while", "avoiding", "obstacles",
            "machine", "learning", "algorithm", "processes", "data", "efficiently", "using",
            "parallel", "computation", "gpu", "acceleration", "optimization", "performance"
        };
        
        for (int i = 0; i < count; i++) {
            StringBuilder text = new StringBuilder();
            int targetLength = avgLength + random.nextInt(avgLength / 2) - avgLength / 4;
            
            while (text.length() < targetLength) {
                text.append(words[random.nextInt(words.length)]).append(" ");
            }
            
            data.add(text.toString().trim());
        }
        
        return data;
    }
}
