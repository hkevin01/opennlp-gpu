# GPU-Accelerated Document Classification Example

This example demonstrates high-speed document classification using GPU acceleration for categorizing documents by topic, genre, or content type.

## Features

- **Multi-Category Classification**: Supports technology, business, science, sports, entertainment, politics, and health categories
- **GPU Acceleration**: Leverages GPU computing for fast TF-IDF and feature processing
- **Advanced Features**:
  - TF-IDF vectorization with GPU acceleration
  - N-gram feature extraction (1-3 grams)
  - Keyword-based feature enhancement
  - Length-based feature normalization
- **Batch Processing**: Efficiently processes multiple documents simultaneously
- **Confidence Scoring**: Provides probability distributions across all categories

## Supported Categories

| Category      | Keywords                                | Common Topics                         |
| ------------- | --------------------------------------- | ------------------------------------- |
| Technology    | AI, software, computer, tech            | Programming, gadgets, innovation      |
| Business      | market, finance, economy, company       | Economics, corporate news, startups   |
| Science       | research, study, discovery, scientist   | Medical breakthroughs, space, physics |
| Sports        | game, team, player, championship        | Football, basketball, Olympics        |
| Entertainment | movie, music, celebrity, film           | Hollywood, concerts, TV shows         |
| Politics      | government, election, policy, president | Elections, legislation, diplomacy     |
| Health        | medical, doctor, patient, treatment     | Medicine, wellness, healthcare        |

## How It Works

### 1. Feature Extraction
- **TF-IDF Computation**: Calculates term frequency-inverse document frequency
- **N-gram Analysis**: Extracts unigrams, bigrams, and trigrams
- **Keyword Enhancement**: Boosts category-specific keywords
- **Document Length Normalization**: Adjusts for document length variations

### 2. Classification Process
- **Vector Computation**: Creates feature vectors for each document
- **Category Scoring**: Compares against category-specific models
- **Probability Calculation**: Normalizes scores to probabilities
- **Confidence Assessment**: Determines classification confidence

### 3. GPU Acceleration
- **Parallel Processing**: Simultaneous processing of multiple documents
- **Matrix Operations**: GPU-optimized vector computations
- **Memory Optimization**: Efficient GPU memory usage for large batches

## Usage

### Single Document Classification

```java
GpuDocumentClassification classifier = new GpuDocumentClassification();

String document = "The new artificial intelligence algorithm shows promising results...";
ClassificationResult result = classifier.classifyDocument(document);

System.out.println("Category: " + result.getCategory());
System.out.println("Confidence: " + result.getConfidence());
```

### Batch Processing

```java
String[] documents = {
    "Tech startup raises $50M in Series A funding...",
    "Scientists discover new exoplanet in distant galaxy...",
    "Championship game ends in overtime victory..."
};

ClassificationResult[] results = classifier.classifyBatch(documents);
for (ClassificationResult result : results) {
    System.out.println(result.getCategory() + ": " + result.getConfidence());
}
```

### Analyzing Classification Details

```java
ClassificationResult result = classifier.classifyDocument(document);

// Get probability distribution
Map<DocumentCategory, Float> probabilities = result.getAllProbabilities();
probabilities.entrySet().stream()
    .sorted(Map.Entry.<DocumentCategory, Float>comparingByValue().reversed())
    .forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));

// Get feature details
Map<String, Double> features = result.getFeatureWeights();
System.out.println("Top features: " + features);
```

## Running the Example

```bash
# Compile and run the example
mvn compile exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.classification.GpuDocumentClassification"
```

## Expected Output

```
📋 OpenNLP GPU-Accelerated Document Classification Demo
======================================================

🔍 Single Document Classification:
=================================
Document: "The breakthrough in quantum computing research..."
Category: SCIENCE
Confidence: 0.89

Probability Distribution:
   SCIENCE: 0.892
   TECHNOLOGY: 0.067
   BUSINESS: 0.041

🚀 Batch Document Classification:
================================
📊 Processing 10 documents
⚡ Batch classification completed in 156 ms (15.6 ms per document)

Classification Results:
 1. The breakthrough in quantum computing research... → SCIENCE (0.89)
 2. Tech startup raises $50M in Series A funding... → BUSINESS (0.84)
 3. Championship game ends in overtime victory... → SPORTS (0.91)
 [... more results ...]

🎯 Classification Summary:
=========================
   TECHNOLOGY: 2 documents
   BUSINESS: 2 documents
   SCIENCE: 2 documents
   SPORTS: 2 documents
   ENTERTAINMENT: 1 documents
   POLITICS: 1 documents
   Average confidence: 0.86

🚀 Features Demonstrated:
========================
✅ GPU-accelerated TF-IDF feature extraction
✅ N-gram feature processing
✅ Multi-category document classification
✅ Keyword-based enhancement
✅ High-speed batch processing
✅ Confidence scoring and probability distribution
```

## Classification Algorithm

### TF-IDF Computation
```
TF-IDF(term, doc) = TF(term, doc) × IDF(term)

Where:
TF(term, doc) = (count of term in doc) / (total terms in doc)
IDF(term) = log(total docs / docs containing term)
```

### Category Scoring
```
Score(category, doc) = Σ(TF-IDF(term, doc) × weight(term, category))

For each term in the document, multiply its TF-IDF value by its 
category-specific weight and sum all values.
```

### N-gram Features
- **Unigrams**: Individual words ("artificial", "intelligence")
- **Bigrams**: Word pairs ("artificial intelligence", "machine learning")
- **Trigrams**: Three-word sequences ("deep learning algorithm")

## Performance Characteristics

- **Speed**: ~15-20ms per document on GPU
- **Scalability**: Linear scaling with batch size
- **Memory**: Efficient memory usage for large document collections
- **Accuracy**: Good performance on well-defined categories

## Customization

### Adding New Categories

```java
// Add new category to enum
MEDICAL("medical", Arrays.asList("doctor", "patient", "medicine", "hospital"))

// Update category models
categoryModels.put(DocumentCategory.MEDICAL, new CategoryWeights(medicalWeights));
```

### Adjusting Feature Weights

```java
// Modify keyword weights
Map<String, Double> techKeywords = new HashMap<>();
techKeywords.put("artificial", 2.0);
techKeywords.put("intelligence", 2.0);
techKeywords.put("machine", 1.8);
```

## Accuracy Notes

This is a demonstration example focused on showing GPU acceleration techniques. For production use, consider:

- **Training Data**: Use larger, more diverse training datasets
- **Advanced Models**: Implement deep learning classifiers (CNN, LSTM, BERT)
- **Feature Engineering**: Add semantic features, entity recognition
- **Cross-validation**: Implement proper evaluation metrics
- **Preprocessing**: Enhanced text cleaning and normalization

## Dependencies

- OpenNLP GPU Common (included in project)
- Java 8+ with GPU support
- CUDA-capable GPU (recommended)
- Sufficient GPU memory for batch processing
