# GPU-Accelerated Named Entity Recognition Example

This example demonstrates high-speed entity extraction using GPU acceleration to identify and classify named entities in text documents.

## Features

- **Multi-Entity Recognition**: Identifies persons, organizations, locations, dates, money, and miscellaneous entities
- **GPU Acceleration**: Leverages GPU computing for fast pattern matching and feature processing
- **Advanced Features**:
  - Pattern-based entity detection
  - Context-aware classification
  - Confidence scoring for each entity
  - Position tracking and span extraction
- **Batch Processing**: Efficiently processes multiple documents simultaneously
- **Detailed Output**: Provides entity type, confidence, and position information

## Supported Entity Types

| Entity Type  | Description             | Examples                                        |
| ------------ | ----------------------- | ----------------------------------------------- |
| PERSON       | People's names          | "John Smith", "Dr. Jane Doe", "President Obama" |
| ORGANIZATION | Companies, institutions | "Google", "Harvard University", "NATO"          |
| LOCATION     | Places, addresses       | "New York", "Pacific Ocean", "123 Main St"      |
| DATE         | Temporal expressions    | "January 2023", "yesterday", "next week"        |
| MONEY        | Monetary amounts        | "$100", "€50", "5 million dollars"              |
| MISC         | Other entities          | "iPhone", "Java", "Nobel Prize"                 |

## How It Works

### 1. Pattern Recognition
- **Name Patterns**: Identifies capitalized word sequences for person names
- **Organization Indicators**: Detects company suffixes (Inc., Corp., Ltd.)
- **Location Markers**: Recognizes geographical indicators and place names
- **Date Patterns**: Matches temporal expressions and date formats
- **Money Patterns**: Identifies currency symbols and monetary expressions

### 2. Context Analysis
- **Title Detection**: Recognizes titles like "Dr.", "President", "CEO"
- **Preposition Context**: Uses prepositions to identify locations ("in Paris")
- **Verb Context**: Analyzes verbs to determine entity relationships
- **Capitalization**: Leverages capitalization patterns for entity detection

### 3. GPU Acceleration
- **Parallel Processing**: Simultaneous processing of multiple texts
- **Pattern Matching**: GPU-optimized regular expression processing
- **Feature Extraction**: Fast computation of linguistic features

## Usage

### Single Text Processing

```java
GpuNamedEntityRecognition ner = new GpuNamedEntityRecognition();

String text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976.";
EntityResult[] entities = ner.extractEntities(text);

for (EntityResult entity : entities) {
    System.out.println(entity.getText() + " -> " + entity.getType());
}
```

### Batch Processing

```java
String[] texts = {
    "John works at Microsoft in Seattle.",
    "The meeting is scheduled for January 15th.",
    "The company raised $50 million last year."
};

EntityResult[][] batchResults = ner.extractEntitiesBatch(texts);
for (EntityResult[] entities : batchResults) {
    for (EntityResult entity : entities) {
        System.out.println(entity.getText() + " (" + entity.getType() + ")");
    }
}
```

### Analyzing Entity Details

```java
EntityResult[] entities = ner.extractEntities(text);

for (EntityResult entity : entities) {
    System.out.printf("Entity: %s%n", entity.getText());
    System.out.printf("Type: %s%n", entity.getType());
    System.out.printf("Confidence: %.2f%n", entity.getConfidence());
    System.out.printf("Position: [%d-%d]%n", entity.getStartPos(), entity.getEndPos());
    System.out.println("Context: " + entity.getContext());
    System.out.println();
}
```

## Running the Example

```bash
# Compile and run the example
mvn compile exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.ner.GpuNamedEntityRecognition"
```

## Expected Output

```
🏷️ OpenNLP GPU-Accelerated Named Entity Recognition Demo
========================================================

🔍 Single Text Entity Recognition:
=================================
Text: "Apple Inc. was founded by Steve Jobs in Cupertino..."

Extracted Entities:
📍 Apple Inc. (ORGANIZATION) - confidence: 0.95
👤 Steve Jobs (PERSON) - confidence: 0.92
📍 Cupertino (LOCATION) - confidence: 0.88
📍 California (LOCATION) - confidence: 0.90
📅 1976 (DATE) - confidence: 0.85

🚀 Batch Entity Recognition:
===========================
📊 Processing 8 texts
⚡ Batch NER completed in 89 ms (11.1 ms per text)

Batch Results Summary:
1. "John Smith works at Microsoft in Seattle." (3 entities)
   👤 John Smith (PERSON) - 0.89
   🏢 Microsoft (ORGANIZATION) - 0.94
   📍 Seattle (LOCATION) - 0.87

[... more results ...]

📊 Entity Recognition Statistics:
================================
Total entities found: 23
   PERSON: 6 entities
   ORGANIZATION: 5 entities
   LOCATION: 7 entities
   DATE: 3 entities
   MONEY: 2 entities
   MISC: 0 entities
Average confidence: 0.883

🚀 Features Demonstrated:
========================
✅ GPU-accelerated entity recognition
✅ Multi-type entity classification
✅ Pattern-based entity detection
✅ Context-aware recognition
✅ High-speed batch processing
✅ Confidence scoring and position tracking
```

## Recognition Patterns

### Person Names
- **Patterns**: Capitalized first and last names
- **Titles**: Dr., Mr., Mrs., President, CEO
- **Context**: Action verbs, possessive indicators

```java
// Examples
"Dr. John Smith" -> PERSON (title + name pattern)
"Smith said" -> PERSON (name + speech verb)
"John's office" -> PERSON (possessive form)
```

### Organizations
- **Suffixes**: Inc., Corp., Ltd., LLC, University
- **Patterns**: Capitalized company names
- **Context**: Business-related words

```java
// Examples
"Apple Inc." -> ORGANIZATION (name + suffix)
"Harvard University" -> ORGANIZATION (name + institution word)
"the company" -> ORGANIZATION (business context)
```

### Locations
- **Geographical**: Cities, states, countries
- **Addresses**: Street addresses, postal codes
- **Context**: Prepositions (in, at, near)

```java
// Examples
"New York City" -> LOCATION (known place name)
"123 Main Street" -> LOCATION (address pattern)
"in California" -> LOCATION (preposition + place)
```

### Dates
- **Formats**: Month Year, MM/DD/YYYY, relative dates
- **Patterns**: Temporal expressions
- **Context**: Time-related prepositions

```java
// Examples
"January 2023" -> DATE (month + year)
"01/15/2023" -> DATE (date format)
"next week" -> DATE (relative time)
```

## Performance Characteristics

- **Speed**: ~10-15ms per text on GPU
- **Scalability**: Linear scaling with batch size
- **Memory**: Efficient memory usage for large text collections
- **Accuracy**: Good performance on common entity types

## Entity Recognition Pipeline

### 1. Preprocessing
```
Input Text → Tokenization → POS Tagging → Capitalization Analysis
```

### 2. Pattern Matching
```
Tokens → Pattern Recognition → Context Analysis → Entity Candidates
```

### 3. Classification
```
Candidates → Type Classification → Confidence Scoring → Final Entities
```

### 4. Post-processing
```
Entities → Duplicate Removal → Span Merging → Result Formatting
```

## Customization

### Adding New Entity Types

```java
// Add new entity type
PRODUCT("product", Arrays.asList("iPhone", "Windows", "Tesla"))

// Update recognition patterns
Map<String, Double> productPatterns = new HashMap<>();
productPatterns.put("\\b[A-Z][a-z]+ \\d+", 1.0); // Product with version
```

### Adjusting Recognition Patterns

```java
// Modify person name patterns
personPatterns.put("\\b(Dr|Mr|Mrs|Ms)\\.\\s+[A-Z][a-z]+\\s+[A-Z][a-z]+", 2.0);
personPatterns.put("\\b[A-Z][a-z]+\\s+[A-Z][a-z]+\\b", 1.0);
```

## Accuracy Notes

This is a demonstration example focused on showing GPU acceleration techniques. For production use, consider:

- **Training Data**: Use larger, annotated NER datasets (CoNLL, OntoNotes)
- **Advanced Models**: Implement BiLSTM-CRF, BERT-based NER models
- **Feature Engineering**: Add word embeddings, POS tags, dependency parsing
- **Context Windows**: Use larger context windows for better accuracy
- **Disambiguation**: Add entity linking and disambiguation capabilities

## Dependencies

- OpenNLP GPU Common (included in project)
- Java 8+ with GPU support
- CUDA-capable GPU (recommended)
- Regular expression libraries for pattern matching
