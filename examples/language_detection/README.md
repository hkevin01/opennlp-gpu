# GPU-Accelerated Language Detection Example

This example demonstrates high-speed language identification using GPU acceleration to detect the language of text documents.

## Features

- **Multi-language Support**: Detects 12 major languages (English, Spanish, French, German, Italian, Portuguese, Dutch, Russian, Chinese, Japanese, Arabic, Hindi)
- **GPU Acceleration**: Leverages GPU computing for fast parallel processing
- **Multiple Analysis Methods**: 
  - Character n-gram analysis
  - Word frequency analysis
  - Character frequency analysis
- **Batch Processing**: Efficiently processes multiple texts simultaneously
- **Confidence Scoring**: Provides probability distributions for all languages

## Supported Languages

| Language   | Code | Script     | Characteristics                    |
| ---------- | ---- | ---------- | ---------------------------------- |
| English    | en   | Latin      | High frequency of "th", "he", "in" |
| Spanish    | es   | Latin      | High frequency of "es", "de", "en" |
| French     | fr   | Latin      | High frequency of "es", "de", "le" |
| German     | de   | Latin      | High frequency of "er", "en", "ch" |
| Italian    | it   | Latin      | High frequency of "re", "er", "ar" |
| Portuguese | pt   | Latin      | High frequency of "de", "ar", "er" |
| Dutch      | nl   | Latin      | High frequency of "en", "de", "er" |
| Russian    | ru   | Cyrillic   | High frequency of "то", "не", "на" |
| Chinese    | zh   | Chinese    | Logographic characters             |
| Japanese   | ja   | Mixed      | Hiragana, Katakana, Kanji          |
| Arabic     | ar   | Arabic     | Right-to-left script               |
| Hindi      | hi   | Devanagari | Complex script system              |

## How It Works

1. **Feature Extraction**: Analyzes character patterns, word frequencies, and linguistic features
2. **Model Scoring**: Compares input text against language-specific models
3. **GPU Processing**: Utilizes parallel processing for fast batch analysis
4. **Probability Calculation**: Normalizes scores to provide confidence levels

## Usage

### Single Text Detection

```java
GpuLanguageDetection detector = new GpuLanguageDetection();
String text = "Hello, this is a sample text in English.";
LanguageResult result = detector.detectLanguage(text);

System.out.println("Language: " + result.getLanguage().getName());
System.out.println("Confidence: " + result.getConfidence());
```

### Batch Processing

```java
String[] texts = {
    "Hello world",
    "Hola mundo", 
    "Bonjour le monde"
};

LanguageResult[] results = detector.detectLanguageBatch(texts);
for (LanguageResult result : results) {
    System.out.println(result.getLanguage().getName());
}
```

## Running the Example

```bash
# Compile and run the example
mvn compile exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.language_detection.GpuLanguageDetection"
```

## Expected Output

```
🌍 OpenNLP GPU-Accelerated Language Detection Demo
==================================================

🔍 Single Text Language Detection:
=================================
Text: "Hello, this is a sample text in English. The..."
Detected Language: English (en)
Confidence: 0.85

Top 3 Language Probabilities:
   English: 0.847
   Dutch: 0.089
   German: 0.064

🚀 Batch Language Detection:
============================
📊 Processing 11 texts
⚡ Batch detection completed in 45 ms (4.09 ms per text)

Detection Results:
 1. Hello, this is a sample text in English... → English (0.85)
 2. Hola, este es un texto de muestra en esp... → Spanish (0.92)
 3. Bonjour, ceci est un texte d'exemple en ... → French (0.88)
 [... more results ...]

📊 Detection Statistics:
=======================
Languages detected:
   English: 1 texts
   Spanish: 1 texts
   French: 1 texts
   [... more languages ...]
Average confidence: 0.823

🚀 Features Demonstrated:
========================
✅ GPU-accelerated language detection
✅ Support for 12 major languages
✅ Character n-gram analysis
✅ Word frequency analysis
✅ Character frequency analysis
✅ High-speed batch processing
✅ Confidence scoring and probability distribution
✅ Parallel processing capabilities
```

## Technical Details

### Language Models

Each language uses a combination of:
- **Common Words**: Most frequent words in the language
- **Character Frequencies**: Expected frequency of each character
- **Character N-grams**: Most common character sequences

### Performance Features

- **Parallel Processing**: Utilizes multiple cores for batch processing
- **GPU Acceleration**: Leverages GPU memory and compute for feature extraction
- **Optimized Algorithms**: Fast character-based analysis methods
- **Memory Efficient**: Optimized data structures for large-scale processing

## Accuracy Notes

This is a demonstration example focused on showing GPU acceleration techniques. For production use, consider:
- Training on larger, more diverse datasets
- Using more sophisticated machine learning models
- Implementing additional linguistic features
- Adding support for mixed-language texts
- Handling short texts more effectively

## Dependencies

- OpenNLP GPU Common (included in project)
- Java 8+ with GPU support
- CUDA-capable GPU (recommended)
