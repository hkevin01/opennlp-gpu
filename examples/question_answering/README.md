# GPU-Accelerated Question Answering Example

This example demonstrates neural question answering with attention mechanisms using GPU acceleration to answer questions based on provided context passages.

## Features

- **Neural Attention**: Implements attention mechanisms to focus on relevant context parts
- **GPU Acceleration**: Leverages GPU computing for fast parallel processing
- **Answer Extraction**: Intelligent span extraction for precise answers
- **Confidence Scoring**: Provides confidence levels for answer quality
- **Batch Processing**: Efficiently processes multiple questions simultaneously
- **Context Understanding**: Analyzes context to find the most relevant answer spans

## How It Works

### 1. Attention Mechanism
- **Word Embeddings**: Maps words to attention weights based on question relevance
- **Direct Matching**: Identifies exact matches between question and context words
- **Semantic Similarity**: Uses embedding-based similarity for related concepts
- **Position Weighting**: Slightly favors earlier context positions

### 2. Answer Extraction
- **Span Detection**: Identifies potential answer spans of varying lengths
- **Score Calculation**: Combines attention weights and linguistic features
- **Entity Recognition**: Boosts scores for proper nouns and capitalized words
- **Number Detection**: Enhances detection of numerical answers
- **Length Optimization**: Balances answer completeness with conciseness

### 3. GPU Processing
- **Parallel Processing**: Processes multiple questions simultaneously
- **Matrix Operations**: Efficient GPU-based computations for attention
- **Memory Optimization**: Optimized memory usage for large batch processing

## Usage

### Single Question Answering

```java
GpuQuestionAnswering qaSystem = new GpuQuestionAnswering();

String question = "What is the capital of France?";
String context = "France is a country in Western Europe. The capital and largest city of France is Paris...";

QAResult result = qaSystem.answerQuestion(question, context);

System.out.println("Answer: " + result.getAnswer());
System.out.println("Confidence: " + result.getConfidence());
```

### Batch Processing

```java
QuestionAnswerPair[] pairs = {
    new QuestionAnswerPair("What is the capital?", "The capital is Paris..."),
    new QuestionAnswerPair("When was it founded?", "Founded in 1995...")
};

QAResult[] results = qaSystem.answerQuestionsBatch(pairs);
for (QAResult result : results) {
    System.out.println(result.getAnswer());
}
```

### Analyzing Attention Weights

```java
QAResult result = qaSystem.answerQuestion(question, context);
Map<String, Double> attention = result.getAttentionWeights();

// Show which words the model focused on
attention.entrySet().stream()
    .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
    .limit(5)
    .forEach(entry -> System.out.println(entry.getKey() + ": " + entry.getValue()));
```

## Running the Example

```bash
# Compile and run the example
mvn compile exec:java -Dexec.mainClass="org.apache.opennlp.gpu.examples.question_answering.GpuQuestionAnswering"
```

## Expected Output

```
🧠 OpenNLP GPU-Accelerated Question Answering Demo
==================================================

🔍 Single Question Answering:
=============================
Question: What is the capital of France?
Answer: Paris
Confidence: 0.89
Position: [12-13]

Top Attention Weights:
   capital: 0.245
   france: 0.198
   paris: 0.167
   largest: 0.089
   city: 0.076

🚀 Batch Question Answering:
============================
📊 Processing 7 questions
⚡ Batch QA completed in 78 ms (11.14 ms per question)

Question Answering Results:
1. Q: What is the capital of France?...
   A: Paris (confidence: 0.89)

2. Q: When was the company founded?...
   A: 1995 (confidence: 0.82)

3. Q: Who discovered gravity?...
   A: Sir Isaac Newton (confidence: 0.91)

[... more results ...]

📊 Question Answering Statistics:
=================================
Questions processed: 7
Average confidence: 0.835
High-confidence answers: 6 (85.7%)
Average answer length: 2.1 words

🚀 Features Demonstrated:
========================
✅ GPU-accelerated question answering
✅ Neural attention mechanisms
✅ Answer span extraction
✅ Confidence scoring
✅ Attention weight visualization
✅ High-speed batch processing
✅ Context-aware answer selection
✅ Multi-question parallel processing
```

## Question Types Supported

### Factual Questions
- **What**: "What is the capital of France?" → "Paris"
- **Who**: "Who discovered gravity?" → "Sir Isaac Newton"
- **Where**: "Where is the Statue of Liberty?" → "Liberty Island"
- **When**: "When was the company founded?" → "1995"

### Numerical Questions
- **How many**: "How many planets are there?" → "eight"
- **Quantities**: "What is the population?" → "8 million"

### Named Entities
- **People**: Names, titles, professions
- **Places**: Cities, countries, landmarks
- **Organizations**: Companies, institutions
- **Dates**: Years, specific dates

## Technical Architecture

### Attention Mechanism
```
Question: "What is the capital of France?"
Context: "France is a country... The capital is Paris..."

Attention Weights:
capital → 0.25 (high relevance)
france  → 0.20 (question match)
paris   → 0.17 (answer candidate)
country → 0.08 (context)
```

### Answer Extraction Process
1. **Tokenization**: Split context into words
2. **Span Generation**: Create candidate answer spans (1-10 words)
3. **Scoring**: Calculate attention-weighted scores
4. **Ranking**: Select highest-scoring span
5. **Validation**: Apply linguistic filters and bonuses

### GPU Optimization
- **Parallel Question Processing**: Multiple questions processed simultaneously
- **Vectorized Operations**: GPU-accelerated matrix computations
- **Memory Pooling**: Efficient GPU memory management
- **Batch Optimization**: Optimized for large-scale processing

## Accuracy Notes

This is a demonstration example focused on showing GPU acceleration and attention mechanisms. For production use, consider:

- **Advanced Models**: Use transformer-based models (BERT, RoBERTa)
- **Larger Training Data**: Train on comprehensive QA datasets
- **Context Preprocessing**: Better text normalization and segmentation
- **Multi-passage Support**: Handle multiple context passages
- **Answer Validation**: Additional semantic validation steps

## Performance Characteristics

- **Speed**: ~10-15ms per question on GPU
- **Scalability**: Linear scaling with batch size
- **Memory**: Efficient memory usage for large contexts
- **Accuracy**: Demonstrates core QA concepts with reasonable performance

## Dependencies

- OpenNLP GPU Common (included in project)
- Java 8+ with GPU support
- CUDA-capable GPU (recommended)
- Sufficient GPU memory for batch processing
