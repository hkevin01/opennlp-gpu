# OpenNLP Test Data Integration

This directory contains test data and resources for validating GPU acceleration with real OpenNLP models and datasets.

## Available Test Resources

### 1. OpenNLP Official Test Data
- **Sentence Detection**: Sample sentences from OpenNLP test suite
- **Tokenization**: Real tokenization test cases
- **POS Tagging**: Part-of-speech tagging examples
- **Named Entity Recognition**: NER test datasets

### 2. Downloadable Models
The integration tests can automatically download these OpenNLP models:
- Language Detection Model (langdetect-183.bin)
- English Tokenizer Model (en-token.bin)
- English POS Tagger Model (en-pos-maxent.bin)

### 3. Custom Test Datasets
- **Performance Testing**: Varying sizes of text datasets (10, 50, 100, 500, 1000 documents)
- **Accuracy Testing**: Ground truth datasets for validation
- **Stress Testing**: Large documents for memory and performance testing

## Usage

### Download Real OpenNLP Test Data
```java
// Automatically downloads and uses real OpenNLP test data
OpenNLPTestDataIntegration integration = new OpenNLPTestDataIntegration();
integration.runRealModelTests();
```

### Use Specific Test Datasets
```java
// Load specific test dataset
List<String> testTexts = TestDataLoader.loadDataset("sentences");
String[] documents = testTexts.toArray(new String[testTexts.size()]);

// Test GPU acceleration
GpuFeatureExtractor extractor = new GpuFeatureExtractor(provider, config, matrixOp);
float[][] features = extractor.extractNGramFeatures(documents, 2, 1000);
```

### Performance Benchmarking
```java
// Run performance comparison with real data
PerformanceBenchmark benchmark = new PerformanceBenchmark();
benchmark.benchmarkWithRealData(documents);
```

## Test Data Sources

1. **Apache OpenNLP Repository**
   - URL: https://github.com/apache/opennlp
   - Test files: `opennlp-tools/src/test/resources/`

2. **OpenNLP Models**
   - URL: https://dlcdn.apache.org/opennlp/models/
   - Pre-trained models for various NLP tasks

3. **Common NLP Datasets**
   - Brown Corpus samples
   - Penn Treebank samples
   - CoNLL-2003 NER samples

## Adding Your Own Test Data

1. Create a new directory under `test-data/custom/`
2. Add your text files (one document per line)
3. Update the `TestDataLoader` to include your dataset
4. Run integration tests with your data

## File Formats

- **Text files**: One sentence/document per line
- **Training data**: Tab-separated values (word\tpos\tlabel)
- **Models**: OpenNLP binary format (.bin files)
