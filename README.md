# Urdu-Keyword-Extraction-Using-Transformer-Model

This system provides a complete keyword extraction solution with multiple models and comprehensive evaluation.

## Features

- **Multiple Keyword Extraction Models:**
  - RAKE (Rapid Automatic Keyword Extraction)
  - YAKE (Yet Another Keyword Extractor)
  - TF-IDF (Term Frequency-Inverse Document Frequency)
  - KeyBERT (BERT-based keyword extraction)
  - Transformer (BERT-based approach)

- **Comprehensive Evaluation:**
  - Precision, Recall, F1-score
  - Relevance and Diversity metrics
  - Performance comparison across models

- **Visualizations:**
  - Data distribution (Pie charts, Bar charts)
  - Evaluation results (Heatmaps, Scatter plots)
  - Radar charts for model comparison
  - Performance metrics comparison

## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Complete System
```bash
python run_keyword_extraction.py
```

### Option 2: Run Step by Step
```python
from keyword_extraction_system import KeywordExtractionSystem

# Initialize system
system = KeywordExtractionSystem()

# Load your data
system.load_data("test.csv")

# Create visualizations
system.visualize_data_distribution()

# Run evaluation
evaluation_results = system.run_evaluation(top_k=10, max_samples=500)

# Generate comparison charts
system.visualize_evaluation_results()
system.create_radar_chart()

# Generate comprehensive report
report = system.generate_comprehensive_report()
```

### Option 3: Extract Keywords for Specific Text
```python
# Extract keywords using all models
text = "Your text here..."
keywords = system.extract_keywords_all_models(text, top_k=10)

# Extract using specific model
rake_keywords = system.extract_keywords_rake(text, top_k=10)
yake_keywords = system.extract_keywords_yake(text, top_k=10)
tfidf_keywords = system.extract_keywords_tfidf(text, top_k=10)
keybert_keywords = system.extract_keywords_keybert(text, top_k=10)
transformer_keywords = system.extract_keywords_transformer(text, top_k=10)
```

## Data Format

The system expects a CSV file with:
- Text column (containing the documents)
- Optional label/category column

The system will automatically detect text and label columns based on common naming patterns.

## Output Files

The system generates:
- `data_distribution.png` - Dataset visualization
- `text_length_distribution.png` - Text length analysis
- `evaluation_results.png` - Model performance comparison
- `radar_chart.png` - Radar chart for model comparison
- `evaluation_report.json` - Detailed evaluation results

## Model Comparison

The system evaluates models on:
1. **Precision** - Accuracy of extracted keywords
2. **Recall** - Coverage of important keywords
3. **F1-Score** - Balanced measure of precision and recall
4. **Relevance** - Semantic similarity to document
5. **Diversity** - Variety among extracted keywords

## Requirements

- Python 3.7+
- See requirements.txt for package dependencies

## Notes

- For large datasets, the evaluation uses a sample for efficiency
- The system automatically handles missing models with fallbacks
- All visualizations are saved as high-quality PNG files
- The evaluation report includes model rankings and recommendations
