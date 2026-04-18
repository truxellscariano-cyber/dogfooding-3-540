# Sentiment Analysis System

A modular machine learning system for analyzing sentiment in movie reviews.

## Features

- Text preprocessing and cleaning with caching
- TF-IDF vectorization
- Naive Bayes classification
- Model persistence (save/load) with caching
- Batch prediction support
- Comprehensive logging
- Unit tests

## Technology Stack

- Python 3.10
- scikit-learn
- pandas
- numpy
- pytest

## Project Structure

```
sentiment_analysis/
├── __init__.py
├── config.py              # Configuration management
├── data/
│   ├── __init__.py
│   └── loader.py          # Data loading
├── preprocessing/
│   ├── __init__.py
│   └── text_processor.py  # Text preprocessing
├── models/
│   ├── __init__.py
│   ├── trainer.py         # Model training
│   └── predictor.py       # Model prediction with caching
└── utils/
    ├── __init__.py
    └── logger.py          # Logging utilities

main.py                    # Main entry point
tests/                     # Unit tests
├── __init__.py
├── test_preprocessing.py
├── test_trainer.py
└── test_predictor.py
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train and test the model:

```bash
python main.py
```

### Use in your code:

```python
from sentiment_analysis import predict_sentiment, batch_predict

text = "This movie is amazing!"
sentiment, probability = predict_sentiment(text)
print(f"Sentiment: {sentiment}")

texts = ["Great film!", "Terrible movie."]
results = batch_predict(texts)
for result in results:
    print(f"{result['text']}: {result['sentiment']} ({result['confidence']:.2%})")
```

## Running Tests

```bash
pytest tests/ -v
```
