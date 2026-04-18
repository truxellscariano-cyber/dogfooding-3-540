# Sentiment Analysis System

A simple machine learning system for analyzing sentiment in movie reviews.

## Features

- Text preprocessing and cleaning
- TF-IDF vectorization
- Naive Bayes classification
- Model persistence (save/load)
- Batch prediction support

## Technology Stack

- Python 3.10
- scikit-learn
- pandas
- numpy

## Installation

```bash
pip install pandas numpy scikit-learn
```

## Usage

Train and test the model:
```bash
python sentiment_analysis.py
```

Use in your code:
```python
from sentiment_analysis import predict_sentiment

text = "This movie is amazing!"
sentiment, probability = predict_sentiment(text)
print(f"Sentiment: {sentiment}")
```

## Current Issues

This is a basic implementation with several areas that need improvement:

- **Code Structure**: All code is in a single file, making it hard to maintain
- **Performance**: No caching mechanism, preprocessing is repeated
- **Configuration**: Hard-coded parameters scattered throughout the code
- **Error Handling**: Minimal error handling and validation
- **Logging**: Only basic print statements, no proper logging
- **Testing**: No unit tests or integration tests
- **Documentation**: Limited inline documentation

## Future Improvements

- Refactor into modular architecture
- Add configuration management
- Implement proper logging
- Add comprehensive error handling
- Create unit tests
- Optimize performance with caching
- Add API interface
