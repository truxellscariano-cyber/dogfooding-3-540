import pickle
import os
from typing import Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from sentiment_analysis.config import Config
from sentiment_analysis.data.loader import load_data
from sentiment_analysis.preprocessing.text_processor import preprocess_text
from sentiment_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def train_model() -> Tuple[MultinomialNB, TfidfVectorizer]:
    logger.info("\n=== Training Model ===")
    
    df = load_data()
    
    logger.info("Preprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], 
        test_size=Config.TEST_SIZE, 
        random_state=Config.RANDOM_STATE
    )
    
    logger.info("Vectorizing text...")
    vectorizer = TfidfVectorizer(
        max_features=Config.TFIDF_MAX_FEATURES,
        ngram_range=Config.TFIDF_NGRAM_RANGE
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    
    logger.info("Training classifier...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)
    
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    logger.info(f"\nModel Accuracy: %.2f%%", accuracy * 100)
    logger.info("\nClassification Report:\n%s", classification_report(y_test, y_pred))
    
    logger.info(f"\nSaving model to %s...", Config.MODEL_FILE)
    with open(Config.MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Saving vectorizer to %s...", Config.VECTORIZER_FILE)
    with open(Config.VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    logger.info("Training complete!")
    return model, vectorizer
