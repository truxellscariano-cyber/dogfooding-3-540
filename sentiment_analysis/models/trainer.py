import pickle
import os
from typing import Tuple, Optional
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

from ..config import Config
from ..data.loader import DataLoader
from ..preprocessing.text_processor import TextProcessor
from ..utils.logger import get_logger


class ModelTrainer:
    def __init__(self, base_dir: str = ''):
        self.base_dir = base_dir
        self.logger = get_logger('sentiment_analysis.models.trainer')
        self.data_loader = DataLoader(base_dir)
        self.text_processor = TextProcessor()
        self._model: Optional[MultinomialNB] = None
        self._vectorizer: Optional[TfidfVectorizer] = None
    
    def train(self) -> Tuple[MultinomialNB, TfidfVectorizer]:
        self.logger.info("Starting model training")
        
        df = self.data_loader.load_data()
        
        self.logger.info("Preprocessing text data")
        df['cleaned_review'] = df['review'].apply(self.text_processor.preprocess_text)
        
        X_train, X_test, y_train, y_test = train_test_split(
            df['cleaned_review'],
            df['sentiment'],
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        
        self.logger.info("Vectorizing text data")
        self._vectorizer = TfidfVectorizer(
            max_features=Config.MAX_FEATURES,
            ngram_range=Config.NGRAM_RANGE
        )
        X_train_vec = self._vectorizer.fit_transform(X_train)
        X_test_vec = self._vectorizer.transform(X_test)
        
        self.logger.info("Training Naive Bayes classifier")
        self._model = MultinomialNB()
        self._model.fit(X_train_vec, y_train)
        
        y_pred = self._model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        
        self.logger.info(f"Model accuracy: {accuracy:.2%}")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        self._save_model()
        
        return self._model, self._vectorizer
    
    def _save_model(self) -> None:
        if self._model is None or self._vectorizer is None:
            raise ValueError("Model or vectorizer not initialized")
        
        model_path = Config.get_model_path(self.base_dir)
        vectorizer_path = Config.get_vectorizer_path(self.base_dir)
        
        self.logger.info(f"Saving model to {model_path}")
        with open(model_path, 'wb') as f:
            pickle.dump(self._model, f)
        
        self.logger.info(f"Saving vectorizer to {vectorizer_path}")
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self._vectorizer, f)
        
        self.logger.info("Training complete")
    
    def get_model(self) -> Tuple[Optional[MultinomialNB], Optional[TfidfVectorizer]]:
        return self._model, self._vectorizer
