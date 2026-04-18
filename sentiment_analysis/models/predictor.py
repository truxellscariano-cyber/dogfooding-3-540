import pickle
import os
from typing import Tuple, List, Dict, Optional, Union
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from ..config import Config
from ..preprocessing.text_processor import TextProcessor
from ..utils.logger import get_logger


class ModelPredictor:
    _instance: Optional['ModelPredictor'] = None
    _model: Optional[MultinomialNB] = None
    _vectorizer: Optional[TfidfVectorizer] = None
    
    def __new__(cls, base_dir: str = ''):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, base_dir: str = ''):
        if self._initialized:
            return
        
        self.base_dir = base_dir
        self.logger = get_logger('sentiment_analysis.models.predictor')
        self.text_processor = TextProcessor()
        self._initialized = True
    
    def load_model(self) -> Tuple[MultinomialNB, TfidfVectorizer]:
        if self._model is not None and self._vectorizer is not None:
            self.logger.debug("Returning cached model")
            return self._model, self._vectorizer
        
        model_path = Config.get_model_path(self.base_dir)
        vectorizer_path = Config.get_vectorizer_path(self.base_dir)
        
        if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
            raise FileNotFoundError(
                f"Model files not found. Please train the model first. "
                f"Expected: {model_path}, {vectorizer_path}"
            )
        
        self.logger.info(f"Loading model from {model_path}")
        with open(model_path, 'rb') as f:
            self._model = pickle.load(f)
        
        self.logger.info(f"Loading vectorizer from {vectorizer_path}")
        with open(vectorizer_path, 'rb') as f:
            self._vectorizer = pickle.load(f)
        
        return self._model, self._vectorizer
    
    def predict(self, text: str) -> Tuple[str, float]:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")
        
        model, vectorizer = self.load_model()
        
        cleaned_text = self.text_processor.preprocess_text(text)
        text_vec = vectorizer.transform([cleaned_text])
        
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        confidence = max(probability)
        
        return prediction, confidence
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Union[str, float]]]:
        self.logger.info(f"Batch prediction for {len(texts)} texts")
        
        results = []
        for text in texts:
            sentiment, confidence = self.predict(text)
            results.append({
                'text': text,
                'sentiment': sentiment,
                'confidence': confidence
            })
        
        return results
    
    def clear_cache(self) -> None:
        self._model = None
        self._vectorizer = None
        self.text_processor.clear_cache()
        self.logger.debug("Model and preprocessing caches cleared")


_predictor_instance: Optional[ModelPredictor] = None


def _get_predictor(base_dir: str = '') -> ModelPredictor:
    global _predictor_instance
    if _predictor_instance is None:
        _predictor_instance = ModelPredictor(base_dir)
    return _predictor_instance


def predict_sentiment(
    text: str,
    model: Optional[MultinomialNB] = None,
    vectorizer: Optional[TfidfVectorizer] = None,
    base_dir: str = ''
) -> Tuple[str, float]:
    if model is not None and vectorizer is not None:
        text_processor = TextProcessor()
        cleaned_text = text_processor.preprocess_text(text)
        text_vec = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vec)[0]
        probability = model.predict_proba(text_vec)[0]
        confidence = max(probability)
        return prediction, confidence
    
    predictor = _get_predictor(base_dir)
    return predictor.predict(text)


def batch_predict(
    texts: List[str],
    base_dir: str = ''
) -> List[Dict[str, Union[str, float]]]:
    predictor = _get_predictor(base_dir)
    return predictor.predict_batch(texts)
