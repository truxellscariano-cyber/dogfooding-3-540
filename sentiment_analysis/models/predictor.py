import pickle
import os
from typing import Optional, Tuple, List, Dict
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sentiment_analysis.config import Config
from sentiment_analysis.models.trainer import train_model
from sentiment_analysis.preprocessing.text_processor import preprocess_text
from sentiment_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)

_model_cache: Optional[Tuple[MultinomialNB, TfidfVectorizer]] = None


def _load_model_and_vectorizer() -> Tuple[MultinomialNB, TfidfVectorizer]:
    global _model_cache
    
    if _model_cache is not None:
        logger.debug("Using cached model and vectorizer")
        return _model_cache
    
    if not os.path.exists(Config.MODEL_FILE) or not os.path.exists(Config.VECTORIZER_FILE):
        logger.info("Model not found. Training new model...")
        model, vectorizer = train_model()
    else:
        logger.info("Loading saved model...")
        with open(Config.MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(Config.VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
    
    _model_cache = (model, vectorizer)
    return model, vectorizer


def predict_sentiment(
    text: str, 
    model: Optional[MultinomialNB] = None, 
    vectorizer: Optional[TfidfVectorizer] = None
) -> Tuple[str, List[float]]:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Input text must be a non-empty string")
    
    if model is None or vectorizer is None:
        model, vectorizer = _load_model_and_vectorizer()
    
    cleaned_text = preprocess_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0].tolist()
    
    return prediction, probability


def batch_predict(texts: List[str]) -> List[Dict]:
    logger.info("\n=== Batch Prediction ===")
    
    if not isinstance(texts, list):
        raise TypeError("texts must be a list of strings")
    
    model, vectorizer = _load_model_and_vectorizer()
    
    results = []
    for text in texts:
        try:
            prediction, probability = predict_sentiment(text, model, vectorizer)
            results.append({
                'text': text,
                'sentiment': prediction,
                'confidence': max(probability)
            })
        except Exception as e:
            logger.error(f"Error predicting for text: {text}. Error: {e}")
            results.append({
                'text': text,
                'sentiment': 'error',
                'confidence': 0.0,
                'error': str(e)
            })
    
    return results


def clear_cache() -> None:
    global _model_cache
    _model_cache = None
    logger.info("Model cache cleared")
