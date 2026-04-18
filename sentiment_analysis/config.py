from dataclasses import dataclass
from typing import Tuple
import os


@dataclass
class Config:
    DATA_FILE: str = 'movie_reviews.csv'
    MODEL_FILE: str = 'sentiment_model.pkl'
    VECTORIZER_FILE: str = 'vectorizer.pkl'
    
    MAX_FEATURES: int = 5000
    NGRAM_RANGE: Tuple[int, int] = (1, 2)
    TEST_SIZE: float = 0.2
    RANDOM_STATE: int = 42
    
    LOG_LEVEL: str = 'INFO'
    LOG_FORMAT: str = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    SAMPLE_DATA_SIZE: int = 50
    
    @classmethod
    def get_model_path(cls, base_dir: str = '') -> str:
        return os.path.join(base_dir, cls.MODEL_FILE)
    
    @classmethod
    def get_vectorizer_path(cls, base_dir: str = '') -> str:
        return os.path.join(base_dir, cls.VECTORIZER_FILE)
    
    @classmethod
    def get_data_path(cls, base_dir: str = '') -> str:
        return os.path.join(base_dir, cls.DATA_FILE)
