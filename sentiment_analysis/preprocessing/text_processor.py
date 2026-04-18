import re
from typing import List, Union
from functools import lru_cache
from ..utils.logger import get_logger


class TextProcessor:
    def __init__(self):
        self.logger = get_logger('sentiment_analysis.preprocessing')
        self._cache_enabled = True
    
    def preprocess_text(self, text: str) -> str:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")
        
        if self._cache_enabled:
            return self._preprocess_cached(text)
        return self._preprocess(text)
    
    @lru_cache(maxsize=1000)
    def _preprocess_cached(self, text: str) -> str:
        return self._preprocess(text)
    
    def _preprocess(self, text: str) -> str:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    
    def preprocess_batch(self, texts: List[str]) -> List[str]:
        self.logger.info(f"Preprocessing {len(texts)} texts")
        return [self.preprocess_text(text) for text in texts]
    
    def clear_cache(self) -> None:
        self._preprocess_cached.cache_clear()
        self.logger.debug("Preprocessing cache cleared")
    
    def enable_cache(self, enabled: bool = True) -> None:
        self._cache_enabled = enabled
        self.logger.debug(f"Cache {'enabled' if enabled else 'disabled'}")
