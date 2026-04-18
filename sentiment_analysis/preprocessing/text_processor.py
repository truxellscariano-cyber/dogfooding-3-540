import re
from typing import Optional
from functools import lru_cache
from sentiment_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


@lru_cache(maxsize=1000)
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        logger.warning("Text is not a string, converting to string")
        text = str(text)
    
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text
