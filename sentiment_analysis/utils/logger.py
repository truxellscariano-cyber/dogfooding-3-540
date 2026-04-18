import logging
from typing import Optional
from ..config import Config


_loggers: dict = {}


def setup_logger(
    name: str = 'sentiment_analysis',
    level: Optional[str] = None,
    log_format: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger(name)
    
    if logger.handlers:
        return logger
    
    log_level = level or Config.LOG_LEVEL
    log_fmt = log_format or Config.LOG_FORMAT
    
    logger.setLevel(getattr(logging, log_level.upper()))
    
    handler = logging.StreamHandler()
    handler.setLevel(getattr(logging, log_level.upper()))
    formatter = logging.Formatter(log_fmt)
    handler.setFormatter(formatter)
    
    logger.addHandler(handler)
    
    _loggers[name] = logger
    return logger


def get_logger(name: str = 'sentiment_analysis') -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    return setup_logger(name)
