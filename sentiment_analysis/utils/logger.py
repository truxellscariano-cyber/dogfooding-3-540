"""日志工具模块。

提供统一的日志配置和管理功能。
"""

import logging
import sys
from pathlib import Path
from typing import Optional

from sentiment_analysis.config import LoggingConfig


def setup_logger(
    name: str,
    config: Optional[LoggingConfig] = None
) -> logging.Logger:
    """设置并返回配置好的日志记录器。
    
    Args:
        name: 日志记录器名称
        config: 日志配置，如果为None则使用默认配置
        
    Returns:
        配置好的日志记录器
    """
    if config is None:
        config = LoggingConfig()
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, config.level.upper()))
    
    # 清除现有的处理器
    logger.handlers.clear()
    
    formatter = logging.Formatter(config.format)
    
    # 控制台处理器
    if config.console_enabled:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, config.level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if config.file_enabled:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        file_handler = logging.FileHandler(
            log_dir / f"{name}.log",
            encoding="utf-8"
        )
        file_handler.setLevel(getattr(logging, config.level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


class LoggerMixin:
    """日志混入类。
    
    为类提供日志功能的混入类。
    """
    
    _loggers: dict = {}
    
    @property
    def logger(self) -> logging.Logger:
        """获取类的日志记录器。"""
        class_name = self.__class__.__name__
        if class_name not in self._loggers:
            self._loggers[class_name] = setup_logger(
                f"sentiment_analysis.{class_name}"
            )
        return self._loggers[class_name]
