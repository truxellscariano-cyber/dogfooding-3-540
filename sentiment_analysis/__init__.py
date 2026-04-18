"""情感分析系统。

一个基于机器学习的电影评论情感分析系统。

主要功能：
- 文本预处理和清洗
- TF-IDF向量化
- 朴素贝叶斯分类
- 模型持久化（保存/加载）
- 批量预测支持

示例用法：
    >>> from sentiment_analysis import predict_sentiment
    >>> sentiment, confidence = predict_sentiment("This movie is amazing!")
    >>> print(f"情感: {sentiment}, 置信度: {confidence:.2f}")
"""

from sentiment_analysis.models import predict_sentiment, Predictor, ModelTrainer
from sentiment_analysis.data import DataLoader
from sentiment_analysis.preprocessing import TextProcessor
from sentiment_analysis.config import Config

__version__ = "2.0.0"
__all__ = [
    "predict_sentiment",
    "Predictor",
    "ModelTrainer",
    "DataLoader",
    "TextProcessor",
    "Config"
]
