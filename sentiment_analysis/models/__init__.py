"""模型模块。

提供模型训练、预测和持久化功能。
"""

from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.models.predictor import Predictor, predict_sentiment

__all__ = ["ModelTrainer", "Predictor", "predict_sentiment"]
