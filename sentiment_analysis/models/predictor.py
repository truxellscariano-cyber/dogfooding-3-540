"""模型预测模块。

提供单条和批量预测功能，支持模型缓存。
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np

from sentiment_analysis.config import Config
from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.utils.logger import LoggerMixin


class Predictor(LoggerMixin):
    """预测器类。
    
    负责模型预测，支持单条和批量预测，内置缓存机制。
    
    Attributes:
        config: 主配置
        trainer: 模型训练器（包含模型和向量化器）
        _model_cache: 模型缓存字典
    """
    
    _model_cache: Dict[str, ModelTrainer] = {}
    
    def __init__(
        self,
        config: Optional[Config] = None,
        model_path: Optional[str] = None,
        use_cache: bool = True
    ):
        """初始化预测器。
        
        Args:
            config: 主配置
            model_path: 模型路径，如果提供则从该路径加载模型
            use_cache: 是否使用模型缓存
        """
        self.config = config or Config()
        self.use_cache = use_cache
        self.trainer: Optional[ModelTrainer] = None
        
        if model_path:
            self._load_model_from_path(model_path)
        else:
            self._load_default_model()
        
        self.logger.info("预测器初始化完成")
    
    def _load_model_from_path(self, model_path: str) -> None:
        """从指定路径加载模型。
        
        Args:
            model_path: 模型路径
        """
        cache_key = model_path
        
        if self.use_cache and cache_key in self._model_cache:
            self.logger.info(f"从缓存加载模型: {model_path}")
            self.trainer = self._model_cache[cache_key]
            return
        
        self.logger.info(f"从路径加载模型: {model_path}")
        self.trainer = ModelTrainer(self.config)
        self.trainer.load()
        
        if self.use_cache:
            self._model_cache[cache_key] = self.trainer
    
    def _load_default_model(self) -> None:
        """加载默认模型。"""
        cache_key = str(self.config.model_path)
        
        if self.use_cache and cache_key in self._model_cache:
            self.logger.info("从缓存加载默认模型")
            self.trainer = self._model_cache[cache_key]
            return
        
        # 检查模型文件是否存在
        if self.config.model_path.exists():
            self.logger.info("加载默认模型")
            self.trainer = ModelTrainer(self.config)
            self.trainer.load()
            
            if self.use_cache:
                self._model_cache[cache_key] = self.trainer
        else:
            self.logger.warning("默认模型不存在，预测器未加载模型")
            self.trainer = ModelTrainer(self.config)
    
    def predict(
        self,
        text: str,
        return_probability: bool = True
    ) -> Dict[str, Any]:
        """预测单条文本的情感。
        
        Args:
            text: 输入文本
            return_probability: 是否返回概率
            
        Returns:
            包含预测结果的字典
            
        Raises:
            ValueError: 模型未加载时抛出
        """
        if not self.trainer or not self.trainer.is_trained:
            raise ValueError("模型未加载，无法预测")
        
        # 预处理
        processed = self.trainer.text_processor.preprocess(text)
        
        # 向量化
        features = self.trainer.text_processor.transform([processed])
        
        # 预测
        prediction = self.trainer.model.predict(features)[0]
        
        result = {
            "text": text,
            "sentiment": "positive" if prediction == 1 else "negative",
            "label": int(prediction)
        }
        
        if return_probability:
            probabilities = self.trainer.model.predict_proba(features)[0]
            result["confidence"] = float(probabilities[prediction])
            result["probabilities"] = {
                "negative": float(probabilities[0]),
                "positive": float(probabilities[1])
            }
        
        return result
    
    def predict_batch(
        self,
        texts: List[str],
        return_probability: bool = True
    ) -> List[Dict[str, Any]]:
        """批量预测文本情感。
        
        Args:
            texts: 输入文本列表
            return_probability: 是否返回概率
            
        Returns:
            预测结果列表
            
        Raises:
            ValueError: 模型未加载时抛出
        """
        if not self.trainer or not self.trainer.is_trained:
            raise ValueError("模型未加载，无法预测")
        
        if not texts:
            return []
        
        self.logger.info(f"批量预测: {len(texts)} 条文本")
        
        # 批量预处理
        processed_texts = [
            self.trainer.text_processor.preprocess(text)
            for text in texts
        ]
        
        # 批量向量化
        features = self.trainer.text_processor.transform(processed_texts)
        
        # 批量预测
        predictions = self.trainer.model.predict(features)
        
        results = []
        if return_probability:
            probabilities = self.trainer.model.predict_proba(features)
            for i, (text, pred) in enumerate(zip(texts, predictions)):
                results.append({
                    "text": text,
                    "sentiment": "positive" if pred == 1 else "negative",
                    "label": int(pred),
                    "confidence": float(probabilities[i][pred]),
                    "probabilities": {
                        "negative": float(probabilities[i][0]),
                        "positive": float(probabilities[i][1])
                    }
                })
        else:
            for text, pred in zip(texts, predictions):
                results.append({
                    "text": text,
                    "sentiment": "positive" if pred == 1 else "negative",
                    "label": int(pred)
                })
        
        self.logger.info(f"批量预测完成")
        return results
    
    @classmethod
    def clear_cache(cls) -> None:
        """清除模型缓存。"""
        cls._model_cache.clear()
    
    @property
    def is_ready(self) -> bool:
        """检查预测器是否已准备好。"""
        return self.trainer is not None and self.trainer.is_trained


def predict_sentiment(
    text: str,
    model_path: Optional[str] = None,
    return_probability: bool = True
) -> tuple:
    """便捷的预测函数。
    
    这是向后兼容的函数，使用方式与原来相同。
    
    Args:
        text: 输入文本
        model_path: 模型路径
        return_probability: 是否返回概率
        
    Returns:
        (情感标签, 置信度) 的元组
    """
    predictor = Predictor(model_path=model_path)
    result = predictor.predict(text, return_probability=return_probability)
    
    sentiment = result["sentiment"]
    confidence = result.get("confidence", 0.0)
    
    return sentiment, confidence
