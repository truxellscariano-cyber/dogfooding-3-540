"""模型训练模块。

提供模型训练和评估功能。
"""

import pickle
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from sentiment_analysis.config import Config, ModelConfig
from sentiment_analysis.preprocessing.text_processor import TextProcessor
from sentiment_analysis.utils.logger import LoggerMixin


class ModelTrainer(LoggerMixin):
    """模型训练器类。
    
    负责模型训练、评估和保存。
    
    Attributes:
        config: 主配置
        model: 朴素贝叶斯分类器
        text_processor: 文本处理器
        is_trained: 是否已训练
    """
    
    def __init__(
        self,
        config: Optional[Config] = None,
        text_processor: Optional[TextProcessor] = None
    ):
        """初始化模型训练器。
        
        Args:
            config: 主配置
            text_processor: 文本处理器，如果为None则创建新的
        """
        self.config = config or Config()
        self.text_processor = text_processor or TextProcessor(
            self.config.preprocessing,
            self.config.model
        )
        self.model: Optional[MultinomialNB] = None
        self.is_trained = False
        self.logger.info("模型训练器初始化完成")
    
    def train(
        self,
        texts: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """训练模型。
        
        Args:
            texts: 训练文本数组
            labels: 训练标签数组
            
        Returns:
            训练信息字典
        """
        self.logger.info(f"开始训练模型: 样本数={len(texts)}")
        
        # 文本向量化
        X_train = self.text_processor.fit_transform(texts.tolist())
        
        # 创建并训练模型
        self.model = MultinomialNB(alpha=self.config.model.alpha)
        self.model.fit(X_train, labels)
        
        self.is_trained = True
        
        # 交叉验证
        cv_scores = cross_val_score(
            self.model,
            X_train,
            labels,
            cv=self.config.training.cv_folds,
            scoring='accuracy'
        )
        
        train_info = {
            "training_samples": len(texts),
            "feature_dimensions": X_train.shape,
            "cv_mean_accuracy": float(cv_scores.mean()),
            "cv_std_accuracy": float(cv_scores.std()),
        }
        
        self.logger.info(
            f"模型训练完成: 交叉验证准确率={cv_scores.mean():.4f} "
            f"(+/- {cv_scores.std() * 2:.4f})"
        )
        
        return train_info
    
    def evaluate(
        self,
        texts: np.ndarray,
        labels: np.ndarray
    ) -> Dict[str, Any]:
        """评估模型性能。
        
        Args:
            texts: 测试文本数组
            labels: 测试标签数组
            
        Returns:
            评估指标字典
            
        Raises:
            ValueError: 模型未训练时抛出
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，请先调用train方法")
        
        self.logger.info(f"开始评估模型: 测试样本数={len(texts)}")
        
        # 预测
        X_test = self.text_processor.transform(texts.tolist())
        predictions = self.model.predict(X_test)
        
        # 计算指标
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "precision": float(precision_score(labels, predictions, average='binary')),
            "recall": float(recall_score(labels, predictions, average='binary')),
            "f1_score": float(f1_score(labels, predictions, average='binary')),
            "test_samples": len(texts),
        }
        
        # 生成分类报告
        report = classification_report(
            labels,
            predictions,
            target_names=["Negative", "Positive"],
            output_dict=True
        )
        
        # 混淆矩阵
        cm = confusion_matrix(labels, predictions)
        
        self.logger.info(f"评估完成: 准确率={metrics['accuracy']:.4f}")
        self.logger.info(f"精确率={metrics['precision']:.4f}, 召回率={metrics['recall']:.4f}, F1={metrics['f1_score']:.4f}")
        
        return {
            **metrics,
            "classification_report": report,
            "confusion_matrix": cm.tolist()
        }
    
    def save(self, model_path: Optional[Path] = None) -> None:
        """保存模型和向量化器。
        
        Args:
            model_path: 模型保存路径，如果为None则使用配置中的路径
            
        Raises:
            ValueError: 模型未训练时抛出
        """
        if not self.is_trained or self.model is None:
            raise ValueError("模型尚未训练，无法保存")
        
        model_path = model_path or self.config.model_path
        vectorizer_path = self.config.vectorizer_path
        
        self.logger.info(f"保存模型到: {model_path}")
        self.logger.info(f"保存向量化器到: {vectorizer_path}")
        
        # 确保目录存在
        model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存模型
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # 保存向量化器
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.text_processor.vectorizer, f)
        
        self.logger.info("模型和向量化器保存完成")
    
    def load(self, model_path: Optional[Path] = None) -> None:
        """加载模型和向量化器。
        
        Args:
            model_path: 模型加载路径，如果为None则使用配置中的路径
            
        Raises:
            FileNotFoundError: 模型文件不存在时抛出
        """
        model_path = model_path or self.config.model_path
        vectorizer_path = self.config.vectorizer_path
        
        self.logger.info(f"加载模型从: {model_path}")
        self.logger.info(f"加载向量化器从: {vectorizer_path}")
        
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        if not vectorizer_path.exists():
            raise FileNotFoundError(f"向量化器文件不存在: {vectorizer_path}")
        
        # 加载模型
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # 加载向量化器
        with open(vectorizer_path, 'rb') as f:
            self.text_processor.vectorizer = pickle.load(f)
        
        self.is_trained = True
        self.logger.info("模型和向量化器加载完成")
