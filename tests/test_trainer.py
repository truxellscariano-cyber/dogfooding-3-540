"""模型训练模块的单元测试。
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.config import Config, ModelConfig, TrainingConfig


class TestModelTrainer:
    """ModelTrainer类的测试。"""
    
    @pytest.fixture
    def sample_data(self):
        """创建示例数据。"""
        texts = np.array([
            "This movie is absolutely amazing",
            "I loved this film, best ever",
            "Fantastic performance and great plot",
            "Wonderful story, highly recommended",
            "Excellent movie, must watch",
            "Terrible movie, waste of time",
            "I hated this film, very bad",
            "Awful direction and weak story",
            "Disappointing performance from cast",
            "Boring and predictable movie"
        ])
        labels = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        return texts, labels
    
    @pytest.fixture
    def trainer(self):
        """创建模型训练器。"""
        return ModelTrainer()
    
    @pytest.fixture
    def temp_dir(self):
        """创建临时目录。"""
        temp_path = tempfile.mkdtemp()
        yield temp_path
        shutil.rmtree(temp_path)
    
    def test_initialization(self, trainer):
        """测试初始化。"""
        assert trainer.config is not None
        assert trainer.text_processor is not None
        assert trainer.model is None
        assert trainer.is_trained is False
    
    def test_train(self, trainer, sample_data):
        """测试模型训练。"""
        texts, labels = sample_data
        
        train_info = trainer.train(texts, labels)
        
        assert trainer.is_trained is True
        assert trainer.model is not None
        assert "training_samples" in train_info
        assert "cv_mean_accuracy" in train_info
        assert train_info["training_samples"] == len(texts)
        assert 0 <= train_info["cv_mean_accuracy"] <= 1
    
    def test_train_empty_data(self, trainer):
        """测试空数据训练。"""
        texts = np.array([])
        labels = np.array([])
        
        with pytest.raises(Exception):
            trainer.train(texts, labels)
    
    def test_evaluate_before_train(self, trainer, sample_data):
        """测试未训练就评估应抛出异常。"""
        texts, labels = sample_data
        
        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.evaluate(texts, labels)
    
    def test_evaluate(self, trainer, sample_data):
        """测试模型评估。"""
        texts, labels = sample_data
        
        trainer.train(texts, labels)
        eval_results = trainer.evaluate(texts, labels)
        
        assert "accuracy" in eval_results
        assert "precision" in eval_results
        assert "recall" in eval_results
        assert "f1_score" in eval_results
        assert "classification_report" in eval_results
        assert "confusion_matrix" in eval_results
        
        assert 0 <= eval_results["accuracy"] <= 1
        assert 0 <= eval_results["precision"] <= 1
        assert 0 <= eval_results["recall"] <= 1
        assert 0 <= eval_results["f1_score"] <= 1
    
    def test_save_before_train(self, trainer, temp_dir):
        """测试未训练就保存应抛出异常。"""
        with pytest.raises(ValueError, match="模型尚未训练"):
            trainer.save()
    
    def test_save_and_load(self, trainer, sample_data, temp_dir):
        """测试模型保存和加载。"""
        texts, labels = sample_data
        
        # 训练模型
        trainer.train(texts, labels)
        
        # 修改配置以使用临时目录
        trainer.config.model_path = Path(temp_dir) / "model.pkl"
        trainer.config.vectorizer_path = Path(temp_dir) / "vectorizer.pkl"
        
        # 保存模型
        trainer.save()
        
        # 验证文件存在
        assert trainer.config.model_path.exists()
        assert trainer.config.vectorizer_path.exists()
        
        # 创建新的训练器并加载
        new_trainer = ModelTrainer(trainer.config)
        new_trainer.load()
        
        assert new_trainer.is_trained is True
        assert new_trainer.model is not None
        assert new_trainer.text_processor.vectorizer is not None
    
    def test_load_nonexistent_model(self, trainer, temp_dir):
        """测试加载不存在的模型应抛出异常。"""
        trainer.config.model_path = Path(temp_dir) / "nonexistent.pkl"
        
        with pytest.raises(FileNotFoundError):
            trainer.load()
    
    def test_prediction_after_train(self, trainer, sample_data):
        """测试训练后的预测。"""
        texts, labels = sample_data
        
        trainer.train(texts, labels)
        
        # 转换新文本
        test_texts = ["This is a great movie"]
        features = trainer.text_processor.transform(test_texts)
        predictions = trainer.model.predict(features)
        
        assert len(predictions) == 1
        assert predictions[0] in [0, 1]
