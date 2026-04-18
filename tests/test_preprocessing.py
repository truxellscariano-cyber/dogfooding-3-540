"""文本预处理模块的单元测试。
"""

import pytest
import numpy as np

from sentiment_analysis.preprocessing.text_processor import TextProcessor
from sentiment_analysis.config import PreprocessingConfig, ModelConfig


class TestTextProcessor:
    """TextProcessor类的测试。"""
    
    @pytest.fixture
    def processor(self):
        """创建默认的文本处理器。"""
        return TextProcessor()
    
    @pytest.fixture
    def custom_processor(self):
        """创建自定义配置的文本处理器。"""
        prep_config = PreprocessingConfig(
            remove_stopwords=False,
            lowercase=False,
            remove_punctuation=False
        )
        return TextProcessor(prep_config=prep_config)
    
    def test_clean_text_lowercase(self, processor):
        """测试文本清洗 - 小写转换。"""
        text = "HELLO World"
        result = processor.clean_text(text)
        assert result == "hello world"
    
    def test_clean_text_remove_punctuation(self, processor):
        """测试文本清洗 - 移除标点。"""
        text = "Hello, world! How are you?"
        result = processor.clean_text(text)
        assert result == "hello world how are you"
    
    def test_clean_text_remove_url(self, processor):
        """测试文本清洗 - 移除URL。"""
        text = "Check out https://example.com for more info"
        result = processor.clean_text(text)
        assert "https" not in result
        assert "example" not in result
    
    def test_clean_text_remove_html(self, processor):
        """测试文本清洗 - 移除HTML标签。"""
        text = "<p>This is a paragraph</p>"
        result = processor.clean_text(text)
        assert "<p>" not in result
        assert "</p>" not in result
        assert "paragraph" in result
    
    def test_clean_text_normalize_whitespace(self, processor):
        """测试文本清洗 - 规范化空格。"""
        text = "Hello    world   test"
        result = processor.clean_text(text)
        assert result == "hello world test"
    
    def test_remove_stopwords(self, processor):
        """测试停用词移除。"""
        text = "this is test of the stopwords removal"
        result = processor.remove_stopwords(text)
        assert "this" not in result
        assert "is" not in result
        assert "test" in result
    
    def test_preprocess_single_text(self, processor):
        """测试单条文本预处理。"""
        text = "This is an AMAZING movie!!!"
        result = processor.preprocess(text)
        assert "amazing" in result
        assert "movie" in result
        assert "this" not in result
        assert "is" not in result
        assert "an" not in result
    
    def test_preprocess_multiple_texts(self, processor):
        """测试多条文本预处理。"""
        texts = [
            "This is great!",
            "That was terrible..."
        ]
        results = processor.preprocess(texts)
        assert len(results) == 2
        assert "great" in results[0]
        assert "terrible" in results[1]
    
    def test_custom_config_no_lowercase(self, custom_processor):
        """测试自定义配置 - 不转换小写。"""
        text = "HELLO World"
        result = custom_processor.clean_text(text)
        assert "HELLO" in result
    
    def test_fit_transform(self, processor):
        """测试拟合和转换。"""
        texts = [
            "This movie is great",
            "This movie is terrible",
            "Great acting and plot"
        ]
        features = processor.fit_transform(texts)
        
        assert features.shape[0] == 3  # 3条文本
        assert features.shape[1] > 0   # 有特征
        assert processor.vectorizer is not None
    
    def test_transform_without_fit(self, processor):
        """测试未拟合就转换应抛出异常。"""
        texts = ["This is a test"]
        
        with pytest.raises(ValueError, match="向量化器尚未拟合"):
            processor.transform(texts)
    
    def test_transform_after_fit(self, processor):
        """测试拟合后转换。"""
        train_texts = ["This is great", "That is bad"]
        test_texts = ["This is amazing"]
        
        processor.fit_transform(train_texts)
        features = processor.transform(test_texts)
        
        assert features.shape[0] == 1
        assert features.shape[1] > 0
    
    def test_get_feature_names(self, processor):
        """测试获取特征名称。"""
        texts = ["great movie", "terrible film"]
        processor.fit_transform(texts)
        
        feature_names = processor.get_feature_names()
        assert feature_names is not None
        assert len(feature_names) > 0
    
    def test_get_feature_names_before_fit(self, processor):
        """测试拟合前获取特征名称应返回None。"""
        feature_names = processor.get_feature_names()
        assert feature_names is None
    
    def test_empty_text(self, processor):
        """测试空文本处理。"""
        result = processor.clean_text("")
        assert result == ""
    
    def test_non_string_input(self, processor):
        """测试非字符串输入。"""
        result = processor.clean_text(123)
        assert "123" in result
