"""文本预处理模块。

提供文本清洗、标准化和向量化功能。
"""

import re
import string
from typing import List, Optional, Union
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from sentiment_analysis.config import PreprocessingConfig, ModelConfig
from sentiment_analysis.utils.logger import LoggerMixin


class TextProcessor(LoggerMixin):
    """文本处理器类。
    
    负责文本的清洗、标准化和向量化。
    
    Attributes:
        prep_config: 预处理配置
        model_config: 模型配置
        vectorizer: TF-IDF向量化器
    """
    
    # 常用停用词
    DEFAULT_STOPWORDS = {
        'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
        'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
        'to', 'was', 'will', 'with', 'i', 'you', 'your', 'this', 'but',
        'they', 'have', 'had', 'what', 'said', 'each', 'which', 'she',
        'do', 'how', 'their', 'if', 'been', 'being', 'have', 'has', 'had'
    }
    
    def __init__(
        self,
        prep_config: Optional[PreprocessingConfig] = None,
        model_config: Optional[ModelConfig] = None
    ):
        """初始化文本处理器。
        
        Args:
            prep_config: 预处理配置
            model_config: 模型配置
        """
        self.prep_config = prep_config or PreprocessingConfig()
        self.model_config = model_config or ModelConfig()
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.logger.info("文本处理器初始化完成")
    
    def clean_text(self, text: str) -> str:
        """清洗单个文本。
        
        Args:
            text: 原始文本
            
        Returns:
            清洗后的文本
        """
        if not isinstance(text, str):
            text = str(text)
        
        # 转换为小写
        if self.prep_config.lowercase:
            text = text.lower()
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # 移除HTML标签
        text = re.sub(r'<.*?>', '', text)
        
        # 移除标点符号
        if self.prep_config.remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # 移除非字母数字字符（保留空格）
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        
        # 规范化空格
        text = ' '.join(text.split())
        
        return text
    
    def remove_stopwords(self, text: str) -> str:
        """移除停用词。
        
        Args:
            text: 清洗后的文本
            
        Returns:
            移除停用词后的文本
        """
        if not self.prep_config.remove_stopwords:
            return text
        
        words = text.split()
        filtered_words = [
            word for word in words
            if word.lower() not in self.DEFAULT_STOPWORDS
            and len(word) >= self.prep_config.min_word_length
        ]
        return ' '.join(filtered_words)
    
    def preprocess(self, text: Union[str, List[str]]) -> Union[str, List[str]]:
        """预处理文本（清洗+停用词移除）。
        
        Args:
            text: 单个文本或文本列表
            
        Returns:
            预处理后的文本
        """
        if isinstance(text, str):
            cleaned = self.clean_text(text)
            return self.remove_stopwords(cleaned)
        
        return [self.preprocess(t) for t in text]
    
    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """拟合向量化器并转换文本。
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
        """
        self.logger.info("开始拟合向量化器")
        
        # 预处理文本
        processed_texts = self.preprocess(texts)
        
        # 创建并拟合向量化器
        self.vectorizer = TfidfVectorizer(
            max_features=self.model_config.max_features,
            ngram_range=self.model_config.ngram_range,
            preprocessor=None,  # 我们已经预处理了
            tokenizer=None,
            stop_words=None  # 我们已经移除了停用词
        )
        
        features = self.vectorizer.fit_transform(processed_texts)
        
        self.logger.info(
            f"向量化完成: 特征维度={features.shape}, "
            f"词汇表大小={len(self.vectorizer.vocabulary_)}"
        )
        
        return features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """使用已拟合的向量化器转换文本。
        
        Args:
            texts: 文本列表
            
        Returns:
            TF-IDF特征矩阵
            
        Raises:
            ValueError: 向量化器未拟合时抛出
        """
        if self.vectorizer is None:
            raise ValueError("向量化器尚未拟合，请先调用fit_transform")
        
        # 预处理文本
        processed_texts = self.preprocess(texts)
        
        return self.vectorizer.transform(processed_texts)
    
    def get_feature_names(self) -> Optional[List[str]]:
        """获取特征名称（词汇表）。
        
        Returns:
            特征名称列表
        """
        if self.vectorizer is None:
            return None
        return self.vectorizer.get_feature_names_out().tolist()
