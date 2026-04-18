"""配置管理模块。

集中管理所有配置参数，包括模型参数、文件路径等。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class ModelConfig:
    """模型配置类。
    
    Attributes:
        max_features: TF-IDF最大特征数
        ngram_range: N-gram范围
        alpha: 朴素贝叶斯平滑参数
    """
    max_features: int = 5000
    ngram_range: Tuple[int, int] = (1, 2)
    alpha: float = 1.0


@dataclass
class PathConfig:
    """路径配置类。
    
    Attributes:
        model_dir: 模型保存目录
        model_file: 模型文件名
        vectorizer_file: 向量化器文件名
        data_dir: 数据目录
        log_dir: 日志目录
    """
    model_dir: Path = field(default_factory=lambda: Path("models"))
    model_file: str = "sentiment_model.pkl"
    vectorizer_file: str = "vectorizer.pkl"
    data_dir: Path = field(default_factory=lambda: Path("data"))
    log_dir: Path = field(default_factory=lambda: Path("logs"))


@dataclass
class PreprocessingConfig:
    """预处理配置类。
    
    Attributes:
        remove_stopwords: 是否移除停用词
        lowercase: 是否转换为小写
        remove_punctuation: 是否移除标点符号
        min_word_length: 最小词长度
    """
    remove_stopwords: bool = True
    lowercase: bool = True
    remove_punctuation: bool = True
    min_word_length: int = 2


@dataclass
class TrainingConfig:
    """训练配置类。
    
    Attributes:
        test_size: 测试集比例
        random_state: 随机种子
        cv_folds: 交叉验证折数
    """
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


@dataclass
class LoggingConfig:
    """日志配置类。
    
    Attributes:
        level: 日志级别
        format: 日志格式
        file_enabled: 是否启用文件日志
        console_enabled: 是否启用控制台日志
    """
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_enabled: bool = True
    console_enabled: bool = True


class Config:
    """主配置类。
    
    集中管理所有配置参数。
    
    Attributes:
        model: 模型配置
        paths: 路径配置
        preprocessing: 预处理配置
        training: 训练配置
        logging: 日志配置
    """
    
    def __init__(
        self,
        model: ModelConfig = None,
        paths: PathConfig = None,
        preprocessing: PreprocessingConfig = None,
        training: TrainingConfig = None,
        logging_config: LoggingConfig = None
    ):
        self.model = model or ModelConfig()
        self.paths = paths or PathConfig()
        self.preprocessing = preprocessing or PreprocessingConfig()
        self.training = training or TrainingConfig()
        self.logging = logging_config or LoggingConfig()
        
        # 确保目录存在
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """确保必要的目录存在。"""
        self.paths.model_dir.mkdir(parents=True, exist_ok=True)
        self.paths.data_dir.mkdir(parents=True, exist_ok=True)
        self.paths.log_dir.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_path(self) -> Path:
        """获取模型文件完整路径。"""
        return self.paths.model_dir / self.paths.model_file
    
    @model_path.setter
    def model_path(self, value: Path) -> None:
        """设置模型文件路径。"""
        self.paths.model_dir = value.parent
        self.paths.model_file = value.name
    
    @property
    def vectorizer_path(self) -> Path:
        """获取向量化器文件完整路径。"""
        return self.paths.model_dir / self.paths.vectorizer_file
    
    @vectorizer_path.setter
    def vectorizer_path(self, value: Path) -> None:
        """设置向量化器文件路径。"""
        # 向量化器和模型保存在同一目录
        pass


# 默认配置实例
default_config = Config()
