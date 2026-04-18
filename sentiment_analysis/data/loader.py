"""数据加载模块。

提供数据集加载和准备功能。
"""

from typing import Tuple, Optional
import numpy as np
from sklearn.datasets import load_files
from sklearn.model_selection import train_test_split

from sentiment_analysis.config import TrainingConfig
from sentiment_analysis.utils.logger import LoggerMixin


class DataLoader(LoggerMixin):
    """数据加载器类。
    
    负责加载和分割数据集。
    
    Attributes:
        config: 训练配置
        texts: 文本数据
        labels: 标签数据
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """初始化数据加载器。
        
        Args:
            config: 训练配置，如果为None则使用默认配置
        """
        self.config = config or TrainingConfig()
        self.texts: Optional[np.ndarray] = None
        self.labels: Optional[np.ndarray] = None
        self.logger.info("数据加载器初始化完成")
    
    def load_from_files(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """从文件目录加载数据。
        
        Args:
            data_dir: 数据目录路径，应包含以类别命名的子目录
            
        Returns:
            文本数组和标签数组的元组
            
        Raises:
            FileNotFoundError: 数据目录不存在时抛出
            ValueError: 数据加载失败时抛出
        """
        try:
            self.logger.info(f"开始从目录加载数据: {data_dir}")
            data = load_files(data_dir, encoding="utf-8", decode_error="ignore")
            self.texts = np.array(data.data)
            self.labels = np.array(data.target)
            self.logger.info(
                f"数据加载完成: 共 {len(self.texts)} 条样本, "
                f"类别: {data.target_names}"
            )
            return self.texts, self.labels
        except FileNotFoundError:
            self.logger.error(f"数据目录不存在: {data_dir}")
            raise
        except Exception as e:
            self.logger.error(f"数据加载失败: {str(e)}")
            raise ValueError(f"数据加载失败: {str(e)}")
    
    def load_sample_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """加载示例数据用于测试。
        
        Returns:
            文本数组和标签数组的元组
        """
        self.logger.info("加载示例数据")
        
        sample_texts = [
            # 正面评论
            "This movie is absolutely amazing! Great acting and plot.",
            "I loved this film. Best movie I've seen this year!",
            "Fantastic performance by the lead actor. Highly recommended!",
            "Wonderful story and beautiful cinematography. A masterpiece!",
            "Excellent direction and brilliant screenplay. Must watch!",
            "Outstanding movie with great character development.",
            "Incredible film! The visuals are stunning.",
            "Perfect blend of action and emotion. Loved it!",
            "Superb acting and engaging storyline throughout.",
            "A truly remarkable cinematic experience. Five stars!",
            # 负面评论
            "Terrible movie. Complete waste of time.",
            "I hated this film. Poor acting and boring plot.",
            "Awful direction and weak storyline. Don't watch!",
            "Disappointing performance from the cast. Very bad.",
            "Boring and predictable. Not worth your money.",
            "Horrible movie with terrible special effects.",
            "Dull and uninteresting. Fell asleep halfway through.",
            "Worst film I've ever seen. Avoid at all costs!",
            "Pathetic attempt at filmmaking. Complete disaster.",
            "Unbearable to watch. Poorly executed in every aspect.",
        ]
        
        # 0 = 负面, 1 = 正面
        sample_labels = [1] * 10 + [0] * 10
        
        self.texts = np.array(sample_texts)
        self.labels = np.array(sample_labels)
        
        self.logger.info(f"示例数据加载完成: 共 {len(self.texts)} 条样本")
        return self.texts, self.labels
    
    def split_data(
        self,
        texts: Optional[np.ndarray] = None,
        labels: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """分割数据集为训练集和测试集。
        
        Args:
            texts: 文本数组，如果为None则使用已加载的数据
            labels: 标签数组，如果为None则使用已加载的数据
            
        Returns:
            训练文本、测试文本、训练标签、测试标签的元组
            
        Raises:
            ValueError: 没有可用数据时抛出
        """
        texts = texts if texts is not None else self.texts
        labels = labels if labels is not None else self.labels
        
        if texts is None or labels is None:
            raise ValueError("没有可用的数据，请先加载数据")
        
        self.logger.info(
            f"分割数据集: 测试比例={self.config.test_size}, "
            f"随机种子={self.config.random_state}"
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            texts,
            labels,
            test_size=self.config.test_size,
            random_state=self.config.random_state,
            stratify=labels
        )
        
        self.logger.info(
            f"数据分割完成: 训练集={len(X_train)}, 测试集={len(X_test)}"
        )
        
        return X_train, X_test, y_train, y_test
