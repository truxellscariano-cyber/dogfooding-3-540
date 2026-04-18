# Sentiment Analysis System

一个基于机器学习的电影评论情感分析系统。使用 TF-IDF 特征提取和朴素贝叶斯分类器进行正面/负面情感分类。

## 功能特性

- **文本预处理**: 清洗、标准化和向量化文本数据
- **TF-IDF 向量化**: 将文本转换为数值特征
- **朴素贝叶斯分类**: 高效的情感分类算法
- **模型持久化**: 支持模型的保存和加载
- **批量预测**: 支持单条和批量文本预测
- **模型缓存**: 避免重复加载模型，提高性能
- **完整日志**: 使用 Python logging 模块记录运行信息
- **类型注解**: 全面的类型提示支持
- **单元测试**: 完整的测试覆盖

## 技术栈

- Python 3.10+
- scikit-learn (机器学习)
- pandas (数据处理)
- numpy (数值计算)
- pytest (测试框架)

## 项目结构

```
.
├── sentiment_analysis/          # 主包
│   ├── __init__.py
│   ├── config.py               # 配置管理
│   ├── data/                   # 数据模块
│   │   ├── __init__.py
│   │   └── loader.py           # 数据加载
│   ├── preprocessing/          # 预处理模块
│   │   ├── __init__.py
│   │   └── text_processor.py   # 文本处理
│   ├── models/                 # 模型模块
│   │   ├── __init__.py
│   │   ├── trainer.py          # 模型训练
│   │   └── predictor.py        # 模型预测
│   └── utils/                  # 工具模块
│       ├── __init__.py
│       └── logger.py           # 日志工具
├── tests/                      # 单元测试
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_trainer.py
│   └── test_predictor.py
├── main.py                     # 主入口
├── requirements.txt            # 依赖列表
└── README.md                   # 项目说明
```

## 安装

1. 克隆仓库:
```bash
git clone https://github.com/truxellscariano-cyber/dogfooding-3-540.git
cd dogfooding-3-540
```

2. 安装依赖:
```bash
pip install -r requirements.txt
```

## 使用方法

### 命令行工具

#### 训练模型

使用示例数据训练:
```bash
python main.py train
```

从自定义数据目录训练（数据目录应包含以类别命名的子目录）:
```bash
python main.py train --data-dir ./my_data
```

#### 预测单条文本

```bash
python main.py predict "This movie is absolutely amazing!"
```

#### 批量预测

```bash
python main.py batch "Great movie!" "Terrible film." "I loved it!"
```

#### 交互式模式

```bash
python main.py interactive
```

### Python API

#### 基本用法

```python
from sentiment_analysis import predict_sentiment

# 预测单条文本
sentiment, confidence = predict_sentiment("This movie is amazing!")
print(f"情感: {sentiment}, 置信度: {confidence:.2f}")
```

#### 高级用法

```python
from sentiment_analysis import Predictor, DataLoader, ModelTrainer, Config

# 创建配置
config = Config()

# 加载数据
data_loader = DataLoader()
texts, labels = data_loader.load_sample_data()
X_train, X_test, y_train, y_test = data_loader.split_data()

# 训练模型
trainer = ModelTrainer(config)
trainer.train(X_train, y_train)
trainer.save()

# 评估模型
eval_results = trainer.evaluate(X_test, y_test)
print(f"准确率: {eval_results['accuracy']:.4f}")

# 预测
predictor = Predictor(config)
result = predictor.predict("This is a great movie!")
print(f"情感: {result['sentiment']}")
print(f"置信度: {result['confidence']:.4f}")

# 批量预测
texts = ["Great!", "Terrible!", "Amazing!"]
results = predictor.predict_batch(texts)
for r in results:
    print(f"{r['text']}: {r['sentiment']}")
```

## 配置

所有配置参数集中在 `sentiment_analysis/config.py` 中:

```python
from sentiment_analysis.config import Config, ModelConfig

# 自定义配置
model_config = ModelConfig(
    max_features=10000,      # TF-IDF 最大特征数
    ngram_range=(1, 3),      # N-gram 范围
    alpha=0.5                # 朴素贝叶斯平滑参数
)

config = Config(model=model_config)
```

### 配置选项

**ModelConfig**:
- `max_features`: TF-IDF 最大特征数 (默认: 5000)
- `ngram_range`: N-gram 范围 (默认: (1, 2))
- `alpha`: 朴素贝叶斯平滑参数 (默认: 1.0)

**PreprocessingConfig**:
- `remove_stopwords`: 是否移除停用词 (默认: True)
- `lowercase`: 是否转换为小写 (默认: True)
- `remove_punctuation`: 是否移除标点符号 (默认: True)
- `min_word_length`: 最小词长度 (默认: 2)

**TrainingConfig**:
- `test_size`: 测试集比例 (默认: 0.2)
- `random_state`: 随机种子 (默认: 42)
- `cv_folds`: 交叉验证折数 (默认: 5)

## 运行测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试文件
pytest tests/test_preprocessing.py
pytest tests/test_trainer.py
pytest tests/test_predictor.py

# 生成覆盖率报告
pytest tests/ --cov=sentiment_analysis --cov-report=html
```

## 日志

日志文件保存在 `logs/` 目录下，包含详细的运行信息和错误记录。

## 改进点

相比原始单文件实现，重构后的版本有以下改进:

1. **模块化架构**: 将功能拆分为独立的模块，职责清晰
2. **配置管理**: 集中管理所有配置参数
3. **日志系统**: 使用标准 logging 模块替代 print
4. **错误处理**: 完善的异常处理和输入验证
5. **类型注解**: 全面的类型提示
6. **单元测试**: 完整的测试覆盖
7. **模型缓存**: 避免重复加载模型
8. **批量处理**: 优化的批量预测流程
9. **代码规范**: 遵循 PEP 8 规范
10. **文档**: 详细的 docstring 和注释

## 许可证

MIT License
