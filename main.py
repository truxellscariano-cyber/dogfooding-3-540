"""主入口文件。

提供训练和预测功能的主入口。
"""

import argparse
import sys
from typing import Optional

from sentiment_analysis.config import Config
from sentiment_analysis.data.loader import DataLoader
from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.models.predictor import Predictor
from sentiment_analysis.utils.logger import setup_logger


def train_model(data_dir: Optional[str] = None) -> None:
    """训练情感分析模型。
    
    Args:
        data_dir: 数据目录，如果为None则使用示例数据
    """
    logger = setup_logger("main")
    logger.info("=" * 50)
    logger.info("开始训练情感分析模型")
    logger.info("=" * 50)
    
    # 初始化配置
    config = Config()
    
    # 加载数据
    data_loader = DataLoader(config.training)
    
    if data_dir:
        logger.info(f"从目录加载数据: {data_dir}")
        texts, labels = data_loader.load_from_files(data_dir)
    else:
        logger.info("使用示例数据")
        texts, labels = data_loader.load_sample_data()
    
    # 分割数据
    X_train, X_test, y_train, y_test = data_loader.split_data()
    
    # 训练模型
    trainer = ModelTrainer(config)
    train_info = trainer.train(X_train, y_train)
    
    logger.info("\n训练信息:")
    for key, value in train_info.items():
        logger.info(f"  {key}: {value}")
    
    # 评估模型
    logger.info("\n评估模型性能...")
    eval_results = trainer.evaluate(X_test, y_test)
    
    logger.info("\n评估结果:")
    logger.info(f"  准确率: {eval_results['accuracy']:.4f}")
    logger.info(f"  精确率: {eval_results['precision']:.4f}")
    logger.info(f"  召回率: {eval_results['recall']:.4f}")
    logger.info(f"  F1分数: {eval_results['f1_score']:.4f}")
    
    # 保存模型
    trainer.save()
    
    logger.info("\n" + "=" * 50)
    logger.info("模型训练完成并已保存")
    logger.info("=" * 50)


def predict_text(text: str) -> None:
    """预测单条文本的情感。
    
    Args:
        text: 输入文本
    """
    logger = setup_logger("main")
    
    try:
        predictor = Predictor()
        result = predictor.predict(text)
        
        print("\n" + "=" * 50)
        print("预测结果")
        print("=" * 50)
        print(f"文本: {result['text']}")
        print(f"情感: {result['sentiment'].upper()}")
        print(f"标签: {result['label']}")
        if 'confidence' in result:
            print(f"置信度: {result['confidence']:.4f}")
        if 'probabilities' in result:
            print(f"负面概率: {result['probabilities']['negative']:.4f}")
            print(f"正面概率: {result['probabilities']['positive']:.4f}")
        print("=" * 50 + "\n")
        
    except ValueError as e:
        logger.error(f"预测失败: {e}")
        logger.info("请先运行训练命令: python main.py train")
        sys.exit(1)


def predict_batch(texts: list) -> None:
    """批量预测文本情感。
    
    Args:
        texts: 文本列表
    """
    logger = setup_logger("main")
    
    try:
        predictor = Predictor()
        results = predictor.predict_batch(texts)
        
        print("\n" + "=" * 50)
        print("批量预测结果")
        print("=" * 50)
        
        for i, result in enumerate(results, 1):
            print(f"\n[{i}] {result['text'][:50]}...")
            print(f"    情感: {result['sentiment'].upper()}")
            if 'confidence' in result:
                print(f"    置信度: {result['confidence']:.4f}")
        
        print("\n" + "=" * 50)
        
    except ValueError as e:
        logger.error(f"预测失败: {e}")
        logger.info("请先运行训练命令: python main.py train")
        sys.exit(1)


def interactive_mode() -> None:
    """交互式预测模式。"""
    logger = setup_logger("main")
    
    try:
        predictor = Predictor()
        print("\n" + "=" * 50)
        print("情感分析系统 - 交互模式")
        print("输入文本进行预测，输入 'quit' 退出")
        print("=" * 50 + "\n")
        
        while True:
            text = input("请输入文本: ").strip()
            
            if text.lower() == 'quit':
                print("再见！")
                break
            
            if not text:
                print("请输入有效的文本\n")
                continue
            
            try:
                result = predictor.predict(text)
                print(f"情感: {result['sentiment'].upper()}")
                if 'confidence' in result:
                    print(f"置信度: {result['confidence']:.4f}")
                print()
            except Exception as e:
                print(f"预测出错: {e}\n")
                
    except ValueError as e:
        logger.error(f"预测器初始化失败: {e}")
        logger.info("请先运行训练命令: python main.py train")
        sys.exit(1)


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="情感分析系统 - 训练和预测工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  python main.py train                    # 使用示例数据训练模型
  python main.py train --data-dir ./data  # 从目录加载数据训练
  python main.py predict "This is great!" # 预测单条文本
  python main.py interactive              # 交互式预测模式
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument(
        '--data-dir',
        type=str,
        help='数据目录路径（包含以类别命名的子目录）'
    )
    
    # predict 命令
    predict_parser = subparsers.add_parser('predict', help='预测单条文本')
    predict_parser.add_argument(
        'text',
        type=str,
        help='要预测的文本'
    )
    
    # batch 命令
    batch_parser = subparsers.add_parser('batch', help='批量预测')
    batch_parser.add_argument(
        'texts',
        nargs='+',
        help='要预测的文本列表'
    )
    
    # interactive 命令
    subparsers.add_parser('interactive', help='交互式预测模式')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.data_dir)
    elif args.command == 'predict':
        predict_text(args.text)
    elif args.command == 'batch':
        predict_batch(args.texts)
    elif args.command == 'interactive':
        interactive_mode()
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
