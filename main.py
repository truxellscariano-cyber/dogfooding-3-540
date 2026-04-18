from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.models.predictor import predict_sentiment, batch_predict
from sentiment_analysis.utils.logger import setup_logger


def main():
    logger = setup_logger()
    
    logger.info("=" * 50)
    logger.info("Sentiment Analysis System")
    logger.info("=" * 50)
    
    trainer = ModelTrainer()
    model, vectorizer = trainer.train()
    
    logger.info("\n=== Testing Predictions ===")
    test_reviews = [
        "This is an amazing movie with great acting!",
        "Terrible waste of time, very disappointing.",
        "Not bad, but could be better.",
    ]
    
    for review in test_reviews:
        sentiment, confidence = predict_sentiment(review, model, vectorizer)
        logger.info(f"\nReview: {review}")
        logger.info(f"Sentiment: {sentiment}")
        logger.info(f"Confidence: {confidence:.2%}")
    
    logger.info("\n=== Batch Prediction Test ===")
    batch_results = batch_predict(test_reviews)
    for result in batch_results:
        logger.info(f"\nText: {result['text']}")
        logger.info(f"Sentiment: {result['sentiment']}")
        logger.info(f"Confidence: {result['confidence']:.2%}")


if __name__ == "__main__":
    main()
