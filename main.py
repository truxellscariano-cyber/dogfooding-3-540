from sentiment_analysis import train_model, predict_sentiment
from sentiment_analysis.utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    logger.info("=" * 50)
    logger.info("Sentiment Analysis System")
    logger.info("=" * 50)
    
    model, vectorizer = train_model()
    
    logger.info("\n=== Testing Predictions ===")
    test_reviews = [
        "This is an amazing movie with great acting!",
        "Terrible waste of time, very disappointing.",
        "Not bad, but could be better.",
    ]
    
    for review in test_reviews:
        sentiment, prob = predict_sentiment(review, model, vectorizer)
        logger.info(f"\nReview: {review}")
        logger.info(f"Sentiment: {sentiment}")
        logger.info(f"Confidence: {max(prob):.2%}")


if __name__ == "__main__":
    main()
