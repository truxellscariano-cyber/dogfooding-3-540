import os


class Config:
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    DATA_FILE = os.path.join(BASE_DIR, 'movie_reviews.csv')
    MODEL_FILE = os.path.join(BASE_DIR, 'sentiment_model.pkl')
    VECTORIZER_FILE = os.path.join(BASE_DIR, 'vectorizer.pkl')
    
    TFIDF_MAX_FEATURES = 5000
    TFIDF_NGRAM_RANGE = (1, 2)
    
    TEST_SIZE = 0.2
    RANDOM_STATE = 42
    
    LOG_LEVEL = 'INFO'
    LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
