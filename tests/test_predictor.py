import pytest
import os
import tempfile
from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.models.predictor import ModelPredictor, predict_sentiment, batch_predict
from sentiment_analysis.config import Config


class TestModelPredictor:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = ModelTrainer(self.temp_dir)
        self.trainer.train()
        
        ModelPredictor._instance = None
        ModelPredictor._model = None
        ModelPredictor._vectorizer = None
    
    def teardown_method(self):
        model_path = Config.get_model_path(self.temp_dir)
        vectorizer_path = Config.get_vectorizer_path(self.temp_dir)
        data_path = Config.get_data_path(self.temp_dir)
        
        for path in [model_path, vectorizer_path, data_path]:
            if os.path.exists(path):
                os.remove(path)
        
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_predict_positive_sentiment(self):
        predictor = ModelPredictor(self.temp_dir)
        sentiment, confidence = predictor.predict("This movie is amazing and wonderful!")
        
        assert sentiment == "positive"
        assert 0 <= confidence <= 1
    
    def test_predict_negative_sentiment(self):
        predictor = ModelPredictor(self.temp_dir)
        sentiment, confidence = predictor.predict("Terrible movie, waste of time!")
        
        assert sentiment == "negative"
        assert 0 <= confidence <= 1
    
    def test_predict_invalid_input(self):
        predictor = ModelPredictor(self.temp_dir)
        
        with pytest.raises(ValueError):
            predictor.predict(123)
    
    def test_predict_batch(self):
        predictor = ModelPredictor(self.temp_dir)
        texts = [
            "Great movie!",
            "Terrible film!",
            "It was okay."
        ]
        
        results = predictor.predict_batch(texts)
        
        assert len(results) == 3
        for result in results:
            assert 'text' in result
            assert 'sentiment' in result
            assert 'confidence' in result
            assert result['sentiment'] in ['positive', 'negative']
    
    def test_load_model_caching(self):
        predictor = ModelPredictor(self.temp_dir)
        
        model1, vec1 = predictor.load_model()
        model2, vec2 = predictor.load_model()
        
        assert model1 is model2
        assert vec1 is vec2
    
    def test_clear_cache(self):
        predictor = ModelPredictor(self.temp_dir)
        predictor.load_model()
        
        predictor.clear_cache()
        
        assert predictor._model is None
        assert predictor._vectorizer is None
    
    def test_predict_sentiment_function(self):
        sentiment, confidence = predict_sentiment(
            "This is a great movie!",
            base_dir=self.temp_dir
        )
        
        assert sentiment in ['positive', 'negative']
        assert 0 <= confidence <= 1
    
    def test_batch_predict_function(self):
        texts = ["Great!", "Terrible!"]
        results = batch_predict(texts, base_dir=self.temp_dir)
        
        assert len(results) == 2
        for result in results:
            assert 'sentiment' in result
            assert 'confidence' in result
