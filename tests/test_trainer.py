import pytest
import os
import tempfile
from sentiment_analysis.models.trainer import ModelTrainer
from sentiment_analysis.config import Config


class TestModelTrainer:
    def setup_method(self):
        self.temp_dir = tempfile.mkdtemp()
        self.trainer = ModelTrainer(self.temp_dir)
    
    def teardown_method(self):
        model_path = Config.get_model_path(self.temp_dir)
        vectorizer_path = Config.get_vectorizer_path(self.temp_dir)
        data_path = Config.get_data_path(self.temp_dir)
        
        for path in [model_path, vectorizer_path, data_path]:
            if os.path.exists(path):
                os.remove(path)
        
        if os.path.exists(self.temp_dir):
            os.rmdir(self.temp_dir)
    
    def test_train_creates_model_files(self):
        model, vectorizer = self.trainer.train()
        
        assert model is not None
        assert vectorizer is not None
        
        model_path = Config.get_model_path(self.temp_dir)
        vectorizer_path = Config.get_vectorizer_path(self.temp_dir)
        
        assert os.path.exists(model_path)
        assert os.path.exists(vectorizer_path)
    
    def test_train_model_accuracy(self):
        model, vectorizer = self.trainer.train()
        
        test_text = "This movie is great!"
        test_vec = vectorizer.transform([test_text])
        prediction = model.predict(test_vec)
        
        assert prediction[0] in ['positive', 'negative']
    
    def test_get_model_before_train(self):
        model, vectorizer = self.trainer.get_model()
        
        assert model is None
        assert vectorizer is None
    
    def test_get_model_after_train(self):
        self.trainer.train()
        model, vectorizer = self.trainer.get_model()
        
        assert model is not None
        assert vectorizer is not None
