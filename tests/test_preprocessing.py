import pytest
from sentiment_analysis.preprocessing.text_processor import TextProcessor


class TestTextProcessor:
    def setup_method(self):
        self.processor = TextProcessor()
    
    def test_preprocess_text_lowercase(self):
        text = "HELLO World"
        result = self.processor.preprocess_text(text)
        assert result == "hello world"
    
    def test_preprocess_text_remove_special_chars(self):
        text = "Hello! How are you? #123"
        result = self.processor.preprocess_text(text)
        assert result == "hello how are you"
    
    def test_preprocess_text_remove_extra_whitespace(self):
        text = "Hello    world   test"
        result = self.processor.preprocess_text(text)
        assert result == "hello world test"
    
    def test_preprocess_text_empty_string(self):
        text = ""
        result = self.processor.preprocess_text(text)
        assert result == ""
    
    def test_preprocess_text_invalid_input(self):
        with pytest.raises(ValueError):
            self.processor.preprocess_text(123)
    
    def test_preprocess_batch(self):
        texts = ["Hello World!", "Test 123"]
        results = self.processor.preprocess_batch(texts)
        assert len(results) == 2
        assert results[0] == "hello world"
        assert results[1] == "test"
    
    def test_cache_clear(self):
        self.processor.preprocess_text("test")
        self.processor.clear_cache()
        assert self.processor._preprocess_cached.cache_info().currsize == 0
    
    def test_cache_enable_disable(self):
        self.processor.enable_cache(False)
        assert self.processor._cache_enabled is False
        
        self.processor.enable_cache(True)
        assert self.processor._cache_enabled is True
