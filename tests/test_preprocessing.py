import pytest
from sentiment_analysis.preprocessing.text_processor import preprocess_text


def test_preprocess_text_lowercase():
    text = "This Is A TEST"
    result = preprocess_text(text)
    assert result == "this is a test"


def test_preprocess_text_remove_special_chars():
    text = "Hello! How are you? 123"
    result = preprocess_text(text)
    assert result == "hello how are you"


def test_preprocess_text_remove_extra_spaces():
    text = "  hello   world  "
    result = preprocess_text(text)
    assert result == "hello world"


def test_preprocess_text_cache():
    text = "Test caching"
    result1 = preprocess_text(text)
    result2 = preprocess_text(text)
    assert result1 == result2


def test_preprocess_text_empty_string():
    text = ""
    result = preprocess_text(text)
    assert result == ""
