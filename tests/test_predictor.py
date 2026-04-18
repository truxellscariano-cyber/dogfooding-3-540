import pytest
from sentiment_analysis.models.predictor import predict_sentiment, batch_predict, clear_cache


def test_predict_sentiment_valid_input():
    text = "This movie is amazing!"
    sentiment, prob = predict_sentiment(text)
    assert sentiment in ['positive', 'negative']
    assert len(prob) == 2


def test_predict_sentiment_empty_text():
    with pytest.raises(ValueError):
        predict_sentiment("")


def test_predict_sentiment_none_text():
    with pytest.raises(ValueError):
        predict_sentiment(None)


def test_batch_predict():
    texts = ["Great movie!", "Terrible film!"]
    results = batch_predict(texts)
    assert len(results) == 2
    for result in results:
        assert 'text' in result
        assert 'sentiment' in result
        assert 'confidence' in result


def test_cache_clear():
    clear_cache()
