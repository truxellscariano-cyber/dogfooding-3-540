import os
import tempfile
import pytest
from sentiment_analysis.config import Config
from sentiment_analysis.models.trainer import train_model
from sentiment_analysis.data.loader import load_data


def test_load_data():
    df = load_data()
    assert len(df) > 0
    assert 'review' in df.columns
    assert 'sentiment' in df.columns


def test_train_model():
    model, vectorizer = train_model()
    assert model is not None
    assert vectorizer is not None
    assert os.path.exists(Config.MODEL_FILE)
    assert os.path.exists(Config.VECTORIZER_FILE)
