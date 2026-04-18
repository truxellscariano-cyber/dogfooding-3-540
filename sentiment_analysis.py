import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import re
import os

# Global variables
DATA_FILE = 'movie_reviews.csv'
MODEL_FILE = 'sentiment_model.pkl'
VECTORIZER_FILE = 'vectorizer.pkl'

def load_data():
    """Load movie reviews dataset"""
    print("Loading data...")
    if not os.path.exists(DATA_FILE):
        # Create sample data if file doesn't exist
        data = {
            'review': [
                'This movie was absolutely fantastic! I loved every minute of it.',
                'Terrible film, waste of time and money.',
                'Great acting and amazing storyline. Highly recommended!',
                'Boring and predictable. Would not watch again.',
                'One of the best movies I have ever seen!',
                'Awful movie with poor acting and bad script.',
                'Excellent cinematography and compelling characters.',
                'Disappointing and overrated. Not worth watching.',
                'Brilliant performance by the lead actor!',
                'Worst movie of the year. Complete disaster.',
            ] * 50,  # Repeat to have more samples
            'sentiment': ['positive', 'negative', 'positive', 'negative', 'positive',
                         'negative', 'positive', 'negative', 'positive', 'negative'] * 50
        }
        df = pd.DataFrame(data)
        df.to_csv(DATA_FILE, index=False)
        print(f"Created sample dataset: {DATA_FILE}")
    else:
        df = pd.read_csv(DATA_FILE)

    print(f"Loaded {len(df)} reviews")
    return df

def preprocess_text(text):
    """Clean and preprocess text"""
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove extra whitespace
    text = ' '.join(text.split())
    return text

def train_model():
    """Train sentiment analysis model"""
    print("\n=== Training Model ===")

    # Load data
    df = load_data()

    # Preprocess
    print("Preprocessing text...")
    df['cleaned_review'] = df['review'].apply(preprocess_text)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned_review'], df['sentiment'], test_size=0.2, random_state=42
    )

    # Vectorize
    print("Vectorizing text...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # Train model
    print("Training classifier...")
    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    # Evaluate
    y_pred = model.predict(X_test_vec)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {accuracy:.2%}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # Save model
    print(f"\nSaving model to {MODEL_FILE}...")
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)

    print(f"Saving vectorizer to {VECTORIZER_FILE}...")
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)

    print("Training complete!")
    return model, vectorizer

def predict_sentiment(text, model=None, vectorizer=None):
    """Predict sentiment of a single text"""
    # Load model if not provided
    if model is None or vectorizer is None:
        if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
            print("Model not found. Training new model...")
            model, vectorizer = train_model()
        else:
            print("Loading saved model...")
            with open(MODEL_FILE, 'rb') as f:
                model = pickle.load(f)
            with open(VECTORIZER_FILE, 'rb') as f:
                vectorizer = pickle.load(f)

    # Preprocess and predict
    cleaned_text = preprocess_text(text)
    text_vec = vectorizer.transform([cleaned_text])
    prediction = model.predict(text_vec)[0]
    probability = model.predict_proba(text_vec)[0]

    return prediction, probability

def batch_predict(texts):
    """Predict sentiment for multiple texts"""
    print("\n=== Batch Prediction ===")

    # Load model
    if not os.path.exists(MODEL_FILE) or not os.path.exists(VECTORIZER_FILE):
        print("Model not found. Training new model...")
        model, vectorizer = train_model()
    else:
        print("Loading saved model...")
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)

    results = []
    for text in texts:
        prediction, probability = predict_sentiment(text, model, vectorizer)
        results.append({
            'text': text,
            'sentiment': prediction,
            'confidence': max(probability)
        })

    return results

def main():
    """Main function"""
    print("=" * 50)
    print("Sentiment Analysis System")
    print("=" * 50)

    # Train model
    model, vectorizer = train_model()

    # Test predictions
    print("\n=== Testing Predictions ===")
    test_reviews = [
        "This is an amazing movie with great acting!",
        "Terrible waste of time, very disappointing.",
        "Not bad, but could be better.",
    ]

    for review in test_reviews:
        sentiment, prob = predict_sentiment(review, model, vectorizer)
        print(f"\nReview: {review}")
        print(f"Sentiment: {sentiment}")
        print(f"Confidence: {max(prob):.2%}")

if __name__ == "__main__":
    main()
