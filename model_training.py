# model_training.py
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

# Load dataset
df = pd.read_csv("data.csv", on_bad_lines='skip')
df = df.dropna()

# Features and labels
X = df['password']
y = df['strength']

# TF-IDF + Logistic Regression Pipeline
model = Pipeline([
    ('tfidf', TfidfVectorizer(analyzer='char', ngram_range=(2, 3))),
    ('clf', LogisticRegression(max_iter=1000))
])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'password_strength_model.pkl')

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model trained successfully ✅ | Accuracy: {accuracy:.2f}")
