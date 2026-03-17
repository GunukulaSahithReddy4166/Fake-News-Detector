

# train.py
# Train a simple Fake News detection model using TF-IDF + Logistic Regression

import pandas as pd


data = pd.read_csv("data.csv")  # <-- file name in quotes

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib
import argparse

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str, default="news.csv", help="CSV file with columns: text,label")
parser.add_argument("--out", type=str, default="model.pkl", help="Output model filename")
args = parser.parse_args()

# Load dataset
print(f"📂 Loading dataset: {args.data}")
df = pd.read_csv(args.data)

# Expecting columns: 'text' and 'label'
if not {"text", "label"}.issubset(df.columns):
    raise ValueError("Dataset must contain 'text' and 'label' columns")

# Drop missing values
df = df.dropna(subset=["text", "label"])

X = df["text"]
y = df["label"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Build pipeline (TF-IDF + Logistic Regression)
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=20000, stop_words="english")),
    ("clf", LogisticRegression(max_iter=1000))
])

# Train
print("🚀 Training model...")
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(pipeline, args.out)
print(f"💾 Model saved as {args.out}")
