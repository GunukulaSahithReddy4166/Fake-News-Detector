"""
Fake News Detection Model Training Script
==========================================

This script trains a machine learning model to classify news articles as Real or Fake.
It uses TF-IDF vectorization for text feature extraction and Logistic Regression for classification.

How it works:
1. Load the news dataset from CSV
2. Preprocess the text (clean, lowercase, remove punctuation/stopwords)
3. Convert text to TF-IDF vectors
4. Train Logistic Regression classifier
5. Save the trained model and vectorizer for use in the web app
"""

import pandas as pd
import numpy as np
import re
import string
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ============================================================================
# STEP 1: Define text preprocessing functions
# ============================================================================

def clean_text(text):
    """
    Clean and preprocess news text.
    
    Steps:
    - Convert to lowercase
    - Remove punctuation
    - Remove extra whitespace
    - Remove numbers
    
    Args:
        text: Raw news text string
    Returns:
        Cleaned text string
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    return text


# ============================================================================
# STEP 2: Load the dataset
# ============================================================================

print("=" * 60)
print("FAKE NEWS DETECTION - MODEL TRAINING")
print("=" * 60)
print("\n📂 Loading dataset...")

# Load news dataset from CSV file
# The dataset should have two columns: 'text' (news content) and 'label' (real/fake)
df = pd.read_csv("news.csv")

# Display dataset information
print(f"   Total samples: {len(df)}")
print(f"   Real news: {len(df[df['label'] == 'real'])}")
print(f"   Fake news: {len(df[df['label'] == 'fake'])}")

# Check for missing values
df = df.dropna()
print(f"   After cleaning: {len(df)} samples")

# ============================================================================
# STEP 3: Preprocess the text data
# ============================================================================

print("\n🔧 Preprocessing text data...")

# Apply text cleaning to all news articles
df['cleaned_text'] = df['text'].apply(clean_text)

# Remove any empty results after cleaning
df = df[df['cleaned_text'].str.len() > 0]

print(f"   Preprocessing complete! {len(df)} samples ready for training")

# ============================================================================
# STEP 4: Split data into training and testing sets
# ============================================================================

print("\n📊 Splitting data into train/test sets...")

# Separate features (text) and labels (real/fake)
X = df['cleaned_text']
y = df['label']

# Split data: 80% training, 20% testing
# random_state ensures reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y  # Maintain balance of real/fake in both sets
)

print(f"   Training samples: {len(X_train)}")
print(f"   Testing samples: {len(X_test)}")

# ============================================================================
# STEP 5: TF-IDF Vectorization
# ============================================================================

print("\n🔢 Creating TF-IDF features...")

# TF-IDF (Term Frequency-Inverse Document Frequency) converts text to numbers
# It weighs words by how important they are in the document and across all documents
# 
# How TF-IDF works:
# - TF (Term Frequency): How often a word appears in a document
# - IDF (Inverse Document Frequency): How rare/common the word is across all documents
# - TF-IDF = TF * IDF
#
# Parameters:
# - max_df=0.7: Ignore words that appear in more than 70% of documents (too common)
# - min_df=2: Ignore words that appear in fewer than 2 documents (too rare)
# - stop_words='english': Remove common English words (the, is, at, which, on)
# - ngram_range=(1,2): Use single words and pairs of words

vectorizer = TfidfVectorizer(
    max_df=0.7,
    min_df=2,
    stop_words='english',
    ngram_range=(1, 2)  # Unigrams and bigrams
)

# Fit on training data and transform both train and test
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"   Vocabulary size: {len(vectorizer.vocabulary_)}")
print(f"   Feature matrix shape: {X_train_tfidf.shape}")

# ============================================================================
# STEP 6: Train Logistic Regression Model
# ============================================================================

print("\n🤖 Training Logistic Regression model...")

# Logistic Regression is a classification algorithm that predicts binary outcomes
# It works well for text classification because:
# - It's fast to train
# - It provides probability scores
# - It handles high-dimensional data well (like TF-IDF vectors)
#
# Parameters:
# - max_iter=1000: Maximum iterations for finding the solution
# - random_state=42: For reproducibility
# - class_weight='balanced': Handle imbalanced classes

model = LogisticRegression(
    max_iter=1000,
    random_state=42,
    class_weight='balanced'
)

# Train the model on the TF-IDF vectors
model.fit(X_train_tfidf, y_train)

print("   Training complete!")

# ============================================================================
# STEP 7: Evaluate the Model
# ============================================================================

print("\n📈 Evaluating model performance...")

# Make predictions on test data
y_pred = model.predict(X_test_tfidf)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"   Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Detailed classification report
print("\n   Classification Report:")
print("   " + "-" * 50)
report = classification_report(y_test, y_pred)
for line in report.split('\n'):
    if line.strip():
        print("   " + line)

# ============================================================================
# STEP 8: Save the Model and Vectorizer
# ============================================================================

print("\n💾 Saving trained model and vectorizer...")

# Save the trained model using joblib
# This allows us to load it later in the Flask app
joblib.dump(model, "model.pkl")
print("   ✓ model.pkl saved successfully")

# Save the TF-IDF vectorizer
# We need this to transform new text the same way we transformed training data
joblib.dump(vectorizer, "vectorizer.pkl")
print("   ✓ vectorizer.pkl saved successfully")

print("\n" + "=" * 60)
print("✅ MODEL TRAINING COMPLETE!")
print("=" * 60)
print("\nNext steps:")
print("1. Run: python app.py")
print("2. Open: http://127.0.0.1:5000")
print("=" * 60)

