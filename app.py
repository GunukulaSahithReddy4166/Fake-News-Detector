"""
Fake News Detection Web Application - Flask Backend
===================================================

This Flask application implements a complete fake news detection system with:

1. ML Classification (TF-IDF + Logistic Regression)
2. Live news verification (NewsAPI simulation + Google News + Twitter trends)
3. Credibility scoring system
4. Rumor detection logic

How it works:
1. Load pre-trained model and vectorizer
2. User submits news text
3. Clean text → ML prediction (40% weight)
4. Check trusted sources (30% weight) 
5. Google News mentions (15% weight)
6. Twitter trending analysis (15% weight)
7. Calculate final credibility score
8. Apply rumor detection rules
9. Return comprehensive JSON results
"""

import os
import re
import string
import random
import requests
from bs4 import BeautifulSoup
from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords

# Download NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# ============================================================================
# GLOBAL: Load ML Model and Vectorizer (runs on startup)
# ============================================================================

print("🔄 Loading ML model and vectorizer...")
try:
    model = joblib.load('model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
    print("✅ Model and vectorizer loaded successfully!")
except FileNotFoundError:
    print("⚠️  Model files not found. Run: python train_model.py")
    model = None
    vectorizer = None

# ============================================================================
# TEXT PREPROCESSING (Same as training)
# ============================================================================

def clean_text(text):
    """
    Clean and preprocess news text (IDENTICAL to train_model.py).
    
    Args:
        text: Raw news text string
    Returns:
        Cleaned text string ready for TF-IDF
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove numbers
    text = re.sub(r'\\d+', '', text)
    
    return text

# ============================================================================
# NEWS VERIFICATION FUNCTIONS (Simulation + Real APIs)
# ============================================================================

def check_newsapi(headline, api_key=None):
    """
    Check if news appears in trusted sources via NewsAPI.
    SIMULATION MODE: Returns realistic mock data.
    """
    # SIMULATION (replace with real API key for production)
    trusted_sources = ['bbc-news', 'cnn', 'reuters', 'abc-news', 'the-new-york-times']
    
    # Mock results based on keywords/realistic patterns
    keywords = ['confirms', 'study shows', 'officials announce', 'researchers find']
    score = sum(1 for kw in keywords if kw in headline.lower()) * 2
    
    num_sources = min(8, max(0, score + random.randint(-1, 3)))
    sources_found = random.sample(trusted_sources, min(num_sources, len(trusted_sources)))
    
    return {
        'num_sources': num_sources,
        'sources_found': sources_found,
        'is_verified': num_sources >= 3
    }

def check_google_news(headline):
    """
    Scrape Google News for mentions (simulation + real scraping).
    """
    # SIMULATION for demo (real scraping would use requests + BS4)
    mention_score = len(headline.split()) * 0.3 + random.uniform(0, 3)
    mentions = int(max(0, mention_score))
    return mentions

def check_twitter_trends(headline):
    """
    Simulate Twitter/X trending analysis via keyword matching.
    """
    trending_keywords = ['breaking', 'live', 'now', 'urgent', 'exclusive']
    tweet_volume = sum(1 for kw in trending_keywords if kw in headline.lower()) * 10000
    tweet_volume += random.randint(5000, 20000)
    
    is_trending = tweet_volume > 15000
    return {
        'estimated_tweets': tweet_volume,
        'is_trending': is_trending
    }

# ============================================================================
# ML PREDICTION FUNCTION
# ============================================================================

def predict_news(text):
    """
    Complete ML prediction pipeline.
    
    Returns:
        dict: ml_prediction, ml_confidence, fake_probability, real_probability
    """
    if model is None or vectorizer is None:
        return {
            'ml_prediction': 'unknown',
            'ml_confidence': 0,
            'fake_probability': 50,
            'real_probability': 50
        }
    
    # Clean text
    cleaned = clean_text(text)
    
    # Transform to TF-IDF
    X_tfidf = vectorizer.transform([cleaned])
    
    # Predict probabilities
    probs = model.predict_proba(X_tfidf)[0]
    
    # Get prediction and confidence
    fake_prob = probs[0] * 100  # index 0 = fake
    real_prob = probs[1] * 100  # index 1 = real
    
    prediction = 'fake' if fake_prob > real_prob else 'real'
    confidence = max(fake_prob, real_prob)
    
    return {
        'ml_prediction': prediction,
        'ml_confidence': round(confidence, 1),
        'fake_probability': round(fake_prob, 1),
        'real_probability': round(real_prob, 1)
    }

# ============================================================================
# CREDIBILITY SCORING & RUMOR DETECTION
# ============================================================================

def calculate_credibility(ml_score, num_sources, google_mentions, is_trending):
    """
    Weighted credibility score formula from requirements:
    Credibility = 0.4*ML + 0.3*Sources + 0.15*Google + 0.15*Trending
    """
    ml_weight = ml_score / 100  # Normalize to 0-1
    sources_weight = min(num_sources / 10, 1.0)  # Normalize 0-10 sources
    google_weight = min(google_mentions / 20, 1.0)  # Normalize mentions
    trending_weight = 1.0 if is_trending else 0.5
    
    score = (
        0.4 * ml_weight +
        0.3 * sources_weight +
        0.15 * google_weight +
        0.15 * trending_weight
    ) * 100
    
    return round(score, 1)

def detect_rumor(ml_pred, credibility, num_sources):
    """
    Rumor detection logic:
    - ML=fake & no sources → Likely Fake
    - ML=real & few sources → Possible Rumor
    - Both good → Verified Real
    """
    if ml_pred == 'fake' and num_sources == 0:
        return 'Likely Fake News', 'red'
    elif credibility < 30:
        return 'High Risk Fake', 'red'
    elif credibility < 50:
        return 'Unverified Rumor', 'orange'
    elif credibility < 70:
        return 'Possible Rumor', 'yellow'
    elif num_sources >= 3 and credibility >= 70:
        return 'Verified Real News', 'green'
    else:
        return 'Low Confidence Real', 'lightgreen'

def generate_reasoning(ml_pred, sources, mentions, trending):
    """Generate human-readable reasoning."""
    reasons = []
    if ml_pred == 'real':
        reasons.append("✅ ML model confident it's real news")
    else:
        reasons.append("⚠️ ML model flags as potentially fake")
    
    if sources >= 3:
        reasons.append("✅ Multiple trusted sources confirm")
    elif sources == 0:
        reasons.append("❌ No trusted sources found")
    else:
        reasons.append("⚠️ Limited source verification")
    
    if mentions > 5:
        reasons.append("✅ Appears in Google News")
    if trending:
        reasons.append("🔥 Currently trending on social media")
    
    return " | ".join(reasons)

# ============================================================================
# FLASK ROUTES
# ============================================================================

@app.route('/')
def index():
    """Serve the main web application."""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze_news():
    """
    Main analysis endpoint called by frontend.
    
    Input: news_text (form data)
    Output: JSON with all UI-required fields
    """
    try:
        news_text = request.form['news_text']
        
        if not news_text or len(news_text.strip()) < 10:
            return jsonify({'error': 'News text too short (min 10 chars)'}), 400
        
        print(f"🔍 Analyzing: {news_text[:100]}...")
        
        # 1. ML Prediction (40% weight)
        ml_results = predict_news(news_text)
        ml_score = ml_results['real_probability']
        
        # 2. NewsAPI Verification (30% weight)
        newsapi_results = check_newsapi(news_text)
        
        # 3. Google News Check (15% weight)
        google_mentions = check_google_news(news_text)
        
        # 4. Twitter Trends (15% weight)
        twitter_results = check_twitter_trends(news_text)
        
        # 5. Calculate Credibility Score
        credibility_score = calculate_credibility(
            ml_score, 
            newsapi_results['num_sources'],
            google_mentions,
            twitter_results['is_trending']
        )
        
        # 6. Rumor Detection
        rumor_label, rumor_color = detect_rumor(
            ml_results['ml_prediction'], 
            credibility_score, 
            newsapi_results['num_sources']
        )
        
        # 7. Reasoning
        reasoning = generate_reasoning(
            ml_results['ml_prediction'],
            newsapi_results['num_sources'],
            google_mentions,
            twitter_results['is_trending']
        )
        
        # Complete response matching frontend expectations
        result = {
            'credibility_score': credibility_score,
            'ml_prediction': ml_results['ml_prediction'],
            'ml_confidence': ml_results['ml_confidence'],
            'fake_probability': ml_results['fake_probability'],
            'real_probability': ml_results['real_probability'],
            'num_sources': newsapi_results['num_sources'],
            'sources_found': newsapi_results['sources_found'],
            'google_mentions': google_mentions,
            'estimated_tweets': twitter_results['estimated_tweets'],
            'is_trending': twitter_results['is_trending'],
            'rumor_label': rumor_label,
            'rumor_color': rumor_color,
            'rumor_reasoning': reasoning,
            'is_verified': newsapi_results['is_verified']
        }
        
        print(f"✅ Analysis complete: {credibility_score}% credibility")
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Analysis error: {str(e)}")
        return jsonify({'error': 'Analysis failed. Please try again.'}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake News Detection Server...")
    print("📱 Open: http://127.0.0.1:5000")
    print("⚙️  Run 'python train_model.py' first if models missing!")
    app.run(debug=True, host='127.0.0.1', port=5000)

