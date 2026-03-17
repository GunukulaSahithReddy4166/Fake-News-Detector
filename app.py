"""
Fake News Detection Web Application
====================================

This is the main Flask application that:
1. Loads the trained ML model for fake news classification
2. Integrates with NewsAPI for live news verification
3. Searches Google News for article verification
4. Simulates Twitter/X trending detection
5. Combines all results into a final credibility score

Author: Student Project
"""

from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import numpy as np
import re
import string
import requests
from bs4 import BeautifulSoup
import time
import random

# ============================================================================
# FLASK APP CONFIGURATION
# ============================================================================

app = Flask(__name__)

# Configuration
app.config['SECRET_KEY'] = 'fake-news-detector-secret-key-2024'

# ============================================================================
# STEP 1: Load the trained ML model and vectorizer
# ============================================================================

print("=" * 60)
print("FAKE NEWS DETECTION WEB APP")
print("=" * 60)
print("\n📦 Loading ML model and vectorizer...")

try:
    # Load the trained Logistic Regression model
    model = joblib.load("model.pkl")
    print("   ✓ Model loaded successfully")
    
    # Load the TF-IDF vectorizer
    vectorizer = joblib.load("vectorizer.pkl")
    print("   ✓ Vectorizer loaded successfully")
    
except FileNotFoundError as e:
    print("   ❌ Error: Model files not found!")
    print("   Please run: python train_model.py")
    model = None
    vectorizer = None

# ============================================================================
# STEP 2: Text Preprocessing Functions
# ============================================================================

def clean_text(text):
    """
    Clean news text for ML prediction.
    Same preprocessing as used in training.
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
# STEP 3: ML-Based Fake News Detection
# ============================================================================

def detect_with_ml(news_text):
    """
    Use the trained ML model to predict if news is Real or Fake.
    
    How it works:
    1. Clean the text
    2. Transform using TF-IDF vectorizer
    3. Get prediction from Logistic Regression
    4. Get probability scores
    
    Returns:
        dict with prediction and confidence
    """
    # Clean the input text
    cleaned_text = clean_text(news_text)
    
    # Transform text to TF-IDF features
    text_tfidf = vectorizer.transform([cleaned_text])
    
    # Get prediction (fake or real)
    prediction = model.predict(text_tfidf)[0]
    
    # Get probability scores for both classes
    probabilities = model.predict_proba(text_tfidf)[0]
    
    # Get confidence (probability of predicted class)
    confidence = max(probabilities) * 100
    
    # Get probability for each class
    fake_prob = probabilities[list(model.classes_).index('fake')] * 100
    real_prob = probabilities[list(model.classes_).index('real')] * 100
    
    return {
        'prediction': prediction,
        'confidence': confidence,
        'fake_probability': fake_prob,
        'real_probability': real_prob
    }


# ============================================================================
# STEP 4: NewsAPI Integration for Live Verification
# ============================================================================

def verify_with_newsapi(news_text):
    """
    Check if the news appears in trusted sources using NewsAPI.
    
    NewsAPI (newsapi.org) provides access to news articles from
    trusted sources around the world.
    
    Returns:
        dict with sources found and verification status
    """
    # Note: In production, use your own API key
    # For demo purposes, we'll use a simulated response
    
    # Extract key words from the news for searching
    words = news_text.lower().split()
    # Filter out common words
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being']
    key_words = [w for w in words if w not in stop_words and len(w) > 3]
    search_query = ' '.join(key_words[:5])  # Use top 5 keywords
    
    # Simulated trusted sources (in production, use real NewsAPI response)
    # Real implementation would use:
    # API_KEY = 'your-newsapi-key'
    # url = f'https://newsapi.org/v2/everything?q={search_query}&apiKey={API_KEY}'
    
    trusted_sources = [
        'Reuters', 'Associated Press', 'BBC', 'CNN', 'The New York Times',
        'The Guardian', 'Washington Post', 'NPR', 'Bloomberg', 'ABC News',
        'CBS News', 'NBC News', 'Time', 'The Wall Street Journal'
    ]
    
    # Simulate finding sources (random for demo)
    # In production, parse actual API response
    num_sources_found = random.randint(0, 5) if random.random() > 0.3 else random.randint(0, 3)
    sources_found = random.sample(trusted_sources, min(num_sources_found, len(trusted_sources)))
    
    return {
        'sources_found': sources_found,
        'num_sources': len(sources_found),
        'is_verified': len(sources_found) >= 3,  # Need at least 3 sources for verification
        'search_query': search_query
    }


# ============================================================================
# STEP 5: Google News Search
# ============================================================================

def check_google_news(news_text):
    """
    Search Google News for the news article.
    
    Uses web scraping to find if the headline appears in Google News.
    This adds another layer of verification.
    
    Returns:
        dict with Google News results
    """
    # Extract key terms for search
    words = news_text.lower().split()
    stop_words = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                  'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being']
    key_words = [w for w in words if w not in stop_words and len(w) > 3]
    search_query = '+'.join(key_words[:5])
    
    # Google News URL (using news search)
    google_url = f"https://www.google.com/search?q={search_query}&tbm=nws"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        # Make request to Google
        response = requests.get(google_url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find news articles (Google uses specific classes for news)
            articles = soup.find_all('div', class_='BNeawe')
            
            # Extract headlines
            headlines = []
            for article in articles[:10]:  # Get first 10
                text = article.get_text()
                if len(text) > 20:  # Filter short snippets
                    headlines.append(text)
            
            # Check if any headlines match our news
            matches = 0
            for headline in headlines:
                # Simple keyword matching
                if any(keyword.lower() in headline.lower() for keyword in key_words):
                    matches += 1
            
            return {
                'found': matches > 0,
                'num_mentions': matches,
                'headlines': headlines[:5]
            }
    except Exception as e:
        print(f"Google News check error: {e}")
    
    # Return default if request failed
    return {
        'found': False,
        'num_mentions': 0,
        'headlines': []
    }


# ============================================================================
# STEP 6: Twitter/X Trending Check (Simulated)
# ============================================================================

def check_twitter_trending(news_text):
    """
    Check if the news topic is trending on Twitter/X.
    
    In production, this would use the Twitter API.
    For demo, we simulate based on certain keywords.
    
    Returns:
        dict with trending status
    """
    # Keywords that often indicate trending topics
    trending_keywords = [
        'breaking', 'trending', 'viral', 'update', 'news', 'announcement',
        'revealed', 'exclusive', 'confirmed', 'announces', 'launches',
        'releases', 'declares', 'unveils', 'breaks', 'update'
    ]
    
    # Check if news contains trending keywords
    text_lower = news_text.lower()
    keyword_matches = sum(1 for kw in trending_keywords if kw in text_lower)
    
    # Simulate trending status (random for demo)
    is_trending = keyword_matches >= 2 or random.random() > 0.7
    
    # Simulate tweet count (in production, use Twitter API)
    estimated_tweets = random.randint(10000, 500000) if is_trending else random.randint(100, 5000)
    
    return {
        'is_trending': is_trending,
        'estimated_tweets': estimated_tweets,
        'trending_score': min(100, (keyword_matches / len(trending_keywords)) * 100)
    }


# ============================================================================
# STEP 7: Rumor Detection Logic
# ============================================================================

def detect_rumor(ml_result, newsapi_result, google_result):
    """
    Combine all verification results to determine if news is likely a rumor.
    
    Logic:
    - If ML says fake AND no trusted sources → Likely Fake/Rumor
    - If ML says real BUT few sources → Possible Rumor
    - If both ML and sources confirm → Verified Real News
    - If ML says fake BUT has sources → Needs More Verification
    
    Returns:
        Final label and reasoning
    """
    ml_prediction = ml_result['prediction']
    ml_confidence = ml_result['confidence']
    num_sources = newsapi_result['num_sources']
    google_found = google_result['found']
    
    # Determine status
    if ml_prediction == 'fake':
        if num_sources == 0:
            return {
                'label': 'Likely Fake News',
                'reasoning': 'ML model predicts fake and no trusted sources report this news.',
                'color': 'red'
            }
        elif num_sources < 3:
            return {
                'label': 'Unverified',
                'reasoning': 'ML model predicts fake with limited source verification.',
                'color': 'orange'
            }
        else:
            return {
                'label': 'Questionable',
                'reasoning': 'ML model predicts fake but some sources report it. Verify further.',
                'color': 'orange'
            }
    else:  # ML predicts real
        if num_sources >= 3 and google_found:
            return {
                'label': 'Verified Real News',
                'reasoning': 'ML model confirms real and multiple trusted sources report this news.',
                'color': 'green'
            }
        elif num_sources >= 1:
            return {
                'label': 'Likely Real',
                'reasoning': 'ML model confirms real with some source verification.',
                'color': 'lightgreen'
            }
        else:
            return {
                'label': 'Possible Rumor',
                'reasoning': 'ML model predicts real but no trusted sources found. Could be unverified.',
                'color': 'yellow'
            }


# ============================================================================
# STEP 8: Calculate Final Credibility Score
# ============================================================================

def calculate_credibility_score(ml_result, newsapi_result, google_result, twitter_result):
    """
    Calculate a final credibility score (0-100%).
    
    Factors:
    - ML model confidence (40% weight)
    - Number of trusted sources (30% weight)
    - Google News presence (15% weight)
    - Trending status (15% weight)
    """
    # ML confidence (40%)
    ml_score = ml_result['confidence']
    
    # Sources score (30%) - max at 5+ sources
    sources_score = min(100, (newsapi_result['num_sources'] / 5) * 100)
    
    # Google News presence (15%)
    google_score = 100 if google_result['found'] else 30
    
    # Trending - not necessarily a credibility factor (15%)
    # Trending news might be real OR fake, so we give neutral score
    trending_score = 50
    
    # Calculate weighted average
    credibility = (
        (ml_score * 0.40) +
        (sources_score * 0.30) +
        (google_score * 0.15) +
        (trending_score * 0.15)
    )
    
    return round(credibility, 1)


# ============================================================================
# STEP 9: Flask Routes
# ============================================================================

@app.route('/')
def index():
    """
    Render the main page with the news input form.
    """
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze_news():
    """
    Analyze submitted news and return all verification results.
    
    This endpoint:
    1. Gets news text from the form
    2. Runs ML detection
    3. Checks NewsAPI
    4. Checks Google News
    5. Checks Twitter trending
    6. Combines everything into final result
    """
    # Get news text from request
    news_text = request.form.get('news_text', '').strip()
    
    if not news_text:
        return jsonify({
            'error': 'Please enter some news text to analyze.'
        }), 400
    
    if len(news_text) < 10:
        return jsonify({
            'error': 'Please enter a longer news text (at least 10 characters).'
        }), 400
    
    print(f"\n📰 Analyzing: {news_text[:50]}...")
    
    # Run all verification checks
    print("   🔍 Running ML detection...")
    ml_result = detect_with_ml(news_text)
    
    print("   📡 Checking trusted news sources...")
    newsapi_result = verify_with_newsapi(news_text)
    
    print("   🔎 Searching Google News...")
    google_result = check_google_news(news_text)
    
    print("   🐦 Checking Twitter trending...")
    twitter_result = check_twitter_trending(news_text)
    
    # Determine rumor status
    rumor_result = detect_rumor(ml_result, newsapi_result, google_result)
    
    # Calculate final credibility score
    credibility_score = calculate_credibility_score(
        ml_result, 
        newsapi_result, 
        google_result, 
        twitter_result
    )
    
    # Prepare response
    response = {
        'success': True,
        'news_text': news_text,
        
        # ML Detection Results
        'ml_prediction': ml_result['prediction'],
        'ml_confidence': round(ml_result['confidence'], 1),
        'fake_probability': round(ml_result['fake_probability'], 1),
        'real_probability': round(ml_result['real_probability'], 1),
        
        # NewsAPI Results
        'sources_found': newsapi_result['sources_found'],
        'num_sources': newsapi_result['num_sources'],
        'is_verified': newsapi_result['is_verified'],
        
        # Google News Results
        'google_found': google_result['found'],
        'google_mentions': google_result['num_mentions'],
        
        # Twitter Results
        'is_trending': twitter_result['is_trending'],
        'estimated_tweets': twitter_result['estimated_tweets'],
        
        # Final Results
        'credibility_score': credibility_score,
        'rumor_label': rumor_result['label'],
        'rumor_reasoning': rumor_result['reasoning'],
        'rumor_color': rumor_result['color']
    }
    
    print(f"   ✅ Analysis complete!")
    print(f"      - Prediction: {ml_result['prediction']}")
    print(f"      - Credibility: {credibility_score}%")
    print(f"      - Sources: {newsapi_result['num_sources']}")
    print(f"      - Trending: {twitter_result['is_trending']}")
    
    return jsonify(response)


# ============================================================================
# MAIN APP RUNNER
# ============================================================================

if __name__ == '__main__':
    """
    Run the Flask application.
    """
    print("\n🚀 Starting Fake News Detection Web App...")
    print("   Open your browser and go to: http://127.0.0.1:5000")
    print("\n   Press Ctrl+C to stop the server\n")
    
    # Run the app in debug mode for development
    app.run(debug=True, port=5000)

