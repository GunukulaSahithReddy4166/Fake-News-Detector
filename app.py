"""Fake News Detection Flask application."""

import os
import re
import string
import xml.etree.ElementTree as ET
from pathlib import Path

import joblib
import requests
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "vectorizer.pkl"
NEWS_API_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
REQUEST_TIMEOUT = 10

TRUSTED_DOMAINS = {
    "reuters.com", "bbc.com", "bbc.co.uk", "apnews.com", "theguardian.com",
    "nytimes.com", "washingtonpost.com", "cnn.com", "abcnews.go.com", "npr.org",
    "cbsnews.com", "nbcnews.com", "indianexpress.com", "thehindu.com",
    "hindustantimes.com", "timesofindia.indiatimes.com",
}

try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    print(f"Loaded model: {MODEL_PATH}")
except (FileNotFoundError, ValueError, EOFError) as exc:
    print(f"ML artifacts unavailable: {exc}")
    model = None
    vectorizer = None


def clean_text(text):
    """Apply exactly the same text cleaning used during training."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return re.sub(r"\d+", "", text)


def build_search_query(text, max_words=25):
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)
    return " ".join(words[:max_words]).strip()[:450]


def get_domain(url):
    match = re.search(r"https?://(?:www\.)?([^/]+)", url or "", re.I)
    return match.group(1).lower() if match else ""


def is_trusted_domain(domain):
    domain = (domain or "").lower().split(":")[0]
    return any(domain == trusted or domain.endswith("." + trusted)
               for trusted in TRUSTED_DOMAINS)


def check_newsapi(news_text, api_key=None):
    api_key = api_key or os.getenv("NEWS_API_KEY")
    if not api_key:
        return {"status": "unavailable", "error": "NEWS_API_KEY is not configured",
                "num_sources": 0, "sources_found": [], "trusted_sources_found": [],
                "articles": [], "is_verified": False}

    query = build_search_query(news_text)
    if not query:
        return {"status": "no_query", "error": "Could not create a search query",
                "num_sources": 0, "sources_found": [], "trusted_sources_found": [],
                "articles": [], "is_verified": False}
    try:
        response = requests.get(
            NEWS_API_URL,
            params={"q": query, "searchIn": "title,description", "language": "en",
                    "sortBy": "relevancy", "pageSize": 20},
            headers={"X-Api-Key": api_key}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        if data.get("status") != "ok":
            raise ValueError(data.get("message", "NewsAPI returned an error"))

        articles, source_names, trusted_names = [], set(), set()
        for article in data.get("articles", []):
            source = article.get("source") or {}
            name = source.get("name") or "Unknown source"
            url = article.get("url") or ""
            trusted = is_trusted_domain(get_domain(url))
            articles.append({"source": name, "domain": get_domain(url),
                             "title": article.get("title"),
                             "description": article.get("description"), "url": url,
                             "published_at": article.get("publishedAt"), "trusted": trusted})
            source_names.add(name)
            if trusted:
                trusted_names.add(name)
        return {"status": "ok", "error": None, "num_sources": len(source_names),
                "sources_found": sorted(source_names),
                "trusted_sources_found": sorted(trusted_names), "articles": articles[:10],
                "is_verified": len(trusted_names) >= 3}
    except (requests.RequestException, ValueError, KeyError) as exc:
        return {"status": "error", "error": str(exc), "num_sources": 0,
                "sources_found": [], "trusted_sources_found": [], "articles": [],
                "is_verified": False}


def check_google_news(news_text):
    query = build_search_query(news_text)
    if not query:
        return {"status": "no_query", "mentions": 0, "articles": []}
    try:
        response = requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={"q": query, "hl": "en-IN", "gl": "IN", "ceid": "IN:en"},
            headers={"User-Agent": "SafeSource/1.0"}, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        root = ET.fromstring(response.content)
        articles = []
        for item in root.findall(".//item")[:20]:
            source_node = item.find("source")
            articles.append({"title": item.findtext("title") or "",
                             "url": item.findtext("link") or "",
                             "published_at": item.findtext("pubDate") or "",
                             "source": source_node.text if source_node is not None else "Unknown source"})
        return {"status": "ok", "mentions": len(articles), "articles": articles}
    except (requests.RequestException, ET.ParseError) as exc:
        return {"status": "error", "mentions": 0, "articles": [], "error": str(exc)}


def check_twitter_trends(_news_text):
    return {"status": "unavailable", "estimated_tweets": None, "is_trending": False,
            "message": "X/Twitter verification is not enabled; no tweet count is fabricated."}


def predict_news(text):
    """Return ML prediction using the persisted training pipeline."""
    if model is None or vectorizer is None:
        return {"ml_prediction": "unknown", "ml_confidence": 0,
                "fake_probability": 50, "real_probability": 50}
    features = vectorizer.transform([clean_text(text)])
    probabilities = model.predict_proba(features)[0]
    fake_prob = real_prob = 0.0
    for label, probability in zip(model.classes_, probabilities):
        label_text = str(label).lower()
        if label_text in {"fake", "0", "false"}:
            fake_prob = probability * 100
        elif label_text in {"real", "1", "true"}:
            real_prob = probability * 100
    prediction = "fake" if fake_prob > real_prob else "real"
    return {"ml_prediction": prediction, "ml_confidence": round(max(fake_prob, real_prob), 1),
            "fake_probability": round(fake_prob, 1), "real_probability": round(real_prob, 1)}


def calculate_credibility(ml_real_probability, newsapi_sources, google_mentions):
    score = (0.55 * min(max(ml_real_probability, 0) / 100, 1) +
             0.30 * min(newsapi_sources / 5, 1) +
             0.15 * min(google_mentions / 10, 1)) * 100
    return round(score, 1)


def detect_rumor(ml_pred, credibility, num_sources):
    if ml_pred == "fake" and num_sources == 0:
        return "Likely Fake News", "red"
    if credibility < 30:
        return "High Risk Fake", "red"
    if credibility < 50:
        return "Unverified Claim", "orange"
    if credibility < 70:
        return "Needs More Evidence", "yellow"
    if num_sources >= 3 and credibility >= 70:
        return "Strongly Supported", "green"
    return "Low Confidence Real", "lightgreen"


def generate_reasoning(ml_pred, trusted_sources, total_sources, google_mentions):
    if ml_pred == "real":
        reasons = ["ML classifier leans toward real"]
    elif ml_pred == "fake":
        reasons = ["ML classifier flags the claim as potentially fake"]
    else:
        reasons = ["ML classifier unavailable"]
    if trusted_sources >= 3:
        reasons.append(f"Matching coverage found from {trusted_sources} trusted publishers")
    elif total_sources:
        reasons.append(f"Matching coverage found from {total_sources} publisher(s), but independent support is limited")
    else:
        reasons.append("No matching NewsAPI coverage found")
    reasons.append(f"Google News RSS returned {google_mentions} matching result(s)" if google_mentions else
                   "Google News returned no matching results")
    return " | ".join(reasons)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze_news():
    try:
        news_text = request.form.get("news_text", "").strip()
        if len(news_text) < 10:
            return jsonify({"error": "News text too short (min 10 chars)"}), 400
        ml_results = predict_news(news_text)
        newsapi_results = check_newsapi(news_text)
        google_results = check_google_news(news_text)
        twitter_results = check_twitter_trends(news_text)
        total_sources = newsapi_results.get("num_sources", 0)
        trusted_sources = len(newsapi_results.get("trusted_sources_found", []))
        google_mentions = google_results.get("mentions", 0)
        credibility = calculate_credibility(ml_results["real_probability"], total_sources, google_mentions)
        rumor_label, rumor_color = detect_rumor(ml_results["ml_prediction"], credibility, total_sources)
        result = {**ml_results, "credibility_score": credibility, "num_sources": total_sources,
                  "trusted_source_count": trusted_sources,
                  "sources_found": newsapi_results.get("sources_found", []),
                  "trusted_sources_found": newsapi_results.get("trusted_sources_found", []),
                  "newsapi_status": newsapi_results.get("status"),
                  "newsapi_error": newsapi_results.get("error"),
                  "newsapi_articles": newsapi_results.get("articles", []),
                  "google_mentions": google_mentions, "google_status": google_results.get("status"),
                  "google_articles": google_results.get("articles", []),
                  "estimated_tweets": twitter_results["estimated_tweets"],
                  "is_trending": twitter_results["is_trending"], "twitter_status": twitter_results["status"],
                  "twitter_message": twitter_results["message"], "rumor_label": rumor_label,
                  "rumor_color": rumor_color,
                  "rumor_reasoning": generate_reasoning(ml_results["ml_prediction"], trusted_sources,
                                                        total_sources, google_mentions),
                  "is_verified": newsapi_results.get("is_verified", False),
                  "verification_note": "Source coverage is supporting evidence, not proof that a claim is true."}
        return jsonify(result)
    except Exception as exc:
        print(f"Analysis error: {exc}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.getenv("PORT", "5000")))
