"""
Fake News Detection Web Application - Flask Backend
===================================================

Live verification version:
1. ML classification (TF-IDF + Logistic Regression)
2. Real NewsAPI article search
3. Real Google News RSS search
4. No simulated/random source counts
5. No fake Twitter/X tweet-volume estimates

Important:
- NEWS_API_KEY must be provided through an environment variable.
- News coverage is evidence that a claim is being reported, not proof that
  the claim is true.
"""

import os
import re
import string
from urllib.parse import quote_plus
import xml.etree.ElementTree as ET

import requests
from flask import Flask, render_template, request, jsonify
import joblib

app = Flask(__name__)

NEWS_API_URL = "https://newsapi.org/v2/everything"
GOOGLE_NEWS_RSS_URL = "https://news.google.com/rss/search"
REQUEST_TIMEOUT = 10

# Domains that are treated as established/trusted publishers for the UI.
# This is NOT a guarantee that an article from one of these publishers is true.
TRUSTED_DOMAINS = {
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
    "apnews.com",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "cnn.com",
    "abcnews.go.com",
    "npr.org",
    "cbsnews.com",
    "nbcnews.com",
    "indianexpress.com",
    "thehindu.com",
    "hindustantimes.com",
    "timesofindia.indiatimes.com",
}

print("Loading ML model and vectorizer...")
try:
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    print("Model and vectorizer loaded successfully.")
except FileNotFoundError:
    print("Model files not found. Run: python train_model.py")
    model = None
    vectorizer = None


def clean_text(text):
    """Clean text in the same way as the training pipeline."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    text = re.sub(r"\d+", "", text)
    return text


def build_search_query(text, max_words=25):
    """Create a compact search query from the submitted article/headline."""
    words = re.findall(r"[A-Za-z0-9][A-Za-z0-9'’-]*", text)
    query = " ".join(words[:max_words]).strip()
    return query[:450]


def get_domain(url):
    """Return a normalized hostname from a URL."""
    match = re.search(r"https?://(?:www\.)?([^/]+)", url or "", re.I)
    return match.group(1).lower() if match else ""


def is_trusted_domain(domain):
    """Check whether a publisher domain belongs to our trusted-domain list."""
    domain = (domain or "").lower().split(":")[0]
    return any(domain == trusted or domain.endswith("." + trusted)
               for trusted in TRUSTED_DOMAINS)


def check_newsapi(news_text, api_key=None):
    """
    Search NewsAPI for real matching articles.

    NewsAPI's /v2/everything endpoint searches articles by keywords/phrases.
    No random values are generated. If the API is unavailable, the response
    explicitly reports that verification could not be completed.
    """
    api_key = api_key or os.getenv("NEWS_API_KEY")
    if not api_key:
        return {
            "status": "unavailable",
            "error": "NEWS_API_KEY is not configured",
            "num_sources": 0,
            "sources_found": [],
            "articles": [],
            "is_verified": False,
        }

    query = build_search_query(news_text)
    if not query:
        return {
            "status": "no_query",
            "error": "Could not create a search query",
            "num_sources": 0,
            "sources_found": [],
            "articles": [],
            "is_verified": False,
        }

    try:
        response = requests.get(
            NEWS_API_URL,
            params={
                "q": query,
                "searchIn": "title,description",
                "language": "en",
                "sortBy": "relevancy",
                "pageSize": 20,
            },
            headers={"X-Api-Key": api_key},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()
        data = response.json()

        if data.get("status") != "ok":
            return {
                "status": "error",
                "error": data.get("message", "NewsAPI returned an error"),
                "num_sources": 0,
                "sources_found": [],
                "articles": [],
                "is_verified": False,
            }

        articles = []
        source_names = set()
        trusted_source_names = set()

        for article in data.get("articles", []):
            source = article.get("source") or {}
            name = source.get("name") or "Unknown source"
            url = article.get("url") or ""
            domain = get_domain(url)

            item = {
                "source": name,
                "domain": domain,
                "title": article.get("title"),
                "description": article.get("description"),
                "url": url,
                "published_at": article.get("publishedAt"),
                "trusted": is_trusted_domain(domain),
            }
            articles.append(item)
            source_names.add(name)
            if item["trusted"]:
                trusted_source_names.add(name)

        # "num_sources" means distinct publishers with matching coverage,
        # not number of articles and not a fabricated trust score.
        return {
            "status": "ok",
            "error": None,
            "num_sources": len(source_names),
            "sources_found": sorted(source_names),
            "trusted_sources_found": sorted(trusted_source_names),
            "articles": articles[:10],
            # Three or more distinct trusted publishers is only an evidence
            # signal; it is deliberately not called proof of truth.
            "is_verified": len(trusted_source_names) >= 3,
        }

    except requests.RequestException as exc:
        return {
            "status": "error",
            "error": f"NewsAPI request failed: {exc}",
            "num_sources": 0,
            "sources_found": [],
            "articles": [],
            "is_verified": False,
        }
    except (ValueError, KeyError) as exc:
        return {
            "status": "error",
            "error": f"Invalid NewsAPI response: {exc}",
            "num_sources": 0,
            "sources_found": [],
            "articles": [],
            "is_verified": False,
        }


def check_google_news(news_text):
    """
    Search Google News through its RSS search feed.

    This is a real network request. It does not manufacture mention counts.
    """
    query = build_search_query(news_text)
    if not query:
        return {
            "status": "no_query",
            "mentions": 0,
            "articles": [],
        }

    try:
        response = requests.get(
            GOOGLE_NEWS_RSS_URL,
            params={
                "q": query,
                "hl": "en-IN",
                "gl": "IN",
                "ceid": "IN:en",
            },
            headers={"User-Agent": "FakeNewsDetector/1.0"},
            timeout=REQUEST_TIMEOUT,
        )
        response.raise_for_status()

        root = ET.fromstring(response.content)
        articles = []
        for item in root.findall(".//item")[:20]:
            title = item.findtext("title") or ""
            link = item.findtext("link") or ""
            pub_date = item.findtext("pubDate") or ""
            source_node = item.find("source")
            source_name = source_node.text if source_node is not None else "Unknown source"
            articles.append({
                "title": title,
                "url": link,
                "published_at": pub_date,
                "source": source_name,
            })

        return {
            "status": "ok",
            "mentions": len(articles),
            "articles": articles,
        }

    except (requests.RequestException, ET.ParseError) as exc:
        return {
            "status": "error",
            "mentions": 0,
            "articles": [],
            "error": f"Google News request failed: {exc}",
        }


def check_twitter_trends(_news_text):
    """
    Twitter/X is intentionally NOT simulated anymore.

    We do not claim to know tweet volume without a real X API integration.
    """
    return {
        "status": "unavailable",
        "estimated_tweets": None,
        "is_trending": False,
        "message": "X/Twitter verification is not enabled; no tweet count is fabricated.",
    }


def predict_news(text):
    """Run the trained ML classifier."""
    if model is None or vectorizer is None:
        return {
            "ml_prediction": "unknown",
            "ml_confidence": 0,
            "fake_probability": 50,
            "real_probability": 50,
        }

    cleaned = clean_text(text)
    X_tfidf = vectorizer.transform([cleaned])
    probs = model.predict_proba(X_tfidf)[0]

    # Do not assume that classes_[0] is fake and classes_[1] is real.
    fake_prob = 0.0
    real_prob = 0.0
    for class_label, probability in zip(model.classes_, probs):
        label = str(class_label).lower()
        if label in {"fake", "0", "false"}:
            fake_prob = probability * 100
        elif label in {"real", "1", "true"}:
            real_prob = probability * 100

    prediction = "fake" if fake_prob > real_prob else "real"
    confidence = max(fake_prob, real_prob)

    return {
        "ml_prediction": prediction,
        "ml_confidence": round(confidence, 1),
        "fake_probability": round(fake_prob, 1),
        "real_probability": round(real_prob, 1),
    }


def calculate_credibility(ml_score, newsapi_sources, google_mentions):
    """
    Evidence-oriented score for the existing UI.

    This score no longer rewards social-media popularity. It combines the ML
    classifier probability with independently retrieved news coverage.
    It is an assessment signal, not proof that a claim is true.
    """
    ml_weight = max(0.0, min(ml_score / 100, 1.0))
    source_weight = min(newsapi_sources / 5, 1.0)
    google_weight = min(google_mentions / 10, 1.0)

    score = (
        0.55 * ml_weight +
        0.30 * source_weight +
        0.15 * google_weight
    ) * 100
    return round(score, 1)


def detect_rumor(ml_pred, credibility, num_sources):
    """Classify the result using the ML result plus real evidence coverage."""
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
    """Generate reasoning without claiming that coverage proves truth."""
    reasons = []

    if ml_pred == "real":
        reasons.append("ML classifier leans toward real")
    elif ml_pred == "fake":
        reasons.append("ML classifier flags the claim as potentially fake")
    else:
        reasons.append("ML classifier unavailable")

    if trusted_sources >= 3:
        reasons.append(f"Matching coverage found from {trusted_sources} trusted publishers")
    elif total_sources > 0:
        reasons.append(f"Matching coverage found from {total_sources} publisher(s), but independent support is limited")
    else:
        reasons.append("No matching NewsAPI coverage found")

    if google_mentions > 0:
        reasons.append(f"Google News RSS returned {google_mentions} matching result(s)")
    else:
        reasons.append("Google News returned no matching results")

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

        print(f"Analyzing: {news_text[:100]}...")

        # 1. ML prediction
        ml_results = predict_news(news_text)

        # 2. REAL NewsAPI search
        newsapi_results = check_newsapi(news_text)

        # 3. REAL Google News RSS search
        google_results = check_google_news(news_text)

        # 4. No fabricated X/Twitter values
        twitter_results = check_twitter_trends(news_text)

        trusted_source_count = len(newsapi_results.get("trusted_sources_found", []))
        total_source_count = newsapi_results.get("num_sources", 0)
        google_mentions = google_results.get("mentions", 0)

        # Existing frontend field retained for compatibility, but its value is
        # now based only on ML + retrieved news evidence, not social popularity.
        credibility_score = calculate_credibility(
            ml_results["real_probability"],
            total_source_count,
            google_mentions,
        )

        rumor_label, rumor_color = detect_rumor(
            ml_results["ml_prediction"],
            credibility_score,
            total_source_count,
        )

        reasoning = generate_reasoning(
            ml_results["ml_prediction"],
            trusted_source_count,
            total_source_count,
            google_mentions,
        )

        result = {
            "credibility_score": credibility_score,
            "ml_prediction": ml_results["ml_prediction"],
            "ml_confidence": ml_results["ml_confidence"],
            "fake_probability": ml_results["fake_probability"],
            "real_probability": ml_results["real_probability"],
            "num_sources": total_source_count,
            "trusted_source_count": trusted_source_count,
            "sources_found": newsapi_results.get("sources_found", []),
            "trusted_sources_found": newsapi_results.get("trusted_sources_found", []),
            "newsapi_status": newsapi_results.get("status"),
            "newsapi_error": newsapi_results.get("error"),
            "newsapi_articles": newsapi_results.get("articles", []),
            "google_mentions": google_mentions,
            "google_status": google_results.get("status"),
            "google_articles": google_results.get("articles", []),
            "estimated_tweets": twitter_results["estimated_tweets"],
            "is_trending": twitter_results["is_trending"],
            "twitter_status": twitter_results["status"],
            "twitter_message": twitter_results["message"],
            "rumor_label": rumor_label,
            "rumor_color": rumor_color,
            "rumor_reasoning": reasoning,
            "is_verified": newsapi_results.get("is_verified", False),
            "verification_note": (
                "Source coverage indicates that the claim is being reported; it does not by itself prove the claim is true."
            ),
        }

        print(f"Analysis complete: evidence score {credibility_score}%")
        return jsonify(result)

    except Exception as exc:
        print(f"Analysis error: {exc}")
        return jsonify({"error": "Analysis failed. Please try again."}), 500


if __name__ == "__main__":
    print("Starting Fake News Detection Server...")
    print("Open: http://127.0.0.1:5000")
    print("Set NEWS_API_KEY before using live NewsAPI verification.")
    app.run(debug=True, host="127.0.0.1", port=5000)
