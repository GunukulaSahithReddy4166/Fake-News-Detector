import json
from pathlib import Path

import pandas as pd

import app

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "news.csv"
METRICS_PATH = BASE_DIR / "models" / "metrics.json"


def test_model_artifacts_are_loaded():
    assert app.model is not None
    assert app.vectorizer is not None
    assert set(str(label).lower() for label in app.model.classes_) == {"fake", "real"}


def test_known_fake_and_real_examples_have_valid_predictions():
    df = pd.read_csv(DATA_PATH)
    fake_text = df.loc[df["label"].str.lower() == "fake", "text"].iloc[0]
    real_text = df.loc[df["label"].str.lower() == "real", "text"].iloc[0]

    for text, expected_label in [(fake_text, "fake"), (real_text, "real")]:
        result = app.predict_news(text)
        assert result["ml_prediction"] in {"fake", "real"}
        assert 0 <= result["fake_probability"] <= 100
        assert 0 <= result["real_probability"] <= 100
        assert abs(result["fake_probability"] + result["real_probability"] - 100) < 0.2
        print(
            f"Known example | expected={expected_label} | "
            f"predicted={result['ml_prediction']} | "
            f"fake={result['fake_probability']}% | real={result['real_probability']}%"
        )


def test_new_headlines_have_valid_probability_output():
    headlines = [
        "Government announces a new national digital education initiative.",
        "Scientists discover a completely invisible material that bends time.",
        "Local officials publish the annual city budget after a public meeting.",
    ]
    for headline in headlines:
        result = app.predict_news(headline)
        assert result["ml_prediction"] in {"fake", "real"}
        assert 0 <= result["ml_confidence"] <= 100
        assert abs(result["fake_probability"] + result["real_probability"] - 100) < 0.2
        print(
            f"New headline | predicted={result['ml_prediction']} | "
            f"confidence={result['ml_confidence']}% | "
            f"fake={result['fake_probability']}% | real={result['real_probability']}%"
        )


def test_metrics_are_complete_and_finite():
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
    holdout = metrics["holdout"]
    cv = metrics["cross_validation"]["metrics"]

    for key in ["accuracy", "precision_real", "recall_real", "f1_real"]:
        assert 0 <= holdout[key] <= 1

    for metric in ["accuracy", "precision", "recall", "f1"]:
        assert 0 <= cv[metric]["mean"] <= 1
        assert cv[metric]["std"] >= 0
        assert len(cv[metric]["folds"]) == metrics["cross_validation"]["folds"]


def test_short_input_is_rejected():
    client = app.app.test_client()
    response = client.post("/analyze", data={"news_text": "short"})
    assert response.status_code == 400
    assert "error" in response.get_json()


def test_analyze_endpoint_without_external_network(monkeypatch):
    monkeypatch.setattr(
        app,
        "check_newsapi",
        lambda text: {
            "status": "unavailable",
            "error": "NEWS_API_KEY is not configured",
            "num_sources": 0,
            "sources_found": [],
            "trusted_sources_found": [],
            "articles": [],
            "is_verified": False,
        },
    )
    monkeypatch.setattr(
        app,
        "check_google_news",
        lambda text: {"status": "ok", "mentions": 0, "articles": []},
    )
    monkeypatch.setattr(
        app,
        "check_twitter_trends",
        lambda text: {
            "status": "unavailable",
            "estimated_tweets": None,
            "is_trending": False,
            "message": "X/Twitter verification is not enabled",
        },
    )

    client = app.app.test_client()
    response = client.post(
        "/analyze",
        data={"news_text": "A government announces a new national education policy."},
    )
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["ml_prediction"] in {"fake", "real"}
    assert payload["twitter_status"] == "unavailable"
    assert payload["newsapi_status"] == "unavailable"
    assert payload["google_mentions"] == 0
