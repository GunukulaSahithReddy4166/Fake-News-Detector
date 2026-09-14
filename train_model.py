"""Train and evaluate the Fake News Detector text-classification model."""

import json
import re
import string
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.metrics import make_scorer
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "news.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model.pkl"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"
METRICS_PATH = MODEL_DIR / "metrics.json"
RANDOM_STATE = 42
TEST_SIZE = 0.20


def clean_text(text):
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    return re.sub(r"\d+", "", text)


def validate_dataset(df):
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    df = df[["text", "label"]].copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df.dropna(subset=["text", "label"])
    df = df[df["text"].str.len() > 0]
    df["cleaned_text"] = df["text"].apply(clean_text)
    df = df[df["cleaned_text"].str.len() > 0]
    df = df[df["label"].isin(["real", "fake"])]
    duplicate_count = int(df.duplicated(subset=["cleaned_text"]).sum())
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)
    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'real' and 'fake' labels.")
    if df["label"].value_counts().min() < 5:
        raise ValueError("Each class needs at least 5 samples for evaluation.")
    return df, duplicate_count


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            max_df=0.95, min_df=1, stop_words="english",
            ngram_range=(1, 2), sublinear_tf=True
        )),
        ("classifier", LogisticRegression(
            max_iter=2000, random_state=RANDOM_STATE, class_weight="balanced"
        )),
    ])


def main():
    print("=" * 72)
    print("FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 72)
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    print(f"Dataset: {DATA_PATH.relative_to(BASE_DIR)}")
    print(f"Raw samples: {len(df)}")
    df, duplicate_count = validate_dataset(df)
    counts = df["label"].value_counts().to_dict()
    print(f"Usable unique samples: {len(df)}")
    print(f"Class distribution: {counts}")
    print(f"Removed duplicate claims: {duplicate_count}")

    X, y = df["cleaned_text"], df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"Train samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=["fake", "real"], average="binary",
        pos_label="real", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=["fake", "real"])

    print("\n--- Holdout test performance ---")
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%) [real as positive]")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%) [real as positive]")
    print(f"F1-score : {f1:.4f} ({f1 * 100:.2f}%) [real as positive]")
    print("Confusion matrix [actual rows, predicted columns; fake, real]:")
    print(matrix)
    print(classification_report(y_test, y_pred, labels=["fake", "real"], zero_division=0))

    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_recall_fscore_support, average="binary",
                                  pos_label="real", zero_division=0),
    }
    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(lambda yt, yp: precision_recall_fscore_support(
            yt, yp, labels=["fake", "real"], average="binary", pos_label="real", zero_division=0
        )[0]),
        "recall": make_scorer(lambda yt, yp: precision_recall_fscore_support(
            yt, yp, labels=["fake", "real"], average="binary", pos_label="real", zero_division=0
        )[1]),
        "f1": make_scorer(lambda yt, yp: precision_recall_fscore_support(
            yt, yp, labels=["fake", "real"], average="binary", pos_label="real", zero_division=0
        )[2]),
    }
    cv_results = cross_validate(build_pipeline(), X, y, cv=cv, scoring=scoring)
    cv_summary = {}
    print(f"--- {folds}-fold stratified cross-validation ---")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = cv_results[f"test_{metric}"]
        mean, std = float(values.mean()), float(values.std())
        cv_summary[metric] = {"mean": mean, "std": std, "folds": [float(v) for v in values]}
        print(f"{metric.capitalize():9}: {mean:.4f} +/- {std:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    fitted_model = pipeline.named_steps["classifier"]
    fitted_vectorizer = pipeline.named_steps["tfidf"]
    joblib.dump(fitted_model, MODEL_PATH)
    joblib.dump(fitted_vectorizer, VECTORIZER_PATH)
    print(f"Saved: {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"Saved: {VECTORIZER_PATH.relative_to(BASE_DIR)}")

    metrics = {
        "dataset": str(DATA_PATH.relative_to(BASE_DIR)),
        "samples": int(len(df)),
        "class_distribution": {k: int(v) for k, v in counts.items()},
        "removed_duplicates": duplicate_count,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "model": "LogisticRegression",
        "features": "TF-IDF unigram + bigram",
        "holdout": {
            "accuracy": float(accuracy),
            "precision_real": float(precision),
            "recall_real": float(recall),
            "f1_real": float(f1),
            "confusion_matrix_labels": ["fake", "real"],
            "confusion_matrix": matrix.tolist(),
        },
        "cross_validation": {"type": "StratifiedKFold", "folds": folds, "metrics": cv_summary},
        "warning": "Metrics describe this small curated dataset and are not real-world fact-checking accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved: {METRICS_PATH.relative_to(BASE_DIR)}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
