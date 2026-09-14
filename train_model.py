"""
Fake News Detection Model Training Pipeline
===========================================

Trains the text classifier used by the Flask application.

Pipeline:
1. Load data/news.csv
2. Validate and clean the dataset
3. Split into stratified train/test sets
4. Fit TF-IDF only on the training set
5. Train Logistic Regression
6. Evaluate on the untouched test set
7. Run 5-fold stratified cross-validation for a more stable estimate
8. Save model artifacts under models/
9. Save evaluation metrics under models/metrics.json

Important:
This dataset is small and contains short, synthetic/curated claims. The
reported metrics describe performance on this dataset; they are NOT proof
that the model can fact-check arbitrary real-world news.
"""

from pathlib import Path
import json
import re
import string

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
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
    """Apply the same deterministic text cleaning used by the Flask app."""
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())
    text = re.sub(r"\d+", "", text)
    return text


def validate_dataset(df):
    """Validate required columns, labels and usable text rows."""
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

    # Duplicate claims can make a tiny dataset look better than it really is.
    duplicate_count = int(df.duplicated(subset=["cleaned_text"]).sum())
    df = df.drop_duplicates(subset=["cleaned_text"]).reset_index(drop=True)

    if df["label"].nunique() != 2:
        raise ValueError("Dataset must contain both 'real' and 'fake' labels.")
    if df["label"].value_counts().min() < 5:
        raise ValueError("Each class needs at least 5 samples for reliable evaluation.")

    return df, duplicate_count


def build_pipeline():
    """Create the complete text-classification pipeline."""
    vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=1,
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )
    classifier = LogisticRegression(
        max_iter=2000,
        random_state=RANDOM_STATE,
        class_weight="balanced",
    )
    return Pipeline([
        ("tfidf", vectorizer),
        ("classifier", classifier),
    ])


def main():
    print("=" * 72)
    print("FAKE NEWS DETECTION - MODEL TRAINING PIPELINE")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    print(f"\nDataset: {DATA_PATH.relative_to(BASE_DIR)}")
    df = pd.read_csv(DATA_PATH)
    print(f"Raw samples: {len(df)}")

    df, duplicate_count = validate_dataset(df)
    class_counts = df["label"].value_counts().to_dict()
    print(f"Usable unique samples: {len(df)}")
    print(f"Class distribution: {class_counts}")
    print(f"Removed duplicate claims: {duplicate_count}")

    X = df["cleaned_text"]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print(f"\nTrain samples: {len(X_train)}")
    print(f"Test samples:  {len(X_test)}")

    # The final pipeline is fit only on the training split. This prevents
    # test-set vocabulary from leaking into training.
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=["fake", "real"], average="binary", pos_label="real", zero_division=0
    )
    matrix = confusion_matrix(y_test, y_pred, labels=["fake", "real"])

    print("\n--- Holdout test performance ---")
    print(f"Accuracy : {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision * 100:.2f}%) [real as positive]")
    print(f"Recall   : {recall:.4f} ({recall * 100:.2f}%) [real as positive]")
    print(f"F1-score : {f1:.4f} ({f1 * 100:.2f}%) [real as positive]")
    print("Confusion matrix [rows=actual, columns=predicted; fake, real]:")
    print(matrix)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, labels=["fake", "real"], zero_division=0))

    # Cross-validation gives a second view of performance on this very small
    # dataset. TF-IDF is inside the pipeline, so it is refit inside each fold.
    folds = min(5, int(y.value_counts().min()))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        build_pipeline(),
        X,
        y,
        cv=cv,
        scoring={"accuracy": "accuracy", "precision": "precision", "recall": "recall", "f1": "f1"},
        n_jobs=None,
    )

    cv_summary = {}
    print(f"--- {folds}-fold stratified cross-validation ---")
    for metric in ["accuracy", "precision", "recall", "f1"]:
        values = cv_results[f"test_{metric}"]
        mean = float(values.mean())
        std = float(values.std())
        cv_summary[metric] = {"mean": mean, "std": std, "folds": [float(v) for v in values]}
        print(f"{metric.capitalize():9}: {mean:.4f} ± {std:.4f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    # Save the fitted components separately because app.py loads them directly.
    fitted_vectorizer = pipeline.named_steps["tfidf"]
    fitted_model = pipeline.named_steps["classifier"]
    joblib.dump(fitted_model, MODEL_PATH)
    joblib.dump(fitted_vectorizer, VECTORIZER_PATH)

    metrics = {
        "dataset": str(DATA_PATH.relative_to(BASE_DIR)),
        "samples": int(len(df)),
        "class_distribution": {k: int(v) for k, v in class_counts.items()},
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
        "cross_validation": {
            "type": "StratifiedKFold",
            "folds": folds,
            "metrics": cv_summary,
        },
        "warning": "Metrics describe this small curated dataset and should not be interpreted as real-world fact-checking accuracy.",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n--- Saved artifacts ---")
    print(f"Model     : {MODEL_PATH.relative_to(BASE_DIR)}")
    print(f"Vectorizer: {VECTORIZER_PATH.relative_to(BASE_DIR)}")
    print(f"Metrics   : {METRICS_PATH.relative_to(BASE_DIR)}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
