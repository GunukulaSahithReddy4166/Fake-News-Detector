# SafeSource — AI-Powered Fake News Detection

SafeSource is a Flask-based web application that combines **machine-learning text classification** with **external news-source evidence** to assess the credibility of a submitted news claim.

> **Important:** SafeSource provides evidence signals, not a guarantee that a claim is true or false. External coverage and an ML prediction should be interpreted together.

## Features

- **Machine Learning:** TF-IDF text features with Logistic Regression.
- **NewsAPI verification:** Searches real publisher coverage when `NEWS_API_KEY` is configured.
- **Google News RSS:** Searches related coverage through Google News RSS.
- **Trusted-source signal:** Counts distinct publishers and identifies configured trusted publishers.
- **Evidence score:** Combines ML output and external-source evidence into a 0–100 application-level score.
- **Transparent X/Twitter status:** X/Twitter data is not fabricated. The current version reports the integration as unavailable because no live X API is configured.
- **Responsive frontend:** HTML, CSS and vanilla JavaScript frontend connected to Flask's `/analyze` endpoint.
- **Production configuration:** Gunicorn and a `Procfile` are included for Procfile-compatible cloud platforms.

## Architecture

```text
User enters news claim
        │
        ▼
   Flask /analyze
        │
        ├──────────────► ML model
        │                 TF-IDF → Logistic Regression
        │
        ├──────────────► NewsAPI
        │                 publisher/article evidence
        │
        └──────────────► Google News RSS
                          related coverage
        │
        ▼
 Evidence + ML confidence
        │
        ▼
 Evidence score + explanation
        │
        ▼
      Web UI
```

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn, TF-IDF, Logistic Regression, joblib |
| External data | NewsAPI, Google News RSS |
| HTTP/XML | requests, Python XML parser |
| Frontend | HTML5, CSS3, vanilla JavaScript |
| Production server | Gunicorn |
| Deployment configuration | Procfile |

## Project Structure

```text
Fake-News-Detector/
├── .github/
│   └── workflows/
│       └── python-app.yml
├── data/
│   └── news.csv
├── models/
│   └── .gitkeep
├── static/
│   └── style.css
├── templates/
│   └── index.html
├── tests/
│   └── test_app.py
├── app.py
├── train_model.py
├── requirements.txt
├── Procfile
├── .env.example
├── .gitignore
├── README.md
└── TODO.md
```

## How the Analysis Works

### 1. Input preprocessing

The submitted text is normalized before ML inference using the same cleaning logic used during training.

### 2. ML prediction

The trained scikit-learn model receives TF-IDF features and returns a fake/real prediction with model probabilities. These probabilities are **model outputs, not factual truth probabilities**.

### 3. NewsAPI evidence

If `NEWS_API_KEY` is available, SafeSource searches NewsAPI for related articles and counts distinct publishers. A configured list of established publishers is used as a trusted-source signal.

### 4. Google News evidence

The application performs a Google News RSS search and counts related results. This is treated as supporting evidence, not proof of authenticity.

### 5. X/Twitter

The application deliberately does **not** generate simulated tweet counts or trending values. Until a real X API integration is configured, the UI reports social-media verification as unavailable.

### 6. Evidence score

The current backend combines:

```text
ML signal          → 55%
NewsAPI sources    → 30%
Google News signal → 15%
```

The score is an application-level evidence score; it is **not a probability that a claim is objectively true**.

## ML Evaluation — Current Result

The included dataset contains **126 usable examples**:

- Fake: **71**
- Real: **55**
- Holdout training split: **100 train / 26 test**
- Duplicate claims removed: **0**

The current Logistic Regression + TF-IDF model produced the following results in GitHub Actions:

| Metric | Holdout | 5-fold CV mean ± std |
|---|---:|---:|
| Accuracy | **65.38%** | **56.40% ± 7.20%** |
| Precision — real | **62.50%** | **49.88% ± 13.74%** |
| Recall — real | **45.45%** | **30.91% ± 9.27%** |
| F1 — real | **52.63%** | **38.08% ± 11.02%** |

Holdout confusion matrix, with rows = actual and columns = predicted (`fake`, `real`):

```text
[[12, 3],
 [ 6, 5]]
```

These results are **not strong enough to claim production-grade fake-news detection**. The dataset is small and appears to contain simplified/synthetic claims, so the next model improvement should focus on obtaining a larger, more representative labeled dataset rather than claiming a high real-world accuracy.

## Model Testing

Automated tests run after training and verify:

- model/vectorizer artifacts load correctly;
- known fake and known real examples produce valid predictions;
- probabilities are in the expected range and sum to approximately 100%;
- completely new headlines produce valid predictions and confidence values;
- short `/analyze` input is rejected;
- `/analyze` works when external news integrations are unavailable;
- generated evaluation metrics are finite and complete.

The latest GitHub Actions run completed successfully with **6 tests passed**.

Example inference results from the latest CI run:

```text
Known fake → fake, 61.2% fake / 38.8% real
Known real → real, 42.2% fake / 57.8% real

New headline #1 → fake, 59.7% confidence
New headline #2 → fake, 52.4% confidence
New headline #3 → fake, 56.0% confidence
```

The new-headline results are deliberately treated as **low-confidence model signals**, not facts. This testing shows why the current model should not be presented as a definitive fact-checker.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/GunukulaSahithReddy4166/Fake-News-Detector.git
cd Fake-News-Detector
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure NewsAPI

Copy `.env.example` to `.env` for local development and set your NewsAPI key.

```text
NEWS_API_KEY=your_key_here
```

Never commit a real API key. The repository's `.gitignore` excludes `.env`.

### 5. Train the model

```bash
python train_model.py
```

This generates:

```text
models/model.pkl
models/vectorizer.pkl
models/metrics.json
```

### 6. Run locally

```bash
python app.py
```

Open `http://127.0.0.1:5000`.

## API Endpoint

### `POST /analyze`

Accepts form data:

```text
news_text=<headline or article text>
```

The endpoint returns the ML prediction, model probabilities, evidence score, source evidence, Google News results, and integration status fields.

Example response shape:

```json
{
  "ml_prediction": "real",
  "ml_confidence": 84.2,
  "fake_probability": 15.8,
  "real_probability": 84.2,
  "credibility_score": 76,
  "num_sources": 4,
  "sources_found": ["Reuters", "BBC"],
  "google_mentions": 6,
  "twitter_status": "unavailable"
}
```

Values above are illustrative response shapes, not guaranteed results.

## Error Handling and Limitations

- NewsAPI requires a valid API key and is subject to provider plan limits.
- Google News RSS availability and search results can change over time.
- X/Twitter verification is currently disabled rather than simulated.
- ML performance depends heavily on dataset quality, size and representativeness.
- Multiple publisher matches do not automatically prove every claim is true.
- The trusted-publisher list is an evidence heuristic, not a fact-checking authority.
- The application should not be the sole source for high-impact decisions.

## Production Deployment

The repository includes:

```text
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

Before deployment:

1. Connect the repository to a recognized cloud platform such as Render or Railway.
2. Install dependencies from `requirements.txt`.
3. Configure `NEWS_API_KEY` as a platform secret.
4. Generate the ML artifacts during the platform build process or otherwise provide them securely.
5. Start the service with Gunicorn.
6. Verify both the home page and `/analyze` endpoint.
7. Monitor hosting and external API costs.

If hosting or external API usage requires paid service, stopping the deployment is preferable to claiming a continuously live deployment that is not actually active.

## Security Notes

- API keys belong in environment variables, never in source code.
- `.env` is ignored by Git.
- Generated model artifacts are ignored by Git.
- External API failures are surfaced as unavailable/error states instead of fabricated data.

## Development Status

- Repository cleanup — complete
- Backend/API evidence integration — complete
- Frontend/backend compatibility — complete
- Production server configuration — complete
- ML training pipeline — complete
- ML evaluation/testing — complete
- Model quality improvement with a larger representative dataset — **pending**
- End-to-end cloud deployment — pending

## License

MIT License.

## Author

**Sahith Reddy**  
B.Tech CSE — BVRIT Narsapur
