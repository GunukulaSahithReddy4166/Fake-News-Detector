# SafeSource — AI-Powered Fake News Detection

SafeSource is a Flask-based web application that combines **machine-learning text classification** with **external news-source evidence** to assess the credibility of a submitted news claim.

> **Important:** SafeSource provides evidence signals, not a guarantee that a claim is true or false. External coverage and an ML prediction should be interpreted together.

## Features

- **Machine Learning:** TF-IDF text features with a scikit-learn classification model.
- **NewsAPI verification:** Searches real publisher coverage when `NEWS_API_KEY` is configured.
- **Google News RSS:** Searches related coverage through Google News RSS.
- **Trusted-source signal:** Counts distinct publishers and identifies configured trusted publishers.
- **Credibility score:** Combines ML confidence and external-source evidence into a single 0–100 score.
- **Transparent X/Twitter status:** X/Twitter data is not fabricated. The current version reports the integration as unavailable because no live X API is configured.
- **Responsive frontend:** HTML, CSS and vanilla JavaScript frontend connected to Flask's `/analyze` endpoint.
- **Production deployment configuration:** Gunicorn and a `Procfile` are included for platforms that support Procfile-based Python services.

## Architecture

```text
User enters news claim
        │
        ▼
   Flask /analyze
        │
        ├──────────────► ML model
        │                 TF-IDF → classification
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
 Credibility score + explanation
        │
        ▼
      Web UI
```

## Technology Stack

| Layer | Technology |
|---|---|
| Backend | Python, Flask |
| Machine Learning | scikit-learn, TF-IDF, joblib |
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

The submitted text is normalized before ML inference. The application removes unnecessary punctuation/whitespace and prepares the text for the trained vectorizer.

### 2. ML prediction

The trained scikit-learn model receives TF-IDF features and returns a fake/real prediction with probabilities.

### 3. NewsAPI evidence

If `NEWS_API_KEY` is available, SafeSource searches NewsAPI for related articles and counts distinct publishers. A small configured list of established publishers is used as a trusted-source signal.

### 4. Google News evidence

The application performs a Google News RSS search and counts related results. This is treated as supporting evidence, not proof of authenticity.

### 5. X/Twitter

The application deliberately does **not** generate simulated tweet counts or trending values. Until a real X API integration is configured, the UI clearly reports that social-media verification is unavailable.

### 6. Final score

The current backend combines:

```text
ML confidence       → 55%
NewsAPI source      → 30%
Google News signal  → 15%
```

The score is an application-level evidence score; it is **not** a probability that a claim is objectively true.

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

Never commit a real API key. The repository's `.gitignore` is configured to exclude `.env`.

### 5. Train the model

The training pipeline is maintained separately from runtime inference. Run:

```bash
python train_model.py
```

This step will generate the model artifacts required by the application. The exact artifact location is part of the final ML-pipeline cleanup and testing stage.

### 6. Run locally

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

## API Endpoint

### `POST /analyze`

Accepts form data:

```text
news_text=<headline or article text>
```

The endpoint returns JSON containing the ML prediction, probabilities, credibility score, source evidence, Google News results, and integration status fields.

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

- NewsAPI requires a valid API key and is subject to the provider's plan limits and restrictions.
- Google News RSS availability and search results can change over time.
- X/Twitter verification is currently disabled rather than simulated.
- ML accuracy depends heavily on the quality, size and representativeness of the training dataset.
- A news article being reported by several publishers does not automatically prove that every claim in it is true.
- The configured trusted-publisher list is an evidence heuristic, not an endorsement or fact-checking authority.
- The application should not be used as the sole source for high-impact decisions.

## Production Deployment

The repository includes a Gunicorn-based production command in `Procfile`:

```text
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

A Procfile-compatible cloud platform can use this command to start the Flask application. Before deployment, configure the required environment variables in the platform's secret/environment settings.

### Recommended deployment approach

1. Connect the GitHub repository to a recognized cloud platform such as Render or Railway.
2. Configure the Python environment and build/install dependencies from `requirements.txt`.
3. Set `NEWS_API_KEY` as a secret environment variable.
4. Use the repository's `Procfile` to start Gunicorn.
5. Verify the home page and `POST /analyze` flow after deployment.
6. Monitor API limits and hosting costs before keeping a public instance continuously active.

A deployment does not need to remain permanently online for the project to demonstrate production deployment knowledge. If a hosting provider or external API requires paid usage, it is better to stop the service than to claim that an inactive deployment is continuously live.

## Security Notes

- API keys belong in environment variables, never in source code.
- `.env` is ignored by Git.
- Model artifacts should be generated during the controlled build/training process rather than committed accidentally.
- External API failures are surfaced as unavailable/error states instead of being replaced with fabricated data.

## Development Status

The repository is being completed in stages:

- Repository cleanup — complete
- Backend/API evidence integration — complete
- Frontend/backend compatibility — complete
- Production server configuration — complete
- Documentation — complete
- ML training/model artifact pipeline — final cleanup stage
- End-to-end testing — pending
- Cloud deployment verification — pending

## License

MIT License.

## Author

**Sahith Reddy**  
B.Tech CSE — BVRIT Narsapur
