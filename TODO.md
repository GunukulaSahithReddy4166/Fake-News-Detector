# SafeSource — Project Completion Checklist

This checklist tracks the engineering state without claiming features that are not implemented.

## 1. Repository and Codebase

- [x] Remove obsolete launch scripts and temporary files
- [x] Remove ngrok binaries/archives from the repository
- [x] Move the training dataset into `data/news.csv`
- [x] Add `models/` for generated model artifacts
- [x] Add `.env.example`
- [x] Protect secrets and generated artifacts with `.gitignore`

## 2. Backend and Verification

- [x] Flask application and `/analyze` endpoint
- [x] Replace simulated NewsAPI results with real NewsAPI requests
- [x] Replace simulated Google News data with real Google News RSS parsing
- [x] Remove simulated Twitter/X counts
- [x] Add explicit unavailable status for unsupported X/Twitter verification
- [x] Add API error handling and timeouts
- [x] Calculate evidence score from ML and external evidence signals

## 3. Frontend

- [x] Fix frontend/backend mismatch for `estimated_tweets: null`
- [x] Prevent the X/Twitter result card from crashing when unavailable
- [x] Remove random analytics generated with `Math.random()`
- [x] Remove misleading hardcoded live-news/trending claims
- [x] Show API-based verification sources clearly
- [x] Improve safe rendering of returned source names

## 4. Production Configuration

- [x] Clean `requirements.txt`
- [x] Add Gunicorn
- [x] Add `Procfile`
- [x] Configure Flask for production-style startup
- [x] Document environment-variable configuration

## 5. ML Pipeline

- [x] Verify `data/news.csv` schema
- [x] Train from the repository dataset path
- [x] Make model output paths consistent with `models/`
- [x] Make `app.py` load model artifacts from the same paths
- [x] Verify fake/real class handling
- [x] Train the model successfully in GitHub Actions
- [x] Calculate holdout Accuracy, Precision, Recall and F1
- [x] Calculate finite 5-fold cross-validation metrics
- [x] Remove unsupported accuracy claims
- [x] Test known fake and known real examples
- [x] Test completely new headlines
- [x] Check probability ranges and confidence behavior
- [ ] Improve model quality using a larger, representative labeled dataset

### Current ML evaluation

- Dataset: 126 examples
- Holdout accuracy: 65.38%
- Holdout precision (real): 62.50%
- Holdout recall (real): 45.45%
- Holdout F1 (real): 52.63%
- 5-fold CV accuracy: 56.40% ± 7.20%
- 5-fold CV F1 (real): 38.08% ± 11.02%

**Conclusion:** the current model is functional but not strong enough to claim production-grade fact-checking accuracy. Dataset expansion is the next ML improvement.

## 6. Testing

- [x] Run Python lint checks
- [x] Verify generated model artifacts
- [x] Verify Flask imports the trained model
- [x] Test home/Flask application components
- [x] Test valid `/analyze` request with external services mocked
- [x] Test empty/short input
- [x] Test X/Twitter unavailable state
- [x] Test ML prediction probabilities
- [x] Test known fake and real examples
- [x] Test new/random-style headlines
- [x] Review GitHub Actions result
- [x] All automated tests pass: 6/6

## 7. Deployment

- [ ] Deploy to a recognized cloud platform such as Render or Railway
- [ ] Configure `NEWS_API_KEY` as a platform secret
- [ ] Configure model training/artifacts for the deployment build
- [ ] Confirm Gunicorn starts successfully
- [ ] Verify the public home page
- [ ] Verify `/analyze` in the deployed environment
- [ ] Document deployment steps and service limitations
- [ ] Stop/deactivate the service if continuous hosting would create unnecessary cost

## Final Definition of Done

The project is complete when:

1. The ML model can be trained from the repository dataset.
2. The Flask application loads the generated model reliably.
3. NewsAPI and Google News provide real evidence when available.
4. Unsupported X/Twitter functionality is clearly marked unavailable rather than simulated.
5. The frontend never crashes because an optional integration is unavailable.
6. No fake/random statistics are presented as live data.
7. Local and production startup paths are documented.
8. Automated testing passes.
9. The ML model has been evaluated honestly and its limitations are documented.
10. Deployment configuration works on a recognized hosting platform.
