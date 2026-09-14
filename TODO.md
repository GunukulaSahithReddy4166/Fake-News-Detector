# SafeSource — Project Completion Checklist

This checklist tracks the remaining engineering work without claiming features that are not implemented yet.

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
- [x] Calculate credibility from ML and external evidence signals

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
- [x] Document environment-variable configuration
- [ ] Review Flask startup/debug configuration for local vs production use

## 5. ML Pipeline — FINAL ENGINEERING STAGE

- [ ] Verify `data/news.csv` schema and dataset quality
- [ ] Update `train_model.py` to use the new dataset path
- [ ] Make model output paths consistent with `models/`
- [ ] Make `app.py` load model artifacts from the same paths
- [ ] Verify label/class handling for fake vs real predictions
- [ ] Train the model successfully
- [ ] Inspect accuracy, precision, recall and F1 score
- [ ] Avoid unsupported accuracy claims in documentation
- [ ] Test inference with representative examples

## 6. Testing

- [ ] Run Python syntax checks
- [ ] Run the application locally
- [ ] Test the home page
- [ ] Test valid `/analyze` requests
- [ ] Test empty/short input
- [ ] Test NewsAPI missing-key behavior
- [ ] Test NewsAPI error/rate-limit behavior
- [ ] Test Google News RSS failure behavior
- [ ] Confirm X/Twitter unavailable state does not break the UI
- [ ] Test ML prediction after the final model pipeline is completed
- [ ] Test complete UI → Flask → ML/API → results flow
- [ ] Review GitHub Actions result

## 7. Deployment

- [ ] Deploy to a recognized cloud platform such as Render or Railway
- [ ] Configure `NEWS_API_KEY` as a platform secret
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
8. The complete application has been tested end-to-end.
9. Deployment configuration works on a recognized hosting platform.
