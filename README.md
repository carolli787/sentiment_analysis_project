# Twitter Sentiment API

A sentiment classifier for tweets, served by a Django REST Framework API. Send it a sentence and it returns `positive` or `negative` with a confidence score.

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"text": "Just got my new phone and I love it"}'
# {"sentiment":"positive","confidence":0.9...}
```

The full requirements are in [spec_improved.md](spec_improved.md).

## Quick start

Requires Python 3.12. Commands are run from the project root.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Train the model (about 3 minutes; writes models/sentiment_model.joblib and .json)
python -m sentiment.train

# 2. Start the API (loads the model once at startup)
python manage.py runserver

# 3. Call it
curl -X POST http://127.0.0.1:8000/evaluate -H "Content-Type: application/json" -d '{"text": "I love this!"}'

# Run the tests (they don't need a trained model, except one acceptance test that is skipped without it)
python manage.py test

# Measure latency against the running server
python scripts/benchmark.py
```

The data is expected in `trainingandtestdata/` (the Sentiment140 CSV files).

## Project layout

```
sentiment/               ML code, no Django dependency (shared by the API and the notebook)
  preprocessing.py       normalize_text(): the one text-cleaning function used in training and serving
  data.py                load, clean (dedupe), and split the data
  classifier.py          SentimentClassifier with train(), evaluate(), predict(), save(), load()
  train.py               training script: tune on validation, score once on test, save model + metrics
  tests/
sentiment_api/           Django app: HTTP concerns only
  serializers.py         request validation (400s) and response shape
  views.py               POST /evaluate, GET /health
  services.py            loads the classifier once per process
  exceptions.py          JSON 500s, never stack traces
  middleware.py          request logging (status, latency, label; never the text)
  tests/
config/                  Django settings, URLs, WSGI (the model is loaded in wsgi.py at startup)
scripts/benchmark.py     latency benchmark
models/                  trained model (.joblib, not committed) and its metrics (.json)
sentiment_analysis_project.ipynb   data exploration
```

```
 OFFLINE                                              ONLINE
 trainingandtestdata/*.csv                            client
        │                                               │ POST /evaluate {"text": ...}
        ▼                                               ▼
 sentiment.data: load, dedupe, split                  sentiment_api view: validate (400/415/405)
        │                                               │
        ▼                                               ▼
 SentimentClassifier.train()   ◄── same normalize_text ──►  SentimentClassifier.evaluate()
        │                                               │
        ▼                                               ▼
 models/sentiment_model.joblib ── loaded once at startup ──►  {"sentiment", "confidence"}
```

## API

### `POST /evaluate`

Request: `{"text": "<1 to 1000 characters>"}` with `Content-Type: application/json`.

Response `200 OK`: `{"sentiment": "positive" | "negative", "confidence": 0.5–1.0}`. `confidence` is the model's probability for the returned label; values near 0.5 mean it's unsure.

| Status | When |
|--------|------|
| `400` | Invalid JSON; `text` missing, `null`, not a string, empty or whitespace-only, or over 1,000 characters |
| `405` | Any method other than `POST` |
| `415` | Content type is not `application/json` |
| `500` | Unexpected error (JSON body, no details; the error is logged) |

Errors use DRF's standard format, for example `{"text": ["This field may not be blank."]}`.

### `GET /health`

`{"status": "ok"}` once the server is up with the model loaded.

### Why `/evaluate`?

The assignment names the endpoint `POST /evaluate`, so I kept it. A strictly resource-oriented design would use a noun, such as `POST /sentiments` or `POST /sentiment-predictions`. `POST` is the right method either way: the request carries a body, and the result isn't a stored resource that `GET` could retrieve.

## Modeling

**Data.** Sentiment140: 1.6M tweets labeled automatically from emoticons (so the labels are noisy), and a hand-labeled test set of 498 tweets.

**Decisions:**
- **Binary classifier.** The training data has no neutral tweets, so the 139 neutral test tweets are excluded from scoring.
- **Duplicates removed.** Texts that appear with both labels are dropped entirely, and other duplicates are kept once. This prevents the same tweet from appearing in both training and validation, and stops spam from being counted many times.
- **Stratified, shuffled split** (98% train, 2% validation, seed 42). The raw file is sorted by label.
- **Model selection on validation only.** The test set is scored once, after the model is chosen.
- **Preprocessing.** Lowercase, decode HTML entities, replace URLs and @mentions with placeholder tokens, shorten "soooo" to "soo", and treat "don't" and "dont" as the same word. Negation words are kept, since "not good" ≠ "good".
- **Features and model.** TF-IDF over word unigrams and bigrams (bigrams capture phrases like "not good"), with logistic regression. This is the standard strong baseline for short-text classification. It trains in under a minute, predicts in well under a millisecond, and its probabilities give a usable confidence score.

## Results

From `models/sentiment_model.json` (seed 42). Rerunning `python -m sentiment.train` reproduces these numbers.

| Regularization (C) | Validation accuracy |
|--------------------|---------------------|
| 0.25 | 82.06% |
| 0.5 | 82.38% |
| 1.0 | 82.86% |
| **2.0 (chosen)** | **82.87%** |

The curve is flat between 1.0 and 2.0, so larger values aren't worth exploring.

| Set | Rows | Accuracy | Macro F1 |
|-----|------|----------|----------|
| Validation | 31,585 | 82.87% | 82.87% |
| Test (positive and negative only) | 359 | **82.45%** | 82.34% |

The test set's confusion matrix (rows show the true label, columns the predicted label):

|              | pred. negative | pred. positive |
|--------------|---------------:|---------------:|
| **negative** | 134 | 43 |
| **positive** | 20  | 162 |

- Test accuracy is close to validation accuracy, even though the test set was labeled by hand and collected differently. The model handles that shift well.
- Most test errors are negative tweets predicted as positive, and many of those have low confidence (0.5–0.6). They are mostly criticism aimed at a company or a politician (AIG, Cheney, AT&T, GM), sarcasm, or mixed tweets ("Lebron is a Beast, but I'm still cheering 4 the A"). The training data, which is personal tweets labeled from emoticons, has few examples like these.
- The result is in line with the 80–83% that Go et al. (2009) reported for simple models on this same test set.
- **Latency.** On an Apple Silicon laptop running `runserver`, `scripts/benchmark.py` measured p50 0.6 ms and p95 0.8 ms per request. Loading the model at startup takes about 1 second, and the model file is about 20 MB.
