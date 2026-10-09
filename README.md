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

## Modeling

**Data.** Sentiment140: 1.6M tweets; test set of 498 tweets.

**Decisions:**
- **Binary classifier.** The training data has no neutral tweets, so the 139 neutral test tweets are excluded from scoring.
- **Duplicates removed.** Texts that appear with both labels are dropped entirely, and other duplicates are kept once. 
- **Stratified, shuffled split** (98% train, 2% validation, seed 42). The raw file is sorted by label.
- **Model selection on validation only.** The test set is scored once, after the model is chosen.
- **Preprocessing.** Lowercase, decode HTML entities, replace URLs and @mentions with placeholder tokens, etc. 
- **Features and model.** TF-IDF over word unigrams and bigrams, with logistic regression. 

## Results

From `models/sentiment_model.json` (seed 42). Rerunning `python -m sentiment.train` reproduces these numbers.

| Regularization (C) | Validation accuracy |
|--------------------|---------------------|
| 0.25 | 82.06% |
| 0.5 | 82.38% |
| 1.0 | 82.86% |
| **2.0 (chosen)** | **82.87%** |


| Set | Rows | Accuracy | Macro F1 |
|-----|------|----------|----------|
| Validation | 31,585 | 82.87% | 82.87% |
| Test (positive and negative only) | 359 | **82.45%** | 82.34% |

The test set's confusion matrix (rows show the true label, columns the predicted label):

|              | pred. negative | pred. positive |
|--------------|---------------:|---------------:|
| **negative** | 134 | 43 |
| **positive** | 20  | 162 |

(Spec coding) Made with Claude Code. 
