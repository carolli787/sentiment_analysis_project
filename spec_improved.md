# Twitter Sentiment Classifier: Spec

## 1. Problem statement and intended use

Build a sentiment classifier for tweets. Given one tweet (or a short message like a tweet), it returns whether the sentiment is **positive** or **negative**. The model is served through a REST API.

**Intended use**
- Short, informal, English text of roughly tweet length (up to about 280 characters): tweets, chat messages, short comments.
- Scoring one message per request.

**Out of scope (misuse cases)**
- Long-form text such as articles, reviews or emails. The model was never trained on these, so results won't be reliable.
- Non-English text.
- Making decisions about individual people (for example moderation, hiring or credit) based on the predicted sentiment of what they wrote.
- Detecting neutral text, sarcasm, emotion categories (anger, joy, and so on), or sentiment toward a specific target ("I love the phone but hate the carrier").

## 2. Goals and non-goals

**Goals**
1. A `train` function that fits the classifier on the labeled training data and saves the trained model to disk.
2. An `evaluate` function that takes one sentence and returns its predicted sentiment.
3. A RESTful API that exposes `evaluate` through `POST /evaluate`, using correct HTTP methods and status codes.
4. Clean, documented code. `train` and `evaluate` can be methods on a class or standalone functions, in a notebook or a script.

**Non-goals**
- State-of-the-art accuracy. Clear, well-justified modeling choices matter more than the last few points of accuracy.
- A neutral class (see §3).
- Batch scoring, authentication, rate limiting or production deployment.
- Retraining through the API. Training runs offline.

## 3. Task definition

**Granularity:** one label per input text (document-level), not per sentence or per target.

**Label scheme**

| Raw value in data | Label      | Used for training? |
|-------------------|------------|--------------------|
| `0`               | `negative` | Yes                |
| `2`               | `neutral`  | No: only in the test file, excluded from scoring |
| `4`               | `positive` | Yes                |

The classifier is **binary**. It always returns `positive` or `negative`, plus a confidence score between 0 and 1.

**Hard-case rules**
- **Empty or whitespace-only input:** reject with `400`. Don't guess a label.
- **Input with no recognizable words** (only URLs, @mentions, or punctuation): return a prediction. The confidence score will usually be close to 0.5, and that low confidence is the signal to the caller.
- **Very long input:** reject anything over a fixed limit (proposed: 1,000 characters) with `400`.
- **Emoticons and emoji:** the training labels came from emoticons, and the emoticons were then **removed** from the training text. The model has therefore never seen them, so its prediction for a text like "ok :(" depends only on the words. This is a known limitation, not a bug.
- **Mixed or neutral sentiment:** the model will still pick one of the two labels. The confidence score reflects the uncertainty.
- **Negation** ("not good", "dont like"): the preprocessing and features must not discard negation words (for example, by removing stopwords that include "not").

## 4. Data

**Source:** Sentiment140 (Go, Bhayani and Huang, 2009), stored in `trainingandtestdata/`.

| File | Rows | Labels | Notes |
|------|------|--------|-------|
| `training.1600000.processed.noemoticon.csv` | 1,600,000 | 800k negative / 800k positive | Labeled automatically from emoticons (distant supervision), so the labels are noisy. Rows are **sorted by label**. |
| `testdata.manual.2009.06.14.csv` | 498 | 177 negative / 182 positive / 139 neutral | Labeled by hand. Collected around specific search queries (products, companies, people). |

**Format:** CSV, no header row, `latin-1` encoding. Columns: `polarity, id, date, query, user, text`. Only `text` (input) and `polarity` (label) are used as model inputs. The date, query and user columns are not used as features.

**Cleaning rules** (applied before splitting)
- Drop exact duplicate texts. Where the same text appears with both labels, drop every copy.
- Shuffle the data before any split or subsample, because the file is sorted by label.
- The same text normalization must run during training and at inference time (one shared function).

**Splits**
- **Train / validation:** a stratified split of the cleaned training file (proposed: 98% / 2%), fixed random seed. Use validation for model selection and tuning.
- **Test:** the 359 positive and negative rows of the manual test file. Use it **once**, for the final reported numbers only. Because it was collected differently from the training data, it also measures how well the model handles a shift in data.

**PII:** the data contains Twitter usernames (`user` column) and @mentions in the text.
- Don't use usernames as features.
- The API must not log the raw input text.
- Don't commit the data files or any derived files that contain raw tweets to version control.

**Versioning:** record the source file names, row counts after cleaning, the random seed and the split sizes alongside each saved model.

## 5. How success will be measured

All metrics are computed on the binary (positive/negative) rows only.

| Metric | Where measured | Why |
|--------|----------------|-----|
| Accuracy | Validation and test | Classes are balanced, so accuracy is meaningful |
| Macro F1 | Validation and test | Catches a model that favors one class |
| Confusion matrix | Test | Shows which kind of error dominates |
| Inference latency (p50 / p95, single request) | Local API benchmark | "Brownie points" for performance |

**Baseline to beat:** a TF-IDF + logistic regression model trained on the same split. Any more complex model must beat this baseline on validation to justify its extra cost.

## 6. Acceptance criteria

Proposed targets. Adjust them after the baseline is measured.

- [ ] Test accuracy ≥ 80% on the 359 binary test rows (roughly the level of published simple baselines on this data set).
- [ ] Gap between validation accuracy and test accuracy is reported and explained.
- [ ] `train` runs end-to-end from the raw CSV files and saves the model to disk. Running it again with the same seed reproduces the same metrics.
- [ ] `evaluate("I love this!")` returns `positive` and `evaluate("This is the worst day ever")` returns `negative`.
- [ ] `POST /evaluate` meets the contract in §7, including error cases.
- [ ] The API loads the model once at server startup (in `config/wsgi.py`), not on every request.
- [ ] p95 latency for one request is under 100 ms on a laptop once the model is loaded (stretch goal).
- [ ] Automated tests cover the preprocessing function, `evaluate`, and the API's success and error responses.

## 7. API specification

Framework: **Django + Django REST Framework (DRF)**. Avoma uses this stack, so reviewers can read the code easily, and DRF's built-in parsing and validation cover most of the error cases below.

**Design**
- A Django app (for example `sentiment_api`) with one `APIView` whose `post` method checks the input, calls `classifier.evaluate(text)` and returns the response. The view contains no ML logic.
- Input is checked by a serializer:
  ```python
  class EvaluateRequestSerializer(serializers.Serializer):
      text = serializers.CharField(max_length=1000)  # trims whitespace; rejects missing, blank or too-long text
  ```
- The ML code (preprocessing, `train`, `evaluate`) lives in a plain Python module that doesn't depend on Django, so the notebook and the API share it.
- No database is needed. Don't define Django models.
- The assignment names the endpoint `/evaluate`, so that name is kept. A purely noun-based REST design would use something like `POST /sentiments`. The README mentions this choice.

### `POST /evaluate`

Evaluates one sentence and returns its sentiment. The request is handled synchronously.

**Request**
```json
{ "text": "Just got my new phone and I love it" }
```

**Response: `200 OK`**
```json
{
  "sentiment": "positive",
  "confidence": 0.93
}
```
`confidence` is the model's probability for the returned label (always ≥ 0.5).

**Errors**

| Status | When | Handled by |
|--------|------|------------|
| `400 Bad Request` | Body is not valid JSON, or `text` is missing, `null`, a list or object, empty or whitespace-only, or over the length limit (DRF's `CharField` turns numbers into strings and accepts them) | DRF parser and serializer |
| `415 Unsupported Media Type` | `Content-Type` is not `application/json` | DRF (`parser_classes = [JSONParser]`) |
| `405 Method Not Allowed` | Any method other than `POST` on `/evaluate` | DRF `APIView` |
| `500 Internal Server Error` | Unexpected failure. The response body must not include a stack trace (`DEBUG = False`). | Django |

If the model file can't be loaded, the server fails at startup with a clear error message, instead of starting and then failing on requests.

Error bodies use DRF's standard format:
```json
{ "text": ["This field may not be blank."] }
```
```json
{ "detail": "Unsupported media type \"text/plain\" in request." }
```

### `GET /health` (optional)

Returns `200 OK` with `{"status": "ok"}` once the model is loaded, so the server can be checked before sending requests.

## 8. Operational aspects

- **Model loading:** the trained model is saved as a file (for example with `joblib`). The server loads it once at startup in `config/wsgi.py` and keeps it in memory. (Not in `AppConfig.ready()`, which also runs for management commands such as `manage.py test` that don't need a trained model.) Slow startup is acceptable. Request latency should be low.
- **Configuration:** the model path, `SECRET_KEY` and `DEBUG` come from environment variables (read in `settings.py`), not hard-coded values.
- **Logging:** log each request's status code, latency and predicted label. Never log the input text.
- **Reproducibility:** pin dependency versions (`requirements.txt` or `pyproject.toml`) and fix random seeds.
- **How to run:** the README explains how to install dependencies, train the model, start the API and call it with `curl`.

## 9. Code guidelines

- All code is documented: a docstring on every public function and class, and comments explaining *why* where the reason isn't obvious.
- Code follows standard Python style (PEP 8) and uses type hints.
- The notebook is for exploration and explaining decisions. Code the API needs (preprocessing, model loading, `evaluate`) lives in an importable module, so the notebook and the API share the same code.

## 10. Open questions

1. Which model family to try after the baseline: linear models only, or also a fine-tuned transformer? A transformer may be more accurate but is slower and larger.
2. Should the API return only the label, or the label plus confidence as proposed above?
3. Is the 80% test accuracy target right, once the baseline result is known?
