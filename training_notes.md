# Training notes

Reference notes on what `python -m sentiment.train` does, how to read its output, and the ideas behind the model.

## The training run, step by step

Example output from one run:

```
Cleaning: CleaningReport(rows_in=1600000, conflicting_rows_dropped=6895, duplicate_rows_dropped=13864, rows_out=1579241)
Split sizes: train=1547656 validation=31585 test=359
regularization=0.25: validation accuracy=0.8206 macro F1=0.8206
regularization=0.5:  validation accuracy=0.8238 macro F1=0.8238
regularization=1.0:  validation accuracy=0.8286 macro F1=0.8285
regularization=2.0:  validation accuracy=0.8287 macro F1=0.8287
Chose regularization=2.0. Test accuracy=0.8245 macro F1=0.8234
```

### 1. Cleaning

[`clean_training_data`](sentiment/data.py) only removes duplicate tweets. It doesn't change any text.

- **Conflicting rows (6,895 dropped):** tweets with the exact same text that appear once as positive and once as negative. At least one label must be wrong and there's no way to tell which, so every copy is removed.
- **Duplicate rows (13,864 dropped):** for texts that appear more than once with the same label, only the first copy is kept. This stops repeated tweets (often spam) from counting many times, and stops the same tweet from landing in both the training and validation sets, which would make the validation score look better than it really is.

1,600,000 − 6,895 − 13,864 = 1,579,241 rows remain.

### 2. Train, validation and test sets

The cleaned data is shuffled and split 98% / 2% into **train** (1,547,656) and **validation** (31,585). The **test** set doesn't come from that split. It's a separate, hand-labeled file, `testdata.manual.2009.06.14.csv`. That file has 498 tweets; the 139 neutral ones are dropped because the model only predicts positive or negative, which leaves 359.

| Set | Job |
|---|---|
| Train | The model learns from this. |
| Validation | Used to pick the best setting (regularization). Because the winner is chosen by looking at these scores, the validation number is slightly optimistic. |
| Test | Used once, at the very end, on data that played no part in any decision. It gives the honest estimate of how the model does on new tweets. |

The test set also matters because the training labels in Sentiment140 were generated automatically from emoticons like `:)` and `:(`, while the test tweets were labeled by people. The test score shows whether the model agrees with human judgment and not just with the emoticon rule.

### 3. Choosing the regularization value

One model is trained per regularization value, and the one with the highest **validation accuracy** wins ([train.py](sentiment/train.py)). F1 isn't used in the decision, although here the two agree.

1.0 and 2.0 are effectively tied: 0.8286 vs 0.8287 is a difference of about 3 tweets out of 31,585. Choosing 2.0 is fine, but it isn't meaningfully better.

### 4. Reading the confusion matrix

Rows are the **true** label and columns are the **predicted** label, both in the order [negative, positive]. Test set:

```
                    predicted neg   predicted pos
actually negative       134              43        ← 43 negatives wrongly called positive
actually positive        20             162        ← 20 positives wrongly called negative
```

- The diagonal (134 + 162 = 296) is correct: 296 / 359 = 82.45% accuracy.
- The validation matrix reads the same way: 12,931 and 13,245 correct, with 2,828 and 2,581 errors. Those errors are fairly balanced.
- On the test set the model makes **twice as many mistakes on negative tweets** as on positive ones (43 vs 20), so it leans toward saying "positive". With only 359 test tweets, that could partly be chance. Accuracy alone wouldn't show this.
- **Macro F1** is the F1 score worked out separately for each class and then averaged. It's slightly lower than accuracy on the test set (0.8234 vs 0.8245) because of that imbalance between the two kinds of mistake.

## Logistic regression

The model works in two stages ([classifier.py](sentiment/classifier.py)).

**Stage 1: turn text into numbers (TF-IDF).** Every word and two-word phrase in the vocabulary becomes a column, such as `love`, `sad`, `not good` or `!`. A tweet becomes a row of numbers showing which of those terms it contains. TF-IDF gives a higher value to terms that are rare across all tweets, because rare terms say more than common words like "the".

**Stage 2: logistic regression.** The model learns one **weight** per term. Positive weights push toward "positive" and negative weights push toward "negative". To classify a tweet it:

1. **Adds up the evidence.** It multiplies each term's value by its weight and sums them. For *"I love this but the ending was not good"*, the sum might look like:

   ```
   love      +2.1
   not good  −1.8
   ending    −0.1
   ...
   total     +0.3
   ```

2. **Turns the total into a probability** with the *sigmoid* function, which squashes any number into the range 0 to 1:
   - a total of 0 gives 50%
   - a large positive total gives close to 100% positive
   - a large negative total gives close to 0%

   Here, +0.3 gives about 57% positive. That number becomes the `confidence` in the API response.

3. **Decides.** Above 50% means positive; otherwise negative.

**Training** is the search for the weights that make these probabilities match the true labels as closely as possible across all 1.5 million tweets. The model is penalized most when it's confidently wrong.

Despite the name, logistic regression is a *classification* model. The "regression" part refers to the weighted sum underneath.

### Why logistic regression was chosen

- **It's a strong standard baseline for short text.** Sentiment in tweets is mostly carried by individual words and short phrases. A weighted sum of those captures most of the signal, and bigrams handle simple negation like "not good".
- **It's fast.** It trains in under a minute per setting and predicts in well under a millisecond, which suits an API.
- **It gives real probabilities.** That's where the confidence score comes from. A linear SVM would be about as accurate but doesn't natively output probabilities.
- **It's easy to inspect.** You can list the most positive and most negative weights and see exactly what the model learned.

The main tradeoff: it ignores word order beyond pairs and can't follow sarcasm or longer context. A fine-tuned transformer model such as BERT would likely score several points higher, but would be much slower and heavier to run.

## Regularization

Regularization adds a second goal to training. Without it, the only goal is to fit the training labels. With it, the goal becomes:

> fit the training labels **+** keep the weights small

Every unit of weight has a cost, so the model only gives a term a large weight when that term earns it across many tweets.

**Example.** Say some username appears in just 3 tweets, and all 3 happen to be negative. Without a penalty, the model could give that username a large negative weight to get those 3 tweets perfectly right. That's memorizing noise, and it would wrongly push any future tweet mentioning that user toward negative. With a penalty, a big weight on that username costs more than it gains, because it only helps 3 tweets. A word like "love" appears in tens of thousands of positive tweets, so a big weight there easily pays for itself. Regularization keeps the strong, consistent signals and shrinks the flukes.

**The `regularization` value is scikit-learn's `C`, the *inverse* of regularization strength.** It controls how expensive weights are:

| C | Penalty | Effect on weights | Effect on predictions |
|---|---|---|---|
| 0.25 | strong | all weights shrink, rare terms shrink most | totals stay near 0, so probabilities stay near 50%: more cautious, less confident |
| 2.0 | weak | weights can grow larger | more confident probabilities, more risk of fitting noise |

### Reading the results with this in mind

Validation accuracy went 0.8206 → 0.8238 → 0.8286 → 0.8287 as C increased. With 1.5 million tweets, many terms have enough evidence to deserve larger weights, so loosening the penalty helped. The gain almost disappeared between 1.0 and 2.0, which suggests the benefit has levelled off. Going higher would likely gain nothing, and at some point the model would start fitting noise and validation accuracy would drop.

Because the weights feed directly into the probabilities, C also changes how confident the model's answers are, not just whether they're right. Models with C = 0.25 and C = 2.0 might give the same answer for a tweet, but with different confidence numbers.
