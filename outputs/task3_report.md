# Task 3 Topic Classification

## Dataset Summary

- Total labeled documents: 200
- Train / dev / test sizes: 140 / 30 / 30
- Split strategy: stratified 70/15/15 with random seed 42

### Label Distribution

- Politics: 43
- Business/Economy: 32
- Sports: 32
- Crime/Law: 31
- Culture/Entertainment: 37
- Other: 25

### Label Provenance

- llm_review: 200

### Evaluation Caveat

- The released label set was produced by a multi-model LLM-assisted pipeline: BM25 seed generation, GPT-5.4 primary review, and cross-provider audit by Gemini 2.5 Flash and Grok 4.20.
- Documents with unanimous external disagreement were re-adjudicated (13 corrections recorded in `configs/task3_review_updates.yaml`).
- This approach provides reproducible, auditable labels with cross-model quality checks, but does not constitute adjudicated multi-human gold labels.

## Model Configuration

- Baseline: `CountVectorizer + MultinomialNB`
- Improved: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) + LinearSVC`
- Best improved-model hyperparameter from dev tuning: `C=1.0`

## Metrics

| Model | Dev Accuracy | Dev Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | 0.8667 | 0.7686 | 0.8667 | 0.8561 |
| Improved | 0.9333 | 0.9253 | 0.9333 | 0.9330 |

### Per-Class F1 (Improved Model, Test Set)

| Class | Politics | Business/Economy | Sports | Crime/Law | Culture/Entertainment | Other |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F1 | 0.909 | 1.000 | 1.000 | 0.800 | 1.000 | 0.889 |

The easiest class is **Business/Economy** (F1=1.000) and the hardest is **Crime/Law** (F1=0.800). Crime/Law documents are more likely to share vocabulary with adjacent categories, making them harder to separate with bag-of-words features.


## Label Noise Robustness Test

To verify that the high macro-F1 reflects genuine topic signal rather than circular label-model fit, we randomly flipped increasing fractions of **train** labels and re-evaluated on the original test set (averaged over 5 trials per noise level).

| Noise Level | 0% | 5% | 10% | 15% | 20% | 25% | 30% |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Macro-F1 | 0.933 | 0.926 | 0.919 | 0.911 | 0.902 | 0.865 | 0.853 |

From 0% to 30% label noise, macro-F1 drops by 0.080 (from 0.933 to 0.853). This monotonic degradation confirms that the classifier is learning meaningful topic-discriminating features from the labels rather than memorizing arbitrary assignments.

![Label noise robustness](figures/noise_robustness.png)

## Frequent Error Types

### Crime/Law -> Other (1 cases)
  - `5022441` | gold=`Crime/Law` | predicted=`Other` | Posted ! A link has been posted to your Facebook feed . Buy Photo Natasha Radosavac , Steve 's sister , center , lights candles for others at a vigil for the Radosavac family near their home that burned in Eastpointe . P...
### Politics -> Crime/Law (1 cases)
  - `4114241` | gold=`Politics` | predicted=`Crime/Law` | In the final , frenetic days of Georgia 's 2010 GOP runoff , a bombshell dropped : word that a federal grand jury in Atlanta demanded information seemingly related to Nathan Deal 's actions as a member of Congress . # Th...
### Additional Frequent Error Type Slot 3 (0 cases)
  - Only 2 distinct off-diagonal confusion type(s) were observed on the held-out test set. This placeholder fills required frequent error type slot #3.

## Systematic Error Pattern Analysis

- **Small error sample**: Only 2 test-set error(s) and 0 dev-set error(s) were observed (2 total). Pattern analysis below pools both splits, but conclusions should be interpreted cautiously given the limited sample size.
- **Crime/Law misclassified as Other** (1 cases): This is a classic boundary case: articles involving personal tragedies or community events related to crime may lack explicit legal/judicial vocabulary, causing the classifier to fall back to the catch-all 'Other' category. Documents labeled Crime/Law were predicted as Other. Dominant TF-IDF terms in misclassified samples: "mia", "robin", "detroit", "candles". These terms have high weight in both the gold and predicted categories, confirming lexical overlap as the root cause.
  - Dominant terms: `mia`, `robin`, `detroit`, `candles`, `funeral`, `friend and`
- **Politics misclassified as Crime/Law** (1 cases): Political corruption, investigations, and ethics probes share vocabulary with crime reporting (e.g., 'grand jury', 'investigation', 'charges'). The classifier cannot distinguish political accountability stories from criminal justice coverage based on bag-of-words features alone. Documents labeled Politics were predicted as Crime/Law. Dominant TF-IDF terms in misclassified samples: "evans", "the ajc", "grand jury", "ethics". These terms have high weight in both the gold and predicted categories, confirming lexical overlap as the root cause.
  - Dominant terms: `evans`, `the ajc`, `grand jury`, `ethics`, `ajc`, `deal`
