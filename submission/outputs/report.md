# COMP 4221 NLP Course Project Report

## Project Scope

This project implements the full noisy-news workflow required by the assignment: light preprocessing for an unlabeled corpus, retrieval with two classical bag-of-words ranking models, and a small supervised topic-classification experiment built on a directly reviewed label set that was later refined through a consensus-assisted external LLM audit workflow.

## Task 1: Corpus Audit and Preprocessing

- Documents: `871`
- Total cleaned tokens: `1,147,881`
- Average document length: `1317.89`
- Median document length: `1250.00`
- Vocabulary size before preprocessing: `57,401`
- Vocabulary size after preprocessing: `47,998`

The corpus parser splits documents with the leading `@@<numeric_id>` marker. Cleaning is intentionally light: repeated gap noise is mapped to `<GAP>`, text is lowercased, and whitespace is normalized while punctuation, stopwords, and rare tokens are preserved for downstream interpretability.

## Task 2: Retrieval

- Average TF-IDF `P@5`: `0.76`
- Average BM25 `P@5`: `0.78`
- TF-IDF wins: `3`
- BM25 wins: `3`
- Ties: `4`
- Best BM25 advantage: `Which articles report on basketball, boxing, golf, baseball, or other sports competitions and athlete performance?` (`0.80` vs `0.40`)
- Best TF-IDF advantage: `Which articles cover movies, television, theater, books, or entertainment industry figures?` (`0.80` vs `0.60`)

The retrieval comparison shows a narrow BM25 advantage overall, but the result is query-dependent rather than absolute. BM25 performs better on compact, topic-heavy queries, while TF-IDF remains stronger when exact lexical overlap dominates the information need.

## Task 2 Qrels Validation

- Validation passed: `True`
- Covered ranked rows: `100/100`
- Missing qrels rows: `0`
- Explicit label mismatches: `0`


## Task 3: Topic Classification

- Labeled documents: `200`
- Labeling method: `single_reviewer_curated_with_external_llm_audit` from `direct_review`
- Review summary path: `/Users/nagato/workspace/project/outputs/task3_label_review_summary.json`

- Review tag: `direct_review_v2`
- Corrected labels after direct review: `19`

- Additional changes from the previous reviewed release: `13`
- Baseline test accuracy / macro-F1: `0.8667` / `0.8561`
- Improved test accuracy / macro-F1: `0.9333` / `0.9330`
- Selected SVM `C`: `1.0`

### Label Distribution

- Politics: 43
- Culture/Entertainment: 37
- Sports: 32
- Business/Economy: 32
- Crime/Law: 31
- Other: 25


### External LLM Audit

- Providers: `vertex, xai`
- Shared audited documents: `200`
- Cross-provider agreement rate: `0.905`
- Unanimous disagreements against the released labels: `0`

### Main Error Pattern

- Crime/Law -> Other (1 cases): `5022441`
- Politics -> Crime/Law (1 cases): `4114241`

The classification experiment now uses a directly reviewed label file that was then re-adjudicated on a subset of externally flagged disagreements. It is therefore stronger than the earlier direct-review-only release, though it still falls short of adjudicated multi-human gold labels.

## Reproducibility Notes

- The current Task 3 labels were directly reviewed after document inspection and then re-adjudicated on unanimous external-audit disagreements, and the correction summary is stored in `outputs/task3_label_review_summary.json`.
- The external audit itself is preserved in `outputs/task3_external_reviews/` and summarized in `outputs/task3_llm_audit_summary.json`.
- The earlier bootstrap recipe remains available in `configs/task3_bootstrap.yaml` as an auxiliary seed-generation artifact, but it is no longer the authoritative label source.
- Retrieval judgments are stored in a model-independent qrels file keyed by query-document pairs.
- The repository can be rebuilt with shell scripts rather than notebooks, and all main entry points expose explicit CLI arguments.

## Figures

![Document length histogram](figures/doc_length_hist.png)

![Zipf plot](figures/zipf_plot.png)

![Task 3 confusion matrix](figures/confusion_matrix.png)
