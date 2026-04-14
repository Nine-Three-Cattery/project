# COMP 4221 NLP Course Project — Final Implementation Plan

> **Goal**: Build a reproducible, well-documented NLP pipeline on a noisy news corpus.
> Scoring priorities: clear methodology > reproducibility > thoughtful analysis > raw performance.

---

## 1. Principles

- Use **only** the provided `sample.txt` and `text_news.txt`. No external data corpora.
- **No external pretrained models** in the main pipeline. Local open-source libraries (scikit-learn, rank_bm25, etc.) are allowed.
- All human-judgment deliverables (queries, relevance labels, topic annotations) are completed by the author, not by any LLM.
- All final results must be traceable to: the given corpus + author's code + author's manual judgments.

---

## 2. Task Selection

| Task | Choice | Rationale |
|------|--------|-----------|
| Task 1 | Light, non-destructive preprocessing | Preserves information for downstream tasks |
| Task 2 | Unigram TF-IDF (baseline) vs BM25 (improved) | Directly matches assignment examples; easy to explain |
| Task 3 | **Track A** — Topic Classification with Manual Labels | More conventional, confusion matrix & macro-F1 are natural fits |

---

## 3. Repository Structure

```
project/
├── data/
│   ├── raw/
│   │   ├── sample.txt
│   │   └── text_news.txt
│   ├── processed/
│   │   ├── text_news_cleaned.jsonl      # internal format
│   │   └── text_news_cleaned.txt        # submission format (@@id style)
│   ├── annotations/
│   │   └── trackA_labels.jsonl
│   └── queries.json
├── src/
│   ├── task1_preprocess.py
│   ├── task2_retrieval.py
│   ├── compute_p_at_5.py
│   ├── task3_classification.py
│   └── utils/
│       ├── tokenizer.py                 # shared tokenize() for TF-IDF & BM25
│       ├── annotator.py
│       └── labels.yaml
├── scripts/
│   ├── run_task1.sh
│   ├── run_retrieval.sh
│   └── run_task3.sh
├── outputs/
│   ├── task1_stats.md                   # source for task1_stats.pdf
│   ├── retrieval_results/
│   │   ├── top5_evaluation.csv
│   │   ├── summary.csv
│   │   └── per_query/
│   ├── figures/
│   │   ├── doc_length_hist.png
│   │   ├── zipf_plot.png
│   │   └── confusion_matrix.png
│   └── split.json
├── docs/
│   └── annotation_guideline.md
├── requirements.txt
└── README.md
```

---

## 4. Phase 1 — Task 1: Preprocessing & Corpus Audit

### 4.1 Parser

- Split on `@@` followed by a numeric ID.
- Extract `doc_id` (the numeric portion) and `raw_text`.

### 4.2 Cleaning Steps

| Step | Action | Reason |
|------|--------|--------|
| Gap noise | Map `@ @ @ @ @ @ @ @ @ @` → `<GAP>` | Preserves structure without polluting vocabulary |
| Case | Lowercase all text | Standard normalization |
| Whitespace | Collapse multiple spaces / normalize `\r\n` | Clean formatting |
| Stopwords | **Not removed** at this stage | Task 2 retrieval needs them for BM25 `avgdl`; Task 3 vectorizers handle this internally |
| Punctuation | **Not removed** at this stage | Sentence splitting in Task 2 snippets depends on punctuation |
| Rare tokens | **Not removed** at this stage | Over-aggressive pruning hurts retrieval recall |

### 4.3 Output Formats

**Internal** — `text_news_cleaned.jsonl`:
```json
{"doc_id": "3000041", "raw_text": "...", "clean_text": "..."}
```

**Submission** — `text_news_cleaned.txt`:
Keep the **exact same line structure** as the original `text_news.txt`:
```
@@3000041  cleaned text for doc 1 ...
@@3000341  cleaned text for doc 2 ...
```
Each document stays on a **single line**. `@@id` and cleaned text on the same line with two-space separation.

### 4.4 Statistics to Report

- Number of documents
- Total token count
- Average document length (tokens)
- Median document length (tokens)
- Vocabulary size **before** preprocessing
- Vocabulary size **after** light cleaning
- 3 before/after examples

### 4.5 Visualizations

- **Document length distribution histogram** (mandatory)
- **Zipf's law plot** — log-log word frequency distribution (recommended)

### 4.6 Supervised vs Unsupervised Judgment

Write explicitly:
- The corpus contains no label columns, no topic/category metadata.
- Document IDs (`@@3000041`) are numeric identifiers, not meaningful labels.
- Therefore the corpus is an **unsupervised (unlabeled) text collection**.
- This is why Task 3 requires manual annotation to create labels.

### 4.7 Deliverables

- `src/task1_preprocess.py` (corresponds to required `preprocess.py`)
- `data/processed/text_news_cleaned.jsonl`
- `data/processed/text_news_cleaned.txt`
- `outputs/task1_stats.md` (export to `task1_stats.pdf`)
- `outputs/figures/doc_length_hist.png`
- `outputs/figures/zipf_plot.png`

---

## 5. Phase 2A — Task 2: Query Scaffold

### 5.1 Format

Create `data/queries.json` with 10 slots. Final queries **must be authored by the student**.

### 5.2 Query Writing Guidelines

- Write as **natural-language information needs**, not keyword lists.
- Cover diverse topics present in the corpus.
- Examples of acceptable style:
  - "Which documents discuss economic growth or inflation trends?"
  - "Find reports about sports matches or athlete performance."
  - "Which articles cover criminal cases, court decisions, or law enforcement?"

---

## 6. Phase 2B — Task 3: Annotation Setup

### 6.1 Label Scheme

Start with a **pilot browse** of 30–50 random documents to determine the actual topic distribution. Then choose between:

**Option A (5 classes)**:
- Politics
- Business/Economy
- Sports
- Culture/Entertainment
- Other

**Option B (6 classes)** — if pilot confirms substantial Crime/Law coverage:
- Politics
- Business/Economy
- Sports
- Crime/Law
- Culture/Entertainment
- Other

Decision rule: if the pilot shows ≥10% of sampled documents are clearly about crime, law enforcement, courts, or prison, use 6 classes. Otherwise stay with 5.

### 6.2 Annotation Volume

| Parameter | Value |
|-----------|-------|
| Minimum (assignment) | 120 documents |
| Initial batch | 160 documents (random sample) |
| Re-sample trigger | Any class < 15 documents |
| Re-sample size | 20 documents per round |
| Max re-sample rounds | 2 |
| Target total | **200 documents** |

All re-sampling uses **random selection**, never targeted by class.

### 6.3 Annotation Tool (`annotator.py`)

- Read from `text_news_cleaned.jsonl`, display **raw_text**
- Keyboard shortcuts for each label
- Undo support
- Incremental save to `data/annotations/trackA_labels.jsonl`

### 6.4 Annotation Guideline (`docs/annotation_guideline.md`)

Must include:
- Definition of each label
- Inclusion / exclusion rules per label
- How to handle mixed-topic documents
- When to assign `Other`
- Revision log for label set changes

---

## 7. Phase 3 — Task 2: Retrieval Pipeline

### 7.1 Shared Tokenizer

Write a unified `tokenize(text)` function in `src/utils/tokenizer.py`.
**Both TF-IDF and BM25 must use this same function.** This guarantees fair comparison.

The tokenizer should:
- Lowercase
- Handle punctuation (strip or split)
- Apply consistent stopword policy
- **Filter out `<gap>` tokens** — they appear in nearly every document and have no discriminative power. Including them would pollute IDF scores and inflate BM25 `avgdl`.

TF-IDF integration:
```python
TfidfVectorizer(tokenizer=tokenize, preprocessor=lambda x: x, token_pattern=None)
```

BM25 integration:
```python
corpus_tokenized = [tokenize(doc) for doc in documents]
bm25 = BM25Okapi(corpus_tokenized)
```

### 7.2 Baseline — Unigram TF-IDF

- `TfidfVectorizer` with the shared tokenizer
- Unigram only
- Cosine similarity for ranking

### 7.3 Improved — BM25

- `rank_bm25.BM25Okapi` with the shared tokenizer
- Default parameters (k1=1.5, b=0.75) unless dev tuning justifies changes

### 7.4 Snippet Strategy

Simplified query-aware snippets (do not over-engineer):

1. Find the first sentence containing the most query tokens
2. If the sentence is too short, extend to adjacent sentences
3. If sentence splitting is unreliable, fall back to a 30–50 token window
4. Cap at **250 characters**

### 7.5 Output

**Main table** — `outputs/retrieval_results/top5_evaluation.csv`:

| Field | Description |
|-------|-------------|
| Query | The query text |
| Model | `tfidf` or `bm25` |
| Rank | 1–5 |
| Doc_ID | Document identifier |
| Snippet | Query-aware excerpt |
| Is_Relevant | `Y` / `N` / blank (filled by student) |

**Per-query detail files** — `outputs/retrieval_results/per_query/query_N.json`

**Summary** — `outputs/retrieval_results/summary.csv`:
- Baseline wins / Improved wins / Ties

### 7.6 Evaluation

- Student manually marks relevance for **at least 5 queries** (all 5 top-5 results must be labeled).
- `compute_p_at_5.py` computes Precision@5 for both systems on fully-labeled queries only.

### 7.7 Error Analysis

From the results, select and write up:
- **1 success case**: a query where the improved system clearly outperformed the baseline, with explanation of why.
- **1 failure case**: a query where neither system returned relevant results, or the improved system performed worse, with analysis.

---

## 8. Phase 4 — Task 3: Classification

### 8.1 Data Split

- Performed **only after all annotations are complete**
- `stratify=y`
- Fixed `random_state=42`
- **Ratio: 70 / 15 / 15** (train / dev / test)
  - With 200 docs: ~140 train, ~30 dev, ~30 test
  - Larger train set benefits small-data learning
- Save split indices to `outputs/split.json`

### 8.2 Models

**Baseline**:
- `CountVectorizer`
- `MultinomialNB`

**Improved**:
- `TfidfVectorizer(ngram_range=(1,2), min_df=2, sublinear_tf=True)`
- `LinearSVC(C=1.0)`

Minimal dev tuning allowed (e.g., try C ∈ {0.1, 1.0, 10.0}). Do not turn this into a hyperparameter sweep.

### 8.3 Metrics

- Accuracy
- Macro-F1
- Confusion matrix (saved as `outputs/figures/confusion_matrix.png`)

### 8.4 Error Analysis — "3 Frequent Error Types"

Do **not** write generic FP/FN descriptions. Instead:

1. Identify the **top 3 confusion pairs** from the confusion matrix
2. For each pair, show 1–2 representative misclassified documents:
   - `doc_id / gold_label / predicted_label / text snippet`
3. Explain **why** the confusion is understandable (e.g., overlapping vocabulary, short documents, mixed topics)

Special note on `Other`: if `Other` is a major confusion source (likely), discuss its inherent ambiguity as one of the 3 error types. The `Other` category has no clear semantic boundary, so low precision there is expected — acknowledge it rather than hiding it.

### 8.5 Optional Supplement

- 5-fold stratified cross-validation result as a robustness check
- This goes in an appendix, **not** as the primary evaluation
- The primary evaluation must use the train/dev/test split (as the assignment requires)

---

## 9. Dependencies

```
numpy
pandas
scikit-learn
rank_bm25
matplotlib
seaborn
pyyaml
tqdm
rich
```

Discipline:
- Do not import libraries not listed in `requirements.txt`
- Do not include unused libraries in `requirements.txt`

---

## 10. README Structure

```markdown
## Environment
pip install -r requirements.txt
Python >= 3.10

## Task 1: Preprocessing
bash scripts/run_task1.sh

## Task 2: Retrieval
1. Author finalizes data/queries.json (10 natural-language queries)
2. bash scripts/run_retrieval.sh
3. Author fills Is_Relevant in top5_evaluation.csv (≥5 queries)
4. python src/compute_p_at_5.py

## Task 3: Classification
1. Author runs annotator to label documents
   python src/utils/annotator.py
2. bash scripts/run_task3.sh

## Notes
- Author's annotation files are included for reproducibility
- src/task1_preprocess.py corresponds to the required preprocess.py
- outputs/task1_stats.md is the source for task1_stats.pdf
```

---

## 11. Final Submission Checklist

| Item | Source | Submit As |
|------|--------|-----------|
| Preprocessing script | `src/task1_preprocess.py` | `preprocess.py` |
| Corpus stats | `outputs/task1_stats.md` | `task1_stats.pdf` |
| Cleaned corpus | `data/processed/text_news_cleaned.txt` | `text_news_cleaned.txt` |
| Retrieval code | `src/task2_retrieval.py` | as-is |
| P@5 script | `src/compute_p_at_5.py` | as-is |
| Queries | `data/queries.json` | as-is |
| Relevance judgments | `top5_evaluation.csv` | as-is |
| Classification code | `src/task3_classification.py` | as-is |
| Annotations | `data/annotations/trackA_labels.jsonl` | as-is |
| Annotation guideline | `docs/annotation_guideline.md` | as-is or PDF |
| README | `README.md` | as-is |
| Report | (written separately) | `report.pdf` |

---

## 12. Report Template

### Task 1
- How the parser works
- Why light cleaning was chosen
- Why `<GAP>` mapping
- Supervised vs unsupervised judgment with evidence
- What the statistics and plots show

### Task 2
- Why TF-IDF as baseline
- Why BM25 as improved
- Why both share a single tokenizer
- Precision@5 results table
- 1 success case with explanation
- 1 failure case with explanation

### Task 3
- Why Track A
- Label definitions and annotation guideline summary
- How train/dev/test was split (ratio, seed, stratification)
- Baseline vs improved: accuracy, macro-F1, confusion matrix
- 3 frequent error types with concrete examples

### Discussion
- On a small, manually-annotated course corpus, we emphasize interpretability, reproducibility, and qualitative error analysis rather than chasing complex models.
- Acknowledge small test set limitation and label distribution constraints.

---

## 13. Execution Timeline

| Day | Deliverables |
|-----|-------------|
| **Day 1** | Project skeleton, Task 1 complete (parser, stats, plots, cleaned corpus), annotator tool, guideline draft |
| **Day 2** | Student finalizes 10 queries, retrieval pipeline, top-5 CSV, student does relevance judgments (≥5 queries), P@5, select success/failure cases |
| **Day 3** | Student completes 160–200 annotations, train classifiers, confusion matrix, 3 error types, optional CV supplement |
| **Day 4** | Write report, verify PDF formatting, finalize README, assemble submission package |
