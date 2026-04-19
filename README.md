# COMP 4221 NLP Course Project

Chinese version: see [README.zh-CN.md](README.zh-CN.md).

This repository implements the full course project workflow for the noisy news corpus assignment:

- Task 1: corpus audit and light preprocessing
- Task 2: document retrieval with unigram TF-IDF and BM25
- Task 3: small supervised topic classification with LLM-assisted six-label annotation and cross-provider external audit

The project is organized as a script-first, reproducible pipeline. All generated artifacts that are required by the assignment are stored in the repository after a successful run.

## Environment

The safest setup is a local virtual environment because some system Python installations block direct package installation into the interpreter. The project shell scripts are Bash entrypoints. On Windows, run them from Git Bash; PowerShell and CMD are not supported entrypoints for the `.sh` wrappers.

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Windows with Git Bash:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The project targets Python 3.10+ and was exercised with Python 3.14.
After the virtual environment exists, the Bash scripts auto-detect the project `.venv` on macOS/Linux and Windows Git Bash. If detection fails, pass the interpreter explicitly:

```bash
PYTHON=/c/path/to/project/.venv/Scripts/python.exe bash scripts/run_full_pipeline.sh
```

The `requirements.txt` file is version-pinned to the exact package versions used for the current reproducible run.
For a stronger environment snapshot that also freezes transitive dependencies, build `requirements-lock.txt` from a clean temporary virtual environment:

```bash
bash scripts/build_lockfile.sh
```

That script also writes `.python-version` so the interpreter version used for the lock snapshot is explicit.

## Repository Layout

```text
project/
├── .python-version
├── configs/
│   ├── task3_bootstrap.yaml      # Task 3 weak-supervision recipe
│   └── task3_review_updates.yaml # LLM audit re-adjudication records
├── data/
│   ├── raw/                   # raw assignment corpora
│   ├── processed/             # cleaned corpus artifacts
│   ├── annotations/           # Task 2 qrels and Task 3 labels
│   └── queries.json           # 10 retrieval queries
├── docs/
│   └── annotation_guideline.md
├── outputs/
│   ├── figures/
│   ├── retrieval_results/
│   ├── report.md
│   ├── report.pdf
│   ├── split.json
│   ├── task1_stats.md
│   ├── task1_stats.pdf
│   ├── task2_analysis.md
│   ├── task3_bootstrap_summary.json # optional seed-label artifact
│   ├── task3_external_reviews/      # optional provider-specific second-pass reviews
│   ├── task3_llm_audit_summary.json
│   ├── task3_llm_audit_summary.md
│   ├── task3_label_review_summary.json
│   ├── task3_metrics.json
│   ├── task3_predictions.csv
│   └── task3_report.md
├── requirements-lock.txt
├── scripts/
│   ├── build_lockfile.sh
│   ├── package_submission.sh
│   ├── prepare_submission.sh
│   ├── run_task3_external_audit.sh
│   ├── run_task1.sh
│   ├── run_retrieval.sh
│   ├── run_task3.sh
│   └── run_full_pipeline.sh
├── src/
│   ├── apply_task3_review_updates.py
│   ├── build_task3_llm_audit.py
│   ├── task1_preprocess.py
│   ├── task2_retrieval.py
│   ├── compute_p_at_5.py
│   ├── task3_external_review.py
│   ├── validate_retrieval_qrels.py
│   ├── verify_submission.py
│   ├── build_task2_analysis.py
│   ├── build_final_report.py
│   ├── task3_classification.py
│   └── utils/
└── preprocess.py              # compatibility shim for assignment submission
```

## Reproducing the Main Results

Run the commands below from the repository root.

### Full End-to-End Run

```bash
bash scripts/run_full_pipeline.sh
```

This convenience script executes Task 1, Task 2, Task 3, and the consolidated final report generator in sequence. The optional external Task 3 audit remains a separate workflow because it depends on third-party APIs.

### Task 1: Preprocessing and Corpus Audit

```bash
bash scripts/run_task1.sh
```

Generated artifacts:

- `data/processed/text_news_cleaned.jsonl`
- `data/processed/text_news_cleaned.txt`
- `outputs/task1_stats.md`
- `outputs/task1_stats.pdf`
- `outputs/figures/doc_length_hist.png`
- `outputs/figures/zipf_plot.png`

Notes:

- The parser splits documents by the leading `@@<numeric_id>` marker.
- Cleaning is intentionally light: the repeated gap placeholder is mapped to `<GAP>`, text is lowercased, and whitespace is normalized.
- The PDF is generated directly from Python so the deliverable does not depend on a platform-specific external PDF engine.

### Task 2: Retrieval

```bash
bash scripts/run_retrieval.sh
```

Generated artifacts:

- `data/queries.json`
- `data/annotations/retrieval_qrels.csv`
- `outputs/retrieval_results/top5_evaluation.csv`
- `outputs/retrieval_results/per_query/*.json`
- `outputs/retrieval_results/qrels_validation.json`
- `outputs/retrieval_results/qrels_validation.md`
- `outputs/retrieval_results/per_query_scores.csv`
- `outputs/retrieval_results/summary.csv`
- `outputs/task2_analysis.md`

Notes:

- Both TF-IDF and BM25 use the same shared tokenizer from `src/utils/tokenizer.py`.
- The shared retrieval tokenizer lowercases text, strips punctuation consistently, and removes only the `<GAP>` placeholder token.
- Canonical relevance judgments are stored in `data/annotations/retrieval_qrels.csv` as model-independent query-document labels and merged into the top-5 evaluation CSV for convenience.
- `bash scripts/run_retrieval.sh` now runs retrieval, qrels validation, Precision@5 scoring, and Task 2 analysis generation as one reproducible workflow.
- `python3 -m src.compute_p_at_5` still prefers the canonical qrels file and uses the evaluation CSV only as a fallback.
- Snippets are query-aware and capped at 250 characters.

### Task 3: Topic Classification

```bash
bash scripts/run_task3.sh
```

Generated artifacts:

- `data/annotations/trackA_labels.jsonl`
- `outputs/task3_label_review_summary.json`
- `outputs/split.json`
- `outputs/figures/confusion_matrix.png`
- `outputs/task3_metrics.json`
- `outputs/task3_predictions.csv`
- `outputs/task3_report.md`

Optional interactive annotation workflow:

```bash
python3 -m src.utils.annotator --sample-size 20
```

Notes:

- `scripts/run_task3.sh` trains and evaluates on the current released label file without regenerating seed labels.
- The earlier bootstrap recipe remains available in `configs/task3_bootstrap.yaml` as an auxiliary seed-generation artifact, but it is no longer the authoritative Task 3 label source.
- If you want to regenerate bootstrap seed labels for comparison, use `bash scripts/bootstrap_task3_labels.sh`; it now writes to a separate seed-label file instead of overwriting the released label file.
- The label set is fixed to six classes: `Politics`, `Business/Economy`, `Sports`, `Crime/Law`, `Culture/Entertainment`, and `Other`.
- The released labels were produced by a three-stage LLM-assisted pipeline (BM25 seed → GPT-5.4 review → Gemini/Grok cross-provider audit). Documents with unanimous external disagreement were re-adjudicated (13 corrections in `configs/task3_review_updates.yaml`).
- This approach provides reproducible, auditable labels with cross-model quality checks, but does not constitute adjudicated multi-human gold labels. LLM-generated labels may favor lexically salient topics, potentially inflating bag-of-words classifier accuracy.
- The split is stratified 70/15/15 with `random_state=42`.
- The baseline model is `CountVectorizer + MultinomialNB`.
- The improved model is `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) + LinearSVC`.
- A **label noise robustness test** randomly flips 5–30% of train labels and retrains, confirming that macro-F1 degrades monotonically (0.933 → 0.853), which validates that the classifier learns genuine topic signal.
- A **per-class F1 breakdown** identifies which categories are hardest for the bag-of-words classifier.
- **Systematic error pattern analysis** extracts dominant TF-IDF terms from misclassified documents, pooling test and dev errors, with pair-specific domain insights.

### Optional Task 3 External LLM Audit

This workflow requests an independent second-pass review from external vendor models and compares those judgments against the released Task 3 labels. It is useful for label auditing, but it does not turn the dataset into adjudicated human gold labels.

Set provider API keys as environment variables and then run:

```bash
export VERTEX_AI_API_KEY="..."
export XAI_API_KEY="..."
bash scripts/run_task3_external_audit.sh
```

Generated artifacts:

- `outputs/task3_external_reviews/vertex.jsonl`
- `outputs/task3_external_reviews/xai.jsonl`
- `outputs/task3_llm_audit_summary.json`
- `outputs/task3_llm_audit_summary.md`

Notes:

- The released Task 3 label file is not overwritten by this audit workflow.
- `src/task3_external_review.py` supports `--limit`, `--doc-ids`, and `--resume` for safe incremental runs.
- API keys are read only from environment variables and should never be committed into the repository.
- The current released labels have already been reconciled against the unanimous external disagreements from this audit workflow, so the latest audit summary should show zero remaining consensus conflicts with the published label file.

## Current Results Snapshot

### Task 1

- Documents: `871`
- Total cleaned tokens: `1,147,881`
- Average document length: `1317.89`
- Median document length: `1250`
- Vocabulary size before preprocessing: `57,401`
- Vocabulary size after preprocessing: `47,998`

### Task 2

- Average `Precision@5` for TF-IDF: `0.76`
- Average `Precision@5` for BM25: `0.78`
- Baseline wins: `3`
- Improved wins: `3`
- Ties: `4`

### Task 3

- Baseline test accuracy: `0.8667`
- Baseline test macro-F1: `0.8561`
- Improved test accuracy: `0.9333`
- Improved test macro-F1: `0.9330`

### Task 3 External Audit

- Cross-provider agreement rate: `0.905`
- Remaining unanimous disagreements against the released label set: `0`

## Final Report

```bash
"$PYTHON" -m src.build_final_report
```

Generated artifacts:

- `outputs/report.md`
- `outputs/report.pdf`

The final report aggregates current Task 1, Task 2, and Task 3 outputs into one submission-friendly document and embeds the main figures from the project.

## Submission Preparation

To rebuild the pipeline, remove cache artifacts, validate required deliverables, and create a clean submission bundle under `submission/`, run:

```bash
bash scripts/prepare_submission.sh
```

Validation is also available as a standalone command:

```bash
"$PYTHON" -m src.verify_submission --root .
"$PYTHON" -m src.verify_submission --root submission
```

To create a ready-to-upload archive after the clean submission bundle exists, run:

```bash
bash scripts/package_submission.sh
```

## Assignment Deliverables Mapping

| Assignment item | File in this repository |
| --- | --- |
| `preprocess.py` | `preprocess.py` |
| `task1_stats.pdf` | `outputs/task1_stats.pdf` |
| `text_news_cleaned.txt` | `data/processed/text_news_cleaned.txt` |
| Retrieval code | `src/task2_retrieval.py` |
| Precision@5 script | `src/compute_p_at_5.py` |
| Qrels validation script | `src/validate_retrieval_qrels.py` |
| Queries | `data/queries.json` |
| Relevance judgments | `data/annotations/retrieval_qrels.csv` |
| Final project report | `outputs/report.pdf` |
| Classification code | `src/task3_classification.py` |
| Annotations | `data/annotations/trackA_labels.jsonl` |
| Annotation guideline | `docs/annotation_guideline.md` |
| Optional external audit | `src/task3_external_review.py`, `src/build_task3_llm_audit.py` |

## Implementation Notes

- `sample.txt` is copied into `data/raw/` for debugging parity with the original assignment files.
- Retrieval judgments are preserved in a canonical qrels file keyed by query-document pairs, so re-running retrieval does not depend on the previous result ordering or model-specific rank positions.
- Retrieval qrels are also validated automatically against the current top-5 result table, and the validation summary is saved under `outputs/retrieval_results/`.
- Task 3 labels were produced by a three-stage LLM-assisted pipeline (BM25 seed generation, GPT-5.4 primary review, Gemini/Grok cross-provider audit), providing reproducible and auditable annotation with cross-model quality checks rather than single-human manual labeling.
- The earlier Task 3 bootstrap parameters still live in `configs/task3_bootstrap.yaml`, but that seed-generation recipe is no longer the authoritative label source.
- The optional Task 3 external-audit workflow stores provider-specific review outputs separately from the released label file so second-pass checks never overwrite the authoritative dataset.
- `requirements.txt` pins top-level dependencies, while `requirements-lock.txt` and `.python-version` provide a stronger frozen environment snapshot when needed.
- All shell scripts source `scripts/detect_python.sh` to automatically find a Python ≥ 3.10 interpreter (checking project and active virtual environments in Unix `bin/` and Windows Git Bash `Scripts/` layouts, then Homebrew and system Python). Override with `PYTHON=/path/to/python3` if needed.
- `scripts/prepare_submission.sh` creates a clean `submission/` bundle after rebuilding outputs and stripping cache artifacts.
- The code includes detailed English comments and docstrings, and the repository intentionally avoids notebooks in favor of scriptable workflows.
- Task 2 analysis includes **per-query case studies** showing actual top-3 retrieved documents for the best BM25 and TF-IDF advantage queries, with content-specific explanations of why each model wins.
