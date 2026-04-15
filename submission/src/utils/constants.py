"""Project-wide constants and canonical paths."""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
ANNOTATIONS_DIR = DATA_DIR / "annotations"
CONFIGS_DIR = PROJECT_ROOT / "configs"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
RETRIEVAL_RESULTS_DIR = OUTPUTS_DIR / "retrieval_results"
PER_QUERY_DIR = RETRIEVAL_RESULTS_DIR / "per_query"
TASK3_EXTERNAL_REVIEWS_DIR = OUTPUTS_DIR / "task3_external_reviews"

DOCS_DIR = PROJECT_ROOT / "docs"
SUBMISSION_DIR = PROJECT_ROOT / "submission"

DEFAULT_RANDOM_SEED = 42

RAW_SAMPLE_PATH = RAW_DIR / "sample.txt"
RAW_CORPUS_PATH = RAW_DIR / "text_news.txt"

CLEANED_JSONL_PATH = PROCESSED_DIR / "text_news_cleaned.jsonl"
CLEANED_TXT_PATH = PROCESSED_DIR / "text_news_cleaned.txt"
QUERIES_PATH = DATA_DIR / "queries.json"
LABELS_PATH = ANNOTATIONS_DIR / "trackA_labels.jsonl"
RETRIEVAL_QRELS_PATH = ANNOTATIONS_DIR / "retrieval_qrels.csv"
TASK3_BOOTSTRAP_LABELS_PATH = ANNOTATIONS_DIR / "trackA_labels_bootstrap_seed.jsonl"
TASK3_BOOTSTRAP_CONFIG_PATH = CONFIGS_DIR / "task3_bootstrap.yaml"

TASK1_STATS_MD_PATH = OUTPUTS_DIR / "task1_stats.md"
TASK1_STATS_PDF_PATH = OUTPUTS_DIR / "task1_stats.pdf"
DOC_LENGTH_FIGURE_PATH = FIGURES_DIR / "doc_length_hist.png"
ZIPF_FIGURE_PATH = FIGURES_DIR / "zipf_plot.png"
CONFUSION_MATRIX_PATH = FIGURES_DIR / "confusion_matrix.png"

TOP5_EVALUATION_PATH = RETRIEVAL_RESULTS_DIR / "top5_evaluation.csv"
RETRIEVAL_SUMMARY_PATH = RETRIEVAL_RESULTS_DIR / "summary.csv"
RETRIEVAL_PER_QUERY_SCORES_PATH = RETRIEVAL_RESULTS_DIR / "per_query_scores.csv"
RETRIEVAL_QRELS_VALIDATION_JSON_PATH = RETRIEVAL_RESULTS_DIR / "qrels_validation.json"
RETRIEVAL_QRELS_VALIDATION_MD_PATH = RETRIEVAL_RESULTS_DIR / "qrels_validation.md"
TASK2_ANALYSIS_PATH = OUTPUTS_DIR / "task2_analysis.md"

SPLIT_PATH = OUTPUTS_DIR / "split.json"
TASK3_METRICS_PATH = OUTPUTS_DIR / "task3_metrics.json"
TASK3_REPORT_PATH = OUTPUTS_DIR / "task3_report.md"
TASK3_PREDICTIONS_PATH = OUTPUTS_DIR / "task3_predictions.csv"
TASK3_BOOTSTRAP_SUMMARY_PATH = OUTPUTS_DIR / "task3_bootstrap_summary.json"
TASK3_LABEL_REVIEW_SUMMARY_PATH = OUTPUTS_DIR / "task3_label_review_summary.json"
TASK3_LLM_AUDIT_JSON_PATH = OUTPUTS_DIR / "task3_llm_audit_summary.json"
TASK3_LLM_AUDIT_MD_PATH = OUTPUTS_DIR / "task3_llm_audit_summary.md"
FINAL_REPORT_MD_PATH = OUTPUTS_DIR / "report.md"
FINAL_REPORT_PDF_PATH = OUTPUTS_DIR / "report.pdf"

LABEL_DEFINITIONS_PATH = PROJECT_ROOT / "src" / "utils" / "labels.yaml"
ANNOTATION_GUIDELINE_PATH = DOCS_DIR / "annotation_guideline.md"
REQUIREMENTS_LOCK_PATH = PROJECT_ROOT / "requirements-lock.txt"
PYTHON_VERSION_PATH = PROJECT_ROOT / ".python-version"

RAW_GAP_MARKER = "@ @ @ @ @ @ @ @ @ @"
GAP_TOKEN = "<GAP>"

TASK3_LABELS = [
    "Politics",
    "Business/Economy",
    "Sports",
    "Crime/Law",
    "Culture/Entertainment",
    "Other",
]

LABEL_SHORTCUTS = OrderedDict(
    [
        ("p", "Politics"),
        ("b", "Business/Economy"),
        ("s", "Sports"),
        ("c", "Crime/Law"),
        ("e", "Culture/Entertainment"),
        ("o", "Other"),
    ]
)
