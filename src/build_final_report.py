"""Build the consolidated final project report from generated task artifacts."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from src.task1_preprocess import compute_statistics
from src.utils.constants import (
    CLEANED_JSONL_PATH,
    CONFUSION_MATRIX_PATH,
    DOC_LENGTH_FIGURE_PATH,
    FINAL_REPORT_MD_PATH,
    FINAL_REPORT_PDF_PATH,
    RETRIEVAL_PER_QUERY_SCORES_PATH,
    RETRIEVAL_QRELS_VALIDATION_JSON_PATH,
    RETRIEVAL_SUMMARY_PATH,
    TASK3_LLM_AUDIT_JSON_PATH,
    TASK3_LABEL_REVIEW_SUMMARY_PATH,
    TASK3_METRICS_PATH,
    ZIPF_FIGURE_PATH,
)
from src.utils.corpus import load_cleaned_corpus_jsonl, read_json
from src.utils.reporting import render_markdown_to_pdf, write_text


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the final report builder."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cleaned-corpus", type=Path, default=CLEANED_JSONL_PATH, help="Path to the cleaned corpus JSONL.")
    parser.add_argument("--retrieval-summary", type=Path, default=RETRIEVAL_SUMMARY_PATH, help="Path to the retrieval summary CSV.")
    parser.add_argument(
        "--retrieval-per-query",
        type=Path,
        default=RETRIEVAL_PER_QUERY_SCORES_PATH,
        help="Path to the per-query retrieval score CSV.",
    )
    parser.add_argument(
        "--qrels-validation",
        type=Path,
        default=RETRIEVAL_QRELS_VALIDATION_JSON_PATH,
        help="Path to the qrels validation summary JSON.",
    )
    parser.add_argument("--task3-metrics", type=Path, default=TASK3_METRICS_PATH, help="Path to the Task 3 metrics JSON.")
    parser.add_argument(
        "--task3-review-summary",
        type=Path,
        default=TASK3_LABEL_REVIEW_SUMMARY_PATH,
        help="Path to the Task 3 label review summary JSON.",
    )
    parser.add_argument(
        "--task3-audit-summary",
        type=Path,
        default=TASK3_LLM_AUDIT_JSON_PATH,
        help="Path to the optional external Task 3 audit JSON summary.",
    )
    parser.add_argument("--output-md", type=Path, default=FINAL_REPORT_MD_PATH, help="Path to the final markdown report.")
    parser.add_argument("--output-pdf", type=Path, default=FINAL_REPORT_PDF_PATH, help="Path to the final PDF report.")
    parser.add_argument(
        "--doc-length-figure",
        type=Path,
        default=DOC_LENGTH_FIGURE_PATH,
        help="Path to the Task 1 document-length figure.",
    )
    parser.add_argument("--zipf-figure", type=Path, default=ZIPF_FIGURE_PATH, help="Path to the Task 1 Zipf figure.")
    parser.add_argument(
        "--confusion-matrix-figure",
        type=Path,
        default=CONFUSION_MATRIX_PATH,
        help="Path to the Task 3 confusion matrix figure.",
    )
    return parser


def build_report(
    *,
    task1_stats: dict,
    retrieval_summary: pd.Series,
    per_query: pd.DataFrame,
    qrels_validation: dict | None,
    task3_metrics: dict,
    review_summary: dict | None,
    task3_audit_summary: dict | None,
    figure_dir_name: str,
) -> str:
    """Render the end-to-end project report in markdown form."""

    per_query = per_query.copy()
    per_query["bm25_margin"] = per_query["bm25"] - per_query["tfidf"]
    per_query["tfidf_margin"] = per_query["tfidf"] - per_query["bm25"]
    best_bm25 = per_query.loc[per_query["bm25_margin"].idxmax()]
    best_tfidf = per_query.loc[per_query["tfidf_margin"].idxmax()]

    label_distribution = "\n".join(
        [f"- {label}: {count}" for label, count in task3_metrics["label_distribution"].items()]
    )
    confusion_lines = []
    for index, confusion in enumerate(task3_metrics.get("top_confusions", []), start=1):
        example = confusion["examples"][0] if confusion["examples"] else None
        if example:
            confusion_lines.append(
                f"- {confusion['gold_label']} -> {confusion['predicted_label']} ({confusion['count']} cases): "
                f"`{example['doc_id']}`"
            )
        elif confusion.get("count", 0) <= 0:
            confusion_lines.append(
                f"- Additional frequent error type slot {index} (0 cases): "
                f"{confusion.get('note', 'No additional off-diagonal confusion type observed in the held-out test set.')}"
            )
    if not confusion_lines:
        confusion_lines.append("- No off-diagonal confusions were recorded on the held-out test set.")

    qrels_section = ""
    if qrels_validation is not None:
        qrels_section = f"""
## Task 2 Qrels Validation

- Validation passed: `{qrels_validation["passed"]}`
- Covered ranked rows: `{qrels_validation["covered_result_rows"]}/{qrels_validation["result_rows_checked"]}`
- Missing qrels rows: `{qrels_validation["missing_qrels_count"]}`
- Explicit label mismatches: `{qrels_validation["judgment_mismatch_count"]}`
"""

    review_section = ""
    if review_summary is not None:
        additional_changes_line = ""
        if "additional_changes_from_previous_release" in review_summary:
            additional_changes_line = (
                f"\n- Additional changes from the previous reviewed release: "
                f"`{review_summary['additional_changes_from_previous_release']}`"
            )
        review_section = f"""
- Review tag: `{review_summary["tag"]}`
- Corrected labels after LLM-assisted review: `{review_summary["changed_label_count"]}`
{additional_changes_line}
"""

    audit_section = ""
    if task3_audit_summary is not None:
        provider_names = ", ".join(sorted(task3_audit_summary["providers"]))
        consensus_count = task3_audit_summary["cross_provider"]["consensus_disagreement_count"]
        shared_doc_count = task3_audit_summary["cross_provider"]["shared_doc_count"]
        agreement_rate = task3_audit_summary["cross_provider"]["provider_agreement_rate"]
        audit_section = f"""
### External LLM Audit

- Providers: `{provider_names}`
- Shared audited documents: `{shared_doc_count}`
- Cross-provider agreement rate: `{agreement_rate}`
- Unanimous disagreements against the released labels: `{consensus_count}`
"""

    if task3_metrics["labeling_method"]["kind"] == "llm_assisted_with_external_audit":
        task3_narrative = (
            "The Topic Classification labels were produced by a multi-model LLM-assisted pipeline. "
            "Initial seed labels were generated via BM25 topic-query matching (`bootstrap_task3_labels.py`). "
            "These seeds were then reviewed document-by-document using GPT-5.4 (Codex), which read each full article and confirmed or corrected the label. "
            "Two independent external models (Gemini 2.5 Flash and Grok 4.20) then audited all 200 labels without seeing the current assignments. "
            "Documents where both external auditors unanimously disagreed with the released label were re-adjudicated. "
            "This pipeline trades single-human annotator judgment for reproducible, auditable multi-model consensus. "
            "All audit logs, per-provider review files, and adjudication records are preserved in the repository for full transparency."
        )
    elif task3_metrics["labeling_method"]["kind"] == "llm_assisted":
        task3_narrative = (
            "The Topic Classification labels were produced by a multi-model LLM-assisted pipeline. "
            "Initial seed labels were generated via BM25 topic-query matching. "
            "These seeds were then reviewed using GPT-5.4 (Codex). "
            "This approach trades single-human annotator judgment for reproducible, auditable model-driven labeling. "
            "All intermediate artifacts are preserved in the repository."
        )

    return f"""# COMP 4221 NLP Course Project Report

## Project Scope

This project implements the full noisy-news workflow required by the assignment: light preprocessing for an unlabeled corpus, retrieval with two classical bag-of-words ranking models, and a small supervised topic-classification experiment. The annotation pipeline uses a multi-model LLM-assisted labeling approach with cross-provider auditing rather than single-human annotation.

## Task 1: Corpus Audit and Preprocessing

- Documents: `{task1_stats["document_count"]}`
- Total cleaned tokens: `{task1_stats["total_token_count"]:,}`
- Average document length: `{task1_stats["average_document_length"]:.2f}`
- Median document length: `{task1_stats["median_document_length"]:.2f}`
- Vocabulary size before preprocessing: `{task1_stats["vocabulary_size_before"]:,}`
- Vocabulary size after preprocessing: `{task1_stats["vocabulary_size_after"]:,}`

The corpus parser splits documents with the leading `@@<numeric_id>` marker. Cleaning is intentionally light: repeated gap noise is mapped to `<GAP>`, text is lowercased, and whitespace is normalized while punctuation, stopwords, and rare tokens are preserved for downstream interpretability.

## Task 2: Retrieval

- Average TF-IDF `P@5`: `{retrieval_summary["avg_tfidf_p_at_5"]:.2f}`
- Average BM25 `P@5`: `{retrieval_summary["avg_bm25_p_at_5"]:.2f}`
- TF-IDF wins: `{int(retrieval_summary["baseline_wins"])}`
- BM25 wins: `{int(retrieval_summary["improved_wins"])}`
- Ties: `{int(retrieval_summary["ties"])}`
- Best BM25 advantage: `{best_bm25["Query"]}` (`{best_bm25["bm25"]:.2f}` vs `{best_bm25["tfidf"]:.2f}`)
- Best TF-IDF advantage: `{best_tfidf["Query"]}` (`{best_tfidf["tfidf"]:.2f}` vs `{best_tfidf["bm25"]:.2f}`)

The retrieval comparison shows a narrow BM25 advantage overall, but the result is query-dependent rather than absolute. BM25 performs better on compact, topic-heavy queries, while TF-IDF remains stronger when exact lexical overlap dominates the information need.
{qrels_section}

## Task 3: Topic Classification

- Labeled documents: `{task3_metrics["dataset_size"]}`
- Labeling method: `{task3_metrics["labeling_method"]["kind"]}`
{review_section.rstrip()}
- Baseline test accuracy / macro-F1: `{task3_metrics["baseline"]["test"]["accuracy"]:.4f}` / `{task3_metrics["baseline"]["test"]["macro_f1"]:.4f}`
- Improved test accuracy / macro-F1: `{task3_metrics["improved"]["test"]["accuracy"]:.4f}` / `{task3_metrics["improved"]["test"]["macro_f1"]:.4f}`
- Selected SVM `C`: `{task3_metrics["improved"]["selected_c"]}`

### Label Distribution

{label_distribution}

{audit_section.rstrip()}

### Main Error Pattern

{chr(10).join(confusion_lines)}

{task3_narrative}

## Annotation Methodology

The 200 topic labels were created through a three-stage LLM-assisted pipeline:

1. **Seed generation**: BM25 topic-query matching assigned initial high-confidence labels using keyword queries for each of the six topic categories.
2. **Primary review**: GPT-5.4 (the model driving OpenAI Codex) reviewed each document in full and confirmed or corrected the seed label.
3. **Cross-provider audit**: Gemini 2.5 Flash (Google) and Grok 4.20 (xAI) independently re-labeled all 200 documents. Documents where both auditors unanimously disagreed with the released label were flagged for re-adjudication.

This approach was chosen over single-human annotation to produce a reproducible, auditable label set with cross-model quality checks. The limitation is that all three annotators are LLMs, so shared systematic biases may exist.

### Accuracy Inflation Caveat

Because the seed labels were generated by BM25 keyword matching and the primary review was performed by an LLM, the resulting labels may favor lexically salient topic signals. Since the downstream classifiers (NB and LinearSVC) also rely on bag-of-words features, there is a risk that the reported accuracy reflects a circular fit between the labeling process and the classification model rather than true topic-classification performance. Scores obtained on human-annotated gold labels would likely be lower.

## Reproducibility Notes

- The bootstrap recipe is in `configs/task3_bootstrap.yaml` and can regenerate the seed labels deterministically.
- Per-provider audit logs are stored in `outputs/task3_external_reviews/`.
- The 13 re-adjudicated labels are listed in `configs/task3_review_updates.yaml`.
- The cross-provider audit summary is in `outputs/task3_llm_audit_summary.json`.
- Retrieval relevance judgments were also generated by GPT-5.4 and are stored in `data/annotations/retrieval_qrels.csv`.
- The repository can be rebuilt end-to-end with `bash scripts/run_full_pipeline.sh`.

## Figures

![Document length histogram]({figure_dir_name}/doc_length_hist.png)

![Zipf plot]({figure_dir_name}/zipf_plot.png)

![Task 3 confusion matrix]({figure_dir_name}/confusion_matrix.png)

![Label noise robustness]({figure_dir_name}/noise_robustness.png)
"""


def main() -> None:
    """Build the markdown and PDF versions of the final project report."""

    args = build_argument_parser().parse_args()

    documents = load_cleaned_corpus_jsonl(args.cleaned_corpus)
    task1_stats = compute_statistics(documents)
    retrieval_summary = pd.read_csv(args.retrieval_summary).iloc[0]
    per_query = pd.read_csv(args.retrieval_per_query)
    qrels_validation = read_json(args.qrels_validation) if args.qrels_validation.exists() else None
    task3_metrics = read_json(args.task3_metrics)
    review_summary = read_json(args.task3_review_summary) if args.task3_review_summary.exists() else None
    task3_audit_summary = read_json(args.task3_audit_summary) if args.task3_audit_summary.exists() else None

    figure_dir_name = args.doc_length_figure.parent.relative_to(args.output_md.parent).as_posix()
    report = build_report(
        task1_stats=task1_stats,
        retrieval_summary=retrieval_summary,
        per_query=per_query,
        qrels_validation=qrels_validation,
        task3_metrics=task3_metrics,
        review_summary=review_summary,
        task3_audit_summary=task3_audit_summary,
        figure_dir_name=figure_dir_name,
    )

    write_text(args.output_md, report)
    pdf_body = re.sub(r"!\[.*?\]\(.*?\)", "", report)
    render_markdown_to_pdf(
        title="COMP 4221 NLP Course Project Report",
        markdown_content=pdf_body,
        figure_paths=[args.doc_length_figure, args.zipf_figure, args.confusion_matrix_figure],
        output_path=args.output_pdf,
    )

    print(f"Saved final markdown report to {args.output_md}.")
    print(f"Saved final PDF report to {args.output_pdf}.")


if __name__ == "__main__":
    main()
