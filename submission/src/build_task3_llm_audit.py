"""Summarize independent Task 3 second-pass reviews from external LLM providers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from src.utils.constants import (
    LABELS_PATH,
    TASK3_EXTERNAL_REVIEWS_DIR,
    TASK3_LABELS,
    TASK3_LLM_AUDIT_JSON_PATH,
    TASK3_LLM_AUDIT_MD_PATH,
)
from src.utils.corpus import ensure_parent_dir, write_json


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the Task 3 external-audit summarizer."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reviews-dir",
        type=Path,
        default=TASK3_EXTERNAL_REVIEWS_DIR,
        help="Directory that contains provider-specific JSONL review files.",
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=LABELS_PATH,
        help="Current released Task 3 label file used to compute agreement statistics.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=TASK3_LLM_AUDIT_JSON_PATH,
        help="Path to the machine-readable audit summary JSON.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=TASK3_LLM_AUDIT_MD_PATH,
        help="Path to the markdown audit summary.",
    )
    return parser


def load_review_rows(path: Path) -> list[dict]:
    """Read a provider review JSONL file into a list of dictionaries."""

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_current_labels(path: Path) -> dict[str, str]:
    """Load the current released Task 3 labels keyed by document ID."""

    labels: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            labels[row["doc_id"]] = row["label"]
    return labels


def summarize_provider(rows: list[dict], current_labels: dict[str, str]) -> dict:
    """Compute agreement and disagreement statistics for one provider."""

    successful_rows = [row for row in rows if row.get("review_status") == "ok"]
    successful_by_doc = {row["doc_id"]: row for row in successful_rows}
    successful = list(successful_by_doc.values())

    error_rows = [row for row in rows if row.get("review_status") != "ok"]
    unresolved_errors = [row for row in error_rows if row["doc_id"] not in successful_by_doc]
    enriched_successful = []
    for row in successful:
        current_label = current_labels.get(row["doc_id"], row.get("source_label"))
        enriched_successful.append({**row, "current_label": current_label})

    disagreements = [row for row in enriched_successful if row["reviewed_label"] != row["current_label"]]

    confusion_counts = Counter((row["current_label"], row["reviewed_label"]) for row in disagreements)
    confusion_rows = [
        {
            "source_label": source_label,
            "reviewed_label": reviewed_label,
            "count": count,
        }
        for (source_label, reviewed_label), count in confusion_counts.most_common()
    ]

    reviewed_label_distribution = Counter(row["reviewed_label"] for row in successful)
    return {
        "provider": successful[0]["provider"] if successful else (error_rows[0]["provider"] if error_rows else "unknown"),
        "model": successful[0]["model"] if successful else (error_rows[0].get("model") if error_rows else None),
        "successful_reviews": len(successful),
        "error_rows": len(unresolved_errors),
        "agreement_count": sum(1 for row in enriched_successful if row["reviewed_label"] == row["current_label"]),
        "disagreement_count": len(disagreements),
        "agreement_rate": round(
            sum(1 for row in enriched_successful if row["reviewed_label"] == row["current_label"]) / len(successful),
            6,
        )
        if successful
        else None,
        "reviewed_label_distribution": {
            label: int(reviewed_label_distribution.get(label, 0)) for label in TASK3_LABELS
        },
        "confusion_counts": confusion_rows,
        "top_disagreements": [
            {
                "doc_id": row["doc_id"],
                "source_label": row["current_label"],
                "reviewed_label": row["reviewed_label"],
                "confidence": row["confidence"],
                "justification": row["justification"],
            }
            for row in sorted(disagreements, key=lambda row: (-float(row["confidence"]), row["doc_id"]))[:10]
        ],
    }


def build_cross_provider_summary(provider_rows: dict[str, list[dict]], current_labels: dict[str, str]) -> dict:
    """Measure where providers agree with each other and where they jointly challenge a source label."""

    successful_by_provider: dict[str, dict[str, dict]] = {}
    for provider_name, rows in provider_rows.items():
        successful_by_provider[provider_name] = {
            row["doc_id"]: row for row in rows if row.get("review_status") == "ok"
        }

    if not successful_by_provider:
        return {
            "shared_doc_count": 0,
            "provider_agreement_count": 0,
            "provider_agreement_rate": None,
            "consensus_disagreement_count": 0,
            "consensus_disagreements": [],
        }

    shared_doc_ids = set.intersection(*(set(rows) for rows in successful_by_provider.values()))
    provider_agreement_count = 0
    consensus_disagreements: list[dict] = []

    for doc_id in sorted(shared_doc_ids):
        provider_labels = {
            provider_name: successful_by_provider[provider_name][doc_id]["reviewed_label"]
            for provider_name in sorted(successful_by_provider)
        }
        unique_labels = set(provider_labels.values())
        if len(unique_labels) == 1:
            provider_agreement_count += 1

        source_label = current_labels.get(doc_id, next(iter(successful_by_provider.values()))[doc_id]["source_label"])
        if len(unique_labels) == 1:
            consensus_label = next(iter(unique_labels))
            if consensus_label != source_label:
                consensus_disagreements.append(
                    {
                        "doc_id": doc_id,
                        "source_label": source_label,
                        "consensus_label": consensus_label,
                        "provider_labels": provider_labels,
                        "provider_confidences": {
                            provider_name: successful_by_provider[provider_name][doc_id]["confidence"]
                            for provider_name in sorted(successful_by_provider)
                        },
                        "provider_justifications": {
                            provider_name: successful_by_provider[provider_name][doc_id]["justification"]
                            for provider_name in sorted(successful_by_provider)
                        },
                    }
                )

    return {
        "shared_doc_count": len(shared_doc_ids),
        "provider_agreement_count": provider_agreement_count,
        "provider_agreement_rate": round(provider_agreement_count / len(shared_doc_ids), 6) if shared_doc_ids else None,
        "consensus_disagreement_count": len(consensus_disagreements),
        "consensus_disagreements": consensus_disagreements[:20],
    }


def build_markdown(summary: dict) -> str:
    """Render the audit summary as a readable markdown report."""

    provider_sections = []
    for provider_name, provider_summary in sorted(summary["providers"].items()):
        disagreement_lines = []
        for row in provider_summary["top_disagreements"]:
            disagreement_lines.append(
                f"- `{row['doc_id']}` | source=`{row['source_label']}` | provider=`{row['reviewed_label']}` | "
                f"confidence={row['confidence']:.3f} | {row['justification']}"
            )
        if not disagreement_lines:
            disagreement_lines.append("- No disagreements were recorded.")

        provider_sections.append(
            f"""## {provider_name}

- Model: `{provider_summary["model"]}`
- Successful reviews: `{provider_summary["successful_reviews"]}`
- Error rows: `{provider_summary["error_rows"]}`
- Agreement count: `{provider_summary["agreement_count"]}`
- Disagreement count: `{provider_summary["disagreement_count"]}`
- Agreement rate: `{provider_summary["agreement_rate"]}`

### Top Disagreements

{chr(10).join(disagreement_lines)}
"""
        )

    consensus_lines = []
    for row in summary["cross_provider"]["consensus_disagreements"]:
        labels = ", ".join([f"{provider}={label}" for provider, label in row["provider_labels"].items()])
        consensus_lines.append(
            f"- `{row['doc_id']}` | source=`{row['source_label']}` | consensus=`{row['consensus_label']}` | {labels}"
        )
    if not consensus_lines:
        consensus_lines.append("- No unanimous cross-provider disagreements were recorded.")

    return f"""# Task 3 External LLM Audit

This report summarizes provider-independent second-pass reviews over the released Task 3 labels. The external audit is useful for spotting suspicious labels, but it does not convert the dataset into adjudicated human gold labels.

## Cross-Provider Summary

- Providers reviewed: `{", ".join(sorted(summary["providers"]))}`
- Shared successful reviews: `{summary["cross_provider"]["shared_doc_count"]}`
- Provider-to-provider agreement count: `{summary["cross_provider"]["provider_agreement_count"]}`
- Provider-to-provider agreement rate: `{summary["cross_provider"]["provider_agreement_rate"]}`
- Unanimous disagreements against the released label set: `{summary["cross_provider"]["consensus_disagreement_count"]}`

### Unanimous Disagreements

{chr(10).join(consensus_lines)}

{chr(10).join(provider_sections)}
"""


def main() -> None:
    """Build the JSON and markdown summaries for the external Task 3 audit."""

    args = build_argument_parser().parse_args()
    review_files = sorted(args.reviews_dir.glob("*.jsonl"))
    if not review_files:
        raise ValueError(f"No provider review files were found in {args.reviews_dir}.")

    current_labels = load_current_labels(args.labels)
    provider_rows = {
        review_file.stem: load_review_rows(review_file)
        for review_file in review_files
    }
    provider_summaries = {
        provider_name: summarize_provider(rows, current_labels)
        for provider_name, rows in provider_rows.items()
    }
    summary = {
        "comparison_label_source": str(args.labels),
        "providers": provider_summaries,
        "cross_provider": build_cross_provider_summary(provider_rows, current_labels),
    }

    write_json(args.output_json, summary)
    ensure_parent_dir(args.output_md)
    args.output_md.write_text(build_markdown(summary), encoding="utf-8")

    print(f"Saved Task 3 audit summary JSON to {args.output_json}.")
    print(f"Saved Task 3 audit summary markdown to {args.output_md}.")


if __name__ == "__main__":
    main()
