"""Validate the canonical retrieval qrels against the ranked evaluation output."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from src.utils.constants import (
    QUERIES_PATH,
    RETRIEVAL_QRELS_PATH,
    RETRIEVAL_QRELS_VALIDATION_JSON_PATH,
    RETRIEVAL_QRELS_VALIDATION_MD_PATH,
    TOP5_EVALUATION_PATH,
)
from src.utils.corpus import read_json, write_json
from src.utils.reporting import write_text


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the retrieval qrels validation workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH, help="Path to the query definition JSON file.")
    parser.add_argument("--qrels", type=Path, default=RETRIEVAL_QRELS_PATH, help="Path to the canonical qrels CSV.")
    parser.add_argument(
        "--results",
        type=Path,
        default=TOP5_EVALUATION_PATH,
        help="Path to the ranked top-5 retrieval output CSV.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=RETRIEVAL_QRELS_VALIDATION_JSON_PATH,
        help="Path to the machine-readable validation summary JSON.",
    )
    parser.add_argument(
        "--md-output",
        type=Path,
        default=RETRIEVAL_QRELS_VALIDATION_MD_PATH,
        help="Path to the human-readable validation report markdown.",
    )
    return parser


def normalize_label(value: str) -> str:
    """Normalize relevance labels to stable uppercase values."""

    return str(value).strip().upper()


def build_markdown_report(summary: dict) -> str:
    """Render the validation summary as a short markdown report."""

    per_query_lines = "\n".join(
        [
            f"- {row['query_id']}: qrels={row['judged_documents']} | top5_unique_docs={row['top5_unique_documents']} | "
            f"covered_top5_docs={row['covered_top5_documents']}"
            for row in summary["per_query"]
        ]
    )
    missing_lines = "\n".join([f"- {item}" for item in summary["missing_qrels_examples"]]) or "- none"
    mismatch_lines = "\n".join([f"- {item}" for item in summary["judgment_mismatch_examples"]]) or "- none"

    return f"""# Retrieval Qrels Validation

## Status

- Validation passed: `{summary['passed']}`
- Qrels rows: `{summary['qrels_rows']}`
- Unique query-document pairs: `{summary['unique_query_doc_pairs']}`
- Result rows checked: `{summary['result_rows_checked']}`
- Top-5 rows covered by qrels: `{summary['covered_result_rows']}`
- Missing qrels rows: `{summary['missing_qrels_count']}`
- Conflicting explicit labels between qrels and result CSV: `{summary['judgment_mismatch_count']}`

## Per-Query Coverage

{per_query_lines}

## Missing Qrels Examples

{missing_lines}

## Explicit Label Mismatch Examples

{mismatch_lines}
"""


def main() -> None:
    """Validate the qrels file and emit JSON plus markdown summaries."""

    args = build_argument_parser().parse_args()

    queries = read_json(args.queries)["queries"]
    query_id_by_text = {query["text"]: query["query_id"] for query in queries}
    query_text_by_id = {query["query_id"]: query["text"] for query in queries}

    qrels_rows = list(csv.DictReader(args.qrels.open("r", encoding="utf-8", newline="")))
    results_rows = list(csv.DictReader(args.results.open("r", encoding="utf-8", newline="")))

    required_qrels_columns = {"Query_ID", "Query", "Doc_ID", "Is_Relevant"}
    missing_columns = required_qrels_columns - set(qrels_rows[0].keys()) if qrels_rows else required_qrels_columns
    if missing_columns:
        raise ValueError(f"Qrels file is missing required columns: {sorted(missing_columns)}")

    qrels_by_key: dict[tuple[str, str], str] = {}
    duplicate_keys: list[str] = []
    invalid_labels: list[str] = []

    for row in qrels_rows:
        label = normalize_label(row["Is_Relevant"])
        key = (row["Query_ID"], row["Doc_ID"])

        if label not in {"Y", "N"}:
            invalid_labels.append(f"{key} -> {label}")
            continue

        if query_text_by_id.get(row["Query_ID"]) != row["Query"]:
            raise ValueError(f"Qrels query text does not match query_id for row: {row}")

        if key in qrels_by_key:
            duplicate_keys.append(f"{key}")
            continue

        qrels_by_key[key] = label

    if duplicate_keys:
        raise ValueError(f"Duplicate query-document pairs found in qrels: {duplicate_keys[:10]}")
    if invalid_labels:
        raise ValueError(f"Invalid relevance labels found in qrels: {invalid_labels[:10]}")

    missing_qrels: list[str] = []
    mismatches: list[str] = []
    covered_result_rows = 0
    per_query_top5_docs: defaultdict[str, set[str]] = defaultdict(set)

    for row in results_rows:
        query_id = query_id_by_text.get(row["Query"])
        if not query_id:
            raise ValueError(f"Could not resolve query ID for retrieval row: {row}")

        doc_id = row["Doc_ID"]
        key = (query_id, doc_id)
        per_query_top5_docs[query_id].add(doc_id)

        if key not in qrels_by_key:
            missing_qrels.append(f"{query_id}:{doc_id}")
            continue

        covered_result_rows += 1
        explicit_label = normalize_label(row.get("Is_Relevant", ""))
        if explicit_label in {"Y", "N"} and explicit_label != qrels_by_key[key]:
            mismatches.append(f"{query_id}:{doc_id} csv={explicit_label} qrels={qrels_by_key[key]}")

    judged_counts = Counter(query_id for query_id, _doc_id in qrels_by_key)
    per_query = []
    for query in queries:
        query_id = query["query_id"]
        top5_docs = per_query_top5_docs.get(query_id, set())
        covered_docs = sum(1 for doc_id in top5_docs if (query_id, doc_id) in qrels_by_key)
        per_query.append(
            {
                "query_id": query_id,
                "judged_documents": judged_counts.get(query_id, 0),
                "top5_unique_documents": len(top5_docs),
                "covered_top5_documents": covered_docs,
            }
        )

    summary = {
        "passed": not missing_qrels and not mismatches,
        "qrels_rows": len(qrels_rows),
        "unique_query_doc_pairs": len(qrels_by_key),
        "result_rows_checked": len(results_rows),
        "covered_result_rows": covered_result_rows,
        "missing_qrels_count": len(missing_qrels),
        "judgment_mismatch_count": len(mismatches),
        "missing_qrels_examples": missing_qrels[:10],
        "judgment_mismatch_examples": mismatches[:10],
        "per_query": per_query,
    }

    write_json(args.json_output, summary)
    write_text(args.md_output, build_markdown_report(summary))

    print(f"Validated qrels against {len(results_rows)} retrieval rows.")
    print(f"Saved validation JSON to {args.json_output}.")
    print(f"Saved validation report to {args.md_output}.")


if __name__ == "__main__":
    main()
