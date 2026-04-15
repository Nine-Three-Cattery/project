"""Compute Precision@5 for the judged retrieval results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.utils.constants import (
    RETRIEVAL_PER_QUERY_SCORES_PATH,
    RETRIEVAL_QRELS_PATH,
    RETRIEVAL_SUMMARY_PATH,
    TOP5_EVALUATION_PATH,
)
from src.utils.corpus import ensure_parent_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the Precision@5 computation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=TOP5_EVALUATION_PATH, help="Path to the top-5 retrieval output CSV.")
    parser.add_argument(
        "--qrels",
        type=Path,
        default=RETRIEVAL_QRELS_PATH,
        help="Path to the canonical relevance judgments CSV.",
    )
    parser.add_argument(
        "--per-query-output",
        type=Path,
        default=RETRIEVAL_PER_QUERY_SCORES_PATH,
        help="Path to the per-query Precision@5 output CSV.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RETRIEVAL_SUMMARY_PATH,
        help="Path to the aggregate summary CSV.",
    )
    return parser


def normalize_label(value: str) -> str:
    """Normalize binary relevance labels to stable uppercase values."""

    return str(value).strip().upper()


def load_judged_frame(input_path: Path, qrels_path: Path) -> pd.DataFrame:
    """Load retrieval outputs and attach canonical judgments when available."""

    frame = pd.read_csv(input_path, dtype={"Rank": int})

    if qrels_path.exists():
        qrels = pd.read_csv(qrels_path)
        qrels["Is_Relevant"] = qrels["Is_Relevant"].fillna("").map(normalize_label)
        qrels = qrels.loc[qrels["Is_Relevant"].isin({"Y", "N"})].copy()

        merge_keys = ["Query", "Doc_ID"]
        if "Model" in qrels.columns:
            # Backward compatibility for legacy qrels files created before the model-independent format.
            qrels = qrels.drop(columns=["Model"])

        if qrels.duplicated(subset=merge_keys).any():
            duplicates = qrels.loc[qrels.duplicated(subset=merge_keys), merge_keys]
            raise ValueError(f"Duplicate qrels rows found for keys: {duplicates.to_dict(orient='records')}")

        frame = frame.drop(columns=["Is_Relevant"], errors="ignore").merge(
            qrels[["Query", "Doc_ID", "Is_Relevant"]],
            on=merge_keys,
            how="left",
        )
    else:
        frame["Is_Relevant"] = frame["Is_Relevant"].fillna("").map(normalize_label)

    return frame


def main() -> None:
    """Compute per-query and aggregate Precision@5 summaries."""

    args = build_argument_parser().parse_args()

    frame = load_judged_frame(args.input, args.qrels)

    grouped = []
    for (query, model), group in frame.groupby(["Query", "Model"], sort=False):
        if len(group) != 5:
            continue
        if not group["Is_Relevant"].isin({"Y", "N"}).all():
            continue

        precision = float((group["Is_Relevant"] == "Y").sum() / 5.0)
        grouped.append({"Query": query, "Model": model, "P@5": precision})

    if not grouped:
        raise ValueError("No fully judged query/model groups were found in the evaluation CSV.")

    per_query = pd.DataFrame(grouped)
    pivot = per_query.pivot(index="Query", columns="Model", values="P@5").reset_index()
    pivot.columns.name = None

    if "tfidf" not in pivot.columns or "bm25" not in pivot.columns:
        raise ValueError("Both tfidf and bm25 scores are required to produce the summary.")

    def winner(row: pd.Series) -> str:
        if row["tfidf"] > row["bm25"]:
            return "tfidf"
        if row["bm25"] > row["tfidf"]:
            return "bm25"
        return "tie"

    pivot["winner"] = pivot.apply(winner, axis=1)
    ensure_parent_dir(args.per_query_output)
    pivot.to_csv(args.per_query_output, index=False)

    summary = pd.DataFrame(
        [
            {
                "evaluated_queries": int(len(pivot)),
                "avg_tfidf_p_at_5": float(pivot["tfidf"].mean()),
                "avg_bm25_p_at_5": float(pivot["bm25"].mean()),
                "baseline_wins": int((pivot["winner"] == "tfidf").sum()),
                "improved_wins": int((pivot["winner"] == "bm25").sum()),
                "ties": int((pivot["winner"] == "tie").sum()),
            }
        ]
    )
    ensure_parent_dir(args.summary_output)
    summary.to_csv(args.summary_output, index=False)

    print(pivot.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
