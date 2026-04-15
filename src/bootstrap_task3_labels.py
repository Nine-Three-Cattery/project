"""Generate a reproducible bootstrapped label set for Task 3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml
from rank_bm25 import BM25Okapi

from src.utils.constants import (
    CLEANED_JSONL_PATH,
    PROJECT_ROOT,
    TASK3_BOOTSTRAP_LABELS_PATH,
    TASK3_BOOTSTRAP_CONFIG_PATH,
    TASK3_BOOTSTRAP_SUMMARY_PATH,
)
from src.utils.corpus import ensure_parent_dir, load_cleaned_corpus_jsonl
from src.utils.tokenizer import tokenize


DEFAULT_BOOTSTRAP_CONFIG = {
    "source": "bootstrapped",
    "tag": "bootstrap_bm25_margin_v1",
    "margin_threshold": 10.0,
    "top_n_per_topic": 34,
    "other_count": 30,
    "label_queries": {
        "Politics": "government election congress president senate policy political campaign diplomacy administration lawmakers voters minister mayor governor",
        "Business/Economy": "economy economic market markets business company companies inflation trade stocks earnings revenue finance investor investors jobs federal reserve bank banks",
        "Sports": "basketball baseball football golf boxing soccer tennis hockey olympics tournament championship season player players coach team teams game games match matches score scored athlete athletes",
        "Crime/Law": "police court courts judge judges jury prison jail criminal crime charged charges arrest arrested trial lawsuit lawsuits prosecutor prosecutors investigation violence abuse prison inmate inmates",
        "Culture/Entertainment": "movie movies film films television tv theater theatre broadway actor actress actors music musician musicians book books author authors novel novels show shows entertainment arts dance musical review",
    },
}


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for deterministic Task 3 label bootstrapping."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CLEANED_JSONL_PATH, help="Path to the cleaned corpus JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK3_BOOTSTRAP_LABELS_PATH,
        help="Path to the bootstrapped seed-label JSONL file.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=TASK3_BOOTSTRAP_CONFIG_PATH,
        help="Path to the YAML bootstrap configuration file.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=TASK3_BOOTSTRAP_SUMMARY_PATH,
        help="Path to the bootstrap summary JSON file.",
    )
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=None,
        help="Optional override for the minimum score margin required for topic-specific label assignment.",
    )
    parser.add_argument(
        "--top-n-per-topic",
        type=int,
        default=None,
        help="Optional override for the number of high-confidence documents to keep for each major topic.",
    )
    parser.add_argument(
        "--other-count",
        type=int,
        default=None,
        help="Optional override for the number of low-topic-score documents to assign to the Other class.",
    )
    return parser


def load_bootstrap_config(path: Path) -> dict:
    """Load and validate the YAML bootstrap configuration."""

    config = dict(DEFAULT_BOOTSTRAP_CONFIG)
    if path.exists():
        loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config.update({key: value for key, value in loaded.items() if value is not None})

    label_queries = config.get("label_queries")
    if not isinstance(label_queries, dict) or not label_queries:
        raise ValueError("Bootstrap configuration must define a non-empty 'label_queries' mapping.")

    for required_key in ("source", "tag", "margin_threshold", "top_n_per_topic", "other_count"):
        if required_key not in config:
            raise ValueError(f"Bootstrap configuration is missing required key: {required_key}")

    return config


def sort_indices(indices: np.ndarray, primary_scores: np.ndarray, documents: list) -> list[int]:
    """Sort candidate indices by descending score and then by stable document ID."""

    sortable = [int(index) for index in indices]
    sortable.sort(key=lambda index: (-float(primary_scores[index]), documents[index].doc_id))
    return sortable


def build_bootstrap_labels(
    *,
    corpus_path: Path,
    source: str,
    tag: str,
    label_queries: dict[str, str],
    margin_threshold: float,
    top_n_per_topic: int,
    other_count: int,
) -> tuple[list[dict], dict]:
    """Create the bootstrapped label rows and a machine-readable summary."""

    documents = load_cleaned_corpus_jsonl(corpus_path)
    tokenized_corpus = [tokenize(document.clean_text) for document in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    labels = list(label_queries.keys())
    score_matrix = np.vstack([bm25.get_scores(tokenize(query)) for query in label_queries.values()]).T
    top_topic_indices = score_matrix.argmax(axis=1)
    max_scores = score_matrix.max(axis=1)
    sorted_scores = np.sort(score_matrix, axis=1)
    margins = sorted_scores[:, -1] - sorted_scores[:, -2]

    label_rows: list[dict] = []
    selected_indices: set[int] = set()
    per_label_doc_ids: dict[str, list[str]] = {}

    for label_index, label_name in enumerate(labels):
        candidate_indices = np.where((top_topic_indices == label_index) & (margins >= margin_threshold))[0]
        sorted_candidates = sort_indices(candidate_indices, max_scores, documents)
        selected_for_label = sorted_candidates[:top_n_per_topic]

        if len(selected_for_label) < top_n_per_topic:
            raise ValueError(
                f"Not enough high-confidence documents for label {label_name}: "
                f"needed {top_n_per_topic}, found {len(selected_for_label)}."
            )

        per_label_doc_ids[label_name] = [documents[index].doc_id for index in selected_for_label]
        for index in selected_for_label:
            selected_indices.add(index)
            label_rows.append(
                {
                    "doc_id": documents[index].doc_id,
                    "label": label_name,
                    "source": source,
                    "annotated_at": tag,
                    "bootstrap_rule": "high_confidence_topic_match",
                    "bootstrap_score": round(float(max_scores[index]), 6),
                    "bootstrap_margin": round(float(margins[index]), 6),
                }
            )

    remaining_indices = [index for index in range(len(documents)) if index not in selected_indices]
    remaining_indices.sort(key=lambda index: (float(max_scores[index]), documents[index].doc_id))
    other_indices = remaining_indices[:other_count]

    if len(other_indices) < other_count:
        raise ValueError(f"Not enough remaining documents to populate Other: needed {other_count}, found {len(other_indices)}.")

    per_label_doc_ids["Other"] = [documents[index].doc_id for index in other_indices]
    for index in other_indices:
        label_rows.append(
            {
                "doc_id": documents[index].doc_id,
                "label": "Other",
                "source": source,
                "annotated_at": tag,
                "bootstrap_rule": "lowest_topic_confidence",
                "bootstrap_score": round(float(max_scores[index]), 6),
                "bootstrap_margin": round(float(margins[index]), 6),
            }
        )

    label_rows.sort(key=lambda row: (row["label"], row["doc_id"]))

    try:
        relative_corpus = str(corpus_path.relative_to(PROJECT_ROOT))
    except (ValueError, AttributeError):
        relative_corpus = str(corpus_path)

    summary = {
        "source": source,
        "tag": tag,
        "corpus_path": relative_corpus,
        "margin_threshold": margin_threshold,
        "top_n_per_topic": top_n_per_topic,
        "other_count": other_count,
        "label_queries": label_queries,
        "label_counts": {label: len(doc_ids) for label, doc_ids in per_label_doc_ids.items()},
        "doc_ids": per_label_doc_ids,
        "other_max_score_cutoff": round(float(max(max_scores[index] for index in other_indices)), 6),
    }
    return label_rows, summary


def write_jsonl(path: Path, rows: list[dict]) -> None:
    """Write the bootstrapped label rows to JSONL."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def main() -> None:
    """Generate the Task 3 label file and the companion bootstrap summary."""

    args = build_argument_parser().parse_args()
    config = load_bootstrap_config(args.config)

    margin_threshold = args.margin_threshold if args.margin_threshold is not None else float(config["margin_threshold"])
    top_n_per_topic = args.top_n_per_topic if args.top_n_per_topic is not None else int(config["top_n_per_topic"])
    other_count = args.other_count if args.other_count is not None else int(config["other_count"])

    label_rows, summary = build_bootstrap_labels(
        corpus_path=args.corpus,
        source=str(config["source"]),
        tag=str(config["tag"]),
        label_queries=dict(config["label_queries"]),
        margin_threshold=margin_threshold,
        top_n_per_topic=top_n_per_topic,
        other_count=other_count,
    )
    try:
        summary["config_path"] = str(args.config.relative_to(PROJECT_ROOT))
    except (ValueError, AttributeError):
        summary["config_path"] = str(args.config)

    write_jsonl(args.output, label_rows)
    ensure_parent_dir(args.summary_output)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Saved {len(label_rows)} bootstrapped labels to {args.output}.")
    print(f"Saved bootstrap summary to {args.summary_output}.")


if __name__ == "__main__":
    main()
