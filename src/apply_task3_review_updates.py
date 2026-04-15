"""Apply Task 3 label adjudications recorded in a YAML update file."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import yaml

from src.utils.constants import (
    LABELS_PATH,
    TASK3_LABELS,
    TASK3_LABEL_REVIEW_SUMMARY_PATH,
)
from src.utils.corpus import ensure_parent_dir


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for reproducible Task 3 review updates."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="Path to the released Task 3 label JSONL.")
    parser.add_argument(
        "--updates",
        type=Path,
        required=True,
        help="YAML file that records adjudication updates.",
    )
    parser.add_argument("--output", type=Path, default=LABELS_PATH, help="Output label JSONL path.")
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=TASK3_LABEL_REVIEW_SUMMARY_PATH,
        help="Path to the updated review summary JSON.",
    )
    return parser


def load_jsonl_rows(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dictionaries."""

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_update_config(path: Path) -> dict:
    """Load the YAML review-update configuration."""

    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "updates" not in config or not isinstance(config["updates"], list):
        raise ValueError("The review-update YAML must contain an 'updates' list.")
    return config


def compute_label_distribution(rows: list[dict], label_key: str) -> dict[str, int]:
    """Compute a stable label distribution using the canonical label order."""

    counter = Counter(row[label_key] for row in rows)
    return {label: int(counter.get(label, 0)) for label in TASK3_LABELS}


def main() -> None:
    """Apply the recorded adjudication updates and rebuild the summary JSON."""

    args = build_argument_parser().parse_args()
    rows = load_jsonl_rows(args.labels)
    config = load_update_config(args.updates)

    updates_by_doc_id = {update["doc_id"]: update for update in config["updates"]}
    row_index = {row["doc_id"]: row for row in rows}

    for update in config["updates"]:
        doc_id = update["doc_id"]
        if doc_id not in row_index:
            raise ValueError(f"Update references unknown doc_id {doc_id}.")

        row = row_index[doc_id]
        if row["label"] != update["old_label"]:
            raise ValueError(
                f"Update for doc_id {doc_id} expected old label {update['old_label']}, "
                f"but found {row['label']} in the current label file."
            )

    previous_label_distribution = compute_label_distribution(rows, "label")

    for row in rows:
        row["review_tag"] = config["tag"]
        if row["doc_id"] not in updates_by_doc_id:
            continue

        update = updates_by_doc_id[row["doc_id"]]
        row["label"] = update["new_label"]
        row["annotated_at"] = str(config["reviewed_at"])
        row["annotation_method"] = str(config["annotation_method"])
        row["review_reason"] = update["reason"]
        row["audit_consensus_label"] = update["new_label"]
        row["audit_support"] = list(config.get("audit_support", []))

    final_label_distribution = compute_label_distribution(rows, "label")
    total_changes_from_initial = sum(1 for row in rows if row["label"] != row.get("initial_label"))

    ensure_parent_dir(args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")

    summary = {
        "source": "llm_review",
        "tag": config["tag"],
        "reviewed_at": str(config["reviewed_at"]),
        "annotation_method": str(config["annotation_method"]),
        "review_scope": str(config["review_scope"]),
        "based_on_previous_tag": str(config["based_on_previous_tag"]),
        "updates_config_path": str(args.updates),
        "initial_source_distribution": Counter(row.get("initial_source", "unknown") for row in rows),
        "final_source_distribution": Counter(row.get("source", "unknown") for row in rows),
        "initial_label_distribution": compute_label_distribution(rows, "initial_label"),
        "previous_release_label_distribution": previous_label_distribution,
        "final_label_distribution": final_label_distribution,
        "changed_label_count": total_changes_from_initial,
        "additional_changes_from_previous_release": len(config["updates"]),
        "changed_labels": [
            {
                "doc_id": update["doc_id"],
                "old_label": update["old_label"],
                "new_label": update["new_label"],
                "reason": update["reason"],
            }
            for update in config["updates"]
        ],
    }

    ensure_parent_dir(args.summary_output)
    args.summary_output.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Applied {len(config['updates'])} review updates to {args.output}.")
    print(f"Saved updated review summary to {args.summary_output}.")


if __name__ == "__main__":
    main()
