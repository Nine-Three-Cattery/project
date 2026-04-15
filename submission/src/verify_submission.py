"""Validate that the repository or submission bundle contains the required deliverables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_PATHS = [
    "README.md",
    "preprocess.py",
    "requirements.txt",
    "src/task1_preprocess.py",
    "src/task2_retrieval.py",
    "src/compute_p_at_5.py",
    "src/task3_classification.py",
    "data/processed/text_news_cleaned.txt",
    "data/annotations/retrieval_qrels.csv",
    "data/annotations/trackA_labels.jsonl",
    "outputs/task1_stats.pdf",
    "outputs/report.pdf",
]

OPTIONAL_PATHS = [
    ".python-version",
    "requirements-lock.txt",
    "outputs/task3_llm_audit_summary.json",
    "outputs/task3_llm_audit_summary.md",
    "outputs/task3_external_reviews",
]

CACHE_PATTERNS = [
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".DS_Store",
]


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for repository or bundle validation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository or bundle root to validate.")
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional JSON summary output path.",
    )
    return parser


def find_cache_paths(root: Path) -> list[str]:
    """Find cache-like files and directories that should not be shipped."""

    cache_hits: list[str] = []
    for pattern in CACHE_PATTERNS:
        for path in root.rglob(pattern):
            cache_hits.append(str(path.relative_to(root)))
    return sorted(set(cache_hits))


def main() -> None:
    """Validate the required files and detect stray cache artifacts."""

    args = build_argument_parser().parse_args()
    root = args.root.resolve()

    missing_required = [path for path in REQUIRED_PATHS if not (root / path).exists()]
    present_optional = [path for path in OPTIONAL_PATHS if (root / path).exists()]
    cache_hits = find_cache_paths(root)

    payload = {
        "root": str(root),
        "missing_required": missing_required,
        "present_optional": present_optional,
        "cache_hits": cache_hits,
        "passed": not missing_required and not cache_hits,
    }

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    if missing_required or cache_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
