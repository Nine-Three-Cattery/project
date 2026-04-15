"""Interactive annotation helper for Task 3 topic labeling."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
import yaml

from src.utils.constants import CLEANED_JSONL_PATH, DEFAULT_RANDOM_SEED, LABEL_DEFINITIONS_PATH, LABELS_PATH
from src.utils.corpus import CorpusDocument, load_cleaned_corpus_jsonl


console = Console()


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the interactive annotation tool."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=CLEANED_JSONL_PATH, help="Path to the cleaned corpus JSONL.")
    parser.add_argument("--output", type=Path, default=LABELS_PATH, help="Path to the annotation JSONL output.")
    parser.add_argument(
        "--labels-yaml",
        type=Path,
        default=LABEL_DEFINITIONS_PATH,
        help="Path to the label definitions YAML file.",
    )
    parser.add_argument("--sample-size", type=int, default=200, help="Number of documents to queue for annotation.")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for document sampling.")
    return parser


def load_label_config(path: Path) -> list[dict]:
    """Load label names and shortcuts from the YAML configuration."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    return payload["labels"]


def load_existing_annotations(path: Path) -> list[dict]:
    """Read any existing annotation records so the tool can resume safely."""

    if not path.exists():
        return []

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_annotations(path: Path, rows: list[dict]) -> None:
    """Rewrite the JSONL annotation file after every change."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def sample_documents(documents: list[CorpusDocument], excluded_doc_ids: set[str], sample_size: int, seed: int) -> list[CorpusDocument]:
    """Select a deterministic random subset of unlabeled documents."""

    candidates = [document for document in documents if document.doc_id not in excluded_doc_ids]
    random.Random(seed).shuffle(candidates)
    return candidates[:sample_size]


def main() -> None:
    """Run the interactive document annotation loop."""

    args = build_argument_parser().parse_args()
    label_config = load_label_config(args.labels_yaml)
    label_map = {row["key"]: row["name"] for row in label_config}
    commands = {"u": "undo", "x": "skip", "q": "quit"}

    existing_rows = load_existing_annotations(args.output)
    existing_doc_ids = {row["doc_id"] for row in existing_rows}
    documents = load_cleaned_corpus_jsonl(args.input)
    queue = sample_documents(documents, existing_doc_ids, args.sample_size, args.seed)
    session_history: list[str] = []

    if not queue:
        console.print("[bold yellow]No unlabeled documents are available for annotation.[/bold yellow]")
        return

    console.print("[bold]Task 3 Annotation Tool[/bold]")
    console.print("Shortcuts:")
    for row in label_config:
        console.print(f"  {row['key']} -> {row['name']}")
    console.print("  u -> undo last annotation")
    console.print("  x -> skip current document")
    console.print("  q -> quit")

    index = 0
    while index < len(queue):
        document = queue[index]
        progress = f"{index + 1}/{len(queue)}"
        preview = document.raw_text if len(document.raw_text) <= 1800 else document.raw_text[:1800] + "..."
        console.print(Panel(preview, title=f"Document {document.doc_id} ({progress})", expand=False))

        choice = Prompt.ask("Label", choices=list(label_map) + list(commands))

        if choice == "u":
            if not session_history:
                console.print("[yellow]Nothing to undo in the current session.[/yellow]")
                continue

            last_doc_id = session_history.pop()
            existing_rows = [row for row in existing_rows if row["doc_id"] != last_doc_id]
            save_annotations(args.output, existing_rows)
            queue.insert(index, next(document for document in documents if document.doc_id == last_doc_id))
            console.print(f"[yellow]Removed annotation for document {last_doc_id}.[/yellow]")
            continue

        if choice == "x":
            index += 1
            continue

        if choice == "q":
            break

        existing_rows.append(
            {
                "doc_id": document.doc_id,
                "label": label_map[choice],
                "source": "manual",
                "annotated_at": "interactive_session",
            }
        )
        session_history.append(document.doc_id)
        save_annotations(args.output, existing_rows)
        index += 1

    console.print(f"[bold green]Saved {len(existing_rows)} annotations to {args.output}.[/bold green]")


if __name__ == "__main__":
    main()
