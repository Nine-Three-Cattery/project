"""Corpus loading, parsing, cleaning, and JSONL helpers."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from .constants import GAP_TOKEN


DOC_START_PATTERN = re.compile(r"(?m)^@@\s*(?P<doc_id>\d+)\s+")
GAP_PATTERN = re.compile(r"(?:@\s+){9}@")
LEXICAL_TOKEN_PATTERN = re.compile(r"<GAP>|[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?|@")


@dataclass(slots=True)
class CorpusDocument:
    """Internal representation of a single document in the news corpus."""

    doc_id: str
    raw_text: str
    clean_text: str


def ensure_parent_dir(path: Path) -> None:
    """Create the parent directory for a file path if it does not exist."""

    path.parent.mkdir(parents=True, exist_ok=True)


def normalize_whitespace(text: str) -> str:
    """Collapse noisy whitespace while preserving the document content itself."""

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def replace_gap_markers(text: str, replacement: str = GAP_TOKEN) -> str:
    """Map the repeated gap placeholder to a single symbolic token."""

    return GAP_PATTERN.sub(replacement, text)


def clean_document_text(text: str) -> str:
    """Apply the light, non-destructive cleaning policy used across the project."""

    text = replace_gap_markers(text, GAP_TOKEN)
    text = normalize_whitespace(text)
    return text.lower()


def parse_corpus_text(text: str) -> list[CorpusDocument]:
    """Split the raw corpus into documents keyed by their leading numeric ID."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    matches = list(DOC_START_PATTERN.finditer(normalized))
    documents: list[CorpusDocument] = []

    if not matches:
        raise ValueError("No documents were found in the corpus input.")

    for index, match in enumerate(matches):
        doc_id = match.group("doc_id")
        content_start = match.end()
        content_end = matches[index + 1].start() if index + 1 < len(matches) else len(normalized)
        raw_text = normalize_whitespace(normalized[content_start:content_end])
        clean_text = clean_document_text(raw_text)
        documents.append(CorpusDocument(doc_id=doc_id, raw_text=raw_text, clean_text=clean_text))

    return documents


def load_raw_corpus(path: Path) -> list[CorpusDocument]:
    """Read and parse a raw corpus file into structured documents."""

    return parse_corpus_text(path.read_text(encoding="utf-8", errors="replace"))


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    """Write dictionaries to JSONL with stable UTF-8 encoding."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False))
            handle.write("\n")


def write_cleaned_corpus_jsonl(path: Path, documents: Iterable[CorpusDocument]) -> None:
    """Persist the cleaned corpus in JSONL form for downstream tasks."""

    rows = (asdict(document) for document in documents)
    write_jsonl(path, rows)


def write_cleaned_corpus_txt(path: Path, documents: Iterable[CorpusDocument]) -> None:
    """Persist the cleaned corpus using the required @@id single-line format."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        for document in documents:
            handle.write(f"@@{document.doc_id}  {document.clean_text}\n")


def load_cleaned_corpus_jsonl(path: Path) -> list[CorpusDocument]:
    """Read the cleaned JSONL corpus back into document objects."""

    documents: list[CorpusDocument] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            documents.append(CorpusDocument(**row))
    return documents


def lexical_tokens(text: str, *, lowercase: bool = False) -> list[str]:
    """Tokenize text for corpus statistics without removing punctuation globally."""

    if lowercase:
        text = text.lower()
    return LEXICAL_TOKEN_PATTERN.findall(text)


def read_json(path: Path) -> dict:
    """Load a JSON file into a dictionary."""

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: dict | list) -> None:
    """Write JSON with indentation so artifacts remain reviewable in git."""

    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
