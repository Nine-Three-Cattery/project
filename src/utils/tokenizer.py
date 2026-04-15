"""Shared tokenization logic for retrieval and lexical overlap helpers."""

from __future__ import annotations

import re

from .constants import GAP_TOKEN


TOKEN_PATTERN = re.compile(r"[a-z0-9]+(?:'[a-z0-9]+)?")


def tokenize(text: str) -> list[str]:
    """Tokenize text consistently for both TF-IDF and BM25 retrieval."""

    text = text.lower().replace(GAP_TOKEN.lower(), " ")
    tokens = TOKEN_PATTERN.findall(text)
    return [token for token in tokens if token != "gap"]


def tokenize_for_lexical_overlap(text: str) -> list[str]:
    """Expose a clearly named helper for query-sentence lexical overlap checks."""

    return tokenize(text)
