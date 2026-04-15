"""Task 2: document retrieval with TF-IDF and BM25."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.constants import (
    CLEANED_JSONL_PATH,
    DEFAULT_RANDOM_SEED,
    PER_QUERY_DIR,
    QUERIES_PATH,
    RETRIEVAL_QRELS_PATH,
    TOP5_EVALUATION_PATH,
)
from src.utils.corpus import (
    CorpusDocument,
    ensure_parent_dir,
    load_cleaned_corpus_jsonl,
    read_json,
    replace_gap_markers,
    write_json,
)
from src.utils.tokenizer import tokenize, tokenize_for_lexical_overlap


SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")


@dataclass(slots=True)
class SearchHit:
    """A ranked retrieval hit."""

    doc_id: str
    score: float
    snippet: str


class TfidfRetriever:
    """Thin wrapper around a TF-IDF matrix and cosine similarity retrieval."""

    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self.doc_ids = [document.doc_id for document in documents]
        self.vectorizer = TfidfVectorizer(
            tokenizer=tokenize,
            preprocessor=lambda text: text,
            lowercase=False,
            token_pattern=None,
            ngram_range=(1, 1),
        )
        self.document_matrix = self.vectorizer.fit_transform(document.clean_text for document in documents)

    def search(self, query: str, top_k: int) -> list[tuple[CorpusDocument, float]]:
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.document_matrix).ravel()
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[index], float(scores[index])) for index in top_indices]


class Bm25Retriever:
    """Thin wrapper around rank_bm25 for fair tokenizer sharing with TF-IDF."""

    def __init__(self, documents: list[CorpusDocument]) -> None:
        self.documents = documents
        self.doc_ids = [document.doc_id for document in documents]
        self.tokenized_corpus = [tokenize(document.clean_text) for document in documents]
        self.model = BM25Okapi(self.tokenized_corpus)

    def search(self, query: str, top_k: int) -> list[tuple[CorpusDocument, float]]:
        scores = self.model.get_scores(tokenize(query))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.documents[index], float(scores[index])) for index in top_indices]


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the retrieval workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CLEANED_JSONL_PATH, help="Path to cleaned corpus JSONL.")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH, help="Path to the queries JSON file.")
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=TOP5_EVALUATION_PATH,
        help="Path to the combined top-5 evaluation CSV.",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=PER_QUERY_DIR,
        help="Directory used for per-query JSON result dumps.",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=RETRIEVAL_QRELS_PATH,
        help="Path to the canonical retrieval relevance judgments CSV.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of results to keep for each query and model.")
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Reserved seed argument for deterministic extensions.",
    )
    return parser


def load_queries(path: Path) -> list[dict[str, str]]:
    """Read query definitions from the project JSON schema."""

    payload = read_json(path)
    queries = payload.get("queries", [])
    if not queries:
        raise ValueError(f"No queries were found in {path}.")
    return queries


def split_sentences(text: str) -> list[str]:
    """Split text into approximate sentences for lightweight snippet extraction."""

    text = replace_gap_markers(text, " ")
    text = re.sub(r"\s+", " ", text).strip()
    sentences = [sentence.strip() for sentence in SENTENCE_SPLIT_PATTERN.split(text) if sentence.strip()]
    return sentences


def fallback_window(text: str, max_words: int = 45) -> str:
    """Return a stable token window when sentence boundaries are unreliable."""

    text = replace_gap_markers(text, " ")
    words = re.sub(r"\s+", " ", text).strip().split()
    return " ".join(words[:max_words]).strip()


def build_snippet(document: CorpusDocument, query_text: str, max_chars: int = 250) -> str:
    """Generate a short query-aware snippet from a retrieved document."""

    query_tokens = set(tokenize_for_lexical_overlap(query_text))
    sentences = split_sentences(document.raw_text)

    if not sentences:
        return fallback_window(document.raw_text)[:max_chars].rstrip()

    best_index = 0
    best_score = -1
    for index, sentence in enumerate(sentences):
        sentence_tokens = set(tokenize_for_lexical_overlap(sentence))
        overlap = len(query_tokens & sentence_tokens)
        if overlap > best_score:
            best_score = overlap
            best_index = index

    snippet = sentences[best_index]

    if len(snippet) < 100 and best_index + 1 < len(sentences):
        snippet = f"{snippet} {sentences[best_index + 1]}"
    if len(snippet) < 100 and best_index > 0:
        snippet = f"{sentences[best_index - 1]} {snippet}"

    if best_score <= 0:
        snippet = fallback_window(document.raw_text)

    snippet = re.sub(r"\s+", " ", snippet).strip()
    return snippet[:max_chars].rstrip()


def normalize_relevance_label(value: str) -> str:
    """Normalize binary relevance labels to stable uppercase values."""

    return str(value).strip().upper()


def load_known_judgments(
    *,
    output_csv_path: Path,
    qrels_path: Path,
    queries: list[dict[str, str]],
) -> dict[tuple[str, str], str]:
    """Load canonical query-document judgments, supplementing them with manual CSV edits."""

    judgments: dict[tuple[str, str], str] = {}
    query_id_by_text = {query["text"]: query["query_id"] for query in queries}

    def ingest(reader: csv.DictReader, source_name: str) -> None:
        for row in reader:
            label = normalize_relevance_label(row.get("Is_Relevant", ""))
            if label not in {"Y", "N"}:
                continue

            query_id = row.get("Query_ID") or query_id_by_text.get(row.get("Query", ""))
            if not query_id:
                raise ValueError(f"Could not resolve a query ID for judged row in {source_name}: {row}")

            key = (query_id, row["Doc_ID"])
            existing = judgments.get(key)
            if existing is not None and existing != label:
                raise ValueError(
                    "Conflicting relevance labels were found for the same query-document pair "
                    f"in {source_name}: key={key}, existing={existing}, new={label}."
                )

            judgments[key] = label

    if qrels_path.exists():
        with qrels_path.open("r", encoding="utf-8", newline="") as handle:
            ingest(csv.DictReader(handle), str(qrels_path))

    if output_csv_path.exists():
        with output_csv_path.open("r", encoding="utf-8", newline="") as handle:
            ingest(csv.DictReader(handle), str(output_csv_path))

    return judgments


def write_results_csv(path: Path, rows: list[dict[str, str]], known_judgments: dict[tuple[str, str], str]) -> None:
    """Write the combined retrieval results CSV using the required schema."""

    ensure_parent_dir(path)
    fieldnames = ["Query", "Model", "Rank", "Doc_ID", "Snippet", "Is_Relevant"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            key = (row["_query_id"], row["Doc_ID"])
            writer.writerow(
                {
                    "Query": row["Query"],
                    "Model": row["Model"],
                    "Rank": row["Rank"],
                    "Doc_ID": row["Doc_ID"],
                    "Snippet": row["Snippet"],
                    "Is_Relevant": known_judgments.get(key, ""),
                }
            )


def write_qrels_csv(
    path: Path,
    known_judgments: dict[tuple[str, str], str],
    queries: list[dict[str, str]],
) -> None:
    """Persist canonical retrieval judgments independent of rank and model."""

    ensure_parent_dir(path)
    query_text_by_id = {query["query_id"]: query["text"] for query in queries}
    fieldnames = ["Query_ID", "Query", "Doc_ID", "Is_Relevant"]

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for query_id, doc_id in sorted(known_judgments):
            writer.writerow(
                {
                    "Query_ID": query_id,
                    "Query": query_text_by_id[query_id],
                    "Doc_ID": doc_id,
                    "Is_Relevant": known_judgments[(query_id, doc_id)],
                }
            )


def collect_rows(
    query_id: str,
    query_text: str,
    model_name: str,
    ranked_results: list[tuple[CorpusDocument, float]],
    top_k: int,
) -> list[dict[str, str]]:
    """Convert ranked search output into the CSV row schema."""

    rows: list[dict[str, str]] = []
    for rank, (document, score) in enumerate(ranked_results[:top_k], start=1):
        rows.append(
            {
                "_query_id": query_id,
                "Query": query_text,
                "Model": model_name,
                "Rank": str(rank),
                "Doc_ID": document.doc_id,
                "Snippet": build_snippet(document, query_text),
                "Is_Relevant": "",
                "_score": score,
            }
        )
    return rows


def main() -> None:
    """Run the retrieval pipeline and write all Task 2 retrieval artifacts."""

    args = build_argument_parser().parse_args()

    documents = load_cleaned_corpus_jsonl(args.corpus)
    queries = load_queries(args.queries)
    tfidf = TfidfRetriever(documents)
    bm25 = Bm25Retriever(documents)
    known_judgments = load_known_judgments(output_csv_path=args.output_csv, qrels_path=args.qrels, queries=queries)

    all_rows: list[dict[str, str]] = []
    args.per_query_dir.mkdir(parents=True, exist_ok=True)

    for query in queries:
        query_id = query["query_id"]
        query_text = query["text"]

        tfidf_rows = collect_rows(query_id, query_text, "tfidf", tfidf.search(query_text, args.top_k), args.top_k)
        bm25_rows = collect_rows(query_id, query_text, "bm25", bm25.search(query_text, args.top_k), args.top_k)

        all_rows.extend(tfidf_rows)
        all_rows.extend(bm25_rows)

        per_query_payload = {
            "query_id": query_id,
            "text": query_text,
            "results": {
                "tfidf": [
                    {
                        "rank": int(row["Rank"]),
                        "doc_id": row["Doc_ID"],
                        "score": row["_score"],
                        "snippet": row["Snippet"],
                    }
                    for row in tfidf_rows
                ],
                "bm25": [
                    {
                        "rank": int(row["Rank"]),
                        "doc_id": row["Doc_ID"],
                        "score": row["_score"],
                        "snippet": row["Snippet"],
                    }
                    for row in bm25_rows
                ],
            },
        }
        write_json(args.per_query_dir / f"{query_id}.json", per_query_payload)

    write_qrels_csv(args.qrels, known_judgments, queries)
    write_results_csv(args.output_csv, all_rows, known_judgments)

    print(f"Loaded {len(documents)} cleaned documents.")
    print(f"Ran retrieval for {len(queries)} queries.")
    print(f"Saved combined results to {args.output_csv}.")
    print(f"Saved canonical relevance judgments to {args.qrels}.")
    print(f"Saved per-query JSON files to {args.per_query_dir}.")


if __name__ == "__main__":
    main()
