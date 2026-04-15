"""Task 1: corpus audit, preprocessing, and reporting."""

from __future__ import annotations

import argparse
import re
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from src.utils.constants import (
    CLEANED_JSONL_PATH,
    CLEANED_TXT_PATH,
    DEFAULT_RANDOM_SEED,
    DOC_LENGTH_FIGURE_PATH,
    GAP_TOKEN,
    RAW_CORPUS_PATH,
    TASK1_STATS_MD_PATH,
    TASK1_STATS_PDF_PATH,
    ZIPF_FIGURE_PATH,
)
from src.utils.corpus import (
    CorpusDocument,
    ensure_parent_dir,
    lexical_tokens,
    load_raw_corpus,
    write_cleaned_corpus_jsonl,
    write_cleaned_corpus_txt,
)
from src.utils.reporting import render_markdown_to_pdf, write_text


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the Task 1 pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=RAW_CORPUS_PATH, help="Path to the raw corpus file.")
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=CLEANED_JSONL_PATH,
        help="Path for the cleaned corpus JSONL output.",
    )
    parser.add_argument(
        "--output-txt",
        type=Path,
        default=CLEANED_TXT_PATH,
        help="Path for the cleaned corpus text output.",
    )
    parser.add_argument(
        "--stats-md",
        type=Path,
        default=TASK1_STATS_MD_PATH,
        help="Path for the markdown statistics report.",
    )
    parser.add_argument(
        "--stats-pdf",
        type=Path,
        default=TASK1_STATS_PDF_PATH,
        help="Path for the PDF statistics report.",
    )
    parser.add_argument(
        "--doc-length-figure",
        type=Path,
        default=DOC_LENGTH_FIGURE_PATH,
        help="Path for the document length histogram.",
    )
    parser.add_argument(
        "--zipf-figure",
        type=Path,
        default=ZIPF_FIGURE_PATH,
        help="Path for the Zipf plot.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_RANDOM_SEED,
        help="Reserved seed argument for deterministic examples and future extensions.",
    )
    return parser


def compute_statistics(documents: list[CorpusDocument]) -> dict:
    """Compute the descriptive corpus statistics requested by the assignment."""

    raw_token_lists = [lexical_tokens(document.raw_text, lowercase=False) for document in documents]
    cleaned_token_lists = [lexical_tokens(document.clean_text, lowercase=False) for document in documents]

    raw_vocabulary = {token for tokens in raw_token_lists for token in tokens}
    cleaned_vocabulary = {token for tokens in cleaned_token_lists for token in tokens}
    cleaned_lengths = [len(tokens) for tokens in cleaned_token_lists]
    cleaned_frequency_counter = Counter(token for tokens in cleaned_token_lists for token in tokens)

    return {
        "document_count": len(documents),
        "total_token_count": sum(cleaned_lengths),
        "average_document_length": statistics.mean(cleaned_lengths),
        "median_document_length": statistics.median(cleaned_lengths),
        "vocabulary_size_before": len(raw_vocabulary),
        "vocabulary_size_after": len(cleaned_vocabulary),
        "doc_lengths": cleaned_lengths,
        "token_frequencies": cleaned_frequency_counter,
    }


def create_doc_length_histogram(lengths: list[int], output_path: Path) -> None:
    """Plot the distribution of document lengths after cleaning."""

    ensure_parent_dir(output_path)

    plt.figure(figsize=(10, 6))
    try:
        import seaborn as sns

        sns.histplot(lengths, bins=30, color="#2f6f95", edgecolor="white")
    except ImportError:
        plt.hist(lengths, bins=30, color="#2f6f95", edgecolor="white")

    plt.title("Document Length Distribution")
    plt.xlabel("Tokens per document")
    plt.ylabel("Document count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def create_zipf_plot(token_frequencies: Counter[str], output_path: Path) -> None:
    """Plot the token frequency distribution on a log-log scale."""

    ensure_parent_dir(output_path)

    sorted_frequencies = sorted(token_frequencies.values(), reverse=True)
    ranks = range(1, len(sorted_frequencies) + 1)

    plt.figure(figsize=(10, 6))
    plt.loglog(ranks, sorted_frequencies, color="#8c2f39")
    plt.title("Zipf Plot of Cleaned Corpus")
    plt.xlabel("Token rank (log scale)")
    plt.ylabel("Token frequency (log scale)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def build_markdown_report(documents: list[CorpusDocument], statistics_payload: dict, figure_dir_name: str) -> str:
    """Build the detailed markdown report for Task 1."""

    example_sections = []
    for document in documents[:3]:
        raw_preview = document.raw_text[:260] + ("..." if len(document.raw_text) > 260 else "")
        clean_preview = document.clean_text[:260] + ("..." if len(document.clean_text) > 260 else "")
        example_sections.append(
            "\n".join(
                [
                    f"### Document {document.doc_id}",
                    f"- Raw: `{raw_preview}`",
                    f"- Cleaned: `{clean_preview}`",
                ]
            )
        )

    report = f"""# Task 1 Corpus Audit and Preprocessing

## Corpus Summary

| Metric | Value |
| --- | ---: |
| Number of documents | {statistics_payload["document_count"]} |
| Total token count | {statistics_payload["total_token_count"]:,} |
| Average document length | {statistics_payload["average_document_length"]:.2f} |
| Median document length | {statistics_payload["median_document_length"]:.2f} |
| Vocabulary size before preprocessing | {statistics_payload["vocabulary_size_before"]:,} |
| Vocabulary size after preprocessing | {statistics_payload["vocabulary_size_after"]:,} |

## Parser and Cleaning Decisions

- The parser locates document boundaries using the leading `@@<numeric_id>` pattern.
- Document identifiers are preserved as metadata but excluded from the cleaned document text.
- Cleaning is intentionally light and non-destructive:
  - repeated gap noise is mapped to `{GAP_TOKEN}`
  - all text is lowercased
  - whitespace is normalized
  - punctuation, stopwords, and rare tokens are retained in the cleaned corpus output
- The cleaned corpus therefore remains suitable for both retrieval and downstream inspection.

## Supervision Assessment

The corpus is best treated as an unlabeled, unsupervised collection. The files contain document identifiers, but they do not expose class labels, topic metadata, or any pre-defined supervision signal. The IDs are therefore bookkeeping keys rather than semantic annotations. Any supervised experiment built on top of this corpus requires labels to be created after inspection.

## Before and After Examples

{chr(10).join(example_sections)}

## Figures

![Document length histogram]({figure_dir_name}/doc_length_hist.png)

![Zipf plot]({figure_dir_name}/zipf_plot.png)
"""
    return report


def main() -> None:
    """Run the Task 1 preprocessing workflow from raw corpus to report artifacts."""

    args = build_argument_parser().parse_args()

    documents = load_raw_corpus(args.input)
    statistics_payload = compute_statistics(documents)

    write_cleaned_corpus_jsonl(args.output_jsonl, documents)
    write_cleaned_corpus_txt(args.output_txt, documents)

    create_doc_length_histogram(statistics_payload["doc_lengths"], args.doc_length_figure)
    create_zipf_plot(statistics_payload["token_frequencies"], args.zipf_figure)

    figure_dir_name = args.doc_length_figure.parent.relative_to(args.stats_md.parent).as_posix()
    markdown_report = build_markdown_report(documents, statistics_payload, figure_dir_name)
    write_text(args.stats_md, markdown_report)

    pdf_body = re.sub(r"!\[.*?\]\(.*?\)", "", markdown_report)
    render_markdown_to_pdf(
        title="Task 1 Corpus Audit and Preprocessing",
        markdown_content=pdf_body,
        figure_paths=[args.doc_length_figure, args.zipf_figure],
        output_path=args.stats_pdf,
    )

    print(f"Processed {len(documents)} documents from {args.input}.")
    print(f"Saved cleaned corpus JSONL to {args.output_jsonl}.")
    print(f"Saved cleaned corpus TXT to {args.output_txt}.")
    print(f"Saved markdown report to {args.stats_md}.")
    print(f"Saved PDF report to {args.stats_pdf}.")


if __name__ == "__main__":
    main()
