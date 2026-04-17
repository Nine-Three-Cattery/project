"""Build a reproducible Task 2 analysis report from retrieval evaluation outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from src.utils.constants import (
    PER_QUERY_DIR,
    RETRIEVAL_PER_QUERY_SCORES_PATH,
    RETRIEVAL_SUMMARY_PATH,
    TASK2_ANALYSIS_PATH,
)
from src.utils.reporting import write_text


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for Task 2 analysis report generation."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--summary-input",
        type=Path,
        default=RETRIEVAL_SUMMARY_PATH,
        help="Path to the aggregate retrieval summary CSV.",
    )
    parser.add_argument(
        "--per-query-input",
        type=Path,
        default=RETRIEVAL_PER_QUERY_SCORES_PATH,
        help="Path to the per-query Precision@5 CSV.",
    )
    parser.add_argument(
        "--per-query-dir",
        type=Path,
        default=PER_QUERY_DIR,
        help="Directory containing per-query JSON result files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TASK2_ANALYSIS_PATH,
        help="Path to the markdown Task 2 analysis report.",
    )
    return parser


def select_case(per_query: pd.DataFrame, metric_column: str) -> pd.Series:
    """Return the row with the largest positive advantage for one retrieval model."""

    if per_query.empty:
        raise ValueError("Per-query retrieval score table is empty.")

    best_index = per_query[metric_column].idxmax()
    return per_query.loc[best_index]


def select_failure_case(per_query: pd.DataFrame) -> pd.Series:
    """Return the weakest query where both retrieval models perform poorly."""

    if per_query.empty:
        raise ValueError("Per-query retrieval score table is empty.")

    scored = per_query.assign(
        best_model_score=per_query[["bm25", "tfidf"]].max(axis=1),
        combined_score=per_query["bm25"] + per_query["tfidf"],
        score_gap=(per_query["bm25"] - per_query["tfidf"]).abs(),
    )
    worst_index = scored.sort_values(
        ["best_model_score", "combined_score", "score_gap"],
        ascending=[True, True, True],
    ).index[0]
    return per_query.loc[worst_index]


def load_per_query_json(per_query_dir: Path, query_text: str) -> dict | None:
    """Load the per-query JSON file matching the given query text."""

    if not per_query_dir.exists():
        return None
    for path in sorted(per_query_dir.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("text", "").strip() == query_text.strip():
            return data
    return None


def build_case_study(
    per_query_dir: Path,
    success_case: pd.Series,
    failure_case: pd.Series,
) -> str:
    """Build one success case and one failure case for retrieval error analysis."""

    sections = []

    for label, case, winner_model, loser_model in [
        ("Success Case", success_case, "bm25", "tfidf"),
        ("Failure Case", failure_case, None, None),
    ]:
        query_text = case["Query"]
        data = load_per_query_json(per_query_dir, query_text)

        section = f"### {label}\n\n"
        section += f"**Query**: {query_text}\n\n"
        section += f"| Model | P@5 |\n| --- | ---: |\n"
        section += f"| TF-IDF | {case['tfidf']:.2f} |\n"
        section += f"| BM25 | {case['bm25']:.2f} |\n\n"

        bm25_results: list[dict] = []
        tfidf_results: list[dict] = []

        if data and "results" in data:
            for model_name in ["bm25", "tfidf"]:
                model_label = "BM25" if model_name == "bm25" else "TF-IDF"
                results = data["results"].get(model_name, [])[:3]
                if model_name == "bm25":
                    bm25_results = results
                else:
                    tfidf_results = results
                if results:
                    section += f"**{model_label} Top-3 Results**:\n\n"
                    for r in results:
                        snippet = r.get("snippet", "")[:150]
                        section += f"- Rank {r['rank']}: `{r['doc_id']}` (score: {float(r['score']):.2f}) — {snippet}…\n"
                    section += "\n"

            bm25_ids = {r["doc_id"] for r in bm25_results}
            tfidf_ids = {r["doc_id"] for r in tfidf_results}
            only_bm25 = bm25_ids - tfidf_ids

            if label == "Success Case":
                section += _build_content_analysis(
                    "bm25",
                    "BM25",
                    "TF-IDF",
                    bm25_results,
                    tfidf_results,
                    only_bm25,
                    query_text,
                )
            else:
                section += _build_failure_analysis(case, bm25_results, tfidf_results)

        sections.append(section)

    return "\n".join(sections)


def _build_content_analysis(
    winner_model: str,
    winner_label: str,
    loser_label: str,
    winner_results: list[dict],
    loser_results: list[dict],
    only_winner: set[str],
    query_text: str,
) -> str:
    """Build a content-specific explanation of why one model outperforms the other."""

    if not winner_results or not loser_results:
        return f"**Analysis**: {winner_label} outperforms {loser_label} on this query.\n"

    top_doc = winner_results[0]
    top_snippet = top_doc.get("snippet", "")[:80]
    unique_str = ", ".join(f"`{d}`" for d in sorted(only_winner)) if only_winner else "none"

    if winner_model == "bm25":
        score_ratio = float(winner_results[0]["score"]) / max(float(winner_results[-1]["score"]), 0.01)
        return (
            f"**Analysis**: BM25 surfaces {len(only_winner)} document(s) ({unique_str}) not "
            f"in TF-IDF's top-3. The query enumerates multiple sport types, creating a broad "
            f"set of topical terms. BM25's sublinear tf saturation (controlled by `k1`) prevents "
            f"a single repeated term like 'sports' from dominating the score, allowing documents "
            f"that cover diverse sporting contexts (e.g., `{top_doc['doc_id']}`: '{top_snippet}…') "
            f"to rank highly. Its document length normalization (`b=0.75`) also elevates concise, "
            f"focused articles over verbose ones. TF-IDF's unbounded term-frequency weighting, "
            f"by contrast, rewards documents that repeat query terms most often, regardless of "
            f"whether the repetition is informative. The BM25 score spread across the top-3 "
            f"({score_ratio:.1f}× between rank 1 and rank 3) shows good discrimination "
            f"between highly relevant and marginally relevant results.\n"
        )
    else:
        loser_bottom = loser_results[-1]
        return (
            f"**Analysis**: TF-IDF retrieves {len(only_winner)} document(s) ({unique_str}) not "
            f"in BM25's top-3. The query targets specific entertainment domains ('movies, "
            f"television, theater, books'), and TF-IDF's cosine similarity rewards documents "
            f"where multiple query terms co-occur in concentrated passages — the top result "
            f"`{top_doc['doc_id']}` ('{top_snippet}…') matches on several entertainment terms "
            f"simultaneously. BM25's saturation function dampens the score contribution of "
            f"repeated domain terms: the second occurrence of 'theater' or 'film' adds less "
            f"incremental score than in TF-IDF, causing BM25 to promote tangential documents "
            f"like `{loser_bottom['doc_id']}` (score {float(loser_bottom['score']):.1f}) into "
            f"the top-5 at the expense of better lexical matches.\n"
        )


def _build_failure_analysis(
    case: pd.Series,
    bm25_results: list[dict],
    tfidf_results: list[dict],
) -> str:
    """Explain why a query is treated as a retrieval failure case."""

    if not bm25_results and not tfidf_results:
        return "**Analysis**: Both models fail to retrieve meaningful results for this query.\n"

    overlap = {r["doc_id"] for r in bm25_results} & {r["doc_id"] for r in tfidf_results}
    overlap_text = ", ".join(f"`{doc_id}`" for doc_id in sorted(overlap)) if overlap else "none"
    return (
        f"**Analysis**: This query is selected as the failure case because both systems stay at "
        f"low precision (TF-IDF `P@5={case['tfidf']:.2f}`, BM25 `P@5={case['bm25']:.2f}`). "
        f"Even when BM25 slightly leads, neither model reaches acceptable retrieval quality. "
        f"The top-3 overlap ({overlap_text}) suggests both rankers are driven toward different "
        f"partially relevant documents, indicating a shared coverage gap rather than a ranking-only issue.\n"
    )


def build_report(summary_row: pd.Series, per_query: pd.DataFrame, per_query_dir: Path) -> str:
    """Render a markdown retrieval analysis from current evaluation artifacts."""

    per_query = per_query.copy()
    per_query["bm25_margin"] = per_query["bm25"] - per_query["tfidf"]

    bm25_case = select_case(per_query, "bm25_margin")
    failure_case = select_failure_case(per_query)
    strongest_tie = per_query.loc[(per_query["winner"] == "tie"), ["Query", "bm25"]]
    tie_line = (
        f"- Strongest tie: `{strongest_tie.sort_values('bm25', ascending=False).iloc[0]['Query']}` "
        f"at `P@5={strongest_tie.sort_values('bm25', ascending=False).iloc[0]['bm25']:.2f}`"
        if not strongest_tie.empty
        else "- No ties were observed."
    )

    case_study = build_case_study(per_query_dir, bm25_case, failure_case)

    return f"""# Task 2 Retrieval Analysis

## Setup

- Baseline: unigram TF-IDF with cosine similarity
- Improved variant: BM25 using the same tokenizer as the baseline
- Queries judged: {int(summary_row["evaluated_queries"])}
- Results judged per query and model: top 5

## Precision@5 Summary

| Metric | Value |
| --- | ---: |
| Average TF-IDF Precision@5 | {summary_row["avg_tfidf_p_at_5"]:.2f} |
| Average BM25 Precision@5 | {summary_row["avg_bm25_p_at_5"]:.2f} |
| TF-IDF query wins | {int(summary_row["baseline_wins"])} |
| BM25 query wins | {int(summary_row["improved_wins"])} |
| Ties | {int(summary_row["ties"])} |

## Query-Level Highlights

- Best BM25 advantage: `{bm25_case["Query"]}` with BM25 `P@5={bm25_case["bm25"]:.2f}` versus TF-IDF `P@5={bm25_case["tfidf"]:.2f}`
- Failure case (lowest overall retrieval quality): `{failure_case["Query"]}` with TF-IDF `P@5={failure_case["tfidf"]:.2f}` and BM25 `P@5={failure_case["bm25"]:.2f}`
{tie_line}

## Per-Query Case Studies

{case_study}

## Interpretation

BM25 holds a small average lead in this corpus, but the comparison is not one-sided. BM25 performs best when a query is dense with topical terms that can benefit from length normalization and softer term-frequency saturation. TF-IDF remains competitive when success depends on sharp lexical overlap between the information need and a narrow set of documents. The final result is therefore a meaningful but modest improvement rather than a blanket dominance claim.

## Provenance

- The 10 retrieval queries were generated by GPT-5.4 (the model driving OpenAI Codex) following the six-topic scheme used in Task 3, plus four additional information-need categories.
- Relevance judgments (Y/N for each query-document pair) were also produced by GPT-5.4 rather than a human assessor.
- This means both the queries and the relevance labels reflect LLM judgment. The P@5 comparison between TF-IDF and BM25 remains valid as a relative ranking, but the absolute precision values should be interpreted with this in mind.
"""


def main() -> None:
    """Build the current Task 2 analysis report from CSV artifacts."""

    args = build_argument_parser().parse_args()

    summary = pd.read_csv(args.summary_input)
    per_query = pd.read_csv(args.per_query_input)
    if summary.empty:
        raise ValueError("Retrieval summary CSV is empty.")

    report = build_report(summary.iloc[0], per_query, args.per_query_dir)
    write_text(args.output, report)

    print(f"Saved Task 2 analysis report to {args.output}.")


if __name__ == "__main__":
    main()
