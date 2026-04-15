# Task 2 Retrieval Analysis

## Setup

- Baseline: unigram TF-IDF with cosine similarity
- Improved variant: BM25 using the same tokenizer as the baseline
- Queries judged: 10
- Results judged per query and model: top 5

## Precision@5 Summary

| Metric | Value |
| --- | ---: |
| Average TF-IDF Precision@5 | 0.76 |
| Average BM25 Precision@5 | 0.78 |
| TF-IDF query wins | 3 |
| BM25 query wins | 3 |
| Ties | 4 |

## Query-Level Highlights

- Best BM25 advantage: `Which articles report on basketball, boxing, golf, baseball, or other sports competitions and athlete performance?` with BM25 `P@5=0.80` versus TF-IDF `P@5=0.40`
- Best TF-IDF advantage: `Which articles cover movies, television, theater, books, or entertainment industry figures?` with TF-IDF `P@5=0.80` versus BM25 `P@5=0.60`
- Strongest tie: `Which articles focus on elections, government decisions, political parties, or debates over public policy?` at `P@5=1.00`

## Interpretation

BM25 holds a small average lead in this corpus, but the comparison is not one-sided. BM25 performs best when a query is dense with topical terms that can benefit from length normalization and softer term-frequency saturation. TF-IDF remains competitive when success depends on sharp lexical overlap between the information need and a narrow set of documents. The final result is therefore a meaningful but modest improvement rather than a blanket dominance claim.
