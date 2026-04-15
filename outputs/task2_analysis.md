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

## Per-Query Case Studies

### Best BM25 Advantage

**Query**: Which articles report on basketball, boxing, golf, baseball, or other sports competitions and athlete performance?

| Model | P@5 |
| --- | ---: |
| TF-IDF | 0.40 |
| BM25 | 0.80 |

**BM25 Top-3 Results**:

- Rank 1: `3049341` (score: 37.52) — In newspapers and on TV , black families see more success stories in sports than in any other field .…
- Rank 2: `3032641` (score: 30.72) — Since 1962 , 27 Stags have gone on to play professional basketball , football or baseball . Since 1986 , 99 have gone on to play college athletics .…
- Rank 3: `3000041` (score: 30.49) — Scoring in the event is based on a modified Stableford system , in which points are awarded for the players ' performance on each hole : 5 for eagle ,…

**TF-IDF Top-3 Results**:

- Rank 1: `3049341` (score: 0.16) — In newspapers and on TV , black families see more success stories in sports than in any other field .…
- Rank 2: `3041741` (score: 0.11) — And so it grieves us when we find that sports figures are not practicing those same virtues off the court or off the field as we expect them to practi…
- Rank 3: `3031741` (score: 0.11) — " I was totally ignorant to boxing other than throwing punches , " says Hopkins , who sued promoter Butch Lewis for breach of contract in 1997 and won…

**Analysis**: BM25 surfaces 2 document(s) (`3000041`, `3032641`) not in TF-IDF's top-3. The query enumerates multiple sport types, creating a broad set of topical terms. BM25's sublinear tf saturation (controlled by `k1`) prevents a single repeated term like 'sports' from dominating the score, allowing documents that cover diverse sporting contexts (e.g., `3049341`: 'In newspapers and on TV , black families see more success stories in sports than…') to rank highly. Its document length normalization (`b=0.75`) also elevates concise, focused articles over verbose ones. TF-IDF's unbounded term-frequency weighting, by contrast, rewards documents that repeat query terms most often, regardless of whether the repetition is informative. The BM25 score spread across the top-3 (1.2× between rank 1 and rank 3) shows good discrimination between highly relevant and marginally relevant results.

### Best TF-IDF Advantage

**Query**: Which articles cover movies, television, theater, books, or entertainment industry figures?

| Model | P@5 |
| --- | ---: |
| TF-IDF | 0.80 |
| BM25 | 0.60 |

**TF-IDF Top-3 Results**:

- Rank 1: `3005041` (score: 0.12) — , " which received strong reviews , was also deemed ineligible because it was mounted in an Off Broadway theater , the same one that housed " Cabaret …
- Rank 2: `3040841` (score: 0.10) — It 's the illness of longtime economic turn makes tickets too expensive ( $ 65 for top musicals on weekend nights ) for the middle class , which in tu…
- Rank 3: `3016841` (score: 0.09) — And if I teach a film course , speak at a cinema club , lecture at a museum , or do a guest shot on television , more viewing may come with the job .…

**BM25 Top-3 Results**:

- Rank 1: `3016841` (score: 21.14) — And if I teach a film course , speak at a cinema club , lecture at a museum , or do a guest shot on television , more viewing may come with the job .…
- Rank 2: `3018541` (score: 19.25) — In deference to differing computer and connection capabilities , both The Fly Thru and - with several screen resolutions for the video imagery , varyi…
- Rank 3: `4061241` (score: 17.90) — The success of March of the Penguins in 2006 has revitalized the nature documentary field as well , a mainstay of science entertainment , says Jean-Fr…

**Analysis**: TF-IDF retrieves 2 document(s) (`3005041`, `3040841`) not in BM25's top-3. The query targets specific entertainment domains ('movies, television, theater, books'), and TF-IDF's cosine similarity rewards documents where multiple query terms co-occur in concentrated passages — the top result `3005041` (', " which received strong reviews , was also deemed ineligible because it was mo…') matches on several entertainment terms simultaneously. BM25's saturation function dampens the score contribution of repeated domain terms: the second occurrence of 'theater' or 'film' adds less incremental score than in TF-IDF, causing BM25 to promote tangential documents like `4061241` (score 17.9) into the top-5 at the expense of better lexical matches.


## Interpretation

BM25 holds a small average lead in this corpus, but the comparison is not one-sided. BM25 performs best when a query is dense with topical terms that can benefit from length normalization and softer term-frequency saturation. TF-IDF remains competitive when success depends on sharp lexical overlap between the information need and a narrow set of documents. The final result is therefore a meaningful but modest improvement rather than a blanket dominance claim.

## Provenance

- The 10 retrieval queries were generated by GPT-5.4 (the model driving OpenAI Codex) following the six-topic scheme used in Task 3, plus four additional information-need categories.
- Relevance judgments (Y/N for each query-document pair) were also produced by GPT-5.4 rather than a human assessor.
- This means both the queries and the relevance labels reflect LLM judgment. The P@5 comparison between TF-IDF and BM25 remains valid as a relative ranking, but the absolute precision values should be interpreted with this in mind.
