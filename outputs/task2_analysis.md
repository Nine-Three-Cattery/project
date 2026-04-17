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
- Failure case (lowest overall retrieval quality): `Which articles describe charities, social services, food pantries, or community support programs?` with TF-IDF `P@5=0.20` and BM25 `P@5=0.40`
- Strongest tie: `Which articles focus on elections, government decisions, political parties, or debates over public policy?` at `P@5=1.00`

## Per-Query Case Studies

### Success Case

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

### Failure Case

**Query**: Which articles describe charities, social services, food pantries, or community support programs?

| Model | P@5 |
| --- | ---: |
| TF-IDF | 0.20 |
| BM25 | 0.40 |

**BM25 Top-3 Results**:

- Rank 1: `4132041` (score: 21.04) — Donations of non-perishable or canned food items will be accepted in the lobby and will be donated to local food banks and charities .…
- Rank 2: `3047541` (score: 19.34) — " A lot of Republicans argued for volunteerism as a substitute for cuts in social services -- which is totally inadequate , " said Pablo Eisenberg , d…
- Rank 3: `3068041` (score: 17.64) — Clinton 's explanations of her public conduct in Arkansas and Washington were just sort of putting her best foot forward as a social crusader , not as…

**TF-IDF Top-3 Results**:

- Rank 1: `3051741` (score: 0.08) — " Social Security by the numbers 36 Number of campaign stops made by President Bush to push his Social Security proposals ( since the 2004 election ) …
- Rank 2: `5015741` (score: 0.06) — The 10 nominees come from diverse backgrounds , and include representatives from public health , parenting and education groups , child welfare expert…
- Rank 3: `3009741` (score: 0.05) — " The oil-for-food deal is based on UN Security Council Resolution 986 , which allows Iraq to sell $ 2 billion worth of oil every six months .…

**Analysis**: This query is selected as the failure case because both systems stay at low precision (TF-IDF `P@5=0.20`, BM25 `P@5=0.40`). Even when BM25 slightly leads, neither model reaches acceptable retrieval quality. The top-3 overlap (none) suggests both rankers are driven toward different partially relevant documents, indicating a shared coverage gap rather than a ranking-only issue.


## Interpretation

BM25 holds a small average lead in this corpus, but the comparison is not one-sided. BM25 performs best when a query is dense with topical terms that can benefit from length normalization and softer term-frequency saturation. TF-IDF remains competitive when success depends on sharp lexical overlap between the information need and a narrow set of documents. The final result is therefore a meaningful but modest improvement rather than a blanket dominance claim.

## Provenance

- The 10 retrieval queries were generated by GPT-5.4 (the model driving OpenAI Codex) following the six-topic scheme used in Task 3, plus four additional information-need categories.
- Relevance judgments (Y/N for each query-document pair) were also produced by GPT-5.4 rather than a human assessor.
- This means both the queries and the relevance labels reflect LLM judgment. The P@5 comparison between TF-IDF and BM25 remains valid as a relative ranking, but the absolute precision values should be interpreted with this in mind.
