# Task 3 Annotation Guideline

## Purpose

This guideline defines the six topic labels used for Track A topic classification and documents the LLM-assisted annotation pipeline that produced the released label set.

## Annotation Pipeline

The 200 topic labels were produced through a three-stage LLM-assisted pipeline rather than single-human manual annotation:

### Stage 1: Seed Generation

`bootstrap_task3_labels.py` uses BM25 topic-query matching to assign initial high-confidence labels. For each of the five named topic categories, a keyword query is scored against the full corpus. Documents with a clear margin between the best and second-best topic score are assigned to the winning category. The lowest-scoring documents are assigned to `Other`. The configuration is stored in `configs/task3_bootstrap.yaml`.

### Stage 2: Primary Review (GPT-5.4 / Codex)

GPT-5.4, the model driving OpenAI Codex, reviewed each of the 200 documents in full. For each document, the model read the complete article text and either confirmed or corrected the bootstrap seed label. The resulting labels were tagged as `source: llm_review` with `annotation_method: llm_review`.

### Stage 3: Cross-Provider External Audit

Two independent external models audited all 200 labels without seeing the current assignments:

- **Gemini 2.5 Flash** (Google Vertex AI) — agreement rate with released labels: 93.5%
- **Grok 4.20** (xAI) — agreement rate with released labels: 95.5%
- **Cross-provider agreement** (Gemini vs Grok): 90.5%

Documents where both external auditors unanimously agreed on a label different from the released one were flagged for re-adjudication. The 13 resulting corrections are recorded in `configs/task3_review_updates.yaml`. These re-adjudicated labels are tagged with `annotation_method: direct_document_review_with_external_llm_audit`.

### Limitations

- All three annotators are LLMs. Shared systematic biases (e.g., preferring more salient topics, struggling with mixed-topic roundup articles) may exist.
- The pipeline is reproducible and auditable, but it does not constitute adjudicated multi-human gold labels.
- The cross-provider agreement rate (90.5%) provides an indirect quality signal but not a direct guarantee of correctness.

## General Labeling Rules

- Assign exactly one label to each document.
- Label the document according to its dominant news angle, not just the first named entity.
- Prefer the label that best describes the central event, conflict, or information need served by the article.
- If a document is genuinely mixed and no single theme dominates, use `Other`.
- Use the full article context when possible. Headlines and first sentences are useful, but they are not always sufficient for edge cases.

## Label Definitions

### Politics

Use `Politics` for documents whose main focus is government activity, elections, public policy, diplomacy, legislative conflict, political campaigning, or ideologically framed disputes.

Include:
- elections, parties, campaigns, voting disputes
- government appointments, resignations, official policy decisions
- international diplomacy or politically framed geopolitical reporting

Exclude:
- corporate regulation stories whose center of gravity is business performance
- crime stories involving public officials when the primary issue is prosecution or investigation

### Business/Economy

Use `Business/Economy` for documents centered on firms, industry trends, jobs, markets, finance, trade, prices, or macroeconomic conditions.

Include:
- company strategy, mergers, product commercialization, earnings, and market moves
- inflation, employment, wages, consumer demand, trade, and broader economic policy impacts
- education or infrastructure stories when the dominant angle is cost, budgeting, or financial viability

Exclude:
- political stories where economic policy is secondary to partisan conflict
- technology stories whose main point is innovation rather than business consequences

### Sports

Use `Sports` for coverage of competitions, tournaments, athletes, coaches, teams, game outcomes, rankings, or sports administration.

Include:
- match results, player performance, records, injuries, transfers
- commentary on sports culture when the article remains anchored to teams or athletic competition

Exclude:
- celebrity features where the sports connection is incidental and the main focus is entertainment or culture

### Crime/Law

Use `Crime/Law` for documents centered on illegal conduct, policing, prosecution, courts, corrections, public safety incidents, or formal legal disputes.

Include:
- criminal charges, investigations, trials, sentencing, prison conditions
- police misconduct, jail abuse, civil rights cases with a legal or enforcement center
- lawsuits and patent disputes when the article is primarily about the litigation itself

Exclude:
- political scandals where the main emphasis is electoral or governmental impact rather than legal process

### Culture/Entertainment

Use `Culture/Entertainment` for coverage of film, television, music, theater, books, museums, celebrity profiles, reviews, and arts criticism.

Include:
- actor, author, musician, or performer profiles
- reviews of movies, shows, books, or stage productions
- arts institutions, awards, festivals, and entertainment business reporting when the cultural product remains central

Exclude:
- business articles about media companies where the main point is corporate finance or strategy

### Other

Use `Other` only when the article does not fit the five primary labels or remains too mixed after full reading.

Common examples:
- local community features that are primarily human-interest
- lifestyle or advice pieces
- health and wellness columns without a strong politics, business, crime, or culture frame
- infrastructure or education stories that do not clearly resolve into politics or business

## Tie-Breaking Rules

- If a story involves both policy and courts, use `Crime/Law` when the article emphasizes charges, evidence, sentencing, prisons, or legal procedure. Use `Politics` when the article emphasizes public office, governing consequences, or electoral impact.
- If a story involves media companies or publishing, use `Culture/Entertainment` when the article is about the creative work or public reception. Use `Business/Economy` when the article is about market performance, corporate deals, or revenue.
- If a sports article profiles an athlete but the piece still centers on the sport, keep it in `Sports`.

## Revision Log

- Version 1.0: Established the six-label scheme with `Crime/Law` separated from `Other` because the corpus contains enough policing, courts, and prison-related coverage to justify a dedicated category.
- Version 1.1: Applied BM25 bootstrap seed labels and GPT-5.4 primary review over the 200-document label set.
- Version 1.2: Re-adjudicated the subset of documents for which independent external LLM reviewers (Gemini 2.5 Flash and Grok 4.20) unanimously disagreed with the released label, producing the `direct_review_v2` label set.
