# Task 1 Corpus Audit and Preprocessing

## Corpus Summary

| Metric | Value |
| --- | ---: |
| Number of documents | 871 |
| Total token count | 1,147,881 |
| Average document length | 1317.89 |
| Median document length | 1250.00 |
| Vocabulary size before preprocessing | 57,401 |
| Vocabulary size after preprocessing | 47,998 |

## Parser and Cleaning Decisions

- The parser locates document boundaries using the leading `@@<numeric_id>` pattern.
- Document identifiers are preserved as metadata but excluded from the cleaned document text.
- Cleaning is intentionally light and non-destructive:
  - repeated gap noise is mapped to `<GAP>`
  - all text is lowercased
  - whitespace is normalized
  - punctuation, stopwords, and rare tokens are retained in the cleaned corpus output
- The cleaned corpus therefore remains suitable for both retrieval and downstream inspection.

## Supervision Assessment

The corpus is best treated as an unlabeled, unsupervised collection. The files contain document identifiers, but they do not expose class labels, topic metadata, or any pre-defined supervision signal. The IDs are therefore bookkeeping keys rather than semantic annotations. Any supervised experiment built on top of this corpus requires labels to be created after inspection.

## Before and After Examples

### Document 3000041
- Raw: `Basketball United States Wins Bronze Medal : The United States used another comeback yesterday to take the bronze medal at the world championships in Buenos Aires , beating Puerto Rico by 107-105 in overtime . Kenny Anderson scored 34 points and made two free ...`
- Cleaned: `basketball united states wins bronze medal : the united states used another comeback yesterday to take the bronze medal at the world championships in buenos aires , beating puerto rico by 107-105 in overtime . kenny anderson scored 34 points and made two free ...`
### Document 3000341
- Raw: `In the airy sitting room of the ranch house where he has lived for 48 years , Byron Nelson holds a picture in his gnarled hands and peers back through time . In the frame , a lean young man is in the perfect golf position , his arms extended and the shaft of h...`
- Cleaned: `in the airy sitting room of the ranch house where he has lived for 48 years , byron nelson holds a picture in his gnarled hands and peers back through time . in the frame , a lean young man is in the perfect golf position , his arms extended and the shaft of h...`
### Document 3000641
- Raw: `It was just a normal summer day when Allan Houston stopped his workout in midchurn , stepped off the stationary bike and ducked into a film room at the Knicks ' practice facility . Just once , he wanted to see it for himself . Not the highlight version , but t...`
- Cleaned: `it was just a normal summer day when allan houston stopped his workout in midchurn , stepped off the stationary bike and ducked into a film room at the knicks ' practice facility . just once , he wanted to see it for himself . not the highlight version , but t...`

## Figures

![Document length histogram](figures/doc_length_hist.png)

![Zipf plot](figures/zipf_plot.png)
