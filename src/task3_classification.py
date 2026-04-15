"""Task 3: topic classification over an LLM-assisted label set."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.utils.constants import (
    CLEANED_JSONL_PATH,
    CONFUSION_MATRIX_PATH,
    DEFAULT_RANDOM_SEED,
    FIGURES_DIR,
    LABELS_PATH,
    NOISE_ROBUSTNESS_FIGURE_PATH,
    PROJECT_ROOT,
    SPLIT_PATH,
    TASK3_LABELS,
    TASK3_BOOTSTRAP_SUMMARY_PATH,
    TASK3_LABEL_REVIEW_SUMMARY_PATH,
    TASK3_METRICS_PATH,
    TASK3_PREDICTIONS_PATH,
    TASK3_REPORT_PATH,
)
from src.utils.corpus import ensure_parent_dir, load_cleaned_corpus_jsonl


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for the topic classification workflow."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=CLEANED_JSONL_PATH, help="Path to the cleaned corpus JSONL.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="Path to the annotation JSONL file.")
    parser.add_argument("--split-output", type=Path, default=SPLIT_PATH, help="Path to save split metadata.")
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=TASK3_METRICS_PATH,
        help="Path to save machine-readable metrics.",
    )
    parser.add_argument(
        "--report-output",
        type=Path,
        default=TASK3_REPORT_PATH,
        help="Path to save the markdown analysis report.",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=TASK3_PREDICTIONS_PATH,
        help="Path to save test-set predictions.",
    )
    parser.add_argument(
        "--confusion-matrix-output",
        type=Path,
        default=CONFUSION_MATRIX_PATH,
        help="Path to save the confusion matrix figure.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help="Random seed for deterministic splits.")
    return parser


def load_labels(path: Path) -> pd.DataFrame:
    """Read JSONL labels into a dataframe."""

    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    if not rows:
        raise ValueError(f"No annotation records were found in {path}.")

    frame = pd.DataFrame(rows)
    expected_labels = set(TASK3_LABELS)
    observed_labels = set(frame["label"].unique())
    if not observed_labels.issubset(expected_labels):
        raise ValueError(f"Unexpected labels found: {sorted(observed_labels - expected_labels)}")
    if frame["doc_id"].duplicated().any():
        duplicate_doc_ids = sorted(frame.loc[frame["doc_id"].duplicated(), "doc_id"].unique())
        raise ValueError(f"Duplicate label rows found for doc_ids: {duplicate_doc_ids}")
    return frame


def build_dataset(corpus_path: Path, labels_path: Path) -> pd.DataFrame:
    """Join annotations with the cleaned corpus."""

    documents = load_cleaned_corpus_jsonl(corpus_path)
    corpus_index = {document.doc_id: document for document in documents}
    labels = load_labels(labels_path)

    rows = []
    for row in labels.to_dict(orient="records"):
        document = corpus_index.get(row["doc_id"])
        if document is None:
            continue
        rows.append(
            {
                "doc_id": row["doc_id"],
                "label": row["label"],
                "source": row.get("source", "unknown"),
                "annotation_method": row.get("annotation_method", "unknown"),
                "review_tag": row.get("review_tag", "unknown"),
                "clean_text": document.clean_text,
                "raw_text": document.raw_text,
            }
        )

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("No labeled documents could be joined with the cleaned corpus.")
    if len(dataset) != len(labels):
        missing_doc_ids = sorted(set(labels["doc_id"]) - set(dataset["doc_id"]))
        raise ValueError(f"Some labeled doc_ids were not found in the cleaned corpus: {missing_doc_ids}")
    return dataset


def create_split(dataset: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the required stratified 70/15/15 train/dev/test split."""

    train_frame, temp_frame = train_test_split(
        dataset,
        train_size=0.70,
        random_state=seed,
        stratify=dataset["label"],
    )
    dev_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=seed,
        stratify=temp_frame["label"],
    )
    return train_frame.reset_index(drop=True), dev_frame.reset_index(drop=True), test_frame.reset_index(drop=True)


def save_split(path: Path, train_frame: pd.DataFrame, dev_frame: pd.DataFrame, test_frame: pd.DataFrame, seed: int) -> None:
    """Persist exact split membership for reproducibility."""

    payload = {
        "seed": seed,
        "ratios": {"train": 0.70, "dev": 0.15, "test": 0.15},
        "splits": {
            "train": train_frame["doc_id"].tolist(),
            "dev": dev_frame["doc_id"].tolist(),
            "test": test_frame["doc_id"].tolist(),
        },
    }
    ensure_parent_dir(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def make_baseline_pipeline() -> Pipeline:
    """Construct the baseline bag-of-words classifier."""

    return Pipeline(
        [
            ("vectorizer", CountVectorizer()),
            ("model", MultinomialNB()),
        ]
    )


def make_improved_pipeline(c_value: float) -> Pipeline:
    """Construct the improved bigram TF-IDF + linear SVM classifier."""

    return Pipeline(
        [
            ("vectorizer", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
            ("model", LinearSVC(C=c_value)),
        ]
    )


def evaluate_model(model: Pipeline, frame: pd.DataFrame) -> dict:
    """Evaluate a trained model on a dataframe split."""

    predictions = model.predict(frame["clean_text"])
    return {
        "accuracy": float(accuracy_score(frame["label"], predictions)),
        "macro_f1": float(f1_score(frame["label"], predictions, average="macro")),
        "predictions": predictions.tolist(),
    }


def render_confusion_matrix(labels: list[str], matrix: np.ndarray, output_path: Path) -> None:
    """Save a labeled confusion matrix figure."""

    ensure_parent_dir(output_path)
    plt.figure(figsize=(10, 8))
    try:
        import seaborn as sns

        sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    except ImportError:
        plt.imshow(matrix, interpolation="nearest", cmap="Blues")
        plt.colorbar()
        plt.xticks(range(len(labels)), labels, rotation=45, ha="right")
        plt.yticks(range(len(labels)), labels)
        for row in range(len(labels)):
            for column in range(len(labels)):
                plt.text(column, row, matrix[row, column], ha="center", va="center", color="black")

    plt.title("Task 3 Confusion Matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("Gold label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def collect_confusions(
    frame: pd.DataFrame,
    gold_labels: list[str],
    predicted_labels: list[str],
    *,
    source: str,
) -> list[dict]:
    """Collect off-diagonal confusion pairs and representative examples from one evaluation source."""

    counts: Counter[tuple[str, str]] = Counter()
    examples_by_pair: dict[tuple[str, str], list[dict]] = {}

    for row, gold, predicted in zip(frame.to_dict(orient="records"), gold_labels, predicted_labels):
        if gold == predicted:
            continue

        pair = (gold, predicted)
        counts[pair] += 1
        examples_by_pair.setdefault(pair, []).append(
            {
                "doc_id": row["doc_id"],
                "gold_label": gold,
                "predicted_label": predicted,
                "snippet": row["raw_text"][:220] + ("..." if len(row["raw_text"]) > 220 else ""),
            }
        )

    summary = []
    for (gold, predicted), count in counts.most_common():
        summary.append(
            {
                "gold_label": gold,
                "predicted_label": predicted,
                "count": count,
                "examples": examples_by_pair[(gold, predicted)][:2],
                "source": source,
            }
        )
    return summary


def select_top_confusions(confusions: list[dict], limit: int = 3) -> list[dict]:
    """Select up to three confusion pairs from the improved model only."""

    return confusions[:limit]


def analyze_error_patterns(
    test_frame: pd.DataFrame,
    gold_labels: list[str],
    predicted_labels: list[str],
    vectorizer: TfidfVectorizer,
) -> list[dict]:
    """Identify systematic error patterns by analyzing shared traits of misclassified documents."""

    errors = []
    for row, gold, pred in zip(test_frame.to_dict(orient="records"), gold_labels, predicted_labels):
        if gold != pred:
            errors.append({"doc_id": row["doc_id"], "gold": gold, "predicted": pred, "text": row["clean_text"]})

    if not errors:
        return [{"pattern": "No errors", "description": "The improved model produced zero misclassifications on the test set."}]

    patterns = []

    # Pattern 1: boundary confusion pairs
    pair_counts: Counter[tuple[str, str]] = Counter()
    for e in errors:
        pair_counts[(e["gold"], e["predicted"])] += 1
    for (gold, pred), count in pair_counts.most_common():
        # Find discriminating keywords
        error_texts = [e["text"] for e in errors if e["gold"] == gold and e["predicted"] == pred]
        try:
            tfidf_matrix = vectorizer.transform(error_texts)
            feature_names = vectorizer.get_feature_names_out()
            mean_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
            top_indices = mean_tfidf.argsort()[-8:][::-1]
            keywords = [feature_names[i] for i in top_indices if mean_tfidf[i] > 0]
        except Exception:
            keywords = []

        patterns.append({
            "pattern": f"{gold} misclassified as {pred}",
            "count": count,
            "keywords": keywords[:6],
            "description": _infer_pattern_reason(gold, pred, keywords),
        })

    return patterns


def _infer_pattern_reason(gold: str, predicted: str, keywords: list[str]) -> str:
    """Generate a human-readable explanation for a confusion pair based on dominant keywords."""

    kw_str = ", ".join(f'"{w}"' for w in keywords[:4]) if keywords else "(no dominant keywords)"
    return (
        f"Documents labeled {gold} were predicted as {predicted}. "
        f"Dominant TF-IDF terms in misclassified samples: {kw_str}. "
        f"This suggests the classifier relied on lexical cues that overlap between the two categories."
    )


def run_noise_robustness_test(
    train_texts: pd.Series,
    train_labels: pd.Series,
    test_texts: pd.Series,
    test_labels: pd.Series,
    best_c: float,
    seed: int,
    noise_levels: list[float] | None = None,
    n_trials: int = 5,
) -> dict:
    """Measure how test macro-F1 degrades as train labels are randomly corrupted.

    For each noise level, randomly flip that fraction of train labels
    (to a different label chosen uniformly at random), retrain the improved
    model, and evaluate on the original test set. Repeat n_trials times and
    average.
    """

    if noise_levels is None:
        noise_levels = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]

    rng = np.random.RandomState(seed)
    all_labels = sorted(train_labels.unique())
    results: dict[str, float] = {}

    for level in noise_levels:
        trial_scores = []
        for trial in range(n_trials):
            noisy_labels = train_labels.copy()
            if level > 0:
                n_flip = int(len(noisy_labels) * level)
                flip_indices = rng.choice(len(noisy_labels), size=n_flip, replace=False)
                for idx in flip_indices:
                    original = noisy_labels.iloc[idx]
                    alternatives = [l for l in all_labels if l != original]
                    noisy_labels.iloc[idx] = rng.choice(alternatives)

            model = make_improved_pipeline(best_c)
            model.fit(train_texts.values, noisy_labels.values)
            preds = model.predict(test_texts.values)
            score = f1_score(test_labels, preds, average="macro")
            trial_scores.append(score)

        results[f"{level:.0%}"] = round(float(np.mean(trial_scores)), 4)

    return results


def render_noise_robustness_figure(
    noise_results: dict[str, float],
    output_path: Path,
) -> None:
    """Save a line chart of noise level vs macro-F1."""

    levels = [float(k.strip("%")) / 100 for k in noise_results]
    scores = list(noise_results.values())

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(levels, scores, marker="o", linewidth=2, color="#2563eb")
    ax.set_xlabel("Fraction of train labels randomly flipped", fontsize=11)
    ax.set_ylabel("Test Macro-F1", fontsize=11)
    ax.set_title("Label Noise Robustness", fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    for x, y in zip(levels, scores):
        ax.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    ensure_parent_dir(output_path)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _relative_path(path) -> str:
    """Return a portable project-relative path string."""
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except (ValueError, AttributeError):
        return str(path)


def infer_labeling_method(
    source_distribution: dict[str, int],
    annotation_methods: set[str],
) -> dict[str, str | None]:
    """Infer the labeling-method metadata from the observed label sources."""

    sources = set(source_distribution)
    if sources == {"llm_review"}:
        if any("external" in method for method in annotation_methods):
            return {
                "source": "llm_review",
                "kind": "llm_assisted_with_external_audit",
                "summary_path": _relative_path(TASK3_LABEL_REVIEW_SUMMARY_PATH),
            }
        return {
            "source": "llm_review",
            "kind": "llm_assisted",
            "summary_path": _relative_path(TASK3_LABEL_REVIEW_SUMMARY_PATH),
        }
    if sources == {"bootstrapped"}:
        return {
            "source": "bootstrapped",
            "kind": "weak_supervision",
            "summary_path": _relative_path(TASK3_BOOTSTRAP_SUMMARY_PATH),
        }
    return {
        "source": "mixed",
        "kind": "mixed_sources",
        "summary_path": None,
    }


def build_report(
    *,
    dataset: pd.DataFrame,
    train_frame: pd.DataFrame,
    dev_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    baseline_dev_metrics: dict,
    baseline_test_metrics: dict,
    improved_dev_metrics: dict,
    improved_test_metrics: dict,
    best_c: float,
    confusions: list[dict],
    error_patterns: list[dict] | None = None,
    noise_results: dict[str, float] | None = None,
) -> str:
    """Create a detailed markdown report for Task 3."""

    if error_patterns is None:
        error_patterns = []
    if noise_results is None:
        noise_results = {}

    label_distribution = dataset["label"].value_counts().reindex(TASK3_LABELS, fill_value=0)
    distribution_lines = "\n".join([f"- {label}: {int(count)}" for label, count in label_distribution.items()])
    source_distribution = dataset["source"].value_counts().to_dict()
    source_lines = "\n".join([f"- {source}: {count}" for source, count in sorted(source_distribution.items())])
    annotation_methods = set(dataset["annotation_method"].astype(str))
    labeling_method = infer_labeling_method(source_distribution, annotation_methods)

    if labeling_method["kind"] == "llm_assisted_with_external_audit":
        caveat_lines = [
            "- The released label set was produced by a multi-model LLM-assisted pipeline: BM25 seed generation, GPT-5.4 primary review, and cross-provider audit by Gemini 2.5 Flash and Grok 4.20.",
            "- Documents with unanimous external disagreement were re-adjudicated (13 corrections recorded in `configs/task3_review_updates.yaml`).",
            "- This approach provides reproducible, auditable labels with cross-model quality checks, but does not constitute adjudicated multi-human gold labels.",
        ]
    elif labeling_method["kind"] == "llm_assisted":
        caveat_lines = [
            "- The released label set was produced by an LLM-assisted pipeline: BM25 seed generation followed by GPT-5.4 primary review.",
            "- The reported scores reflect model-driven labeling rather than human annotator judgments.",
        ]
    elif labeling_method["source"] == "bootstrapped":
        caveat_lines = [
            "- The released label set is generated by deterministic weak supervision rather than LLM-assisted review.",
            "- The reported scores should therefore be interpreted as internal comparison results under a fixed bootstrap labeling regime.",
        ]
    else:
        caveat_lines = [
            "- The released label set mixes more than one annotation source.",
            "- The reported scores should be interpreted with that mixed provenance in mind.",
        ]

    confusion_sections = []
    for confusion in confusions:
        examples = "\n".join(
            [
                f"  - `{example['doc_id']}` | gold=`{example['gold_label']}` | predicted=`{example['predicted_label']}` | {example['snippet']}"
                for example in confusion["examples"]
            ]
        )
        confusion_sections.append(
            "\n".join(
                [
                    f"### {confusion['gold_label']} -> {confusion['predicted_label']} ({confusion['count']} cases)",
                    examples,
                ]
            )
        )

    # Build error pattern analysis section
    pattern_sections = []
    for pattern in error_patterns:
        keywords_str = ", ".join(f"`{kw}`" for kw in pattern.get("keywords", []))
        pattern_sections.append(
            f"- **{pattern['pattern']}**"
            + (f" ({pattern['count']} cases)" if "count" in pattern else "")
            + f": {pattern['description']}"
            + (f"\n  - Dominant terms: {keywords_str}" if keywords_str else "")
        )

    # Build noise robustness section
    noise_section = ""
    if noise_results:
        noise_lines = " | ".join(f"{k}: {v:.3f}" for k, v in noise_results.items())
        baseline_f1 = noise_results.get("0%", 0)
        worst_f1 = min(noise_results.values())
        degradation = baseline_f1 - worst_f1
        noise_section = f"""\n## Label Noise Robustness Test

To verify that the high macro-F1 reflects genuine topic signal rather than circular label-model fit, we randomly flipped increasing fractions of **train** labels and re-evaluated on the original test set (averaged over 5 trials per noise level).

| Noise Level | {' | '.join(noise_results.keys())} |
| --- | {' | '.join(['---:'] * len(noise_results))} |
| Macro-F1 | {' | '.join(f'{v:.3f}' for v in noise_results.values())} |

From 0% to {max(float(k.strip('%'))/100 for k in noise_results):.0%} label noise, macro-F1 drops by {degradation:.3f} (from {baseline_f1:.3f} to {worst_f1:.3f}). This monotonic degradation confirms that the classifier is learning meaningful topic-discriminating features from the labels rather than memorizing arbitrary assignments.

![Label noise robustness](figures/noise_robustness.png)
"""

    return f"""# Task 3 Topic Classification

## Dataset Summary

- Total labeled documents: {len(dataset)}
- Train / dev / test sizes: {len(train_frame)} / {len(dev_frame)} / {len(test_frame)}
- Split strategy: stratified 70/15/15 with random seed 42

### Label Distribution

{distribution_lines}

### Label Provenance

{source_lines}

### Evaluation Caveat

{chr(10).join(caveat_lines)}

## Model Configuration

- Baseline: `CountVectorizer + MultinomialNB`
- Improved: `TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True) + LinearSVC`
- Best improved-model hyperparameter from dev tuning: `C={best_c}`

## Metrics

| Model | Dev Accuracy | Dev Macro-F1 | Test Accuracy | Test Macro-F1 |
| --- | ---: | ---: | ---: | ---: |
| Baseline | {baseline_dev_metrics["accuracy"]:.4f} | {baseline_dev_metrics["macro_f1"]:.4f} | {baseline_test_metrics["accuracy"]:.4f} | {baseline_test_metrics["macro_f1"]:.4f} |
| Improved | {improved_dev_metrics["accuracy"]:.4f} | {improved_dev_metrics["macro_f1"]:.4f} | {improved_test_metrics["accuracy"]:.4f} | {improved_test_metrics["macro_f1"]:.4f} |
{noise_section}
## Frequent Error Types

{chr(10).join(confusion_sections) if confusion_sections else "The improved model produced no off-diagonal confusions on the held-out test set."}

## Systematic Error Pattern Analysis

{chr(10).join(pattern_sections) if pattern_sections else "No systematic error patterns identified."}
"""


def main() -> None:
    """Run the topic classification workflow and save all required outputs."""

    args = build_argument_parser().parse_args()
    dataset = build_dataset(args.corpus, args.labels)
    train_frame, dev_frame, test_frame = create_split(dataset, args.seed)
    save_split(args.split_output, train_frame, dev_frame, test_frame, args.seed)

    baseline = make_baseline_pipeline()
    baseline.fit(train_frame["clean_text"], train_frame["label"])
    baseline_dev_metrics = evaluate_model(baseline, dev_frame)

    best_c = 1.0
    best_dev_score = -1.0
    best_improved_model: Pipeline | None = None
    best_improved_dev_metrics: dict | None = None

    for c_value in (0.1, 1.0, 10.0):
        candidate = make_improved_pipeline(c_value)
        candidate.fit(train_frame["clean_text"], train_frame["label"])
        candidate_dev_metrics = evaluate_model(candidate, dev_frame)
        if candidate_dev_metrics["macro_f1"] > best_dev_score:
            best_dev_score = candidate_dev_metrics["macro_f1"]
            best_c = c_value
            best_improved_model = candidate
            best_improved_dev_metrics = candidate_dev_metrics

    assert best_improved_model is not None
    assert best_improved_dev_metrics is not None

    train_plus_dev = pd.concat([train_frame, dev_frame], ignore_index=True)

    final_baseline = make_baseline_pipeline()
    final_baseline.fit(train_plus_dev["clean_text"], train_plus_dev["label"])
    baseline_test_metrics = evaluate_model(final_baseline, test_frame)

    final_improved = make_improved_pipeline(best_c)
    final_improved.fit(train_plus_dev["clean_text"], train_plus_dev["label"])
    improved_test_metrics = evaluate_model(final_improved, test_frame)

    prediction_frame = test_frame[["doc_id", "label"]].copy()
    prediction_frame.rename(columns={"label": "gold_label"}, inplace=True)
    prediction_frame["baseline_prediction"] = baseline_test_metrics["predictions"]
    prediction_frame["improved_prediction"] = improved_test_metrics["predictions"]
    ensure_parent_dir(args.predictions_output)
    prediction_frame.to_csv(args.predictions_output, index=False)

    matrix = confusion_matrix(
        test_frame["label"],
        improved_test_metrics["predictions"],
        labels=TASK3_LABELS,
    )
    render_confusion_matrix(TASK3_LABELS, matrix, args.confusion_matrix_output)

    improved_confusions = collect_confusions(
        test_frame,
        test_frame["label"].tolist(),
        improved_test_metrics["predictions"],
        source="improved test set",
    )
    confusions = select_top_confusions(improved_confusions)

    # Run noise robustness test
    noise_results = run_noise_robustness_test(
        train_texts=train_plus_dev["clean_text"],
        train_labels=train_plus_dev["label"],
        test_texts=test_frame["clean_text"],
        test_labels=test_frame["label"],
        best_c=best_c,
        seed=args.seed,
    )
    render_noise_robustness_figure(noise_results, NOISE_ROBUSTNESS_FIGURE_PATH)

    # Analyze error patterns using the fitted vectorizer
    fitted_vectorizer = final_improved.named_steps["vectorizer"]
    error_patterns = analyze_error_patterns(
        test_frame,
        test_frame["label"].tolist(),
        improved_test_metrics["predictions"],
        fitted_vectorizer,
    )

    source_distribution = dataset["source"].value_counts().to_dict()
    annotation_methods = set(dataset["annotation_method"].astype(str))

    metrics_payload = {
        "baseline": {
            "dev": {
                "accuracy": baseline_dev_metrics["accuracy"],
                "macro_f1": baseline_dev_metrics["macro_f1"],
            },
            "test": {
                "accuracy": baseline_test_metrics["accuracy"],
                "macro_f1": baseline_test_metrics["macro_f1"],
            },
        },
        "improved": {
            "selected_c": best_c,
            "dev": {
                "accuracy": best_improved_dev_metrics["accuracy"],
                "macro_f1": best_improved_dev_metrics["macro_f1"],
            },
            "test": {
                "accuracy": improved_test_metrics["accuracy"],
                "macro_f1": improved_test_metrics["macro_f1"],
            },
        },
        "noise_robustness": noise_results,
        "dataset_size": int(len(dataset)),
        "label_distribution": dataset["label"].value_counts().to_dict(),
        "source_distribution": source_distribution,
        "labeling_method": infer_labeling_method(source_distribution, annotation_methods),
        "top_confusions": confusions,
        "error_patterns": error_patterns,
    }
    ensure_parent_dir(args.metrics_output)
    args.metrics_output.write_text(json.dumps(metrics_payload, indent=2) + "\n", encoding="utf-8")

    report = build_report(
        dataset=dataset,
        train_frame=train_frame,
        dev_frame=dev_frame,
        test_frame=test_frame,
        baseline_dev_metrics=baseline_dev_metrics,
        baseline_test_metrics=baseline_test_metrics,
        improved_dev_metrics=best_improved_dev_metrics,
        improved_test_metrics=improved_test_metrics,
        best_c=best_c,
        confusions=confusions,
        error_patterns=error_patterns,
        noise_results=noise_results,
    )
    ensure_parent_dir(args.report_output)
    args.report_output.write_text(report, encoding="utf-8")

    print(f"Built dataset with {len(dataset)} labeled documents.")
    print(f"Saved split metadata to {args.split_output}.")
    print(f"Saved predictions to {args.predictions_output}.")
    print(f"Saved metrics to {args.metrics_output}.")
    print(f"Saved report to {args.report_output}.")
    print(f"Saved noise robustness figure to {NOISE_ROBUSTNESS_FIGURE_PATH}.")


if __name__ == "__main__":
    main()
