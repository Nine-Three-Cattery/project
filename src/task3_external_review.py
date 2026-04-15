"""Run an external LLM audit over the released Task 3 labels without overwriting them."""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request

import yaml

from src.utils.constants import (
    CLEANED_JSONL_PATH,
    LABEL_DEFINITIONS_PATH,
    LABELS_PATH,
    TASK3_EXTERNAL_REVIEWS_DIR,
    TASK3_LABELS,
)
from src.utils.corpus import ensure_parent_dir, load_cleaned_corpus_jsonl


DEFAULT_MODELS = {
    "vertex": "gemini-2.5-flash",
    "xai": "grok-4.20-reasoning",
}

DEFAULT_API_ENV = {
    "vertex": "VERTEX_AI_API_KEY",
    "xai": "XAI_API_KEY",
}

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {
            "type": "string",
            "enum": TASK3_LABELS,
            "description": "One dominant topic label from the fixed Task 3 label set.",
        },
        "confidence": {
            "type": "number",
            "description": "Model confidence between 0.0 and 1.0.",
        },
        "justification": {
            "type": "string",
            "description": "Short explanation anchored in the article's dominant topic.",
        },
    },
    "required": ["label", "confidence", "justification"],
    "additionalProperties": False,
}


@dataclass(slots=True)
class ReviewItem:
    """The data needed to send one document to an external audit model."""

    doc_id: str
    source_label: str
    raw_text: str


def build_argument_parser() -> argparse.ArgumentParser:
    """Create the CLI for provider-specific Task 3 external review."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=sorted(DEFAULT_MODELS), required=True, help="External model provider.")
    parser.add_argument("--corpus", type=Path, default=CLEANED_JSONL_PATH, help="Path to the cleaned corpus JSONL.")
    parser.add_argument("--labels", type=Path, default=LABELS_PATH, help="Path to the released Task 3 label JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Provider-specific JSONL output path. Defaults to outputs/task3_external_reviews/<provider>.jsonl.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Optional provider model override. Defaults to a stable fast audit model per provider.",
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default=None,
        help="Environment variable that stores the API key. Defaults to a provider-specific variable.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of documents to review. Useful for smoke tests.",
    )
    parser.add_argument(
        "--doc-ids",
        type=str,
        default="",
        help="Optional comma-separated subset of doc IDs to review.",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=8000,
        help="Maximum article characters to send to the external model for each review.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=180,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for transient provider failures.",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=0.5,
        help="Pause between requests to avoid hammering provider APIs.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file and skip documents with successful prior reviews.",
    )
    return parser


def load_review_items(corpus_path: Path, labels_path: Path) -> list[ReviewItem]:
    """Join released labels with the raw corpus text for audit prompting."""

    documents = {document.doc_id: document for document in load_cleaned_corpus_jsonl(corpus_path)}

    items: list[ReviewItem] = []
    with labels_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue

            row = json.loads(line)
            document = documents.get(row["doc_id"])
            if document is None:
                raise ValueError(f"Label row references unknown doc_id {row['doc_id']}.")

            items.append(
                ReviewItem(
                    doc_id=row["doc_id"],
                    source_label=row["label"],
                    raw_text=document.raw_text,
                )
            )
    return items


def load_label_guidance(path: Path) -> str:
    """Render the fixed label definitions into a compact prompt block."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    labels = payload.get("labels", [])
    guidance_lines = []
    for label in labels:
        guidance_lines.append(f"- {label['name']}: {label['definition']}")
    return "\n".join(guidance_lines)


def build_prompts(item: ReviewItem, label_guidance: str, max_chars: int) -> tuple[str, str]:
    """Create the provider-agnostic instruction pair for one audit request."""

    article_text = item.raw_text.strip()
    if len(article_text) > max_chars:
        article_text = article_text[:max_chars].rstrip() + "\n\n[TRUNCATED AFTER MAX-CHAR LIMIT]"

    system_prompt = (
        "You are validating a six-class news-topic dataset. "
        "Choose exactly one dominant label for the article. "
        "Do not hedge with multiple labels. "
        "Return only JSON that matches the requested schema."
    )
    user_prompt = f"""Assign one label to the news article below.

Valid labels and definitions:
{label_guidance}

Rules:
- Base the decision on the article's dominant topic, not a minor side angle.
- If the story is too mixed or does not fit the five main themes, use Other.
- Ignore any existing dataset label. This is an independent second-pass review.
- Return only a JSON object with fields: label, confidence, justification.

Article ID: {item.doc_id}
Article text:
\"\"\"
{article_text}
\"\"\"
"""
    return system_prompt, user_prompt


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse the first JSON object found in a provider response."""

    text = text.strip()
    if not text:
        raise ValueError("Provider returned an empty response.")

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Provider response is not valid JSON: {text[:240]!r}") from None
        return json.loads(text[start : end + 1])


def normalize_review_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize the provider response into a stable local schema."""

    label = str(payload.get("label", "")).strip()
    if label not in TASK3_LABELS:
        raise ValueError(f"Provider returned an unexpected label: {label!r}")

    confidence = payload.get("confidence")
    if isinstance(confidence, str):
        normalized_confidence = confidence.strip().lower()
        confidence_mapping = {
            "very high": 0.95,
            "high": 0.85,
            "medium": 0.60,
            "moderate": 0.60,
            "low": 0.35,
            "very low": 0.15,
        }
        if normalized_confidence in confidence_mapping:
            confidence = confidence_mapping[normalized_confidence]
        else:
            try:
                confidence = float(normalized_confidence)
            except ValueError as exc:
                raise ValueError(f"Provider returned a non-numeric confidence: {confidence!r}") from exc
    elif not isinstance(confidence, (int, float)):
        raise ValueError(f"Provider returned a non-numeric confidence: {confidence!r}")

    confidence = max(0.0, min(1.0, float(confidence)))

    justification = str(payload.get("justification", "")).strip()
    if not justification:
        raise ValueError("Provider returned an empty justification.")

    return {
        "reviewed_label": label,
        "confidence": round(confidence, 6),
        "justification": justification,
    }


def build_vertex_request(model: str, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    """Build the Vertex AI REST endpoint and payload for one review."""

    endpoint = f"https://aiplatform.googleapis.com/v1/publishers/google/models/{model}:generateContent"
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "text": f"{system_prompt}\n\n{user_prompt}",
                    }
                ],
            }
        ],
        "generationConfig": {
            "temperature": 0.0,
            "responseMimeType": "application/json",
        },
    }
    return endpoint, payload


def build_xai_request(model: str, system_prompt: str, user_prompt: str) -> tuple[str, dict[str, Any]]:
    """Build the xAI Responses API endpoint and payload for one review."""

    endpoint = "https://api.x.ai/v1/responses"
    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "task3_annotation_review",
                "schema": RESPONSE_SCHEMA,
                "strict": True,
            }
        },
    }
    return endpoint, payload


def extract_vertex_text(response_payload: dict[str, Any]) -> str:
    """Extract the model text from a Vertex AI generateContent response."""

    candidates = response_payload.get("candidates", [])
    if not candidates:
        raise ValueError(f"Vertex AI returned no candidates: {response_payload}")

    parts = candidates[0].get("content", {}).get("parts", [])
    for part in parts:
        text = part.get("text")
        if text:
            return text
    raise ValueError(f"Vertex AI returned no text part: {response_payload}")


def extract_xai_text(response_payload: dict[str, Any]) -> str:
    """Extract the model text from an xAI Responses API response."""

    if isinstance(response_payload.get("output_text"), str) and response_payload["output_text"].strip():
        return response_payload["output_text"]

    for item in response_payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return content["text"]
    raise ValueError(f"xAI returned no output text: {response_payload}")


def perform_request(
    *,
    provider: str,
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
) -> tuple[dict[str, Any], str | None]:
    """Execute one provider request and return the normalized payload plus provider response ID."""

    if provider == "vertex":
        endpoint, payload = build_vertex_request(model, system_prompt, user_prompt)
        endpoint = f"{endpoint}?key={api_key}"
        headers = {"Content-Type": "application/json"}
        extractor = extract_vertex_text
    elif provider == "xai":
        endpoint, payload = build_xai_request(model, system_prompt, user_prompt)
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        extractor = extract_xai_text
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    http_request = request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )

    with request.urlopen(http_request, timeout=timeout) as response:
        response_payload = json.loads(response.read().decode("utf-8"))

    text = extractor(response_payload)
    normalized = normalize_review_payload(parse_json_object(text))
    return normalized, response_payload.get("id")


def load_completed_doc_ids(path: Path) -> set[str]:
    """Load successfully reviewed document IDs from an existing provider output."""

    if not path.exists():
        return set()

    completed: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("review_status") == "ok":
                completed.add(str(row["doc_id"]))
    return completed


def append_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    """Append one JSON row without rewriting the full file."""

    ensure_parent_dir(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False))
        handle.write("\n")


def main() -> None:
    """Run one external second-pass review over the released Task 3 labels."""

    args = build_argument_parser().parse_args()
    output_path = args.output or (TASK3_EXTERNAL_REVIEWS_DIR / f"{args.provider}.jsonl")
    model = args.model or DEFAULT_MODELS[args.provider]
    api_key_env = args.api_key_env or DEFAULT_API_ENV[args.provider]
    api_key = os.getenv(api_key_env, "").strip()
    if not api_key:
        raise EnvironmentError(
            f"Missing API key. Set the environment variable {api_key_env} before running this script."
        )

    label_guidance = load_label_guidance(LABEL_DEFINITIONS_PATH)
    items = load_review_items(args.corpus, args.labels)

    requested_doc_ids = {doc_id.strip() for doc_id in args.doc_ids.split(",") if doc_id.strip()}
    if requested_doc_ids:
        items = [item for item in items if item.doc_id in requested_doc_ids]
    if args.limit is not None:
        items = items[: args.limit]

    if not items:
        raise ValueError("No documents matched the requested audit subset.")

    completed_doc_ids = load_completed_doc_ids(output_path) if args.resume else set()

    reviewed_count = 0
    error_count = 0
    skipped_count = 0
    for item in items:
        if item.doc_id in completed_doc_ids:
            skipped_count += 1
            continue

        system_prompt, user_prompt = build_prompts(item, label_guidance, args.max_chars)

        last_error: Exception | None = None
        for attempt in range(1, args.max_retries + 1):
            try:
                normalized, response_id = perform_request(
                    provider=args.provider,
                    api_key=api_key,
                    model=model,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    timeout=args.timeout,
                )
                row = {
                    "doc_id": item.doc_id,
                    "provider": args.provider,
                    "model": model,
                    "source_label": item.source_label,
                    "reviewed_label": normalized["reviewed_label"],
                    "confidence": normalized["confidence"],
                    "justification": normalized["justification"],
                    "agrees_with_source": normalized["reviewed_label"] == item.source_label,
                    "review_status": "ok",
                    "provider_response_id": response_id,
                    "reviewed_at": datetime.now(timezone.utc).isoformat(),
                }
                append_jsonl_row(output_path, row)
                reviewed_count += 1
                last_error = None
                break
            except Exception as exc:
                last_error = exc
                if attempt == args.max_retries:
                    break
                time.sleep(min(10.0, args.sleep_seconds * (2**attempt)))

        if last_error is not None:
            error_row = {
                "doc_id": item.doc_id,
                "provider": args.provider,
                "model": model,
                "source_label": item.source_label,
                "review_status": "error",
                "error": str(last_error),
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            }
            append_jsonl_row(output_path, error_row)
            error_count += 1
            continue

        time.sleep(args.sleep_seconds)

    if reviewed_count == 0 and skipped_count == 0:
        raise RuntimeError("No successful or reusable reviews were available for this run.")

    print(f"Provider: {args.provider}")
    print(f"Model: {model}")
    print(f"Output path: {output_path}")
    print(f"Completed reviews written: {reviewed_count}")
    print(f"Skipped existing successful reviews: {skipped_count}")
    print(f"Error rows written: {error_count}")


if __name__ == "__main__":
    main()
