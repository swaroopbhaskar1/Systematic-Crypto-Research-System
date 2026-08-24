"""Call an LLM to emit schema-valid hypotheses. Invalid rows are dropped."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Protocol

from cq.research.log import content_hash
from cq.research.schema import Hypothesis

MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "generator.txt"


class HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> object: ...


def generate_hypotheses(
    *,
    client: HttpClient,
    calls: int,
    batch_size: int,
    api_key: str | None = None,
    dropped_log: Path | None = None,
) -> tuple[Hypothesis, ...]:
    """Run independent batches and keep unique schema-valid hypotheses."""
    if calls < 1 or batch_size < 1:
        raise ValueError("calls and batch_size must be positive")
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is required")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    kept: list[Hypothesis] = []
    seen: set[str] = set()
    for _ in range(calls):
        raw = _complete(client, key, prompt, batch_size)
        for item in _parse_array(raw):
            hypothesis = _validated_or_logged(item, dropped_log)
            if hypothesis is None:
                continue
            digest = content_hash(hypothesis)
            if digest in seen:
                continue
            seen.add(digest)
            kept.append(hypothesis)
    return tuple(kept)


def _complete(client: HttpClient, api_key: str, prompt: str, batch_size: int) -> str:
    response = client.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 4096,
            "temperature": 1.0,
            "system": prompt,
            "messages": [
                {
                    "role": "user",
                    "content": f"Return exactly {batch_size} hypotheses as a JSON array.",
                }
            ],
        },
    )
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    payload = _as_mapping(getattr(response, "json")())
    return _text_content(payload)


def _parse_array(raw: str) -> list[object]:
    parsed = json.loads(raw)
    if not isinstance(parsed, list):
        raise ValueError("generator output must be a JSON array")
    return parsed


def _validated_or_logged(
    item: object,
    dropped_log: Path | None,
) -> Hypothesis | None:
    try:
        return Hypothesis.model_validate(item)
    except Exception as error:
        if dropped_log is not None:
            dropped_log.parent.mkdir(parents=True, exist_ok=True)
            record = {"error": str(error), "payload": item}
            with dropped_log.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, default=str) + "\n")
        return None


def _text_content(payload: dict[str, object]) -> str:
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("LLM response is missing content")
    first = content[0]
    if not isinstance(first, dict):
        raise ValueError("LLM content is malformed")
    text = first.get("text")
    if not isinstance(text, str) or not text:
        raise ValueError("LLM content is missing text")
    return text


def _as_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError("LLM response must be an object")
    return value
