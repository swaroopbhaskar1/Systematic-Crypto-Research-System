"""Binary critic with a closed kill-reason enum and no scores."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cq.research.schema import Hypothesis

KillReason = Literal[
    "no_forced_participant",
    "logically_incoherent",
    "lookahead_in_rule",
    "data_unavailable",
]
VerdictLabel = Literal["PASS", "KILL"]
MODEL = "claude-sonnet-4-6"
ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "critic.txt"


class HttpClient(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, object],
    ) -> object: ...


class Verdict(BaseModel):
    """PASS/KILL only. A score field is forbidden because it invites ranking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    verdict: VerdictLabel
    kill_reason: KillReason | None
    flags: list[str] = Field(default_factory=list)


def critique_hypothesis(
    hypothesis: Hypothesis,
    *,
    client: HttpClient,
    api_key: str | None = None,
) -> Verdict:
    """Critique one hypothesis from its fields alone."""
    key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is required")
    prompt = PROMPT_PATH.read_text(encoding="utf-8")
    response = client.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={
            "model": MODEL,
            "max_tokens": 1024,
            "temperature": 0.0,
            "system": prompt,
            "messages": [
                {
                    "role": "user",
                    "content": json.dumps(
                        hypothesis.model_dump(mode="json"),
                        sort_keys=True,
                    ),
                }
            ],
        },
    )
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()
    payload = getattr(response, "json")()
    if not isinstance(payload, dict):
        raise ValueError("critic response must be an object")
    content = payload.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("critic response is missing content")
    first = content[0]
    if not isinstance(first, dict) or not isinstance(first.get("text"), str):
        raise ValueError("critic content is malformed")
    verdict = Verdict.model_validate_json(first["text"])
    if verdict.id != hypothesis.id:
        raise ValueError("critic verdict id does not match the hypothesis")
    if verdict.verdict == "KILL" and verdict.kill_reason is None:
        raise ValueError("KILL verdicts require a kill_reason")
    if verdict.verdict == "PASS" and verdict.kill_reason is not None:
        raise ValueError("PASS verdicts cannot carry a kill_reason")
    return verdict
