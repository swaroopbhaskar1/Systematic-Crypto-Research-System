"""One-shot holdout access control."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path

UNLOCK_ENV = "CQ_HOLDOUT_UNLOCK"
SHA_ENV = "CQ_GIT_SHA"
DEFAULT_ROOT = Path("data")
DEFAULT_ACCESS_LOG = Path("research") / "log" / "holdout_access.jsonl"


class HoldoutLockedError(Exception):
    """Raised when holdout data is requested without a matching unlock."""


class HoldoutBurnedError(Exception):
    """Raised when the same hypothesis tries to load holdout twice."""


def load_holdout(
    hypothesis_id: str,
    *,
    root: Path | None = None,
    access_log: Path | None = None,
) -> Path:
    """Load holdout files once per hypothesis after an explicit unlock."""
    _require_unlock(hypothesis_id)
    log_path = access_log or DEFAULT_ACCESS_LOG
    _reject_burned(hypothesis_id, log_path)
    holdout = _holdout_dir(root or DEFAULT_ROOT)
    _append_access(hypothesis_id, log_path)
    return holdout


def _require_unlock(hypothesis_id: str) -> None:
    if not hypothesis_id:
        raise ValueError("hypothesis_id is required")
    if os.environ.get(UNLOCK_ENV) != hypothesis_id:
        raise HoldoutLockedError("CQ_HOLDOUT_UNLOCK must match hypothesis_id")


def _holdout_dir(root: Path) -> Path:
    holdout = root / "HOLDOUT"
    if not holdout.exists():
        raise FileNotFoundError(f"missing holdout directory: {holdout}")
    return holdout


def _reject_burned(hypothesis_id: str, log_path: Path) -> None:
    if not log_path.exists():
        return
    for raw in log_path.read_text(encoding="utf-8").splitlines():
        if not raw:
            continue
        record = json.loads(raw)
        if record.get("hypothesis_id") == hypothesis_id:
            raise HoldoutBurnedError(
                f"holdout already accessed for {hypothesis_id}"
            )


def _append_access(hypothesis_id: str, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hypothesis_id": hypothesis_id,
        "git_sha": _git_sha(),
    }
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def _git_sha() -> str:
    override = os.environ.get(SHA_ENV)
    if override:
        return override
    head = Path(".git") / "HEAD"
    if not head.exists():
        raise FileNotFoundError("git SHA is unavailable")
    content = head.read_text(encoding="utf-8").strip()
    if content.startswith("ref:"):
        ref = Path(".git") / content.split(":", 1)[1].strip()
        return ref.read_text(encoding="utf-8").strip()
    return content
