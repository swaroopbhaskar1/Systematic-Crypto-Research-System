"""Holdout data must be one-shot and physically awkward to touch."""

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from cq.research.holdout import (
    HoldoutBurnedError,
    HoldoutLockedError,
    load_holdout,
)


def _write_holdout_file(root: Path) -> Path:
    holdout = root / "HOLDOUT"
    holdout.mkdir()
    payload = holdout / "panel.parquet"
    frame = pd.DataFrame(
        {
            "ts": [1_720_000_000_000],
            "symbol": ["HOLDUSDT"],
            "close": [1.0],
        }
    )
    frame.to_parquet(payload, index=False)
    payload.chmod(0o400)
    return root


def test_holdout_load_requires_matching_unlock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_holdout_file(tmp_path)
    monkeypatch.delenv("CQ_HOLDOUT_UNLOCK", raising=False)
    with pytest.raises(HoldoutLockedError):
        load_holdout("funding_extreme_001", root=root)

    monkeypatch.setenv("CQ_HOLDOUT_UNLOCK", "other_id")
    with pytest.raises(HoldoutLockedError):
        load_holdout("funding_extreme_001", root=root)


def test_holdout_access_is_logged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_holdout_file(tmp_path)
    log_path = tmp_path / "holdout_access.jsonl"
    monkeypatch.setenv("CQ_HOLDOUT_UNLOCK", "funding_extreme_001")
    monkeypatch.setenv("CQ_GIT_SHA", "abc123def")
    loaded = load_holdout(
        "funding_extreme_001",
        root=root,
        access_log=log_path,
    )
    assert loaded.exists()
    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["hypothesis_id"] == "funding_extreme_001"
    assert record["git_sha"] == "abc123def"
    assert "timestamp" in record


def test_second_holdout_access_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _write_holdout_file(tmp_path)
    log_path = tmp_path / "holdout_access.jsonl"
    monkeypatch.setenv("CQ_HOLDOUT_UNLOCK", "funding_extreme_001")
    monkeypatch.setenv("CQ_GIT_SHA", "deadbeef")
    load_holdout("funding_extreme_001", root=root, access_log=log_path)
    with pytest.raises(HoldoutBurnedError):
        load_holdout("funding_extreme_001", root=root, access_log=log_path)
    assert len(log_path.read_text(encoding="utf-8").splitlines()) == 1


def test_holdout_files_are_owner_read_only(tmp_path: Path) -> None:
    payload = _write_holdout_file(tmp_path) / "HOLDOUT" / "panel.parquet"
    assert stat_mode(payload) == 0o400


def stat_mode(path: Path) -> int:
    return path.stat().st_mode & 0o777


def test_walkforward_and_dev_loaders_cannot_see_holdout_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from cq.research.splits import assert_research_timestamps

    root = _write_holdout_file(tmp_path)
    monkeypatch.chdir(root)
    monkeypatch.delenv("CQ_HOLDOUT_UNLOCK", raising=False)
    holdout_ts = pd.Timestamp("2025-07-01", tz="UTC")
    with pytest.raises(HoldoutLockedError):
        assert_research_timestamps((holdout_ts,))
    assert os.getenv("CQ_HOLDOUT_UNLOCK") is None
