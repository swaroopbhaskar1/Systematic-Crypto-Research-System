"""Advisory static checks for backtester and grammar diffs."""

from __future__ import annotations

import re
from dataclasses import dataclass

_ADDED = re.compile(r"^\+")
_FILE_HEADER = re.compile(r"^\+\+\+")


@dataclass(frozen=True)
class Finding:
    """One advisory match in a diff."""

    rule: str
    line: str


@dataclass(frozen=True)
class AuditReport:
    """Auditor output. Advisory only; tests remain authoritative."""

    findings: tuple[Finding, ...]
    advisory: bool = True


_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("mid_price_fill", re.compile(r"(bid\s*\+\s*ask|\(\s*bid\s*\+\s*ask\s*\)\s*/\s*2|mid[\s_]*price)", re.I)),
    ("negative_shift", re.compile(r"\.shift\(\s*-")),
    ("centered_rolling", re.compile(r"center\s*=\s*True")),
    ("negative_lag", re.compile(r"\blag\s*\([^)]*-\s*\d")),
    ("short_min_periods", re.compile(r"min_periods\s*=\s*[0-9]+")),
    ("price_nan_fill", re.compile(r"(fillna|ffill|bfill|interpolate)\(")),
)


def audit_diff(diff: str) -> AuditReport:
    """Scan added diff lines for honesty-rule violations."""
    findings: list[Finding] = []
    for raw in diff.splitlines():
        if not _ADDED.match(raw) or _FILE_HEADER.match(raw):
            continue
        line = raw[1:]
        for rule, pattern in _RULES:
            if not pattern.search(line):
                continue
            if rule == "short_min_periods" and not _short_window(line):
                continue
            findings.append(Finding(rule=rule, line=line.strip()))
    return AuditReport(tuple(findings))


def _short_window(line: str) -> bool:
    window = re.search(r"rolling\s*\(\s*(\d+)", line)
    periods = re.search(r"min_periods\s*=\s*(\d+)", line)
    if periods is None:
        return False
    if window is None:
        return True
    return int(periods.group(1)) < int(window.group(1))
