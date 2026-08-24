"""Paper/live branching is forbidden outside the execution adapter."""

from pathlib import Path

PRODUCTION_ROOT = Path(__file__).resolve().parents[1] / "cq"


def test_if_paper_exists_only_inside_execution() -> None:
    offenders: list[str] = []
    for path in PRODUCTION_ROOT.rglob("*.py"):
        if "execution" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if "if paper" in text or "if is_paper" in text:
            offenders.append(str(path.relative_to(PRODUCTION_ROOT)))
    assert offenders == []
