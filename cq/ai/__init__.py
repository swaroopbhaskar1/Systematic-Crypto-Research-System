"""LLM-backed hypothesis generation and critique. No agent framework."""

from cq.ai.audit import audit_diff
from cq.ai.critique import Verdict, critique_hypothesis
from cq.ai.generate import generate_hypotheses

__all__ = [
    "Verdict",
    "audit_diff",
    "critique_hypothesis",
    "generate_hypotheses",
]
