"""Research counting, reporting, and holdout controls."""

from cq.research.holdout import HoldoutBurnedError, HoldoutLockedError, load_holdout
from cq.research.log import DuplicateHypothesisError, content_hash, record, test_count
from cq.research.report import ResearchReport, adjusted_sharpe_threshold, render_report
from cq.research.splits import DEV_END, WALKFWD_END

__all__ = [
    "DEV_END",
    "WALKFWD_END",
    "DuplicateHypothesisError",
    "HoldoutBurnedError",
    "HoldoutLockedError",
    "ResearchReport",
    "adjusted_sharpe_threshold",
    "content_hash",
    "load_holdout",
    "record",
    "render_report",
    "test_count",
]
