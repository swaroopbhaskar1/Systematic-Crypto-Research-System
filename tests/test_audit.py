"""Auditor flags lookahead and mid-price fill patterns in code diffs."""

from cq.ai.audit import audit_diff


CLEAN_DIFF = """
+def fill_price(side, bar, spread_bps):
+    half = bar.open * spread_bps / 20_000
+    return bar.open + half if side is Side.BUY else bar.open - half
"""


def test_clean_diff_has_no_findings() -> None:
    report = audit_diff(CLEAN_DIFF)
    assert report.findings == ()
    assert report.advisory is True


def test_auditor_flags_midprice_negative_shift_centered_roll_and_fillna() -> None:
    diff = """
+price = (bid + ask) / 2
+future = close.shift(-1)
+rolled = close.rolling(5, center=True).mean()
+lagged = lag(close, -3)
+partial = close.rolling(10, min_periods=1).std()
+filled = close.fillna(0)
"""
    report = audit_diff(diff)
    reasons = {finding.rule for finding in report.findings}
    assert "mid_price_fill" in reasons
    assert "negative_shift" in reasons
    assert "centered_rolling" in reasons
    assert "negative_lag" in reasons
    assert "short_min_periods" in reasons
    assert "price_nan_fill" in reasons
    assert report.advisory is True
