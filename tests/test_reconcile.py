"""Position reconciliation is a hard mismatch detector."""

from cq.monitor.reconcile import positions_mismatch


def test_reconciliation_detects_and_ignores_within_tolerance() -> None:
    intended = {"ETHUSDT": 1.0, "BTCUSDT": -0.5}
    assert positions_mismatch(intended, {"ETHUSDT": 1.0, "BTCUSDT": -0.5}) is False
    assert positions_mismatch(intended, {"ETHUSDT": 1.0}) is True
    assert (
        positions_mismatch(intended, {"ETHUSDT": 1.0 + 1e-12, "BTCUSDT": -0.5})
        is False
    )
