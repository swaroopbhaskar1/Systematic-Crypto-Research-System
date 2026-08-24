from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from cq.data.ingest import (
    DataAvailability,
    raw_cache_key,
    validate_bar_gaps,
)


def test_raw_cache_key_scopes_every_request_dimension() -> None:
    start = datetime(2024, 1, 1, tzinfo=UTC)
    end = datetime(2024, 1, 2, tzinfo=UTC)
    base = raw_cache_key("binance", "spot", "BTCUSDT", "1h", start, end)

    assert isinstance(base, Path)
    assert base != raw_cache_key("binance", "perp", "BTCUSDT", "1h", start, end)
    assert base != raw_cache_key("binance", "spot", "ETHUSDT", "1h", start, end)
    assert "2024-01-01" in str(base)
    assert "2024-01-02" in str(base)


def test_gap_greater_than_one_bar_fails_loudly() -> None:
    bars = pd.DataFrame(
        {
            "ts": [0, 3_600_000, 10_800_000],
            "open": [1.0, 1.0, 1.0],
            "high": [1.0, 1.0, 1.0],
            "low": [1.0, 1.0, 1.0],
            "close": [1.0, 1.0, 1.0],
            "volume": [1.0, 1.0, 1.0],
            "quote_volume": [1.0, 1.0, 1.0],
        }
    )

    with pytest.raises(ValueError, match="gap"):
        validate_bar_gaps(bars, timeframe="1h")


def test_open_interest_is_explicitly_not_required_historical_data() -> None:
    availability = DataAvailability.binance_open_interest()

    assert not availability.available_for_deep_history
    assert availability.maximum_history_days == 30
    assert "open interest" in availability.reason.lower()
    assert "vision metrics" in availability.reason.lower()
    assert "unavailable" in availability.reason.lower()
