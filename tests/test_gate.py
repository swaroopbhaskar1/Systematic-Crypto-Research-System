import csv
import io
import json
import zipfile
from datetime import date
from pathlib import Path
from typing import Self

import pytest

from cq.data import gate
from cq.data.gate import (
    ArchiveGateResult,
    ArchiveInventory,
    classify_archive_inventory,
    collect_real_archive_inventory,
    evaluate_archive_gate,
)


def funding_zip(timestamps: list[int]) -> bytes:
    data = io.StringIO()
    writer = csv.writer(data, lineterminator="\n")
    writer.writerow(["calc_time", "funding_interval_hours", "last_funding_rate"])
    writer.writerows((timestamp, "8", "0.001") for timestamp in timestamps)
    payload = io.BytesIO()
    with zipfile.ZipFile(payload, "w") as archive:
        archive.writestr("funding.csv", data.getvalue())
    return payload.getvalue()


class FakeArchiveClient:
    def symbol_prefixes(
        self,
        market_type: str,
        data_type: str,
        *,
        period: str = "monthly",
    ) -> tuple[str, ...]:
        if market_type == "spot":
            return (
                "data/spot/monthly/klines/OLDUSDT/",
                "data/spot/monthly/klines/LIVEUSDT/",
                "data/spot/monthly/klines/IGNOREDUSDC/",
            )
        if data_type == "funding":
            return ("data/futures/um/monthly/fundingRate/BTCUSDT/",)
        return ("data/futures/um/monthly/klines/BTCUSDT/",)

    def list_objects(self, prefix: str) -> tuple[str, ...]:
        if "OLDUSDT" in prefix:
            return ("data/spot/monthly/klines/OLDUSDT/1d/OLDUSDT-1d-2024-01.zip",)
        if "LIVEUSDT" in prefix:
            return ("data/spot/daily/klines/LIVEUSDT/1d/LIVEUSDT-1d-2024-02-29.zip",)
        return (
            (
                "data/futures/um/monthly/fundingRate/BTCUSDT/"
                "BTCUSDT-fundingRate-2024-01.zip"
            ),
            "data/futures/um/monthly/fundingRate/BTCUSDT/checksum.txt",
        )

    def download(self, _key: str) -> bytes:
        interval = 28_800_000
        return funding_zip([0, interval, 2 * interval])

    def spot_trading_symbols(self) -> frozenset[str]:
        return frozenset({"LIVEUSDT"})


def test_archive_keys_are_classified_by_market_data_type() -> None:
    inventory = classify_archive_inventory(
        [
            "data/spot/monthly/klines/OLDUSDT/1d/OLDUSDT-1d-2020-01.zip",
            "data/spot/monthly/klines/LIVEUSDT/1d/LIVEUSDT-1d-2024-06.zip",
            "data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-2024-06.zip",
            (
                "data/futures/um/monthly/fundingRate/BTCUSDT/"
                "BTCUSDT-fundingRate-2024-06.zip"
            ),
        ],
        archive_latest=date(2024, 6, 30),
    )

    assert inventory.spot_symbols == frozenset({"OLDUSDT", "LIVEUSDT"})
    assert inventory.perp_symbols == frozenset({"BTCUSDT"})
    assert inventory.funding_symbols == frozenset({"BTCUSDT"})
    assert inventory.delisted_base_assets == frozenset({"OLD"})


def test_daily_funding_paths_are_not_treated_as_real_archives() -> None:
    inventory = classify_archive_inventory(
        [
            (
                "data/futures/um/daily/fundingRate/BTCUSDT/"
                "BTCUSDT-fundingRate-2024-06-01.zip"
            )
        ],
        archive_latest=date(2024, 6, 1),
    )

    assert inventory.funding_symbols == frozenset()
    assert inventory.perp_symbols == frozenset()


def test_gate_reports_real_counts_and_fails_every_short_bucket() -> None:
    inventory = ArchiveInventory(
        spot_symbols=frozenset(f"TOKEN{i}USDT" for i in range(149)),
        perp_symbols=frozenset(f"PERP{i}USDT" for i in range(99)),
        funding_symbols=frozenset(f"PERP{i}USDT" for i in range(99)),
        delisted_base_assets=frozenset(f"OLD{i}" for i in range(49)),
        funding_contiguous_counts={f"PERP{i}USDT": 270 for i in range(99)},
    )

    result = evaluate_archive_gate(inventory)

    assert isinstance(result, ArchiveGateResult)
    assert result.token_count == 149
    assert result.spot_symbol_count == 149
    assert result.perp_symbol_count == 99
    assert result.funding_symbol_count == 99
    assert result.delisted_count == 49
    assert result.funded_perp_count == 99
    assert result.open_interest_disposition == "unavailable"
    assert not result.passed
    assert set(result.failures) == {
        "token_count",
        "delisted_count",
        "funded_perp_count",
    }


def test_gate_passes_only_at_all_thresholds() -> None:
    inventory = ArchiveInventory(
        spot_symbols=frozenset(f"TOKEN{i}USDT" for i in range(150)),
        perp_symbols=frozenset(f"PERP{i}USDT" for i in range(100)),
        funding_symbols=frozenset(f"PERP{i}USDT" for i in range(100)),
        delisted_base_assets=frozenset(f"OLD{i}" for i in range(50)),
        funding_contiguous_counts={f"PERP{i}USDT": 270 for i in range(100)},
    )

    assert evaluate_archive_gate(inventory).passed


def test_gate_reports_latest_observed_archive_date() -> None:
    inventory = ArchiveInventory(
        spot_symbols=frozenset({"BTCUSDT"}),
        perp_symbols=frozenset(),
        funding_symbols=frozenset(),
        delisted_base_assets=frozenset(),
        funding_contiguous_counts={},
        last_archive_dates={"BTCUSDT": date(2024, 6, 30)},
    )

    assert evaluate_archive_gate(inventory).archive_latest_date == "2024-06-30"


def test_real_inventory_collection_filters_symbols_and_counts_funding() -> None:
    inventory = collect_real_archive_inventory(
        FakeArchiveClient(),
        workers=2,  # type: ignore[arg-type]
    )

    assert inventory.spot_symbols == frozenset({"LIVEUSDT", "OLDUSDT"})
    assert inventory.perp_symbols == frozenset({"BTCUSDT"})
    assert inventory.funding_symbols == frozenset({"BTCUSDT"})
    assert inventory.delisted_base_assets == frozenset({"OLD"})
    assert inventory.last_archive_dates["LIVEUSDT"] == date(2024, 2, 29)
    assert inventory.funding_contiguous_counts == {"BTCUSDT": 3}


def test_real_inventory_includes_funding_only_perpetual_archives() -> None:
    class FundingOnlyClient(FakeArchiveClient):
        def symbol_prefixes(
            self,
            market_type: str,
            data_type: str,
            *,
            period: str = "monthly",
        ) -> tuple[str, ...]:
            if market_type == "spot":
                return super().symbol_prefixes(market_type, data_type, period=period)
            if data_type == "funding":
                return (
                    "data/futures/um/monthly/fundingRate/BTCUSDT/",
                    "data/futures/um/monthly/fundingRate/ONLYUSDT/",
                )
            return ("data/futures/um/monthly/klines/BTCUSDT/",)

    inventory = collect_real_archive_inventory(FundingOnlyClient(), workers=2)

    assert inventory.perp_symbols == frozenset({"BTCUSDT", "ONLYUSDT"})


def test_real_inventory_combines_daily_and_monthly_object_dates() -> None:
    class DailyObjectClient(FakeArchiveClient):
        def list_objects(self, prefix: str) -> tuple[str, ...]:
            if "OLDUSDT" in prefix and "/daily/" in prefix:
                return ("data/spot/daily/klines/OLDUSDT/1d/OLDUSDT-1d-2024-02-29.zip",)
            return super().list_objects(prefix)

    inventory = collect_real_archive_inventory(DailyObjectClient(), workers=2)

    assert inventory.last_archive_dates["OLDUSDT"] == date(2024, 2, 29)


def test_real_inventory_uses_current_status_for_recent_delistings() -> None:
    class RecentlyDelistedClient(FakeArchiveClient):
        def list_objects(self, prefix: str) -> tuple[str, ...]:
            if "OLDUSDT" in prefix:
                return ("data/spot/daily/klines/OLDUSDT/1d/OLDUSDT-1d-2024-02-29.zip",)
            return super().list_objects(prefix)

    inventory = collect_real_archive_inventory(RecentlyDelistedClient(), workers=2)

    assert inventory.delisted_base_assets == frozenset({"OLD"})


def test_real_inventory_collection_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="workers"):
        collect_real_archive_inventory(
            FakeArchiveClient(),
            workers=0,  # type: ignore[arg-type]
        )

    client = FakeArchiveClient()
    client.list_objects = lambda _prefix: ()  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="no dated objects"):
        collect_real_archive_inventory(client)  # type: ignore[arg-type]


def test_gate_main_writes_passing_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    inventory = ArchiveInventory(
        spot_symbols=frozenset(f"TOKEN{i}USDT" for i in range(150)),
        perp_symbols=frozenset(f"PERP{i}USDT" for i in range(100)),
        funding_symbols=frozenset(f"PERP{i}USDT" for i in range(100)),
        delisted_base_assets=frozenset(f"OLD{i}" for i in range(50)),
        funding_contiguous_counts={f"PERP{i}USDT": 270 for i in range(100)},
    )

    class StubClient:
        def __enter__(self) -> Self:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(gate, "BinanceArchiveClient", StubClient)
    monkeypatch.setattr(
        gate,
        "collect_real_archive_inventory",
        lambda _client, *, workers: inventory,
    )
    output = tmp_path / "nested" / "gate.json"

    assert gate.main(["--workers", "3", "--output", str(output)]) == 0

    rendered = json.loads(output.read_text(encoding="utf-8"))
    assert rendered["passed"] is True
    assert rendered["failures"] == []
    assert json.loads(capsys.readouterr().out)["funded_perp_count"] == 100
