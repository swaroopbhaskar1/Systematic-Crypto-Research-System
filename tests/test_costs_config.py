"""Adversarial tests for the declarative cost configuration contract.

``config/costs.yaml`` is the single place a reader can audit what the backtest
charged.  These tests exist to make a silently-wrong config impossible: a flat
spread, a quietly-ignored slippage model, or a tax rate leaking into the trade
loop must all fail loudly rather than produce a prettier equity curve.
"""

from dataclasses import FrozenInstanceError, fields
from pathlib import Path

import pytest

from cq.backtest.costs import CostConfig, CostModel

COSTS_CONFIG = Path(__file__).resolve().parents[1] / "config" / "costs.yaml"

SPEC_DECILE_SPREADS = (200.0, 120.0, 80.0, 55.0, 40.0, 30.0, 22.0, 16.0, 11.0, 8.0)

_DEFAULT_BLOCKS: dict[str, str] = {
    "costs": (
        "costs:\n"
        "  spot:\n"
        "    maker_bps: 10\n"
        "    taker_bps: 10\n"
        "  perp:\n"
        "    maker_bps: 2\n"
        "    taker_bps: 5"
    ),
    "spread_bps": (
        "spread_bps:\n"
        "  default: 30\n"
        "  by_liquidity_decile: [200, 120, 80, 55, 40, 30, 22, 16, 11, 8]"
    ),
    "taker_fee_bps": "taker_fee_bps: 10",
    "slippage": 'slippage:\n  model: "square_root"\n  coefficient: 0.6',
    "tax": "tax:\n  short_term_rate: 0.35",
}


def _config_text(**overrides: str | None) -> str:
    """Render a full config, replacing (``str``) or dropping (``None``) blocks."""
    blocks = dict(_DEFAULT_BLOCKS)
    for name, replacement in overrides.items():
        if replacement is None:
            del blocks[name]
        else:
            blocks[name] = replacement
    return "\n".join(blocks.values()) + "\n"


def _write(tmp_path: Path, text: str) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "costs.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def _load(tmp_path: Path, **overrides: str | None) -> CostConfig:
    return CostConfig.from_yaml(_write(tmp_path, _config_text(**overrides)))


# --------------------------------------------------------------------------
# The real file on disk must match the spec literally.
# --------------------------------------------------------------------------


def test_repo_config_parses_and_matches_spec_literals() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    assert config.default_spread_bps == pytest.approx(30.0)
    assert config.taker_fee_bps == pytest.approx(10.0)
    assert config.slippage_model == "square_root"
    assert config.impact_coefficient == pytest.approx(0.6)
    assert config.short_term_rate == pytest.approx(0.35)


def test_repo_config_decile_spreads_are_the_spec_list_keyed_one_through_ten() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    assert set(config.spread_bps_by_decile) == set(range(1, 11))
    assert [config.spread_bps_by_decile[decile] for decile in range(1, 11)] == list(
        SPEC_DECILE_SPREADS
    )
    # Decile 1 is the least liquid and must carry the widest spread.
    assert config.spread_bps_by_decile[1] == pytest.approx(200.0)
    assert config.spread_bps_by_decile[10] == pytest.approx(8.0)


def test_repo_config_keeps_per_market_maker_and_taker_fees() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    assert config.market_fees["spot"]["maker_bps"] == pytest.approx(10.0)
    assert config.market_fees["spot"]["taker_bps"] == pytest.approx(10.0)
    assert config.market_fees["perp"]["maker_bps"] == pytest.approx(2.0)
    assert config.market_fees["perp"]["taker_bps"] == pytest.approx(5.0)


def test_repo_config_spread_is_not_flat_across_deciles() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)
    spreads = [config.spread_bps_by_decile[decile] for decile in range(1, 11)]

    assert len(set(spreads)) == 10
    assert max(spreads) / min(spreads) >= 10.0


# --------------------------------------------------------------------------
# Building a CostModel from config.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(("market", "taker_bps"), [("spot", 10.0), ("perp", 5.0)])
def test_cost_model_uses_market_taker_fee_with_config_spreads(
    market: str, taker_bps: float
) -> None:
    model = CostConfig.from_yaml(COSTS_CONFIG).cost_model(market)

    assert model.taker_bps == pytest.approx(taker_bps)
    assert model.impact_coefficient == pytest.approx(0.6)
    assert [model.spread_bps(decile) for decile in range(1, 11)] == list(
        SPEC_DECILE_SPREADS
    )


def test_cost_model_without_market_uses_top_level_taker_fee() -> None:
    model = CostConfig.from_yaml(COSTS_CONFIG).cost_model()

    assert model.taker_bps == pytest.approx(10.0)


@pytest.mark.parametrize("market", ["cash", "futures", "", "SPOT"])
def test_cost_model_rejects_unknown_market_types(market: str) -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    with pytest.raises(ValueError, match="market"):
        config.cost_model(market)


# --------------------------------------------------------------------------
# The tax rate must never reach the trade loop.
# --------------------------------------------------------------------------


def test_cost_model_has_no_tax_field() -> None:
    model = CostConfig.from_yaml(COSTS_CONFIG).cost_model("spot")

    assert not any("tax" in field.name for field in fields(CostModel))
    assert not hasattr(model, "tax")
    assert not hasattr(model, "short_term_rate")
    assert not hasattr(model, "tax_rate")


def test_trade_cost_is_identical_regardless_of_configured_tax_rate(
    tmp_path: Path,
) -> None:
    taxed = _load(tmp_path / "a", tax="tax:\n  short_term_rate: 0.35")
    untaxed = _load(tmp_path / "b", tax="tax:\n  short_term_rate: 0.0")
    inputs = {
        "notional": 50_000.0,
        "adv": 5_000_000.0,
        "volatility": 0.04,
        "liquidity_decile": 4,
    }

    taxed_cost = taxed.cost_model("spot").trade_cost(**inputs)
    untaxed_cost = untaxed.cost_model("spot").trade_cost(**inputs)

    assert taxed.short_term_rate != untaxed.short_term_rate
    assert taxed_cost == untaxed_cost
    assert taxed_cost.total_deducted == pytest.approx(
        taxed_cost.taker_fee + taxed_cost.impact
    )


# --------------------------------------------------------------------------
# Adversarial parsing.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "entries",
    [
        "[200, 120, 80, 55, 40, 30, 22, 16, 11]",
        "[200, 120, 80, 55, 40, 30, 22, 16, 11, 8, 5]",
        "[]",
        "[30]",
    ],
)
def test_decile_list_must_have_exactly_ten_entries(
    tmp_path: Path, entries: str
) -> None:
    with pytest.raises(ValueError, match="by_liquidity_decile"):
        _load(
            tmp_path,
            spread_bps=f"spread_bps:\n  default: 30\n  by_liquidity_decile: {entries}",
        )


@pytest.mark.parametrize(
    "entries",
    [
        "[8, 11, 16, 22, 30, 40, 55, 80, 120, 200]",
        "[200, 200, 80, 55, 40, 30, 22, 16, 11, 8]",
        "[200, 120, 80, 55, 40, 40, 22, 16, 11, 8]",
        "[200, 120, 80, 55, 40, 30, 22, 16, 8, 11]",
    ],
)
def test_decile_list_must_strictly_decrease(tmp_path: Path, entries: str) -> None:
    with pytest.raises(ValueError, match="decrease"):
        _load(
            tmp_path,
            spread_bps=f"spread_bps:\n  default: 30\n  by_liquidity_decile: {entries}",
        )


@pytest.mark.parametrize(
    "entries",
    [
        "[200, 120, 80, 55, 40, 30, 22, 16, 11, wide]",
        "[200, 120, 80, 55, 40, 30, 22, 16, 11, ]",
        "[200, 120, 80, 55, 40, 30, 22, 16, 11, nan]",
        "[inf, 120, 80, 55, 40, 30, 22, 16, 11, 8]",
        "200, 120, 80, 55, 40, 30, 22, 16, 11, 8",
    ],
)
def test_decile_list_rejects_non_numeric_or_nonfinite_entries(
    tmp_path: Path, entries: str
) -> None:
    with pytest.raises(ValueError):
        _load(
            tmp_path,
            spread_bps=f"spread_bps:\n  default: 30\n  by_liquidity_decile: {entries}",
        )


@pytest.mark.parametrize("entries", ["[0, -1, -2, -3, -4, -5, -6, -7, -8, -9]"])
def test_decile_list_rejects_nonpositive_spreads(tmp_path: Path, entries: str) -> None:
    with pytest.raises(ValueError, match="spread"):
        _load(
            tmp_path,
            spread_bps=f"spread_bps:\n  default: 30\n  by_liquidity_decile: {entries}",
        )


@pytest.mark.parametrize(
    "model",
    ['"linear"', '"sqrt"', '"square-root"', '"SQUARE_ROOT"', '""', "square_root_v2"],
)
def test_unknown_slippage_model_is_rejected_loudly(tmp_path: Path, model: str) -> None:
    with pytest.raises(ValueError, match="square_root"):
        _load(tmp_path, slippage=f"slippage:\n  model: {model}\n  coefficient: 0.6")


def test_unquoted_square_root_model_is_accepted(tmp_path: Path) -> None:
    config = _load(
        tmp_path, slippage="slippage:\n  model: square_root\n  coefficient: 0.6"
    )

    assert config.slippage_model == "square_root"


@pytest.mark.parametrize("rate", ["1.0", "1", "1.0000001", "-0.1", "-0.0001", "2.5"])
def test_tax_rate_outside_zero_to_one_is_rejected(tmp_path: Path, rate: str) -> None:
    with pytest.raises(ValueError, match="short_term_rate"):
        _load(tmp_path, tax=f"tax:\n  short_term_rate: {rate}")


@pytest.mark.parametrize("rate", ["nan", "inf", "-inf", "thirty-five percent"])
def test_tax_rate_must_be_a_finite_number(tmp_path: Path, rate: str) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, tax=f"tax:\n  short_term_rate: {rate}")


@pytest.mark.parametrize("rate", ["0.0", "0", "0.999"])
def test_tax_rate_boundary_values_inside_the_interval_are_accepted(
    tmp_path: Path, rate: str
) -> None:
    config = _load(tmp_path, tax=f"tax:\n  short_term_rate: {rate}")

    assert 0.0 <= config.short_term_rate < 1.0


@pytest.mark.parametrize(
    "block",
    [
        "taker_fee_bps: 10\ntaker_fee_bps: 4",
        "tax:\n  short_term_rate: 0.35\n  short_term_rate: 0.10",
        (
            "spread_bps:\n"
            "  default: 30\n"
            "  default: 5\n"
            "  by_liquidity_decile: [200, 120, 80, 55, 40, 30, 22, 16, 11, 8]"
        ),
    ],
)
def test_duplicate_keys_are_rejected(tmp_path: Path, block: str) -> None:
    text = _config_text() + block + "\n"

    with pytest.raises(ValueError, match="duplicate"):
        CostConfig.from_yaml(_write(tmp_path, text))


def test_duplicate_top_level_section_is_rejected(tmp_path: Path) -> None:
    text = _config_text() + _DEFAULT_BLOCKS["tax"] + "\n"

    with pytest.raises(ValueError, match="duplicate"):
        CostConfig.from_yaml(_write(tmp_path, text))


@pytest.mark.parametrize(
    "block",
    [
        "taker_fee_bps:\t10",
        "tax:\n\tshort_term_rate: 0.35",
        "slippage:\n  model: \"square_root\"\n  coefficient:\t0.6",
    ],
)
def test_tab_characters_are_rejected(tmp_path: Path, block: str) -> None:
    section = block.split(":", 1)[0]
    with pytest.raises(ValueError, match="tab"):
        _load(tmp_path, **{section: block})


@pytest.mark.parametrize(
    "section", ["costs", "spread_bps", "taker_fee_bps", "slippage", "tax"]
)
def test_missing_sections_are_rejected(tmp_path: Path, section: str) -> None:
    with pytest.raises(ValueError, match=section):
        _load(tmp_path, **{section: None})


@pytest.mark.parametrize(
    "block",
    [
        "spread_bps:\n  default: 30",
        "spread_bps:\n  by_liquidity_decile: [200, 120, 80, 55, 40, 30, 22, 16, 11, 8]",
    ],
)
def test_missing_spread_subkeys_are_rejected(tmp_path: Path, block: str) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, spread_bps=block)


@pytest.mark.parametrize(
    "block",
    ['slippage:\n  model: "square_root"', "slippage:\n  coefficient: 0.6"],
)
def test_missing_slippage_subkeys_are_rejected(tmp_path: Path, block: str) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, slippage=block)


def test_unknown_top_level_section_is_rejected(tmp_path: Path) -> None:
    text = _config_text() + "borrow_bps: 12\n"

    with pytest.raises(ValueError, match="unknown"):
        CostConfig.from_yaml(_write(tmp_path, text))


def test_unknown_subkey_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown"):
        _load(tmp_path, tax="tax:\n  short_term_rate: 0.35\n  long_term_rate: 0.20")


@pytest.mark.parametrize(
    "text",
    [
        "spread_bps: 30\n",
        "   spread_bps:\n  default: 30\n",
        "taker_fee_bps\n",
        "",
    ],
)
def test_structurally_invalid_documents_are_rejected(tmp_path: Path, text: str) -> None:
    with pytest.raises(ValueError):
        CostConfig.from_yaml(_write(tmp_path, text))


def test_scalar_where_mapping_expected_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, tax="tax: 0.35")


def test_mapping_where_scalar_expected_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, taker_fee_bps="taker_fee_bps:\n  value: 10")


@pytest.mark.parametrize("value", ["-1", "nan", "inf", "free"])
def test_taker_fee_must_be_a_finite_nonnegative_number(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError):
        _load(tmp_path, taker_fee_bps=f"taker_fee_bps: {value}")


@pytest.mark.parametrize("value", ["-0.1", "nan", "inf", "sqrt"])
def test_impact_coefficient_must_be_a_finite_nonnegative_number(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError):
        _load(
            tmp_path,
            slippage=f'slippage:\n  model: "square_root"\n  coefficient: {value}',
        )


@pytest.mark.parametrize("value", ["0", "-5", "nan", "inf", "20000", "wide"])
def test_default_spread_must_be_a_finite_positive_bps(
    tmp_path: Path, value: str
) -> None:
    with pytest.raises(ValueError):
        _load(
            tmp_path,
            spread_bps=(
                f"spread_bps:\n  default: {value}\n"
                "  by_liquidity_decile: [200, 120, 80, 55, 40, 30, 22, 16, 11, 8]"
            ),
        )


def test_missing_market_taker_fee_is_rejected(tmp_path: Path) -> None:
    config = _load(
        tmp_path,
        costs="costs:\n  spot:\n    maker_bps: 10\n  perp:\n    taker_bps: 5",
    )

    with pytest.raises(ValueError, match="taker_bps"):
        config.cost_model("spot")


def test_comments_and_blank_lines_are_ignored(tmp_path: Path) -> None:
    text = "# leading comment\n\n" + _config_text().replace(
        "taker_fee_bps: 10", "taker_fee_bps: 10  # exchange taker fee"
    )

    config = CostConfig.from_yaml(_write(tmp_path, text))

    assert config.taker_fee_bps == pytest.approx(10.0)


# --------------------------------------------------------------------------
# Immutability and direct construction.
# --------------------------------------------------------------------------


def test_cost_config_is_frozen_and_slotted() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    assert CostConfig.__slots__
    assert not hasattr(config, "__dict__")
    with pytest.raises(FrozenInstanceError):
        config.short_term_rate = 0.0  # type: ignore[misc]


def test_cost_config_decile_mapping_is_read_only() -> None:
    config = CostConfig.from_yaml(COSTS_CONFIG)

    with pytest.raises(TypeError):
        config.spread_bps_by_decile[1] = 1.0  # type: ignore[index]


@pytest.mark.parametrize("rate", [1.0, -0.1, float("nan"), float("inf")])
def test_direct_construction_validates_the_tax_rate(rate: float) -> None:
    with pytest.raises(ValueError, match="short_term_rate"):
        CostConfig(
            default_spread_bps=30.0,
            spread_bps_by_decile=dict(enumerate(SPEC_DECILE_SPREADS, 1)),
            taker_fee_bps=10.0,
            slippage_model="square_root",
            impact_coefficient=0.6,
            short_term_rate=rate,
            market_fees={"spot": {"taker_bps": 10.0}},
        )


def test_direct_construction_rejects_bool_tax_rate() -> None:
    with pytest.raises(TypeError):
        CostConfig(
            default_spread_bps=30.0,
            spread_bps_by_decile=dict(enumerate(SPEC_DECILE_SPREADS, 1)),
            taker_fee_bps=10.0,
            slippage_model="square_root",
            impact_coefficient=0.6,
            short_term_rate=True,  # type: ignore[arg-type]
            market_fees={"spot": {"taker_bps": 10.0}},
        )


def test_direct_construction_rejects_a_flat_spread_curve() -> None:
    with pytest.raises(ValueError, match="decrease"):
        CostConfig(
            default_spread_bps=30.0,
            spread_bps_by_decile={decile: 30.0 for decile in range(1, 11)},
            taker_fee_bps=10.0,
            slippage_model="square_root",
            impact_coefficient=0.6,
            short_term_rate=0.35,
            market_fees={"spot": {"taker_bps": 10.0}},
        )
