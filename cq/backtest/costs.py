"""Deterministic execution-price and transaction-cost primitives."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from math import isfinite, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Self, cast

MAX_PARTICIPATION = 0.02
MAX_ALLOWED_PARTICIPATION = 0.05
_BPS_DENOMINATOR = 10_000.0
_HALF_SPREAD_DENOMINATOR = 20_000.0
_DECILES = frozenset(range(1, 11))


class Side(str, Enum):
    """Trade direction."""

    BUY = "buy"
    SELL = "sell"


@dataclass(frozen=True, slots=True)
class Bar:
    """Market inputs used to price and cap one execution."""

    open: float
    close: float
    quote_volume: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "open", _positive("open", self.open))
        object.__setattr__(self, "close", _positive("close", self.close))
        object.__setattr__(
            self,
            "quote_volume",
            _positive("quote_volume", self.quote_volume),
        )


def fill_price(side: Side, bar: Bar, spread_bps: float) -> float:
    """Apply exactly one adverse half-spread to the bar open."""
    direction = _side_direction(side)
    execution_bar = _validated_bar(bar)
    spread = _positive("spread_bps", spread_bps)
    if spread >= _HALF_SPREAD_DENOMINATOR:
        raise ValueError("spread_bps must be less than 20000")
    return execution_bar.open * (
        1.0 + direction * spread / _HALF_SPREAD_DENOMINATOR
    )


def executable_notional(
    desired_notional: float,
    quote_volume: float,
    max_participation: float = MAX_PARTICIPATION,
) -> float:
    """Return desired notional capped by a validated quote-volume share."""
    desired = _positive("desired_notional", desired_notional)
    volume = _positive("quote_volume", quote_volume)
    participation = _positive("max_participation", max_participation)
    if participation > MAX_ALLOWED_PARTICIPATION:
        raise ValueError("max_participation cannot exceed 0.05")
    return min(desired, participation * volume)


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    """Analytical spread plus costs explicitly deducted from cash."""

    spread: float
    taker_fee: float
    impact: float

    def __post_init__(self) -> None:
        object.__setattr__(self, "spread", _nonnegative("spread", self.spread))
        object.__setattr__(
            self,
            "taker_fee",
            _nonnegative("taker_fee", self.taker_fee),
        )
        object.__setattr__(self, "impact", _nonnegative("impact", self.impact))

    @property
    def total_deducted(self) -> float:
        """Fee and impact only; spread is already paid in the fill price."""
        total = self.taker_fee + self.impact
        return _nonnegative("total_deducted", total)


@dataclass(frozen=True, slots=True)
class CostModel:
    """Immutable taker-fee, spread, and square-root impact model."""

    taker_bps: float
    impact_coefficient: float
    spread_bps_by_decile: Mapping[int, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "taker_bps",
            _nonnegative("taker_bps", self.taker_bps),
        )
        object.__setattr__(
            self,
            "impact_coefficient",
            _nonnegative("impact_coefficient", self.impact_coefficient),
        )
        spreads = _validated_spreads(self.spread_bps_by_decile)
        object.__setattr__(self, "spread_bps_by_decile", spreads)

    def spread_bps(self, liquidity_decile: int) -> float:
        """Return the configured full spread for deciles one through ten."""
        decile = _validated_decile(liquidity_decile)
        return self.spread_bps_by_decile[decile]

    def trade_cost(
        self,
        *,
        notional: float,
        adv: float,
        volatility: float,
        liquidity_decile: int,
    ) -> CostBreakdown:
        """Calculate analytical spread, taker fee, and square-root impact."""
        trade_notional = _positive("notional", notional)
        average_volume = _positive("adv", adv)
        sigma = _positive("volatility", volatility)
        spread = trade_notional * self.spread_bps(liquidity_decile)
        spread /= _HALF_SPREAD_DENOMINATOR
        fee = trade_notional * self.taker_bps / _BPS_DENOMINATOR
        impact = self._impact(trade_notional, average_volume, sigma)
        return CostBreakdown(spread=spread, taker_fee=fee, impact=impact)

    def _impact(self, notional: float, adv: float, volatility: float) -> float:
        participation = notional / adv
        return notional * self.impact_coefficient * volatility * sqrt(participation)

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        *,
        market_type: str | None,
        impact_coefficient: float,
        spread_bps_by_decile: Mapping[int, float],
    ) -> Self:
        """Load the selected market's explicit taker fee from YAML."""
        if market_type not in ("spot", "perp"):
            raise ValueError(f"unsupported market type: {market_type!r}")
        markets = _load_cost_markets(Path(path))
        market = markets.get(market_type)
        if market is None:
            raise ValueError(f"missing market configuration: {market_type}")
        if "taker_bps" not in market:
            raise ValueError(f"missing taker_bps for market: {market_type}")
        return cls(
            taker_bps=market["taker_bps"],
            impact_coefficient=impact_coefficient,
            spread_bps_by_decile=spread_bps_by_decile,
        )


def _positive(name: str, value: object) -> float:
    number = _finite_number(name, value)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive")
    return number


def _nonnegative(name: str, value: object) -> float:
    number = _finite_number(name, value)
    if number < 0.0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _finite_number(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _validated_decile(value: object) -> int:
    if type(value) is not int or value not in _DECILES:
        raise ValueError("liquidity decile must be an integer from 1 through 10")
    return value


def _side_direction(value: object) -> float:
    if not isinstance(value, Side):
        raise TypeError("side must be a Side")
    return 1.0 if value is Side.BUY else -1.0


def _validated_bar(value: object) -> Bar:
    if not isinstance(value, Bar):
        raise TypeError("bar must be a Bar")
    return value


def _validated_spreads(
    spreads: object,
) -> Mapping[int, float]:
    if not isinstance(spreads, Mapping):
        raise TypeError("spread configuration must be a mapping")
    raw_spreads = cast(Mapping[object, object], spreads)
    if any(type(key) is not int for key in raw_spreads):
        raise ValueError("spread decile keys must be integers")
    if set(raw_spreads) != set(_DECILES):
        raise ValueError("spread configuration must contain deciles 1 through 10")
    validated = {
        decile: _positive("spread_bps", raw_spreads[decile])
        for decile in range(1, 11)
    }
    if any(spread >= _HALF_SPREAD_DENOMINATOR for spread in validated.values()):
        raise ValueError("spread_bps must be less than 20000")
    if any(left <= right for left, right in pairwise(validated.values())):
        raise ValueError("spreads must strictly decrease as liquidity decile rises")
    return MappingProxyType(validated)


def _load_cost_markets(path: Path) -> dict[str, dict[str, float]]:
    markets: dict[str, dict[str, float]] = {}
    current_market: str | None = None
    saw_costs = False
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parsed = _parse_yaml_line(raw_line, line_number)
        if parsed is None:
            continue
        indent, key, value = parsed
        if indent == 0:
            saw_costs = _parse_costs_root(key, value, saw_costs)
            current_market = None
        elif indent == 2 and saw_costs:
            current_market = _add_market(markets, key, value)
        elif indent == 4 and current_market is not None:
            _add_market_field(markets[current_market], key, value, line_number)
        else:
            raise ValueError(f"invalid costs YAML structure at line {line_number}")
    if not saw_costs:
        raise ValueError("costs YAML must contain a costs mapping")
    return markets


def _parse_yaml_line(
    raw_line: str,
    line_number: int,
) -> tuple[int, str, str] | None:
    if "\t" in raw_line:
        raise ValueError(f"tabs are not allowed in costs YAML at line {line_number}")
    uncommented = raw_line.split("#", 1)[0].rstrip()
    if not uncommented.strip() or uncommented.strip() == "---":
        return None
    indent = len(uncommented) - len(uncommented.lstrip(" "))
    key, separator, value = uncommented.strip().partition(":")
    if not separator or not key.strip():
        raise ValueError(f"invalid costs YAML at line {line_number}")
    return indent, key.strip(), value.strip()


def _parse_costs_root(key: str, value: str, already_seen: bool) -> bool:
    if key != "costs" or value:
        raise ValueError("costs YAML root must be a costs mapping")
    if already_seen:
        raise ValueError("duplicate costs mapping")
    return True


def _add_market(
    markets: dict[str, dict[str, float]],
    key: str,
    value: str,
) -> str:
    if value:
        raise ValueError(f"market {key!r} must contain a mapping")
    if key in markets:
        raise ValueError(f"duplicate market configuration: {key}")
    markets[key] = {}
    return key


def _add_market_field(
    market: dict[str, float],
    key: str,
    value: str,
    line_number: int,
) -> None:
    if not value:
        raise ValueError(f"missing value for {key!r} at line {line_number}")
    if key in market:
        raise ValueError(f"duplicate market field: {key}")
    try:
        market[key] = float(value)
    except ValueError as error:
        raise ValueError(f"invalid numeric value at line {line_number}") from error
