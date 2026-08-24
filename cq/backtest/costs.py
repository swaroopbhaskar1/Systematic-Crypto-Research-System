"""Deterministic execution-price and transaction-cost primitives."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from math import isfinite, sqrt
from pathlib import Path
from types import MappingProxyType
from typing import Self, TypeAlias, cast

MAX_PARTICIPATION = 0.02
MAX_ALLOWED_PARTICIPATION = 0.05
SQUARE_ROOT_SLIPPAGE_MODEL = "square_root"
MARKET_TYPES = ("spot", "perp")
_BPS_DENOMINATOR = 10_000.0
_HALF_SPREAD_DENOMINATOR = 20_000.0
_DECILES = frozenset(range(1, 11))
_DECILE_COUNT = 10
_INDENT_STEP = 2
_TOP_LEVEL_KEYS = frozenset(
    {"costs", "spread_bps", "taker_fee_bps", "slippage", "tax"}
)
_SPREAD_KEYS = frozenset({"default", "by_liquidity_decile"})
_SLIPPAGE_KEYS = frozenset({"model", "coefficient"})
_TAX_KEYS = frozenset({"short_term_rate"})

_YamlNode: TypeAlias = "str | dict[str, _YamlNode]"
_YamlMapping: TypeAlias = "dict[str, _YamlNode]"


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


@dataclass(frozen=True, slots=True)
class CostConfig:
    """Fully validated contents of ``config/costs.yaml``.

    Holds every cost assumption declaratively so a reader can audit what the
    backtest charged.  ``short_term_rate`` is reporting-only: it is never read
    by :class:`CostModel` and never reaches the trade loop.
    """

    default_spread_bps: float
    spread_bps_by_decile: Mapping[int, float]
    taker_fee_bps: float
    slippage_model: str
    impact_coefficient: float
    short_term_rate: float
    market_fees: Mapping[str, Mapping[str, float]]

    def __post_init__(self) -> None:
        self._set(
            "default_spread_bps",
            _spread_bps("default_spread_bps", self.default_spread_bps),
        )
        self._set(
            "spread_bps_by_decile",
            _validated_spreads(self.spread_bps_by_decile),
        )
        self._set("taker_fee_bps", _nonnegative("taker_fee_bps", self.taker_fee_bps))
        self._set("slippage_model", _validated_slippage_model(self.slippage_model))
        self._set(
            "impact_coefficient",
            _nonnegative("impact_coefficient", self.impact_coefficient),
        )
        self._set("short_term_rate", _validated_tax_rate(self.short_term_rate))
        self._set("market_fees", _validated_market_fees(self.market_fees))

    def _set(self, name: str, value: object) -> None:
        object.__setattr__(self, name, value)

    @classmethod
    def from_yaml(cls, path: str | Path) -> Self:
        """Parse and strictly validate the declarative cost configuration."""
        document = _load_yaml_document(Path(path))
        _reject_unknown_keys(document, _TOP_LEVEL_KEYS, "top-level")
        default_spread, deciles = _read_spread_section(document)
        model, coefficient = _read_slippage_section(document)
        return cls(
            default_spread_bps=default_spread,
            spread_bps_by_decile=deciles,
            taker_fee_bps=_read_number(document, "taker_fee_bps"),
            slippage_model=model,
            impact_coefficient=coefficient,
            short_term_rate=_read_tax_section(document),
            market_fees=_cost_markets(document),
        )

    def cost_model(self, market_type: str | None = None) -> CostModel:
        """Build a CostModel from config alone, with no Python constants.

        ``market_type`` selects that market's taker fee; ``None`` falls back to
        the top-level ``taker_fee_bps``.
        """
        return CostModel(
            taker_bps=self._taker_bps(market_type),
            impact_coefficient=self.impact_coefficient,
            spread_bps_by_decile=self.spread_bps_by_decile,
        )

    def _taker_bps(self, market_type: str | None) -> float:
        if market_type is None:
            return self.taker_fee_bps
        if market_type not in MARKET_TYPES:
            raise ValueError(f"unsupported market type: {market_type!r}")
        market = self.market_fees.get(market_type)
        if market is None:
            raise ValueError(f"missing market configuration: {market_type}")
        if "taker_bps" not in market:
            raise ValueError(f"missing taker_bps for market: {market_type}")
        return market["taker_bps"]


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


def _spread_bps(name: str, value: object) -> float:
    spread = _positive(name, value)
    if spread >= _HALF_SPREAD_DENOMINATOR:
        raise ValueError(f"{name} must be less than 20000")
    return spread


def _validated_tax_rate(value: object) -> float:
    rate = _finite_number("short_term_rate", value)
    if not 0.0 <= rate < 1.0:
        raise ValueError("short_term_rate must be in [0, 1)")
    return rate


def _validated_slippage_model(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("slippage model must be a string")
    if value != SQUARE_ROOT_SLIPPAGE_MODEL:
        raise ValueError(f'slippage model must be "square_root", got {value!r}')
    return value


def _decile_spreads(values: Sequence[float]) -> Mapping[int, float]:
    if len(values) != _DECILE_COUNT:
        raise ValueError("by_liquidity_decile must contain exactly 10 entries")
    return _validated_spreads(dict(enumerate(values, 1)))


def _validated_market_fees(fees: object) -> Mapping[str, Mapping[str, float]]:
    if not isinstance(fees, Mapping):
        raise TypeError("market fees must be a mapping")
    raw_fees = cast(Mapping[object, object], fees)
    if any(not isinstance(market, str) for market in raw_fees):
        raise ValueError("market names must be strings")
    validated = {
        cast(str, market): _validated_market_fields(cast(str, market), values)
        for market, values in raw_fees.items()
    }
    return MappingProxyType(validated)


def _validated_market_fields(market: str, values: object) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise ValueError(f"market {market!r} must contain a mapping")
    raw_fields = cast(Mapping[object, object], values)
    if any(not isinstance(key, str) for key in raw_fields):
        raise ValueError(f"market {market!r} field names must be strings")
    fields = {
        cast(str, key): _nonnegative(f"{market}.{key}", value)
        for key, value in raw_fields.items()
    }
    return MappingProxyType(fields)


def _read_spread_section(
    document: _YamlMapping,
) -> tuple[float, Mapping[int, float]]:
    section = _required_mapping(document, "spread_bps")
    _reject_unknown_keys(section, _SPREAD_KEYS, "spread_bps")
    entries = _inline_float_list(
        "by_liquidity_decile",
        _required_scalar(section, "by_liquidity_decile"),
    )
    return _read_number(section, "default"), _decile_spreads(entries)


def _read_slippage_section(document: _YamlMapping) -> tuple[str, float]:
    section = _required_mapping(document, "slippage")
    _reject_unknown_keys(section, _SLIPPAGE_KEYS, "slippage")
    model = _validated_slippage_model(_unquoted(_required_scalar(section, "model")))
    return model, _read_number(section, "coefficient")


def _read_tax_section(document: _YamlMapping) -> float:
    section = _required_mapping(document, "tax")
    _reject_unknown_keys(section, _TAX_KEYS, "tax")
    return _read_number(section, "short_term_rate")


def _cost_markets(document: _YamlMapping) -> dict[str, dict[str, float]]:
    section = _required_mapping(document, "costs")
    markets: dict[str, dict[str, float]] = {}
    for market, node in section.items():
        if not isinstance(node, dict):
            raise ValueError(f"market {market!r} must contain a mapping")
        markets[market] = {key: _read_number(node, key) for key in node}
    return markets


def _load_cost_markets(path: Path) -> dict[str, dict[str, float]]:
    return _cost_markets(_load_yaml_document(path))


def _load_yaml_document(path: Path) -> _YamlMapping:
    """Parse a strict two-space-indented YAML subset into nested mappings."""
    root: _YamlMapping = {}
    stack: list[tuple[int, _YamlMapping]] = [(0, root)]
    lines = path.read_text(encoding="utf-8").splitlines()
    for line_number, raw_line in enumerate(lines, 1):
        parsed = _parse_yaml_line(raw_line, line_number)
        if parsed is None:
            continue
        indent, key, value = parsed
        container = _container_at(stack, indent, line_number)
        if key in container:
            raise ValueError(f"duplicate key {key!r} at line {line_number}")
        if value:
            container[key] = value
            continue
        child: _YamlMapping = {}
        container[key] = child
        stack.append((indent + _INDENT_STEP, child))
    return root


def _container_at(
    stack: list[tuple[int, _YamlMapping]],
    indent: int,
    line_number: int,
) -> _YamlMapping:
    while len(stack) > 1 and indent < stack[-1][0]:
        stack.pop()
    expected, container = stack[-1]
    if indent != expected:
        raise ValueError(f"invalid costs YAML structure at line {line_number}")
    return container


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


def _reject_unknown_keys(
    container: Mapping[str, object],
    allowed: frozenset[str],
    where: str,
) -> None:
    unknown = sorted(set(container) - allowed)
    if unknown:
        raise ValueError(f"unknown {where} keys in costs YAML: {', '.join(unknown)}")


def _required_mapping(container: _YamlMapping, key: str) -> _YamlMapping:
    node = container.get(key)
    if node is None:
        raise ValueError(f"costs YAML must contain a {key} section")
    if not isinstance(node, dict):
        raise ValueError(f"{key} must be a mapping")
    return node


def _required_scalar(container: _YamlMapping, key: str) -> str:
    node = container.get(key)
    if node is None:
        raise ValueError(f"costs YAML must contain a {key} section")
    if not isinstance(node, str):
        raise ValueError(f"{key} must be a scalar value")
    return node


def _read_number(container: _YamlMapping, key: str) -> float:
    return _number_from_text(key, _required_scalar(container, key))


def _number_from_text(name: str, text: str) -> float:
    try:
        number = float(text)
    except ValueError as error:
        raise ValueError(f"{name} must be a number, got {text!r}") from error
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    return number


def _inline_float_list(name: str, text: str) -> list[float]:
    if not (text.startswith("[") and text.endswith("]")):
        raise ValueError(f"{name} must be an inline list of numbers")
    body = text[1:-1].strip()
    if not body:
        raise ValueError(f"{name} must contain exactly 10 entries")
    return [_number_from_text(name, entry.strip()) for entry in body.split(",")]


def _unquoted(text: str) -> str:
    if len(text) >= 2 and text[0] == text[-1] and text[0] in "\"'":
        return text[1:-1]
    return text
