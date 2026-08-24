# M1 data availability

## Binance USD-M open-interest history

Checked 2026-08-24.

Open interest is intentionally omitted from the M1 long schema and archive
gate. It is explicitly unavailable in `DataAvailability`; no sparse OI column
is created. Binance's official futures connector documents
`GET /futures/data/openInterestHist` with this explicit restriction:
"Only the data of the latest 30 days is available."

Binance Vision separately publishes deep daily USD-M metrics objects at
`data/futures/um/daily/metrics/{SYMBOL}/{SYMBOL}-metrics-YYYY-MM-DD.zip`.
Those objects contain `create_time`, `sum_open_interest`, and
`sum_open_interest_value`, and early files can contain duplicate timestamps.
M1 does not implement that separate best-effort adapter. Any future adapter
must normalize and deduplicate metrics independently rather than substituting
the 30-day REST endpoint.

Evidence:

- [Binance USD-M futures connector source](https://github.com/binance/binance-futures-connector-python/blob/main/binance/um_futures/market.py)
- [Binance Open Interest Statistics API documentation](https://developers.binance.com/docs/derivatives/usds-margined-futures/market-data/rest-api/Open-Interest-Statistics)
- CCXT 4.5.75 exposes the same endpoint but does not provide a deeper archive.

A direct request for a 60-day-old interval was also attempted. The environment
received HTTP 451 because Binance's API is location-restricted, so that request
could not independently observe the empty old interval. This does not weaken
the official 30-day limit, and no mostly-empty historical column is persisted.

## Binance public archive

The archive metadata endpoint remained accessible. A metadata-only observation
on 2026-08-24 returned 3,695 spot kline symbol prefixes, 986 USD-M perpetual
kline prefixes, and 920 USD-M funding-rate prefixes. Restricting the gate to
USDT pairs yielded 723 spot, 832 perpetual-kline, and 833 funding prefixes;
the union is 833 lifetime USDT perpetuals.
These prefix counts are discovery evidence only; they are not substitutes for
the full gate's last-date and exact contiguous-funding checks.
