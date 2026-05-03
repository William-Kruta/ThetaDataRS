import datetime as dt
from collections.abc import Iterable

import polars as pl

from ._common import finite_number, probability_otm_from_delta, right_name


def empty_frame(columns: Iterable[str]) -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.Null for column in columns})


def expiration_date(value: object) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(str(value), "%Y-%m-%d").date()


def dte(expiration: dt.date, today: dt.date) -> int:
    return (expiration - today).days


def mid_price(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def latest_underlying_price(rows: list[dict]) -> float | None:
    for row in rows:
        price = finite_number(row.get("underlying_price"))
        if price is not None and price > 0:
            return price
    return None


def normalize_rows(chain: pl.DataFrame) -> list[dict]:
    if chain.is_empty():
        return []
    rows = chain.to_dicts()
    fallback_underlying_price = latest_underlying_price(rows)
    if fallback_underlying_price is None:
        return rows
    return [
        row if row.get("underlying_price") is not None else {**row, "underlying_price": fallback_underlying_price}
        for row in rows
    ]


def option_rows(rows: list[dict], right: str) -> list[dict]:
    return [row for row in rows if right_name(row.get("right")) == right]


def row_by_strike(rows: list[dict]) -> dict[float, dict]:
    result = {}
    for row in rows:
        strike = finite_number(row.get("strike"))
        if strike is not None:
            result[strike] = row
    return result


def grouped_by_expiration(rows: list[dict]) -> dict[dt.date, list[dict]]:
    grouped = {}
    for row in rows:
        expiration = row.get("expiration")
        if expiration is None:
            continue
        grouped.setdefault(expiration_date(expiration), []).append(row)
    return grouped


def abs_delta(row: dict) -> float | None:
    delta = finite_number(row.get("delta"))
    return abs(delta) if delta is not None else None


def probability_itm_from_delta(row: dict) -> float | None:
    return abs_delta(row)


def probability_otm(row: dict) -> float | None:
    return probability_otm_from_delta(finite_number(row.get("delta")))


def annualize(value: float | None, days_to_expiration: int) -> float | None:
    if value is None or days_to_expiration <= 0:
        return None
    return value * (365 / days_to_expiration)


def pct(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0:
        return None
    return numerator / denominator


def sort_and_limit(
    df: pl.DataFrame,
    *,
    rank_by: str,
    tie_breakers: list[str],
    descending: list[bool],
    top_n: int | None,
) -> pl.DataFrame:
    if df.is_empty():
        return df
    columns = [rank_by, *tie_breakers]
    df = df.sort(columns, descending=descending, nulls_last=True)
    if top_n is not None and top_n > 0:
        df = df.head(top_n)
    return df
