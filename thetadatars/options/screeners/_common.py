import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client
from ..greeks import calculate_american_first_order_greeks
from ..snapshot.greeks_first_order import RateType, get_snapshot_greeks_first_order
from ..snapshot.quote import get_snapshot_quote

Right = Literal["call", "put", "both"]


def parse_expiration(expiration: dt.date | str) -> dt.date | str:
    if isinstance(expiration, dt.datetime):
        return expiration.date()
    if isinstance(expiration, dt.date):
        return expiration
    if expiration == "*":
        return expiration
    return dt.datetime.strptime(expiration, "%Y-%m-%d").date()


def right_name(value: object) -> str:
    right = str(value).strip().lower()
    if right in {"c", "call", "calls"}:
        return "call"
    if right in {"p", "put", "puts"}:
        return "put"
    return right


def finite_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def probability_otm_from_delta(delta: float | None) -> float | None:
    if delta is None:
        return None
    return max(0.0, min(1.0, 1 - abs(delta)))


def is_standard_subscription_error(error: Exception) -> bool:
    message = str(error).lower()
    return (
        "standard subscription" in message
        and "value subscription" in message
    )


def normalize_rate_value(rate_value: float | None) -> float:
    if rate_value is None:
        return 0.0
    rate = float(rate_value)
    return rate / 100 if abs(rate) > 1 else rate


def filter_expiration_dte(
    chain: pl.DataFrame,
    *,
    expiration: dt.date | str,
    today: dt.date,
    min_dte: int | None,
    max_dte: int | None,
) -> pl.DataFrame:
    if expiration != "*" or chain.is_empty() or (min_dte is None and max_dte is None):
        return chain

    chain = chain.with_columns(
        (pl.col("expiration") - pl.lit(today)).dt.total_days().alias("_dte")
    )
    if min_dte is not None:
        chain = chain.filter(pl.col("_dte") >= min_dte)
    if max_dte is not None:
        chain = chain.filter(pl.col("_dte") <= max_dte)
    return chain.drop("_dte")


def get_first_order_chain(
    *,
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    log: logging.Logger,
    right: Right,
    today: dt.date,
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    conn=None,
) -> pl.DataFrame:
    try:
        return get_snapshot_greeks_first_order(
            ticker=ticker,
            expiration=expiration,
            client=client,
            strike="*",
            right=right,
            annual_dividend=annual_dividend,
            rate_type=rate_type,
            rate_value=rate_value,
            stock_price=stock_price,
            version=version,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            use_market_value=use_market_value,
            stale_threshold=stale_threshold,
            conn=conn,
        )
    except Exception as exc:
        if not fallback_to_local_greeks or not is_standard_subscription_error(exc):
            raise
        log.info(
            "Falling back to local American option greeks for %s exp=%s because "
            "ThetaData first-order greeks require a Standard subscription",
            ticker,
            expiration,
        )
        quote_chain = get_snapshot_quote(
            ticker=ticker,
            expiration=expiration,
            client=client,
            strike="*",
            right=right,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            stale_threshold=stale_threshold,
            conn=conn,
        )
        return calculate_american_first_order_greeks(
            quote_chain,
            stock_price=stock_price,
            valuation_date=today,
            risk_free_rate=normalize_rate_value(rate_value),
            annual_dividend=annual_dividend,
            steps=local_greeks_steps,
        )
