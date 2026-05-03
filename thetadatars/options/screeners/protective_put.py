import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import dte, empty_frame, grouped_by_expiration, normalize_rows, sort_and_limit

log = logging.getLogger(__name__)

RankBy = Literal["protection_efficiency", "protected_floor_percent", "hedge_cost_percent", "delta"]

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "strike", "premium", "hedge_cost",
    "hedge_cost_percent", "protected_floor", "protected_floor_percent",
    "max_loss_percent", "protection_efficiency", "dte", "underlying_price",
    "delta", "implied_vol", "theta", "vega", "timestamp",
]


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, stock_price: float | None, max_hedge_cost_percent: float | None) -> pl.DataFrame:
    output = []
    for expiration, rows in grouped_by_expiration(normalize_rows(chain)).items():
        days = dte(expiration, today)
        for row in rows:
            if right_name(row.get("right")) != "put":
                continue
            strike = finite_number(row.get("strike"))
            premium = finite_number(row.get("ask"))
            underlying = stock_price if stock_price is not None else finite_number(row.get("underlying_price"))
            if strike is None or premium is None or premium <= 0 or underlying is None or underlying <= 0:
                continue
            hedge_cost_percent = premium / underlying
            if max_hedge_cost_percent is not None and hedge_cost_percent > max_hedge_cost_percent:
                continue
            protected_floor = strike - premium
            protected_floor_percent = protected_floor / underlying
            max_loss_percent = 1 - protected_floor_percent
            protection_efficiency = protected_floor_percent / hedge_cost_percent if hedge_cost_percent > 0 else None
            output.append({
                "root": row.get("root", ticker),
                "expiration": expiration,
                "strategy": "protective_put",
                "strike": strike,
                "premium": premium,
                "hedge_cost": premium * 100,
                "hedge_cost_percent": hedge_cost_percent,
                "protected_floor": protected_floor,
                "protected_floor_percent": protected_floor_percent,
                "max_loss_percent": max_loss_percent,
                "protection_efficiency": protection_efficiency,
                "dte": days,
                "underlying_price": underlying,
                "delta": finite_number(row.get("delta")),
                "implied_vol": finite_number(row.get("implied_vol")),
                "theta": finite_number(row.get("theta")),
                "vega": finite_number(row.get("vega")),
                "timestamp": row.get("timestamp"),
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_protective_puts(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    max_hedge_cost_percent: float | None = None,
    top_n: int | None = 25,
    rank_by: RankBy = "protection_efficiency",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    conn=None,
) -> pl.DataFrame:
    expiration = parse_expiration(expiration)
    today = dt.date.today()
    if expiration != "*":
        days = dte(expiration, today)
        if min_dte is not None and days < min_dte:
            return empty_frame(_OUTPUT_COLUMNS)
        if max_dte is not None and days > max_dte:
            return empty_frame(_OUTPUT_COLUMNS)
    if client is None:
        client = create_client()
    chain = get_first_order_chain(
        ticker=ticker, expiration=expiration, client=client, log=log, right="put", today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, max_dte=max_dte, strike_range=strike_range,
        min_time=min_time, use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=min_dte, max_dte=max_dte)
    puts = _build_rows(ticker, chain, today=today, stock_price=stock_price, max_hedge_cost_percent=max_hedge_cost_percent)
    return sort_and_limit(puts, rank_by=rank_by, tie_breakers=["protected_floor_percent", "hedge_cost_percent"], descending=[True, True, False], top_n=top_n)


find_best_protective_puts = get_best_protective_puts

__all__ = ["get_best_protective_puts", "find_best_protective_puts"]
