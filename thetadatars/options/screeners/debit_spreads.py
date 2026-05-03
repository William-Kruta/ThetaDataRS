import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import annualize, dte, empty_frame, grouped_by_expiration, normalize_rows, row_by_strike, sort_and_limit

log = logging.getLogger(__name__)

Right = Literal["call", "put", "both"]
RankBy = Literal["annualized_return_on_risk", "return_on_risk", "max_profit", "debit", "probability_itm"]

_OUTPUT_COLUMNS = [
    "root", "expiration", "right", "spread_type", "long_strike", "short_strike",
    "width", "debit", "max_profit", "max_loss", "return_on_risk",
    "annualized_return_on_risk", "breakeven", "probability_itm", "dte",
    "underlying_price", "long_ask", "short_bid", "long_delta", "short_delta",
    "long_implied_vol", "short_implied_vol", "long_timestamp", "short_timestamp",
]


def _probability_itm(long_leg: dict, short_leg: dict) -> float | None:
    long_delta = finite_number(long_leg.get("delta"))
    short_delta = finite_number(short_leg.get("delta"))
    if long_delta is None and short_delta is None:
        return None
    if long_delta is None:
        return abs(short_delta)
    if short_delta is None:
        return abs(long_delta)
    return max(0.0, min(1.0, (abs(long_delta) + abs(short_delta)) / 2))


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, right: Right, min_debit: float, max_debit: float | None, min_width: float | None, max_width: float | None) -> pl.DataFrame:
    rows = normalize_rows(chain)
    if not rows:
        return empty_frame(_OUTPUT_COLUMNS)
    output = []
    for expiration, expiration_rows in grouped_by_expiration(rows).items():
        days = dte(expiration, today)
        for option_right in ("call", "put"):
            if right != "both" and right != option_right:
                continue
            side_rows = sorted([row for row in expiration_rows if right_name(row.get("right")) == option_right], key=lambda row: finite_number(row.get("strike")) or 0)
            by_strike = row_by_strike(side_rows)
            strikes = sorted(by_strike)
            for long_strike in strikes:
                for short_strike in strikes:
                    if option_right == "call" and short_strike <= long_strike:
                        continue
                    if option_right == "put" and short_strike >= long_strike:
                        continue
                    long_leg = by_strike[long_strike]
                    short_leg = by_strike[short_strike]
                    long_ask = finite_number(long_leg.get("ask"))
                    short_bid = finite_number(short_leg.get("bid"))
                    if long_ask is None or short_bid is None:
                        continue
                    width = abs(short_strike - long_strike)
                    debit = long_ask - short_bid
                    max_profit = width - debit
                    if debit < min_debit or max_profit <= 0:
                        continue
                    if max_debit is not None and debit > max_debit:
                        continue
                    if min_width is not None and width < min_width:
                        continue
                    if max_width is not None and width > max_width:
                        continue
                    spread_type = "bull_call_debit" if option_right == "call" else "bear_put_debit"
                    breakeven = long_strike + debit if option_right == "call" else long_strike - debit
                    return_on_risk = max_profit / debit
                    output.append({
                        "root": long_leg.get("root", ticker),
                        "expiration": expiration,
                        "right": option_right,
                        "spread_type": spread_type,
                        "long_strike": long_strike,
                        "short_strike": short_strike,
                        "width": width,
                        "debit": debit,
                        "max_profit": max_profit,
                        "max_loss": debit,
                        "return_on_risk": return_on_risk,
                        "annualized_return_on_risk": annualize(return_on_risk, days),
                        "breakeven": breakeven,
                        "probability_itm": _probability_itm(long_leg, short_leg),
                        "dte": days,
                        "underlying_price": finite_number(long_leg.get("underlying_price")),
                        "long_ask": long_ask,
                        "short_bid": short_bid,
                        "long_delta": finite_number(long_leg.get("delta")),
                        "short_delta": finite_number(short_leg.get("delta")),
                        "long_implied_vol": finite_number(long_leg.get("implied_vol")),
                        "short_implied_vol": finite_number(short_leg.get("implied_vol")),
                        "long_timestamp": long_leg.get("timestamp"),
                        "short_timestamp": short_leg.get("timestamp"),
                    })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_debit_spreads(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "both",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_debit: float = 0.01,
    max_debit: float | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    top_n: int | None = 25,
    rank_by: RankBy = "annualized_return_on_risk",
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
        ticker=ticker, expiration=expiration, client=client, log=log, right=right, today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, max_dte=max_dte, strike_range=strike_range,
        min_time=min_time, use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=min_dte, max_dte=max_dte)
    spreads = _build_rows(ticker, chain, today=today, right=right, min_debit=min_debit, max_debit=max_debit, min_width=min_width, max_width=max_width)
    return sort_and_limit(spreads, rank_by=rank_by, tie_breakers=["max_profit", "probability_itm"], descending=[True, True, True], top_n=top_n)


find_best_debit_spreads = get_best_debit_spreads

__all__ = ["get_best_debit_spreads", "find_best_debit_spreads"]
