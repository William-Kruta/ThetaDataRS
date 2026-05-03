import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import annualize, dte, empty_frame, grouped_by_expiration, normalize_rows, row_by_strike, sort_and_limit

log = logging.getLogger(__name__)

RankBy = Literal["annualized_return_on_risk", "return_on_risk", "credit", "body_distance_percent"]

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "long_put_strike", "body_strike",
    "long_call_strike", "put_width", "call_width", "credit", "max_loss",
    "return_on_risk", "annualized_return_on_risk", "lower_breakeven",
    "upper_breakeven", "body_distance_percent", "dte", "underlying_price",
    "short_put_delta", "short_call_delta",
]


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, min_credit: float, min_width: float | None, max_width: float | None) -> pl.DataFrame:
    output = []
    for expiration, expiration_rows in grouped_by_expiration(normalize_rows(chain)).items():
        puts = row_by_strike([r for r in expiration_rows if right_name(r.get("right")) == "put"])
        calls = row_by_strike([r for r in expiration_rows if right_name(r.get("right")) == "call"])
        body_strikes = sorted(set(puts) & set(calls))
        days = dte(expiration, today)
        for body in body_strikes:
            short_put = puts[body]
            short_call = calls[body]
            sp_bid = finite_number(short_put.get("bid"))
            sc_bid = finite_number(short_call.get("bid"))
            if sp_bid is None or sc_bid is None:
                continue
            for lp, long_put in puts.items():
                lp_ask = finite_number(long_put.get("ask"))
                if lp >= body or lp_ask is None:
                    continue
                put_width = body - lp
                if min_width is not None and put_width < min_width:
                    continue
                if max_width is not None and put_width > max_width:
                    continue
                for lc, long_call in calls.items():
                    lc_ask = finite_number(long_call.get("ask"))
                    if lc <= body or lc_ask is None:
                        continue
                    call_width = lc - body
                    if min_width is not None and call_width < min_width:
                        continue
                    if max_width is not None and call_width > max_width:
                        continue
                    credit = sp_bid + sc_bid - lp_ask - lc_ask
                    max_loss = max(put_width, call_width) - credit
                    if credit < min_credit or max_loss <= 0:
                        continue
                    underlying = finite_number(short_put.get("underlying_price"))
                    body_distance_percent = abs(body - underlying) / underlying if underlying is not None and underlying > 0 else None
                    return_on_risk = credit / max_loss
                    output.append({
                        "root": short_put.get("root", ticker),
                        "expiration": expiration,
                        "strategy": "iron_butterfly",
                        "long_put_strike": lp,
                        "body_strike": body,
                        "long_call_strike": lc,
                        "put_width": put_width,
                        "call_width": call_width,
                        "credit": credit,
                        "max_loss": max_loss,
                        "return_on_risk": return_on_risk,
                        "annualized_return_on_risk": annualize(return_on_risk, days),
                        "lower_breakeven": body - credit,
                        "upper_breakeven": body + credit,
                        "body_distance_percent": body_distance_percent,
                        "dte": days,
                        "underlying_price": underlying,
                        "short_put_delta": finite_number(short_put.get("delta")),
                        "short_call_delta": finite_number(short_call.get("delta")),
                    })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_iron_butterflies(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_credit: float = 0.01,
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
        ticker=ticker, expiration=expiration, client=client, log=log, right="both", today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, max_dte=max_dte, strike_range=strike_range,
        min_time=min_time, use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=min_dte, max_dte=max_dte)
    flies = _build_rows(ticker, chain, today=today, min_credit=min_credit, min_width=min_width, max_width=max_width)
    return sort_and_limit(flies, rank_by=rank_by, tie_breakers=["credit", "body_distance_percent"], descending=[True, True, False], top_n=top_n)


find_best_iron_butterflies = get_best_iron_butterflies

__all__ = ["get_best_iron_butterflies", "find_best_iron_butterflies"]
