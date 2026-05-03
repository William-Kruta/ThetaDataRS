import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration
from ._strategy_utils import dte, empty_frame, grouped_by_expiration, normalize_rows, row_by_strike, sort_and_limit

log = logging.getLogger(__name__)

Side = Literal["long", "short"]
RankBy = Literal["vega_per_dollar", "theta_income", "premium", "body_distance_percent"]

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "side", "strike", "premium", "max_loss",
    "lower_breakeven", "upper_breakeven", "body_distance_percent", "dte",
    "underlying_price", "call_delta", "put_delta", "net_delta", "net_theta",
    "net_vega", "vega_per_dollar", "theta_income",
]


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, side: Side, min_premium: float) -> pl.DataFrame:
    output = []
    for expiration, rows in grouped_by_expiration(normalize_rows(chain)).items():
        calls = row_by_strike([r for r in rows if str(r.get("right")).lower() in {"call", "c", "calls"}])
        puts = row_by_strike([r for r in rows if str(r.get("right")).lower() in {"put", "p", "puts"}])
        days = dte(expiration, today)
        for strike in sorted(set(calls) & set(puts)):
            call = calls[strike]
            put = puts[strike]
            call_price = finite_number(call.get("ask" if side == "long" else "bid"))
            put_price = finite_number(put.get("ask" if side == "long" else "bid"))
            if call_price is None or put_price is None:
                continue
            premium = call_price + put_price
            if premium < min_premium:
                continue
            underlying = finite_number(call.get("underlying_price"))
            body_distance_percent = abs(strike - underlying) / underlying if underlying is not None and underlying > 0 else None
            call_delta = finite_number(call.get("delta"))
            put_delta = finite_number(put.get("delta"))
            net_delta = (call_delta or 0) + (put_delta or 0)
            net_theta = (finite_number(call.get("theta")) or 0) + (finite_number(put.get("theta")) or 0)
            net_vega = (finite_number(call.get("vega")) or 0) + (finite_number(put.get("vega")) or 0)
            if side == "short":
                net_delta *= -1
                net_theta *= -1
                net_vega *= -1
            output.append({
                "root": call.get("root", ticker),
                "expiration": expiration,
                "strategy": f"{side}_straddle",
                "side": side,
                "strike": strike,
                "premium": premium,
                "max_loss": premium * 100 if side == "long" else None,
                "lower_breakeven": strike - premium,
                "upper_breakeven": strike + premium,
                "body_distance_percent": body_distance_percent,
                "dte": days,
                "underlying_price": underlying,
                "call_delta": call_delta,
                "put_delta": put_delta,
                "net_delta": net_delta,
                "net_theta": net_theta,
                "net_vega": net_vega,
                "vega_per_dollar": abs(net_vega) / premium if premium > 0 else None,
                "theta_income": net_theta if side == "short" else -net_theta,
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_straddles(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    side: Side = "long",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_premium: float = 0.01,
    top_n: int | None = 25,
    rank_by: RankBy = "vega_per_dollar",
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
    straddles = _build_rows(ticker, chain, today=today, side=side, min_premium=min_premium)
    return sort_and_limit(straddles, rank_by=rank_by, tie_breakers=["premium", "body_distance_percent"], descending=[True, True, False], top_n=top_n)


find_best_straddles = get_best_straddles

__all__ = ["get_best_straddles", "find_best_straddles"]
