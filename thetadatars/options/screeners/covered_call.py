import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration
from ._strategy_utils import annualize, dte, empty_frame, grouped_by_expiration, normalize_rows, probability_otm, sort_and_limit

log = logging.getLogger(__name__)

RankBy = Literal[
    "annualized_called_return",
    "annualized_premium_yield",
    "risk_adjusted_yield",
    "called_return",
    "premium_yield",
    "premium",
    "probability_otm",
]

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "strike", "premium", "called_price",
    "breakeven", "premium_yield", "annualized_premium_yield", "called_return",
    "annualized_called_return", "probability_otm", "risk_adjusted_yield",
    "dte", "underlying_price", "delta", "implied_vol", "theta", "vega",
    "rho", "epsilon", "lambda", "timestamp",
]


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, stock_price: float | None, include_itm: bool, min_premium: float, min_probability_otm: float | None) -> pl.DataFrame:
    rows = normalize_rows(chain)
    if not rows:
        return empty_frame(_OUTPUT_COLUMNS)
    output = []
    for expiration, expiration_rows in grouped_by_expiration(rows).items():
        days = dte(expiration, today)
        for row in expiration_rows:
            if str(row.get("right")).lower() not in {"call", "c", "calls"}:
                continue
            strike = finite_number(row.get("strike"))
            premium = finite_number(row.get("bid"))
            underlying = stock_price if stock_price is not None else finite_number(row.get("underlying_price"))
            if strike is None or premium is None or premium < min_premium or underlying is None or underlying <= 0:
                continue
            if not include_itm and strike <= underlying:
                continue
            prob_otm = probability_otm(row)
            if min_probability_otm is not None and (prob_otm is None or prob_otm < min_probability_otm):
                continue
            premium_yield = premium / underlying
            called_return = (strike - underlying + premium) / underlying
            annualized_premium_yield = annualize(premium_yield, days)
            risk_adjusted_yield = annualized_premium_yield * prob_otm if annualized_premium_yield is not None and prob_otm is not None else None
            output.append({
                "root": row.get("root", ticker),
                "expiration": expiration,
                "strategy": "covered_call",
                "strike": strike,
                "premium": premium,
                "called_price": strike,
                "breakeven": underlying - premium,
                "premium_yield": premium_yield,
                "annualized_premium_yield": annualized_premium_yield,
                "called_return": called_return,
                "annualized_called_return": annualize(called_return, days),
                "probability_otm": prob_otm,
                "risk_adjusted_yield": risk_adjusted_yield,
                "dte": days,
                "underlying_price": underlying,
                "delta": finite_number(row.get("delta")),
                "implied_vol": finite_number(row.get("implied_vol")),
                "theta": finite_number(row.get("theta")),
                "vega": finite_number(row.get("vega")),
                "rho": finite_number(row.get("rho")),
                "epsilon": finite_number(row.get("epsilon")),
                "lambda": finite_number(row.get("lambda")),
                "timestamp": row.get("timestamp"),
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_covered_calls(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_premium: float = 0.01,
    min_probability_otm: float | None = None,
    include_itm: bool = False,
    top_n: int | None = 25,
    rank_by: RankBy = "risk_adjusted_yield",
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
        ticker=ticker, expiration=expiration, client=client, log=log, right="call", today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, max_dte=max_dte, strike_range=strike_range,
        min_time=min_time, use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=min_dte, max_dte=max_dte)
    covered_calls = _build_rows(ticker, chain, today=today, stock_price=stock_price, include_itm=include_itm, min_premium=min_premium, min_probability_otm=min_probability_otm)
    return sort_and_limit(covered_calls, rank_by=rank_by, tie_breakers=["premium", "probability_otm"], descending=[True, True, True], top_n=top_n)


find_best_covered_calls = get_best_covered_calls

__all__ = ["get_best_covered_calls", "find_best_covered_calls"]
