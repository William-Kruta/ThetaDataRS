import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import dte, empty_frame, normalize_rows, sort_and_limit

log = logging.getLogger(__name__)

Right = Literal["call", "put"]
RankBy = Literal["return_if_assigned", "near_credit_to_debit", "net_delta", "debit", "calendar_days"]

_OUTPUT_COLUMNS = [
    "root", "strategy", "right", "near_expiration", "far_expiration",
    "long_strike", "short_strike", "debit", "max_loss", "near_credit",
    "far_ask", "return_if_assigned", "near_dte", "far_dte", "calendar_days",
    "near_credit_to_debit", "net_delta", "net_theta", "net_vega",
    "underlying_price",
]


def _build_rows(ticker: str, near_chain: pl.DataFrame, far_chain: pl.DataFrame, *, today: dt.date, right: Right, min_debit: float) -> pl.DataFrame:
    near_rows = [r for r in normalize_rows(near_chain) if right_name(r.get("right")) == right]
    far_rows = [r for r in normalize_rows(far_chain) if right_name(r.get("right")) == right]
    output = []
    for far_leg in far_rows:
        long_strike = finite_number(far_leg.get("strike"))
        far_ask = finite_number(far_leg.get("ask"))
        if long_strike is None or far_ask is None:
            continue
        for near_leg in near_rows:
            short_strike = finite_number(near_leg.get("strike"))
            near_bid = finite_number(near_leg.get("bid"))
            if short_strike is None or near_bid is None:
                continue
            if right == "call" and short_strike <= long_strike:
                continue
            if right == "put" and short_strike >= long_strike:
                continue
            debit = far_ask - near_bid
            if debit < min_debit:
                continue
            near_exp = parse_expiration(near_leg.get("expiration"))
            far_exp = parse_expiration(far_leg.get("expiration"))
            assignment_value = abs(short_strike - long_strike)
            return_if_assigned = (assignment_value - debit) / debit if debit > 0 else None
            output.append({
                "root": far_leg.get("root", ticker),
                "strategy": f"{right}_diagonal",
                "right": right,
                "near_expiration": near_exp,
                "far_expiration": far_exp,
                "long_strike": long_strike,
                "short_strike": short_strike,
                "debit": debit,
                "max_loss": debit * 100,
                "near_credit": near_bid,
                "far_ask": far_ask,
                "return_if_assigned": return_if_assigned,
                "near_dte": dte(near_exp, today),
                "far_dte": dte(far_exp, today),
                "calendar_days": dte(far_exp, today) - dte(near_exp, today),
                "near_credit_to_debit": near_bid / debit if debit > 0 else None,
                "net_delta": (finite_number(far_leg.get("delta")) or 0) - (finite_number(near_leg.get("delta")) or 0),
                "net_theta": (finite_number(far_leg.get("theta")) or 0) - (finite_number(near_leg.get("theta")) or 0),
                "net_vega": (finite_number(far_leg.get("vega")) or 0) - (finite_number(near_leg.get("vega")) or 0),
                "underlying_price": finite_number(far_leg.get("underlying_price")),
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_diagonal_spreads(
    ticker: str,
    near_expiration: dt.date | str,
    far_expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "call",
    min_debit: float = 0.01,
    top_n: int | None = 25,
    rank_by: RankBy = "return_if_assigned",
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
    near_expiration = parse_expiration(near_expiration)
    far_expiration = parse_expiration(far_expiration)
    today = dt.date.today()
    if client is None:
        client = create_client()
    kwargs = dict(
        ticker=ticker, client=client, log=log, right=right, today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, strike_range=strike_range, min_time=min_time,
        use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    near_chain = get_first_order_chain(expiration=near_expiration, **kwargs)
    far_chain = get_first_order_chain(expiration=far_expiration, **kwargs)
    diagonals = _build_rows(ticker, near_chain, far_chain, today=today, right=right, min_debit=min_debit)
    return sort_and_limit(diagonals, rank_by=rank_by, tie_breakers=["near_credit_to_debit", "net_delta"], descending=[True, True, True], top_n=top_n)


find_best_diagonal_spreads = get_best_diagonal_spreads

__all__ = ["get_best_diagonal_spreads", "find_best_diagonal_spreads"]
