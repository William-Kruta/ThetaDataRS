import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import dte, empty_frame, normalize_rows, row_by_strike, sort_and_limit

log = logging.getLogger(__name__)

Right = Literal["call", "put", "both"]
RankBy = Literal["theta_edge", "vega_per_debit", "near_credit_to_debit", "debit", "calendar_days"]

_OUTPUT_COLUMNS = [
    "root", "strategy", "right", "near_expiration", "far_expiration", "strike",
    "debit", "max_loss", "near_credit", "far_ask", "near_dte", "far_dte",
    "calendar_days", "near_credit_to_debit", "net_delta", "net_theta",
    "net_vega", "theta_edge", "vega_per_debit", "underlying_price",
]


def _build_rows(ticker: str, near_chain: pl.DataFrame, far_chain: pl.DataFrame, *, today: dt.date, right: Right, min_debit: float) -> pl.DataFrame:
    near_rows = normalize_rows(near_chain)
    far_rows = normalize_rows(far_chain)
    output = []
    for option_right in ("call", "put"):
        if right != "both" and right != option_right:
            continue
        near = row_by_strike([r for r in near_rows if right_name(r.get("right")) == option_right])
        far = row_by_strike([r for r in far_rows if right_name(r.get("right")) == option_right])
        for strike in sorted(set(near) & set(far)):
            near_leg = near[strike]
            far_leg = far[strike]
            near_bid = finite_number(near_leg.get("bid"))
            far_ask = finite_number(far_leg.get("ask"))
            if near_bid is None or far_ask is None:
                continue
            debit = far_ask - near_bid
            if debit < min_debit:
                continue
            near_exp = parse_expiration(near_leg.get("expiration"))
            far_exp = parse_expiration(far_leg.get("expiration"))
            near_days = dte(near_exp, today)
            far_days = dte(far_exp, today)
            net_theta = (finite_number(far_leg.get("theta")) or 0) - (finite_number(near_leg.get("theta")) or 0)
            net_vega = (finite_number(far_leg.get("vega")) or 0) - (finite_number(near_leg.get("vega")) or 0)
            output.append({
                "root": far_leg.get("root", ticker),
                "strategy": f"{option_right}_calendar",
                "right": option_right,
                "near_expiration": near_exp,
                "far_expiration": far_exp,
                "strike": strike,
                "debit": debit,
                "max_loss": debit * 100,
                "near_credit": near_bid,
                "far_ask": far_ask,
                "near_dte": near_days,
                "far_dte": far_days,
                "calendar_days": far_days - near_days,
                "near_credit_to_debit": near_bid / debit if debit > 0 else None,
                "net_delta": (finite_number(far_leg.get("delta")) or 0) - (finite_number(near_leg.get("delta")) or 0),
                "net_theta": net_theta,
                "net_vega": net_vega,
                "theta_edge": -net_theta,
                "vega_per_debit": net_vega / debit if debit > 0 else None,
                "underlying_price": finite_number(far_leg.get("underlying_price")),
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def get_best_calendar_spreads(
    ticker: str,
    near_expiration: dt.date | str,
    far_expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "both",
    min_debit: float = 0.01,
    top_n: int | None = 25,
    rank_by: RankBy = "vega_per_debit",
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
    spreads = _build_rows(ticker, near_chain, far_chain, today=today, right=right, min_debit=min_debit)
    return sort_and_limit(spreads, rank_by=rank_by, tie_breakers=["near_credit_to_debit", "calendar_days"], descending=[True, True, True], top_n=top_n)


find_best_calendar_spreads = get_best_calendar_spreads

__all__ = ["get_best_calendar_spreads", "find_best_calendar_spreads"]
