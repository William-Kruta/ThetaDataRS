import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import filter_expiration_dte, get_first_order_chain, parse_expiration
from ._single_leg import RankBy, build_single_leg_options, empty_frame
from .cash_secured_put import get_best_cash_secured_puts

log = logging.getLogger(__name__)

PutSide = Literal["long", "cash_secured", "cash-secured", "csp"]


def get_best_long_puts(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_premium: float = 0.01,
    max_premium: float | None = None,
    min_delta: float | None = None,
    max_delta: float | None = None,
    min_probability_itm: float | None = None,
    max_breakeven_move_percent: float | None = None,
    min_open_interest: int | None = None,
    include_itm: bool = False,
    top_n: int | None = 25,
    rank_by: RankBy = "delta_per_dollar",
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
    """Find long puts ranked by exposure per dollar or other single-leg metrics."""
    expiration = parse_expiration(expiration)
    today = dt.date.today()

    if expiration != "*":
        dte = (expiration - today).days
        if min_dte is not None and dte < min_dte:
            return empty_frame()
        if max_dte is not None and dte > max_dte:
            return empty_frame()

    if client is None:
        client = create_client()

    chain = get_first_order_chain(
        ticker=ticker,
        expiration=expiration,
        client=client,
        log=log,
        right="put",
        today=today,
        annual_dividend=annual_dividend,
        rate_type=rate_type,
        rate_value=rate_value,
        stock_price=stock_price,
        version=version,
        max_dte=max_dte,
        strike_range=strike_range,
        min_time=min_time,
        use_market_value=use_market_value,
        fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps,
        stale_threshold=stale_threshold,
        conn=conn,
    )
    chain = filter_expiration_dte(
        chain,
        expiration=expiration,
        today=today,
        min_dte=min_dte,
        max_dte=max_dte,
    )

    puts = build_single_leg_options(
        ticker,
        chain,
        option_right="put",
        today=today,
        include_itm=include_itm,
        min_premium=min_premium,
        max_premium=max_premium,
        min_delta=min_delta,
        max_delta=max_delta,
        min_probability_itm=min_probability_itm,
        max_breakeven_move_percent=max_breakeven_move_percent,
        min_open_interest=min_open_interest,
    )
    if puts.is_empty():
        return puts

    puts = puts.sort(
        [rank_by, "probability_itm", "premium"],
        descending=[True, True, False],
        nulls_last=True,
    )
    if top_n is not None and top_n > 0:
        puts = puts.head(top_n)
    return puts


def get_best_puts(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    side: PutSide = "long",
    **kwargs,
) -> pl.DataFrame:
    """Find puts by side.

    ``side="long"`` buys puts and ranks long-option exposure. ``side="cash_secured"``
    sells cash-secured puts and ranks collateral/risk yield.
    """
    if side == "long":
        return get_best_long_puts(ticker, expiration, client=client, **kwargs)
    if side in {"cash_secured", "cash-secured", "csp"}:
        return get_best_cash_secured_puts(ticker, expiration, client=client, **kwargs)
    raise ValueError("side must be one of: 'long', 'cash_secured', 'cash-secured', 'csp'")


find_best_long_puts = get_best_long_puts
find_best_puts = get_best_puts


__all__ = [
    "get_best_puts",
    "get_best_long_puts",
    "get_best_cash_secured_puts",
    "find_best_puts",
    "find_best_long_puts",
]
