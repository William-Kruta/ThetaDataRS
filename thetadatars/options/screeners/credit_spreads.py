import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ..snapshot.greeks_first_order import RateType
from ._common import (
    filter_expiration_dte,
    finite_number,
    get_first_order_chain,
    parse_expiration,
    probability_otm_from_delta,
    right_name,
)

log = logging.getLogger(__name__)

Right = Literal["call", "put", "both"]
RankBy = Literal[
    "annualized_return_on_risk",
    "annualized_risk_adjusted_return",
    "return_on_risk",
    "risk_adjusted_return",
    "credit_to_width",
    "credit",
]

_OUTPUT_COLUMNS = [
    "root",
    "expiration",
    "right",
    "spread_type",
    "short_strike",
    "long_strike",
    "width",
    "credit",
    "max_loss",
    "return_on_risk",
    "annualized_return_on_risk",
    "probability_otm",
    "risk_adjusted_return",
    "annualized_risk_adjusted_return",
    "credit_to_width",
    "breakeven",
    "dte",
    "underlying_price",
    "short_bid",
    "long_ask",
    "short_delta",
    "long_delta",
    "short_implied_vol",
    "long_implied_vol",
    "short_timestamp",
    "long_timestamp",
]


def _latest_underlying_price(rows: list[dict]) -> float | None:
    for row in rows:
        price = finite_number(row.get("underlying_price"))
        if price is not None and price > 0:
            return price
    return None


def _passes_delta_filter(
    delta: float | None,
    min_short_delta: float | None,
    max_short_delta: float | None,
) -> bool:
    if delta is None:
        return min_short_delta is None and max_short_delta is None
    abs_delta = abs(delta)
    if min_short_delta is not None and abs_delta < min_short_delta:
        return False
    if max_short_delta is not None and abs_delta > max_short_delta:
        return False
    return True


def _build_spread_row(
    ticker: str,
    expiration: dt.date,
    right: Literal["call", "put"],
    short_leg: dict,
    long_leg: dict,
    dte: int,
) -> dict | None:
    short_strike = finite_number(short_leg.get("strike"))
    long_strike = finite_number(long_leg.get("strike"))
    short_bid = finite_number(short_leg.get("bid"))
    long_ask = finite_number(long_leg.get("ask"))

    if short_strike is None or long_strike is None or short_bid is None or long_ask is None:
        return None

    width = abs(short_strike - long_strike)
    credit = short_bid - long_ask
    max_loss = width - credit
    if width <= 0 or max_loss <= 0:
        return None

    short_delta = finite_number(short_leg.get("delta"))
    probability_otm = probability_otm_from_delta(short_delta)
    return_on_risk = credit / max_loss
    annualized_return_on_risk = return_on_risk * (365 / dte) if dte > 0 else None
    risk_adjusted_return = (
        return_on_risk * probability_otm if probability_otm is not None else None
    )
    annualized_risk_adjusted_return = (
        annualized_return_on_risk * probability_otm
        if annualized_return_on_risk is not None and probability_otm is not None
        else None
    )
    credit_to_width = credit / width
    breakeven = short_strike + credit if right == "call" else short_strike - credit
    spread_type = "bear_call_credit" if right == "call" else "bull_put_credit"

    return {
        "root": short_leg.get("root", ticker),
        "expiration": expiration,
        "right": right,
        "spread_type": spread_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": width,
        "credit": credit,
        "max_loss": max_loss,
        "return_on_risk": return_on_risk,
        "annualized_return_on_risk": annualized_return_on_risk,
        "probability_otm": probability_otm,
        "risk_adjusted_return": risk_adjusted_return,
        "annualized_risk_adjusted_return": annualized_risk_adjusted_return,
        "credit_to_width": credit_to_width,
        "breakeven": breakeven,
        "dte": dte,
        "underlying_price": finite_number(short_leg.get("underlying_price")),
        "short_bid": short_bid,
        "long_ask": long_ask,
        "short_delta": short_delta,
        "long_delta": finite_number(long_leg.get("delta")),
        "short_implied_vol": finite_number(short_leg.get("implied_vol")),
        "long_implied_vol": finite_number(long_leg.get("implied_vol")),
        "short_timestamp": short_leg.get("timestamp"),
        "long_timestamp": long_leg.get("timestamp"),
    }


def _build_credit_spreads(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    include_itm: bool,
    min_width: float | None,
    max_width: float | None,
    min_credit: float,
    min_credit_to_width: float | None,
    min_bid: float,
    max_long_ask: float | None,
    min_short_delta: float | None,
    max_short_delta: float | None,
) -> pl.DataFrame:
    if chain.is_empty():
        return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})

    rows = chain.to_dicts()
    underlying_price = _latest_underlying_price(rows)
    spreads = []

    expirations = sorted({row.get("expiration") for row in rows if row.get("expiration") is not None})
    for expiration in expirations:
        if isinstance(expiration, dt.datetime):
            expiration_date = expiration.date()
        elif isinstance(expiration, dt.date):
            expiration_date = expiration
        else:
            expiration_date = dt.datetime.strptime(str(expiration), "%Y-%m-%d").date()

        dte = (expiration_date - today).days
        expiration_rows = [row for row in rows if row.get("expiration") == expiration]

        for option_right in ("call", "put"):
            right_rows = [
                row for row in expiration_rows
                if right_name(row.get("right")) == option_right
            ]
            if not right_rows:
                continue

            if underlying_price is not None and not include_itm:
                if option_right == "call":
                    right_rows = [
                        row for row in right_rows
                        if (finite_number(row.get("strike")) or 0) > underlying_price
                    ]
                else:
                    right_rows = [
                        row for row in right_rows
                        if (finite_number(row.get("strike")) or 0) < underlying_price
                    ]

            right_rows = sorted(right_rows, key=lambda row: finite_number(row.get("strike")) or 0)
            for short_leg in right_rows:
                short_strike = finite_number(short_leg.get("strike"))
                short_bid = finite_number(short_leg.get("bid"))
                short_delta = finite_number(short_leg.get("delta"))
                if short_strike is None or short_bid is None:
                    continue
                if short_bid < min_bid:
                    continue
                if not _passes_delta_filter(short_delta, min_short_delta, max_short_delta):
                    continue

                for long_leg in right_rows:
                    long_strike = finite_number(long_leg.get("strike"))
                    long_ask = finite_number(long_leg.get("ask"))
                    if long_strike is None or long_ask is None:
                        continue
                    if max_long_ask is not None and long_ask > max_long_ask:
                        continue
                    if option_right == "call" and long_strike <= short_strike:
                        continue
                    if option_right == "put" and long_strike >= short_strike:
                        continue

                    width = abs(short_strike - long_strike)
                    if min_width is not None and width < min_width:
                        continue
                    if max_width is not None and width > max_width:
                        continue

                    spread = _build_spread_row(ticker, expiration_date, option_right, short_leg, long_leg, dte)
                    if spread is None:
                        continue
                    if spread["credit"] < min_credit:
                        continue
                    if min_credit_to_width is not None and spread["credit_to_width"] < min_credit_to_width:
                        continue
                    spreads.append(spread)

    if not spreads:
        return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})

    return pl.DataFrame(spreads).select(_OUTPUT_COLUMNS)


def get_best_credit_spreads(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "both",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    min_credit: float = 0.01,
    min_credit_to_width: float | None = None,
    min_bid: float = 0.01,
    max_long_ask: float | None = None,
    min_short_delta: float | None = None,
    max_short_delta: float | None = None,
    include_itm: bool = False,
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
    """Find the best yielding vertical option credit spreads for an expiration.

    The screener sells the richer near-the-money leg at bid and buys the farther
    out-of-the-money hedge at ask. Results are ranked by annualized return on
    risk by default.
    """
    expiration = parse_expiration(expiration)
    today = dt.date.today()

    if expiration != "*":
        dte = (expiration - today).days
        if min_dte is not None and dte < min_dte:
            return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})
        if max_dte is not None and dte > max_dte:
            return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})

    if client is None:
        client = create_client()

    chain = get_first_order_chain(
        ticker=ticker,
        expiration=expiration,
        client=client,
        log=log,
        right=right,
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

    spreads = _build_credit_spreads(
        ticker,
        chain,
        today=today,
        include_itm=include_itm,
        min_width=min_width,
        max_width=max_width,
        min_credit=min_credit,
        min_credit_to_width=min_credit_to_width,
        min_bid=min_bid,
        max_long_ask=max_long_ask,
        min_short_delta=min_short_delta,
        max_short_delta=max_short_delta,
    )

    if spreads.is_empty():
        return spreads

    spreads = spreads.sort(
        [rank_by, "credit", "return_on_risk"],
        descending=[True, True, True],
        nulls_last=True,
    )
    if top_n is not None and top_n > 0:
        spreads = spreads.head(top_n)
    return spreads


find_best_credit_spreads = get_best_credit_spreads


__all__ = ["get_best_credit_spreads", "find_best_credit_spreads"]
