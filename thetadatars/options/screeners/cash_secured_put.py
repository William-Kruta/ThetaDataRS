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

RankBy = Literal[
    "annualized_risk_adjusted_return",
    "annualized_return_on_risk",
    "annualized_return_on_collateral",
    "risk_adjusted_return",
    "return_on_risk",
    "return_on_collateral",
    "premium",
    "probability_otm",
    "discount_to_underlying",
]

_OUTPUT_COLUMNS = [
    "root",
    "expiration",
    "strategy",
    "strike",
    "premium",
    "ask",
    "mid",
    "cash_collateral",
    "net_cash_at_risk",
    "return_on_collateral",
    "annualized_return_on_collateral",
    "return_on_risk",
    "annualized_return_on_risk",
    "probability_otm",
    "risk_adjusted_return",
    "annualized_risk_adjusted_return",
    "breakeven",
    "discount_to_underlying",
    "dte",
    "underlying_price",
    "delta",
    "implied_vol",
    "theta",
    "vega",
    "rho",
    "epsilon",
    "lambda",
    "timestamp",
]


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})


def _mid_price(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def _passes_delta_filter(
    delta: float | None,
    min_delta: float | None,
    max_delta: float | None,
) -> bool:
    if delta is None:
        return min_delta is None and max_delta is None
    abs_delta = abs(delta)
    if min_delta is not None and abs_delta < min_delta:
        return False
    if max_delta is not None and abs_delta > max_delta:
        return False
    return True


def _latest_underlying_price(rows: list[dict]) -> float | None:
    for row in rows:
        price = finite_number(row.get("underlying_price"))
        if price is not None and price > 0:
            return price
    return None


def _build_cash_secured_put_row(row: dict, *, ticker: str, expiration: dt.date, dte: int) -> dict | None:
    strike = finite_number(row.get("strike"))
    premium = finite_number(row.get("bid"))
    ask = finite_number(row.get("ask"))
    if strike is None or premium is None or strike <= 0 or premium <= 0:
        return None

    net_cash_at_risk = strike - premium
    if net_cash_at_risk <= 0:
        return None

    delta = finite_number(row.get("delta"))
    underlying_price = finite_number(row.get("underlying_price"))
    probability_otm = probability_otm_from_delta(delta)
    return_on_collateral = premium / strike
    return_on_risk = premium / net_cash_at_risk
    annualized_return_on_collateral = return_on_collateral * (365 / dte) if dte > 0 else None
    annualized_return_on_risk = return_on_risk * (365 / dte) if dte > 0 else None
    risk_adjusted_return = (
        return_on_risk * probability_otm if probability_otm is not None else None
    )
    annualized_risk_adjusted_return = (
        annualized_return_on_risk * probability_otm
        if annualized_return_on_risk is not None and probability_otm is not None
        else None
    )
    discount_to_underlying = (
        (underlying_price - strike) / underlying_price
        if underlying_price is not None and underlying_price > 0
        else None
    )

    return {
        "root": row.get("root", ticker),
        "expiration": expiration,
        "strategy": "cash_secured_put",
        "strike": strike,
        "premium": premium,
        "ask": ask,
        "mid": _mid_price(premium, ask),
        "cash_collateral": strike * 100,
        "net_cash_at_risk": net_cash_at_risk * 100,
        "return_on_collateral": return_on_collateral,
        "annualized_return_on_collateral": annualized_return_on_collateral,
        "return_on_risk": return_on_risk,
        "annualized_return_on_risk": annualized_return_on_risk,
        "probability_otm": probability_otm,
        "risk_adjusted_return": risk_adjusted_return,
        "annualized_risk_adjusted_return": annualized_risk_adjusted_return,
        "breakeven": strike - premium,
        "discount_to_underlying": discount_to_underlying,
        "dte": dte,
        "underlying_price": underlying_price,
        "delta": delta,
        "implied_vol": finite_number(row.get("implied_vol")),
        "theta": finite_number(row.get("theta")),
        "vega": finite_number(row.get("vega")),
        "rho": finite_number(row.get("rho")),
        "epsilon": finite_number(row.get("epsilon")),
        "lambda": finite_number(row.get("lambda")),
        "timestamp": row.get("timestamp"),
    }


def _build_cash_secured_puts(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    include_itm: bool,
    min_premium: float,
    min_return_on_collateral: float | None,
    min_return_on_risk: float | None,
    min_probability_otm: float | None,
    min_discount_to_underlying: float | None,
    min_delta: float | None,
    max_delta: float | None,
) -> pl.DataFrame:
    if chain.is_empty():
        return _empty_frame()

    rows = chain.to_dicts()
    fallback_underlying_price = _latest_underlying_price(rows)
    puts = []

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
        for row in expiration_rows:
            if right_name(row.get("right")) != "put":
                continue
            if row.get("underlying_price") is None and fallback_underlying_price is not None:
                row = {**row, "underlying_price": fallback_underlying_price}

            strike = finite_number(row.get("strike"))
            premium = finite_number(row.get("bid"))
            delta = finite_number(row.get("delta"))
            underlying_price = finite_number(row.get("underlying_price"))
            if strike is None or premium is None:
                continue
            if premium < min_premium:
                continue
            if not include_itm and underlying_price is not None and strike >= underlying_price:
                continue
            if not _passes_delta_filter(delta, min_delta, max_delta):
                continue

            put = _build_cash_secured_put_row(row, ticker=ticker, expiration=expiration_date, dte=dte)
            if put is None:
                continue
            if min_return_on_collateral is not None and put["return_on_collateral"] < min_return_on_collateral:
                continue
            if min_return_on_risk is not None and put["return_on_risk"] < min_return_on_risk:
                continue
            if min_probability_otm is not None:
                if put["probability_otm"] is None or put["probability_otm"] < min_probability_otm:
                    continue
            if min_discount_to_underlying is not None:
                if put["discount_to_underlying"] is None or put["discount_to_underlying"] < min_discount_to_underlying:
                    continue
            puts.append(put)

    if not puts:
        return _empty_frame()
    return pl.DataFrame(puts).select(_OUTPUT_COLUMNS)


def get_best_cash_secured_puts(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_premium: float = 0.01,
    min_return_on_collateral: float | None = None,
    min_return_on_risk: float | None = None,
    min_probability_otm: float | None = None,
    min_discount_to_underlying: float | None = None,
    min_delta: float | None = None,
    max_delta: float | None = None,
    include_itm: bool = False,
    top_n: int | None = 25,
    rank_by: RankBy = "annualized_risk_adjusted_return",
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
    """Find cash-secured puts ranked by yield, risk-adjusted yield, or premium."""
    expiration = parse_expiration(expiration)
    today = dt.date.today()

    if expiration != "*":
        dte = (expiration - today).days
        if min_dte is not None and dte < min_dte:
            return _empty_frame()
        if max_dte is not None and dte > max_dte:
            return _empty_frame()

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

    puts = _build_cash_secured_puts(
        ticker,
        chain,
        today=today,
        include_itm=include_itm,
        min_premium=min_premium,
        min_return_on_collateral=min_return_on_collateral,
        min_return_on_risk=min_return_on_risk,
        min_probability_otm=min_probability_otm,
        min_discount_to_underlying=min_discount_to_underlying,
        min_delta=min_delta,
        max_delta=max_delta,
    )
    if puts.is_empty():
        return puts

    puts = puts.sort(
        [rank_by, "premium", "return_on_risk"],
        descending=[True, True, True],
        nulls_last=True,
    )
    if top_n is not None and top_n > 0:
        puts = puts.head(top_n)
    return puts


find_best_cash_secured_puts = get_best_cash_secured_puts


__all__ = ["get_best_cash_secured_puts", "find_best_cash_secured_puts"]
