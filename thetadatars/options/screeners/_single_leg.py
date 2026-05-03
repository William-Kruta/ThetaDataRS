import datetime as dt
from typing import Literal

import polars as pl

from ._common import finite_number, probability_otm_from_delta, right_name

OptionRight = Literal["call", "put"]
RankBy = Literal[
    "delta_per_dollar",
    "vega_per_dollar",
    "theta_efficiency",
    "probability_itm",
    "intrinsic_value",
    "extrinsic_value",
    "implied_vol",
    "premium",
    "lambda",
]

OUTPUT_COLUMNS = [
    "root",
    "expiration",
    "strategy",
    "right",
    "strike",
    "premium",
    "bid",
    "ask",
    "mid",
    "max_loss",
    "breakeven",
    "breakeven_move_percent",
    "intrinsic_value",
    "extrinsic_value",
    "extrinsic_to_premium",
    "moneyness",
    "dte",
    "underlying_price",
    "delta",
    "abs_delta",
    "probability_itm",
    "probability_otm",
    "implied_vol",
    "theta",
    "vega",
    "rho",
    "epsilon",
    "lambda",
    "delta_per_dollar",
    "vega_per_dollar",
    "theta_efficiency",
    "timestamp",
]


def empty_frame() -> pl.DataFrame:
    return pl.DataFrame(schema={column: pl.Null for column in OUTPUT_COLUMNS})


def _mid_price(bid: float | None, ask: float | None) -> float | None:
    if bid is not None and ask is not None and ask >= bid:
        return (bid + ask) / 2
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def _latest_underlying_price(rows: list[dict]) -> float | None:
    for row in rows:
        price = finite_number(row.get("underlying_price"))
        if price is not None and price > 0:
            return price
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


def _build_single_leg_row(
    row: dict,
    *,
    ticker: str,
    option_right: OptionRight,
    expiration: dt.date,
    dte: int,
) -> dict | None:
    strike = finite_number(row.get("strike"))
    bid = finite_number(row.get("bid"))
    ask = finite_number(row.get("ask"))
    premium = ask
    if strike is None or strike <= 0 or premium is None or premium <= 0:
        return None

    underlying_price = finite_number(row.get("underlying_price"))
    delta = finite_number(row.get("delta"))
    abs_delta = abs(delta) if delta is not None else None
    probability_otm = probability_otm_from_delta(delta)
    probability_itm = abs_delta if abs_delta is not None else None
    intrinsic_value = None
    breakeven = strike + premium if option_right == "call" else strike - premium
    breakeven_move_percent = None
    moneyness = None

    if underlying_price is not None and underlying_price > 0:
        moneyness = underlying_price / strike
        if option_right == "call":
            intrinsic_value = max(underlying_price - strike, 0.0)
            breakeven_move_percent = (breakeven - underlying_price) / underlying_price
        else:
            intrinsic_value = max(strike - underlying_price, 0.0)
            breakeven_move_percent = (underlying_price - breakeven) / underlying_price

    extrinsic_value = premium - intrinsic_value if intrinsic_value is not None else None
    extrinsic_to_premium = (
        extrinsic_value / premium if extrinsic_value is not None and premium > 0 else None
    )
    vega = finite_number(row.get("vega"))
    theta = finite_number(row.get("theta"))
    delta_per_dollar = abs_delta / premium if abs_delta is not None else None
    vega_per_dollar = vega / premium if vega is not None else None
    theta_efficiency = (
        abs_delta / abs(theta) if abs_delta is not None and theta is not None and theta != 0 else None
    )

    return {
        "root": row.get("root", ticker),
        "expiration": expiration,
        "strategy": f"long_{option_right}",
        "right": option_right,
        "strike": strike,
        "premium": premium,
        "bid": bid,
        "ask": ask,
        "mid": _mid_price(bid, ask),
        "max_loss": premium * 100,
        "breakeven": breakeven,
        "breakeven_move_percent": breakeven_move_percent,
        "intrinsic_value": intrinsic_value,
        "extrinsic_value": extrinsic_value,
        "extrinsic_to_premium": extrinsic_to_premium,
        "moneyness": moneyness,
        "dte": dte,
        "underlying_price": underlying_price,
        "delta": delta,
        "abs_delta": abs_delta,
        "probability_itm": probability_itm,
        "probability_otm": probability_otm,
        "implied_vol": finite_number(row.get("implied_vol")),
        "theta": theta,
        "vega": vega,
        "rho": finite_number(row.get("rho")),
        "epsilon": finite_number(row.get("epsilon")),
        "lambda": finite_number(row.get("lambda")),
        "delta_per_dollar": delta_per_dollar,
        "vega_per_dollar": vega_per_dollar,
        "theta_efficiency": theta_efficiency,
        "timestamp": row.get("timestamp"),
    }


def build_single_leg_options(
    ticker: str,
    chain: pl.DataFrame,
    *,
    option_right: OptionRight,
    today: dt.date,
    include_itm: bool,
    min_premium: float,
    max_premium: float | None,
    min_delta: float | None,
    max_delta: float | None,
    min_probability_itm: float | None,
    max_breakeven_move_percent: float | None,
    min_open_interest: int | None = None,
) -> pl.DataFrame:
    if chain.is_empty():
        return empty_frame()

    rows = chain.to_dicts()
    fallback_underlying_price = _latest_underlying_price(rows)
    options = []

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
            if right_name(row.get("right")) != option_right:
                continue
            if row.get("underlying_price") is None and fallback_underlying_price is not None:
                row = {**row, "underlying_price": fallback_underlying_price}

            strike = finite_number(row.get("strike"))
            ask = finite_number(row.get("ask"))
            delta = finite_number(row.get("delta"))
            underlying_price = finite_number(row.get("underlying_price"))
            open_interest = finite_number(row.get("open_interest"))
            if strike is None or ask is None:
                continue
            if ask < min_premium:
                continue
            if max_premium is not None and ask > max_premium:
                continue
            if min_open_interest is not None:
                if open_interest is None or open_interest < min_open_interest:
                    continue
            if not _passes_delta_filter(delta, min_delta, max_delta):
                continue
            if not include_itm and underlying_price is not None:
                if option_right == "call" and strike <= underlying_price:
                    continue
                if option_right == "put" and strike >= underlying_price:
                    continue

            option = _build_single_leg_row(
                row,
                ticker=ticker,
                option_right=option_right,
                expiration=expiration_date,
                dte=dte,
            )
            if option is None:
                continue
            if min_probability_itm is not None:
                if option["probability_itm"] is None or option["probability_itm"] < min_probability_itm:
                    continue
            if max_breakeven_move_percent is not None:
                if (
                    option["breakeven_move_percent"] is None
                    or option["breakeven_move_percent"] > max_breakeven_move_percent
                ):
                    continue
            options.append(option)

    if not options:
        return empty_frame()
    return pl.DataFrame(options).select(OUTPUT_COLUMNS)
