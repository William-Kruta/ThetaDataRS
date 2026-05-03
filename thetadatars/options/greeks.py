import datetime as dt
import math
from typing import Literal

import polars as pl

Right = Literal["call", "put"]


def _parse_date(value: dt.date | dt.datetime | str) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def _right_name(value: object) -> Right | None:
    right = str(value).strip().lower()
    if right in {"c", "call", "calls"}:
        return "call"
    if right in {"p", "put", "puts"}:
        return "put"
    return None


def _number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _mid_price(row: dict) -> float | None:
    bid = _number(row.get("bid"))
    ask = _number(row.get("ask"))
    if bid is not None and ask is not None and bid >= 0 and ask >= bid:
        return (bid + ask) / 2
    if bid is not None and bid > 0:
        return bid
    if ask is not None and ask > 0:
        return ask
    return None


def _dividend_yield(annual_dividend: float | None, stock_price: float | None) -> float:
    if annual_dividend is None:
        return 0.0
    dividend = max(float(annual_dividend), 0.0)
    if dividend <= 1:
        return dividend
    if stock_price is None or stock_price <= 0:
        return 0.0
    return dividend / stock_price


def american_option_price(
    stock_price: float,
    strike: float,
    time_to_expiration: float,
    volatility: float,
    *,
    right: Right,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    steps: int = 150,
) -> float | None:
    """Price an American option with a Cox-Ross-Rubinstein binomial tree."""
    if stock_price <= 0 or strike <= 0:
        return None
    if time_to_expiration <= 0:
        return (
            max(stock_price - strike, 0.0)
            if right == "call"
            else max(strike - stock_price, 0.0)
        )

    volatility = max(volatility, 1e-6)
    steps = max(int(steps), 1)
    step_time = time_to_expiration / steps
    up = math.exp(volatility * math.sqrt(step_time))
    down = 1 / up
    growth = math.exp((risk_free_rate - dividend_yield) * step_time)
    denominator = up - down
    if denominator == 0:
        return None

    probability = (growth - down) / denominator
    probability = min(max(probability, 0.0), 1.0)

    discount = math.exp(-risk_free_rate * step_time)
    values = []
    for i in range(steps + 1):
        node_price = stock_price * (up**i) * (down ** (steps - i))
        payoff = node_price - strike if right == "call" else strike - node_price
        values.append(max(payoff, 0.0))

    for step in range(steps - 1, -1, -1):
        next_values = []
        for i in range(step + 1):
            node_price = stock_price * (up**i) * (down ** (step - i))
            continuation = discount * (
                probability * values[i + 1] + (1 - probability) * values[i]
            )
            exercise = node_price - strike if right == "call" else strike - node_price
            next_values.append(max(continuation, exercise, 0.0))
        values = next_values

    return values[0]


def american_option_implied_volatility(
    market_price: float,
    stock_price: float,
    strike: float,
    time_to_expiration: float,
    *,
    right: Right,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    steps: int = 150,
    tolerance: float = 1e-4,
    max_iterations: int = 80,
) -> tuple[float | None, float | None]:
    intrinsic = (
        max(stock_price - strike, 0.0)
        if right == "call"
        else max(strike - stock_price, 0.0)
    )
    if market_price <= 0 or market_price < intrinsic - tolerance:
        return None, None

    low = 1e-4
    high = 5.0
    low_price = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        low,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    high_price = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        high,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    if low_price is None or high_price is None:
        return None, None
    if market_price <= low_price:
        return low, abs(low_price - market_price)
    while high_price < market_price and high < 10:
        high *= 2
        high_price = american_option_price(
            stock_price,
            strike,
            time_to_expiration,
            high,
            right=right,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            steps=steps,
        )
        if high_price is None:
            return None, None
    if high_price < market_price:
        return None, high_price - market_price

    best_volatility = None
    best_error = None
    for _ in range(max_iterations):
        midpoint = (low + high) / 2
        price = american_option_price(
            stock_price,
            strike,
            time_to_expiration,
            midpoint,
            right=right,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            steps=steps,
        )
        if price is None:
            return None, None
        error = price - market_price
        best_volatility = midpoint
        best_error = error
        if abs(error) <= tolerance:
            break
        if error > 0:
            high = midpoint
        else:
            low = midpoint

    return best_volatility, abs(best_error) if best_error is not None else None


def american_option_first_order_greeks(
    stock_price: float,
    strike: float,
    time_to_expiration: float,
    volatility: float,
    *,
    right: Right,
    risk_free_rate: float = 0.0,
    dividend_yield: float = 0.0,
    steps: int = 150,
) -> dict[str, float | None]:
    price = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    if price is None:
        return {
            "delta": None,
            "theta": None,
            "vega": None,
            "rho": None,
            "epsilon": None,
            "lambda": None,
        }

    stock_bump = max(stock_price * 0.01, 0.01)
    vol_bump = 0.01
    rate_bump = 0.01
    day = 1 / 365

    price_up = american_option_price(
        stock_price + stock_bump,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    price_down = american_option_price(
        max(stock_price - stock_bump, 0.01),
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    theta_price = american_option_price(
        stock_price,
        strike,
        max(time_to_expiration - day, 0.0),
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    vega_up = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility + vol_bump,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    vega_down = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        max(volatility - vol_bump, 1e-4),
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    rho_up = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate + rate_bump,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    rho_down = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate - rate_bump,
        dividend_yield=dividend_yield,
        steps=steps,
    )
    epsilon_up = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=dividend_yield + rate_bump,
        steps=steps,
    )
    epsilon_down = american_option_price(
        stock_price,
        strike,
        time_to_expiration,
        volatility,
        right=right,
        risk_free_rate=risk_free_rate,
        dividend_yield=max(dividend_yield - rate_bump, 0.0),
        steps=steps,
    )

    delta = None
    if price_up is not None and price_down is not None:
        delta = (price_up - price_down) / (
            (stock_price + stock_bump) - max(stock_price - stock_bump, 0.01)
        )

    theta = theta_price - price if theta_price is not None else None
    vega = (
        (vega_up - vega_down) / 2
        if vega_up is not None and vega_down is not None
        else None
    )
    rho = (
        (rho_up - rho_down) / 2 if rho_up is not None and rho_down is not None else None
    )
    epsilon = (
        (epsilon_up - epsilon_down) / 2
        if epsilon_up is not None and epsilon_down is not None
        else None
    )
    leverage = (
        (delta * stock_price / price) if delta is not None and price > 0 else None
    )

    return {
        "delta": delta,
        "theta": theta,
        "vega": vega,
        "rho": rho,
        "epsilon": epsilon,
        "lambda": leverage,
    }


def infer_underlying_price_from_option_chain(
    chain: pl.DataFrame,
    *,
    valuation_date: dt.date | str | None = None,
    risk_free_rate: float = 0.0,
    annual_dividend: float | None = None,
) -> float | None:
    """Estimate spot from call/put mids at matching strikes using parity."""
    if chain.is_empty():
        return None

    today = _parse_date(valuation_date or dt.date.today())
    grouped: dict[tuple[dt.date, float], dict[str, float]] = {}
    for row in chain.to_dicts():
        right = _right_name(row.get("right"))
        strike = _number(row.get("strike"))
        expiration = row.get("expiration")
        mid = _mid_price(row)
        if right is None or strike is None or expiration is None or mid is None:
            continue
        expiration_date = _parse_date(expiration)
        grouped.setdefault((expiration_date, strike), {})[right] = mid

    estimates = []
    for (expiration, strike), prices in grouped.items():
        call_mid = prices.get("call")
        put_mid = prices.get("put")
        if call_mid is None or put_mid is None:
            continue
        time_to_expiration = max((expiration - today).days / 365, 1 / 365)
        dividend_yield = _dividend_yield(annual_dividend, None)
        estimate = (
            call_mid - put_mid + strike * math.exp(-risk_free_rate * time_to_expiration)
        )
        estimate *= math.exp(dividend_yield * time_to_expiration)
        if estimate > 0:
            estimates.append(estimate)

    if not estimates:
        return None
    estimates.sort()
    return estimates[len(estimates) // 2]


def calculate_american_first_order_greeks(
    chain: pl.DataFrame,
    *,
    stock_price: float | None = None,
    valuation_date: dt.date | str | None = None,
    risk_free_rate: float = 0.0,
    annual_dividend: float | None = None,
    steps: int = 150,
) -> pl.DataFrame:
    """Add American-option implied volatility and first-order Greeks to quotes."""
    if chain.is_empty():
        return chain

    today = _parse_date(valuation_date or dt.date.today())
    if stock_price is None:
        stock_price = infer_underlying_price_from_option_chain(
            chain,
            valuation_date=today,
            risk_free_rate=risk_free_rate,
            annual_dividend=annual_dividend,
        )

    rows = []
    for row in chain.to_dicts():
        right = _right_name(row.get("right"))
        strike = _number(row.get("strike"))
        expiration = row.get("expiration")
        market_price = _mid_price(row)

        row["underlying_price"] = stock_price
        row["underlying_timestamp"] = None
        row["delta"] = None
        row["theta"] = None
        row["vega"] = None
        row["rho"] = None
        row["epsilon"] = None
        row["lambda"] = None
        row["implied_vol"] = None
        row["iv_error"] = None

        if (
            right is None
            or strike is None
            or expiration is None
            or market_price is None
        ):
            rows.append(row)
            continue
        if stock_price is None or stock_price <= 0:
            rows.append(row)
            continue

        expiration_date = _parse_date(expiration)
        time_to_expiration = max((expiration_date - today).days / 365, 1 / 365)
        dividend_yield = _dividend_yield(annual_dividend, stock_price)
        implied_vol, iv_error = american_option_implied_volatility(
            market_price,
            stock_price,
            strike,
            time_to_expiration,
            right=right,
            risk_free_rate=risk_free_rate,
            dividend_yield=dividend_yield,
            steps=steps,
        )
        row["implied_vol"] = implied_vol
        row["iv_error"] = iv_error

        if implied_vol is not None:
            greeks = american_option_first_order_greeks(
                stock_price,
                strike,
                time_to_expiration,
                implied_vol,
                right=right,
                risk_free_rate=risk_free_rate,
                dividend_yield=dividend_yield,
                steps=steps,
            )
            row.update(greeks)

        rows.append(row)

    return pl.DataFrame(rows)


__all__ = [
    "american_option_price",
    "american_option_implied_volatility",
    "american_option_first_order_greeks",
    "infer_underlying_price_from_option_chain",
    "calculate_american_first_order_greeks",
]
