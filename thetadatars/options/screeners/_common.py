import datetime as dt
import logging
from typing import Literal

import polars as pl

from ...client import Client
from ...data.cache import CachePolicy
from ...errors import SubscriptionError, classify_thetadata_error
from ..greeks import calculate_american_first_order_greeks
from ..list.expirations import get_options_expiration_list
from ..snapshot.greeks_first_order import RateType, get_snapshot_greeks_first_order
from ..snapshot.quote import get_snapshot_quote

Right = Literal["call", "put", "both"]
GreekSource = Literal["auto", "thetadata", "local", "none"]


def parse_expiration(expiration: dt.date | str) -> dt.date | str:
    if isinstance(expiration, dt.datetime):
        return expiration.date()
    if isinstance(expiration, dt.date):
        return expiration
    if expiration == "*":
        return expiration
    return dt.datetime.strptime(expiration, "%Y-%m-%d").date()


def right_name(value: object) -> str:
    right = str(value).strip().lower()
    if right in {"c", "call", "calls"}:
        return "call"
    if right in {"p", "put", "puts"}:
        return "put"
    return right


def finite_number(value: object) -> float | None:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number:
        return None
    return number


def probability_otm_from_delta(delta: float | None) -> float | None:
    if delta is None:
        return None
    return max(0.0, min(1.0, 1 - abs(delta)))


def is_standard_subscription_error(error: Exception) -> bool:
    if isinstance(error, SubscriptionError):
        return True
    message = str(error).lower()
    return (
        "standard subscription" in message
        and "value subscription" in message
    )


def normalize_rate_value(rate_value: float | None) -> float:
    if rate_value is None:
        return 0.0
    rate = float(rate_value)
    return rate / 100 if abs(rate) > 1 else rate


def resolve_local_greeks_steps(value: int | str) -> int:
    if isinstance(value, str):
        presets = {
            "fast": 35,
            "balanced": 75,
            "accurate": 150,
        }
        try:
            return presets[value]
        except KeyError as exc:
            raise ValueError(
                "local_greeks_steps must be an int or one of: fast, balanced, accurate"
            ) from exc
    return max(int(value), 1)


def _quote_chain_with_empty_greeks(
    chain: pl.DataFrame,
    *,
    stock_price: float | None,
) -> pl.DataFrame:
    columns = {
        "underlying_price": stock_price,
        "underlying_timestamp": None,
        "delta": None,
        "theta": None,
        "vega": None,
        "rho": None,
        "epsilon": None,
        "lambda": None,
        "implied_vol": None,
        "iv_error": None,
    }
    expressions = [
        pl.lit(value).alias(name)
        for name, value in columns.items()
        if name not in chain.columns
    ]
    return chain.with_columns(expressions) if expressions else chain


def _expiration_dates_for_request(
    *,
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    today: dt.date,
    max_dte: int | None,
    stale_threshold: dt.timedelta,
    cache_policy: CachePolicy,
    conn=None,
) -> list[dt.date | str]:
    if expiration != "*":
        return [expiration]

    expirations = get_options_expiration_list(
        ticker,
        client=client,
        stale_threshold=stale_threshold,
        conn=conn,
        cache_policy=cache_policy,
    )
    if expirations.is_empty():
        return []

    values = []
    for row in expirations.select("expiration").to_dicts():
        expiration_date = parse_expiration(row["expiration"])
        if not isinstance(expiration_date, dt.date):
            continue
        days_to_expiration = (expiration_date - today).days
        if days_to_expiration < 0:
            continue
        if max_dte is not None and days_to_expiration > max_dte:
            continue
        values.append(expiration_date)
    return values


def _concat_frames(frames: list[pl.DataFrame]) -> pl.DataFrame:
    non_empty = [frame for frame in frames if not frame.is_empty()]
    return pl.concat(non_empty, how="diagonal_relaxed") if non_empty else pl.DataFrame()


def _snapshot_quotes_for_expirations(
    *,
    ticker: str,
    expirations: list[dt.date | str],
    client: Client,
    right: Right,
    max_dte: int | None,
    strike_range: int | None,
    min_time: dt.time | None,
    stale_threshold: dt.timedelta,
    cache_policy: CachePolicy,
    conn=None,
) -> pl.DataFrame:
    frames = []
    for expiration in expirations:
        frames.append(
            get_snapshot_quote(
                ticker=ticker,
                expiration=expiration,
                client=client,
                strike="*",
                right=right,
                max_dte=max_dte,
                strike_range=strike_range,
                min_time=min_time,
                stale_threshold=stale_threshold,
                cache_policy=cache_policy,
                conn=conn,
            )
        )
    return _concat_frames(frames)


def _snapshot_first_order_greeks_for_expirations(
    *,
    ticker: str,
    expirations: list[dt.date | str],
    client: Client,
    right: Right,
    annual_dividend: float | None,
    rate_type: RateType,
    rate_value: float | None,
    stock_price: float | None,
    version: Literal["latest", "1"],
    max_dte: int | None,
    strike_range: int | None,
    min_time: dt.time | None,
    use_market_value: bool,
    stale_threshold: dt.timedelta,
    cache_policy: CachePolicy,
    conn=None,
) -> pl.DataFrame:
    frames = []
    for expiration in expirations:
        frames.append(
            get_snapshot_greeks_first_order(
                ticker=ticker,
                expiration=expiration,
                client=client,
                strike="*",
                right=right,
                annual_dividend=annual_dividend,
                rate_type=rate_type,
                rate_value=rate_value,
                stock_price=stock_price,
                version=version,
                max_dte=max_dte,
                strike_range=strike_range,
                min_time=min_time,
                use_market_value=use_market_value,
                stale_threshold=stale_threshold,
                cache_policy=cache_policy,
                conn=conn,
            )
        )
    return _concat_frames(frames)


def filter_expiration_dte(
    chain: pl.DataFrame,
    *,
    expiration: dt.date | str,
    today: dt.date,
    min_dte: int | None,
    max_dte: int | None,
) -> pl.DataFrame:
    if expiration != "*" or chain.is_empty() or (min_dte is None and max_dte is None):
        return chain

    chain = chain.with_columns(
        (pl.col("expiration") - pl.lit(today)).dt.total_days().alias("_dte")
    )
    if min_dte is not None:
        chain = chain.filter(pl.col("_dte") >= min_dte)
    if max_dte is not None:
        chain = chain.filter(pl.col("_dte") <= max_dte)
    return chain.drop("_dte")


def get_first_order_chain(
    *,
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    log: logging.Logger,
    right: Right,
    today: dt.date,
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
    greeks_source: GreekSource = "auto",
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int | str = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    diagnostics: dict[str, object] | None = None,
    conn=None,
) -> pl.DataFrame:
    diagnostics = diagnostics if diagnostics is not None else {}
    local_steps = resolve_local_greeks_steps(local_greeks_steps)
    expirations = _expiration_dates_for_request(
        ticker=ticker,
        expiration=expiration,
        client=client,
        today=today,
        max_dte=max_dte,
        stale_threshold=stale_threshold,
        cache_policy=cache_policy,
        conn=conn,
    )
    if not expirations:
        diagnostics["greeks_source"] = greeks_source
        return pl.DataFrame()

    if greeks_source in {"local", "none"}:
        quote_chain = _snapshot_quotes_for_expirations(
            ticker=ticker,
            expirations=expirations,
            client=client,
            right=right,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            stale_threshold=stale_threshold,
            cache_policy=cache_policy,
            conn=conn,
        )
        diagnostics["greeks_source"] = greeks_source
        if greeks_source == "none":
            return _quote_chain_with_empty_greeks(quote_chain, stock_price=stock_price)
        return calculate_american_first_order_greeks(
            quote_chain,
            stock_price=stock_price,
            valuation_date=today,
            risk_free_rate=normalize_rate_value(rate_value),
            annual_dividend=annual_dividend,
            steps=local_steps,
        )

    try:
        chain = _snapshot_first_order_greeks_for_expirations(
            ticker=ticker,
            expirations=expirations,
            client=client,
            right=right,
            annual_dividend=annual_dividend,
            rate_type=rate_type,
            rate_value=rate_value,
            stock_price=stock_price,
            version=version,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            use_market_value=use_market_value,
            stale_threshold=stale_threshold,
            cache_policy=cache_policy,
            conn=conn,
        )
        # ThetaData silently returns an empty DataFrame when the greeks endpoint is
        # unavailable for the account tier, rather than raising a SubscriptionError.
        # Treat that the same as a subscription gate and fall through to the local fallback.
        if chain.is_empty() and fallback_to_local_greeks and greeks_source != "thetadata":
            log.info(
                "Falling back to local American option greeks for %s exp=%s because "
                "ThetaData greeks endpoint returned no data (likely subscription gate)",
                ticker,
                expiration,
            )
            quote_chain = _snapshot_quotes_for_expirations(
                ticker=ticker,
                expirations=expirations,
                client=client,
                right=right,
                max_dte=max_dte,
                strike_range=strike_range,
                min_time=min_time,
                stale_threshold=stale_threshold,
                cache_policy=cache_policy,
                conn=conn,
            )
            diagnostics["greeks_source"] = "local"
            return calculate_american_first_order_greeks(
                quote_chain,
                stock_price=stock_price,
                valuation_date=today,
                risk_free_rate=normalize_rate_value(rate_value),
                annual_dividend=annual_dividend,
                steps=local_steps,
            )
        diagnostics["greeks_source"] = "thetadata"
        return chain
    except Exception as exc:
        if (
            greeks_source == "thetadata"
            or not fallback_to_local_greeks
            or not is_standard_subscription_error(exc)
        ):
            raise classify_thetadata_error(
                exc,
                ticker=ticker,
                endpoint="option_snapshot_greeks_first_order",
                params={
                    "expiration": expiration,
                    "strike": "*",
                    "right": right,
                    "max_dte": max_dte,
                    "strike_range": strike_range,
                    "min_time": min_time,
                },
            ) from exc
        log.info(
            "Falling back to local American option greeks for %s exp=%s because "
            "ThetaData first-order greeks require a Standard subscription",
            ticker,
            expiration,
        )
        quote_chain = _snapshot_quotes_for_expirations(
            ticker=ticker,
            expirations=expirations,
            client=client,
            right=right,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            stale_threshold=stale_threshold,
            cache_policy=cache_policy,
            conn=conn,
        )
        diagnostics["greeks_source"] = "local"
        return calculate_american_first_order_greeks(
            quote_chain,
            stock_price=stock_price,
            valuation_date=today,
            risk_free_rate=normalize_rate_value(rate_value),
            annual_dividend=annual_dividend,
            steps=local_steps,
        )
