import datetime as dt
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

import polars as pl

from ...client import Client, create_client
from ...data.cache import CacheCoverage, CachePolicy, inspect_cache_coverage
from ...data.db import get_connection
from ...errors import (
    InvalidRequestError,
    ThetaDataError,
    TimeoutError as ThetaTimeoutError,
    classify_thetadata_error,
)
from ..snapshot.greeks_first_order import RateType
from ._common import (
    GreekSource,
    filter_expiration_dte,
    finite_number,
    get_first_order_chain,
    parse_expiration,
    probability_otm_from_delta,
    right_name,
)
from ._typed import (
    BatchResult,
    BatchStats,
    CircuitBreakerPolicy,
    RateLimitPolicy,
    RetryPolicy,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
    ScreenerWarning,
    TickerFailure,
    TickerResult,
    TimeoutPolicy,
    WarmCacheResult,
    _CircuitBreaker,
    _RateLimiter,
    _circuit_open_failure,
    _failure_for,
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

_RANK_BY_VALUES = {
    "annualized_risk_adjusted_return",
    "annualized_return_on_risk",
    "annualized_return_on_collateral",
    "risk_adjusted_return",
    "return_on_risk",
    "return_on_collateral",
    "premium",
    "probability_otm",
    "discount_to_underlying",
}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}


@dataclass(frozen=True, slots=True)
class CashSecuredPutRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    min_dte: int | None = None
    max_dte: int | None = None
    min_premium: float = 0.01
    min_return_on_collateral: float | None = None
    min_return_on_risk: float | None = None
    min_probability_otm: float | None = None
    min_discount_to_underlying: float | None = None
    min_delta: float | None = None
    max_delta: float | None = None
    include_itm: bool = False
    top_n: int | None = 25
    rank_by: RankBy = "annualized_risk_adjusted_return"
    annual_dividend: float | None = None
    rate_type: RateType = "sofr"
    rate_value: float | None = None
    stock_price: float | None = None
    version: Literal["latest", "1"] = "latest"
    strike_range: int | None = None
    min_time: dt.time | None = None
    use_market_value: bool = False
    greeks_source: GreekSource = "auto"
    fallback_to_local_greeks: bool = True
    local_greeks_steps: int | Literal["fast", "balanced", "accurate"] = 150
    stale_threshold: dt.timedelta = dt.timedelta(hours=1)
    cache_policy: CachePolicy = "prefer_cache"
    allow_full_chain: bool = False

    def for_ticker(self, ticker: str) -> "CashSecuredPutRequest":
        return replace(self, ticker=ticker)


def _invalid_request(
    message: str,
    *,
    ticker: str | None,
    params: dict[str, object],
) -> InvalidRequestError:
    return InvalidRequestError(
        message,
        ticker=ticker,
        endpoint="screen_cash_secured_puts",
        params=params,
        retryable=False,
        user_message=message,
    )


def _request_params(request: CashSecuredPutRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
        "min_premium": request.min_premium,
        "min_return_on_collateral": request.min_return_on_collateral,
        "min_return_on_risk": request.min_return_on_risk,
        "min_probability_otm": request.min_probability_otm,
        "min_discount_to_underlying": request.min_discount_to_underlying,
        "min_delta": request.min_delta,
        "max_delta": request.max_delta,
        "strike_range": request.strike_range,
        "top_n": request.top_n,
        "rank_by": request.rank_by,
        "greeks_source": request.greeks_source,
        "cache_policy": request.cache_policy,
        "allow_full_chain": request.allow_full_chain,
    }


def _validate_request(
    request: CashSecuredPutRequest,
) -> tuple[dt.date | str, tuple[ScreenerWarning, ...]]:
    params = _request_params(request)
    if not request.ticker:
        raise _invalid_request("ticker is required", ticker=None, params=params)
    try:
        expiration = parse_expiration(request.expiration)
    except ValueError as exc:
        raise _invalid_request(
            "expiration must be '*' or a YYYY-MM-DD date",
            ticker=request.ticker,
            params=params,
        ) from exc
    if request.rank_by not in _RANK_BY_VALUES:
        raise _invalid_request(
            f"rank_by must be one of: {', '.join(sorted(_RANK_BY_VALUES))}",
            ticker=request.ticker,
            params=params,
        )
    if request.greeks_source not in _GREEKS_SOURCE_VALUES:
        raise _invalid_request(
            "greeks_source must be one of: auto, thetadata, local, none",
            ticker=request.ticker,
            params=params,
        )
    if request.cache_policy not in _CACHE_POLICY_VALUES:
        raise _invalid_request(
            "cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache",
            ticker=request.ticker,
            params=params,
        )
    if (
        request.min_dte is not None
        and request.max_dte is not None
        and request.min_dte > request.max_dte
    ):
        raise _invalid_request(
            "min_dte cannot be greater than max_dte",
            ticker=request.ticker,
            params=params,
        )
    if (
        request.min_delta is not None
        and request.max_delta is not None
        and request.min_delta > request.max_delta
    ):
        raise _invalid_request(
            "min_delta cannot be greater than max_delta",
            ticker=request.ticker,
            params=params,
        )
    if request.top_n is not None and request.top_n < 0:
        raise _invalid_request(
            "top_n cannot be negative",
            ticker=request.ticker,
            params=params,
        )

    warnings = []
    if expiration == "*" and request.max_dte is None:
        message = "expiration='*' without max_dte can fetch a full option chain."
        if not request.allow_full_chain:
            raise _invalid_request(
                f"{message} Set max_dte or allow_full_chain=True.",
                ticker=request.ticker,
                params=params,
            )
        warnings.append(ScreenerWarning(code="full_chain", message=message))
    if expiration == "*" and request.strike_range is None:
        warnings.append(
            ScreenerWarning(
                code="unbounded_strikes",
                message="expiration='*' without strike_range can fetch far out-of-the-money contracts.",
            )
        )
    if request.greeks_source == "local":
        warnings.append(
            ScreenerWarning(
                code="local_greeks",
                message="local Greek calculation can be slow across broad chains.",
                severity="info",
            )
        )
    return expiration, tuple(warnings)


def _planned_endpoint_and_params(
    request: CashSecuredPutRequest,
    expiration: dt.date | str,
) -> tuple[str, dict[str, object]]:
    if request.greeks_source in {"local", "none"}:
        return (
            "option_snapshot_quote",
            {
                "expiration": expiration,
                "strike": "*",
                "right": "put",
                "max_dte": request.max_dte,
                "strike_range": request.strike_range,
                "min_time": request.min_time,
            },
        )

    return (
        "option_snapshot_greeks_first_order",
        {
            "expiration": expiration,
            "strike": "*",
            "right": "put",
            "annual_dividend": request.annual_dividend,
            "rate_type": request.rate_type,
            "rate_value": request.rate_value,
            "stock_price": request.stock_price,
            "version": request.version,
            "max_dte": request.max_dte,
            "strike_range": request.strike_range,
            "min_time": request.min_time,
            "use_market_value": request.use_market_value,
        },
    )


def _planned_cost(request: CashSecuredPutRequest, expiration: dt.date | str) -> str:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: CashSecuredPutRequest, expiration: dt.date | str) -> str:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_cash_secured_puts(
    request: CashSecuredPutRequest,
    *,
    conn=None,
) -> ScreenerPlan:
    """Explain the expected cash-secured-put screening cost before fetching data."""
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    endpoint, params = _planned_endpoint_and_params(request, expiration)

    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        if request.cache_policy == "no_cache":
            coverage = CacheCoverage(
                endpoint=endpoint,
                root=ticker,
                params=params,
                covered=False,
                fresh=False,
                reason="no_cache",
            )
            cache_hits = 0
            cache_misses = 1
            upstream_calls: int | None = 1
        else:
            coverage = inspect_cache_coverage(
                conn=conn,
                endpoint=endpoint,
                root=ticker,
                params=params,
                stale_threshold=request.stale_threshold,
            )
            cache_hits = 1 if coverage.covered and coverage.fresh and request.cache_policy != "refresh" else 0
            cache_misses = 0 if cache_hits else 1
            upstream_calls = 0 if cache_hits or request.cache_policy == "cache_only" else 1

        return ScreenerPlan(
            ticker=ticker,
            strategy="cash_secured_put",
            expected_endpoint=endpoint,
            upstream_calls=upstream_calls,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cost=_planned_cost(request, expiration),  # type: ignore[arg-type]
            local_computation=_planned_local_computation(request, expiration),  # type: ignore[arg-type]
            warnings=warnings,
            cache_coverage=(coverage,),
        )
    finally:
        if own_conn:
            conn.close()


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


def _empty_result(
    *,
    ticker: str,
    started_at: float,
    greeks_source: str | None,
    plan: ScreenerPlan,
    cache_policy: CachePolicy,
    warnings: tuple[ScreenerWarning, ...],
) -> ScreenerResult:
    return ScreenerResult(
        data=_empty_frame(),
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=0,
            filtered_rows=0,
            candidate_rows=0,
            returned_rows=0,
            duration_seconds=time.perf_counter() - started_at,
            greeks_source=greeks_source,
            cache_hits=plan.cache_hits,
            cache_misses=plan.cache_misses,
            upstream_calls=plan.upstream_calls,
            cache_policy=cache_policy,
        ),
        warnings=warnings,
    )


def _execute_cash_secured_put_request(
    request: CashSecuredPutRequest,
    *,
    client: Client | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_cash_secured_puts(request, conn=conn)

    if expiration != "*":
        days_to_expiration = (expiration - today).days
        if request.min_dte is not None and days_to_expiration < request.min_dte:
            return _empty_result(
                ticker=ticker,
                started_at=started_at,
                greeks_source=None,
                plan=plan,
                cache_policy=request.cache_policy,
                warnings=warnings,
            )
        if request.max_dte is not None and days_to_expiration > request.max_dte:
            return _empty_result(
                ticker=ticker,
                started_at=started_at,
                greeks_source=None,
                plan=plan,
                cache_policy=request.cache_policy,
                warnings=warnings,
            )

    if client is None:
        client = create_client()

    diagnostics: dict[str, object] = {}
    try:
        chain = get_first_order_chain(
            ticker=ticker,
            expiration=expiration,
            client=client,
            log=log,
            right="put",
            today=today,
            annual_dividend=request.annual_dividend,
            rate_type=request.rate_type,
            rate_value=request.rate_value,
            stock_price=request.stock_price,
            version=request.version,
            max_dte=request.max_dte,
            strike_range=request.strike_range,
            min_time=request.min_time,
            use_market_value=request.use_market_value,
            greeks_source=request.greeks_source,
            fallback_to_local_greeks=request.fallback_to_local_greeks,
            local_greeks_steps=request.local_greeks_steps,
            stale_threshold=request.stale_threshold,
            cache_policy=request.cache_policy,
            diagnostics=diagnostics,
            conn=conn,
        )
    except ThetaDataError:
        raise
    except Exception as exc:
        raise classify_thetadata_error(
            exc,
            ticker=ticker,
            endpoint="screen_cash_secured_puts",
            params=_request_params(request),
        ) from exc

    fetched_rows = len(chain)
    chain = filter_expiration_dte(
        chain,
        expiration=expiration,
        today=today,
        min_dte=request.min_dte,
        max_dte=request.max_dte,
    )
    filtered_rows = len(chain)

    puts = _build_cash_secured_puts(
        ticker,
        chain,
        today=today,
        include_itm=request.include_itm,
        min_premium=request.min_premium,
        min_return_on_collateral=request.min_return_on_collateral,
        min_return_on_risk=request.min_return_on_risk,
        min_probability_otm=request.min_probability_otm,
        min_discount_to_underlying=request.min_discount_to_underlying,
        min_delta=request.min_delta,
        max_delta=request.max_delta,
    )
    candidate_rows = len(puts)

    if not puts.is_empty():
        puts = puts.sort(
            [request.rank_by, "premium", "return_on_risk"],
            descending=[True, True, True],
            nulls_last=True,
        )
        if request.top_n is not None and request.top_n > 0:
            puts = puts.head(request.top_n)

    return ScreenerResult(
        data=puts,
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=fetched_rows,
            filtered_rows=filtered_rows,
            candidate_rows=candidate_rows,
            returned_rows=len(puts),
            duration_seconds=time.perf_counter() - started_at,
            greeks_source=str(diagnostics.get("greeks_source", request.greeks_source)),
            cache_hits=plan.cache_hits,
            cache_misses=plan.cache_misses,
            upstream_calls=plan.upstream_calls,
            cache_policy=request.cache_policy,
        ),
        warnings=warnings,
    )


def screen_cash_secured_puts(
    request: CashSecuredPutRequest,
    client: Client | None = None,
    *,
    conn=None,
) -> ScreenerResult:
    """Screen cash-secured puts from a typed request and return diagnostics."""
    return _execute_cash_secured_put_request(request, client=client, conn=conn)


def _attempt_screen(
    request: CashSecuredPutRequest,
    *,
    client: Client | None,
    retry_policy: RetryPolicy,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            attempt_started = time.perf_counter()
            result = screen_cash_secured_puts(request, client=client, conn=conn)
            if (
                timeout_policy is not None
                and timeout_policy.per_ticker_seconds is not None
                and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds
            ):
                raise ThetaTimeoutError(
                    "Cash-secured-put screening exceeded the per-ticker timeout.",
                    ticker=request.ticker,
                    endpoint="screen_cash_secured_puts",
                    params=_request_params(request),
                    retryable=True,
                    user_message="Cash-secured-put screening exceeded the per-ticker timeout.",
                )
            return result
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ThetaDataError)
                else classify_thetadata_error(
                    exc,
                    ticker=request.ticker,
                    endpoint="screen_cash_secured_puts",
                    params=_request_params(request),
                )
            )
            last_error = error
            if (
                not error.retryable
                or attempts >= max(retry_policy.max_attempts, 1)
                or retry_policy.backoff_seconds <= 0
            ):
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_cash_secured_put_watchlist(
    tickers: list[str],
    request: CashSecuredPutRequest,
    client: Client | None = None,
    *,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    circuit_breaker_policy: CircuitBreakerPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[TickerResult | TickerFailure], None] | None = None,
) -> BatchResult:
    """Screen a watchlist for cash-secured puts with per-ticker failures."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    started_at = time.perf_counter()
    retry_policy = retry_policy or RetryPolicy()
    successes: list[TickerResult] = []
    failures: list[TickerFailure] = []
    frames: list[pl.DataFrame] = []
    tickers = list(tickers)
    rate_limiter = _RateLimiter(rate_limit_policy)
    circuit_breaker = _CircuitBreaker(circuit_breaker_policy)

    def run_one(ticker: str) -> tuple[str, ScreenerResult | TickerFailure]:
        if circuit_breaker.is_open():
            return ticker, _circuit_open_failure(ticker, endpoint="screen_cash_secured_puts")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        task_client = client if client is not None else (client_factory() if client_factory else create_client())
        try:
            result = _attempt_screen(
                ticker_request,
                client=task_client,
                retry_policy=retry_policy,
                timeout_policy=timeout_policy,
            )
            return ticker, result
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(
                ticker,
                exc,
                attempts=max(retry_policy.max_attempts, 1),
                endpoint="screen_cash_secured_puts",
            )

    if concurrency > 1 and len(tickers) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker, item = future.result()
                if isinstance(item, TickerFailure):
                    failures.append(item)
                    if on_progress is not None:
                        on_progress(item)
                    continue
                successes.append(TickerResult(ticker=ticker, stats=item.stats))
                if not item.data.is_empty():
                    frames.append(item.data)
                if on_progress is not None:
                    on_progress(successes[-1])
    else:
        for ticker in tickers:
            _, item = run_one(ticker)
            if isinstance(item, TickerFailure):
                failures.append(item)
                if on_progress is not None:
                    on_progress(item)
                continue
            successes.append(TickerResult(ticker=ticker, stats=item.stats))
            if not item.data.is_empty():
                frames.append(item.data)
            if on_progress is not None:
                on_progress(successes[-1])

    data = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    return BatchResult(
        data=data,
        successes=successes,
        failures=failures,
        stats=BatchStats(
            total=len(tickers),
            succeeded=len(successes),
            failed=len(failures),
            duration_seconds=time.perf_counter() - started_at,
        ),
    )


def _warm_cash_secured_put_request(
    request: CashSecuredPutRequest,
    *,
    client: Client | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_cash_secured_puts(request, conn=conn)

    if client is None:
        client = create_client()

    diagnostics: dict[str, object] = {}
    chain = get_first_order_chain(
        ticker=ticker,
        expiration=expiration,
        client=client,
        log=log,
        right="put",
        today=today,
        annual_dividend=request.annual_dividend,
        rate_type=request.rate_type,
        rate_value=request.rate_value,
        stock_price=request.stock_price,
        version=request.version,
        max_dte=request.max_dte,
        strike_range=request.strike_range,
        min_time=request.min_time,
        use_market_value=request.use_market_value,
        greeks_source=request.greeks_source,
        fallback_to_local_greeks=request.fallback_to_local_greeks,
        local_greeks_steps=request.local_greeks_steps,
        stale_threshold=request.stale_threshold,
        cache_policy=request.cache_policy,
        diagnostics=diagnostics,
        conn=conn,
    )
    duration = time.perf_counter() - started_at
    if (
        timeout_policy is not None
        and timeout_policy.per_ticker_seconds is not None
        and duration > timeout_policy.per_ticker_seconds
    ):
        raise ThetaTimeoutError(
            "Cash-secured-put cache warmup exceeded the per-ticker timeout.",
            ticker=ticker,
            endpoint="warm_cash_secured_put_cache",
            params=_request_params(request),
            retryable=True,
            user_message="Cash-secured-put cache warmup exceeded the per-ticker timeout.",
        )

    filtered = filter_expiration_dte(
        chain,
        expiration=expiration,
        today=today,
        min_dte=request.min_dte,
        max_dte=request.max_dte,
    )
    return ScreenerResult(
        data=pl.DataFrame(),
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=len(chain),
            filtered_rows=len(filtered),
            candidate_rows=0,
            returned_rows=0,
            duration_seconds=duration,
            greeks_source=str(diagnostics.get("greeks_source", request.greeks_source)),
            cache_hits=plan.cache_hits,
            cache_misses=plan.cache_misses,
            upstream_calls=plan.upstream_calls,
            cache_policy=request.cache_policy,
        ),
        warnings=warnings,
    )


def _attempt_warm(
    request: CashSecuredPutRequest,
    *,
    client: Client | None,
    retry_policy: RetryPolicy,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            return _warm_cash_secured_put_request(
                request,
                client=client,
                timeout_policy=timeout_policy,
                conn=conn,
            )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ThetaDataError)
                else classify_thetadata_error(
                    exc,
                    ticker=request.ticker,
                    endpoint="warm_cash_secured_put_cache",
                    params=_request_params(request),
                )
            )
            last_error = error
            if (
                not error.retryable
                or attempts >= max(retry_policy.max_attempts, 1)
                or retry_policy.backoff_seconds <= 0
            ):
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_cash_secured_put_cache(
    tickers: list[str],
    request: CashSecuredPutRequest,
    client: Client | None = None,
    *,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    circuit_breaker_policy: CircuitBreakerPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[TickerResult | TickerFailure], None] | None = None,
    conn=None,
) -> WarmCacheResult:
    """Warm cache inputs for cash-secured-put screening without ranking puts."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    started_at = time.perf_counter()
    retry_policy = retry_policy or RetryPolicy()
    successes: list[TickerResult] = []
    failures: list[TickerFailure] = []
    tickers = list(tickers)
    rate_limiter = _RateLimiter(rate_limit_policy)
    circuit_breaker = _CircuitBreaker(circuit_breaker_policy)

    def run_one(ticker: str) -> tuple[str, ScreenerResult | TickerFailure]:
        if circuit_breaker.is_open():
            return ticker, _circuit_open_failure(ticker, endpoint="screen_cash_secured_puts")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        task_client = client if client is not None else (client_factory() if client_factory else create_client())
        try:
            result = _attempt_warm(
                ticker_request,
                client=task_client,
                retry_policy=retry_policy,
                timeout_policy=timeout_policy,
                conn=conn,
            )
            return ticker, result
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(
                ticker,
                exc,
                attempts=max(retry_policy.max_attempts, 1),
                endpoint="screen_cash_secured_puts",
            )

    if concurrency > 1 and len(tickers) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                ticker, item = future.result()
                if isinstance(item, TickerFailure):
                    failures.append(item)
                    if on_progress is not None:
                        on_progress(item)
                    continue
                successes.append(TickerResult(ticker=ticker, stats=item.stats))
                if on_progress is not None:
                    on_progress(successes[-1])
    else:
        for ticker in tickers:
            _, item = run_one(ticker)
            if isinstance(item, TickerFailure):
                failures.append(item)
                if on_progress is not None:
                    on_progress(item)
                continue
            successes.append(TickerResult(ticker=ticker, stats=item.stats))
            if on_progress is not None:
                on_progress(successes[-1])

    return WarmCacheResult(
        successes=successes,
        failures=failures,
        stats=BatchStats(
            total=len(tickers),
            succeeded=len(successes),
            failed=len(failures),
            duration_seconds=time.perf_counter() - started_at,
        ),
    )


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
    greeks_source: GreekSource = "auto",
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int | Literal["fast", "balanced", "accurate"] = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    conn=None,
) -> pl.DataFrame:
    """Find cash-secured puts ranked by yield, risk-adjusted yield, or premium."""
    request = CashSecuredPutRequest(
        ticker=ticker,
        expiration=expiration,
        min_dte=min_dte,
        max_dte=max_dte,
        min_premium=min_premium,
        min_return_on_collateral=min_return_on_collateral,
        min_return_on_risk=min_return_on_risk,
        min_probability_otm=min_probability_otm,
        min_discount_to_underlying=min_discount_to_underlying,
        min_delta=min_delta,
        max_delta=max_delta,
        include_itm=include_itm,
        top_n=top_n,
        rank_by=rank_by,
        annual_dividend=annual_dividend,
        rate_type=rate_type,
        rate_value=rate_value,
        stock_price=stock_price,
        version=version,
        strike_range=strike_range,
        min_time=min_time,
        use_market_value=use_market_value,
        greeks_source=greeks_source,
        fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps,
        stale_threshold=stale_threshold,
        cache_policy=cache_policy,
        allow_full_chain=True,
    )
    return _execute_cash_secured_put_request(request, client=client, conn=conn).data


find_best_cash_secured_puts = get_best_cash_secured_puts


__all__ = [
    "CashSecuredPutRequest",
    "get_best_cash_secured_puts",
    "find_best_cash_secured_puts",
    "plan_cash_secured_puts",
    "screen_cash_secured_puts",
    "screen_cash_secured_put_watchlist",
    "warm_cash_secured_put_cache",
]
