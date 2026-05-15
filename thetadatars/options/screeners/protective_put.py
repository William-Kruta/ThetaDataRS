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
from ._common import GreekSource, filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import dte, empty_frame, grouped_by_expiration, normalize_rows, sort_and_limit
from ._typed import (
    BatchResult,
    BatchStats,
    CircuitBreakerPolicy,
    PlanCost,
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

RankBy = Literal["protection_efficiency", "protected_floor_percent", "hedge_cost_percent", "delta"]

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "strike", "premium", "hedge_cost",
    "hedge_cost_percent", "protected_floor", "protected_floor_percent",
    "max_loss_percent", "protection_efficiency", "dte", "underlying_price",
    "delta", "implied_vol", "theta", "vega", "timestamp",
]
_RANK_BY_VALUES = {
    "protection_efficiency",
    "protected_floor_percent",
    "hedge_cost_percent",
    "delta",
}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}


@dataclass(frozen=True, slots=True)
class ProtectivePutRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    min_dte: int | None = None
    max_dte: int | None = None
    max_hedge_cost_percent: float | None = None
    top_n: int | None = 25
    rank_by: RankBy = "protection_efficiency"
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

    def for_ticker(self, ticker: str) -> "ProtectivePutRequest":
        return replace(self, ticker=ticker)


def _invalid_request(
    message: str,
    *,
    ticker: str | None,
    params: dict[str, object],
    endpoint: str = "screen_protective_puts",
) -> InvalidRequestError:
    return InvalidRequestError(
        message,
        ticker=ticker,
        endpoint=endpoint,
        params=params,
        retryable=False,
        user_message=message,
    )


def _request_params(request: ProtectivePutRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
        "max_hedge_cost_percent": request.max_hedge_cost_percent,
        "strike_range": request.strike_range,
        "top_n": request.top_n,
        "rank_by": request.rank_by,
        "greeks_source": request.greeks_source,
        "cache_policy": request.cache_policy,
        "allow_full_chain": request.allow_full_chain,
    }


def _validate_request(
    request: ProtectivePutRequest,
    *,
    endpoint: str = "screen_protective_puts",
) -> tuple[dt.date | str, tuple[ScreenerWarning, ...]]:
    params = _request_params(request)
    if not request.ticker:
        raise _invalid_request("ticker is required", ticker=None, params=params, endpoint=endpoint)
    try:
        expiration = parse_expiration(request.expiration)
    except ValueError as exc:
        raise _invalid_request(
            "expiration must be '*' or a YYYY-MM-DD date",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
        ) from exc
    if request.rank_by not in _RANK_BY_VALUES:
        raise _invalid_request(
            f"rank_by must be one of: {', '.join(sorted(_RANK_BY_VALUES))}",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
        )
    if request.greeks_source not in _GREEKS_SOURCE_VALUES:
        raise _invalid_request(
            "greeks_source must be one of: auto, thetadata, local, none",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
        )
    if request.cache_policy not in _CACHE_POLICY_VALUES:
        raise _invalid_request(
            "cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
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
            endpoint=endpoint,
        )
    if request.top_n is not None and request.top_n < 0:
        raise _invalid_request(
            "top_n cannot be negative",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
        )

    warnings = []
    if expiration == "*" and request.max_dte is None:
        message = "expiration='*' without max_dte can fetch a full option chain."
        if not request.allow_full_chain:
            raise _invalid_request(
                f"{message} Set max_dte or allow_full_chain=True.",
                ticker=request.ticker,
                params=params,
                endpoint=endpoint,
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
    request: ProtectivePutRequest,
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


def _planned_cost(request: ProtectivePutRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: ProtectivePutRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_protective_puts(
    request: ProtectivePutRequest,
    *,
    conn=None,
) -> ScreenerPlan:
    """Explain the expected protective-put screening cost before fetching data."""
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
            strategy="protective_put",
            expected_endpoint=endpoint,
            upstream_calls=upstream_calls,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cost=_planned_cost(request, expiration),
            local_computation=_planned_local_computation(request, expiration),
            warnings=warnings,
            cache_coverage=(coverage,),
        )
    finally:
        if own_conn:
            conn.close()


def _build_rows(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    stock_price: float | None,
    max_hedge_cost_percent: float | None,
) -> pl.DataFrame:
    output = []
    for expiration, rows in grouped_by_expiration(normalize_rows(chain)).items():
        days = dte(expiration, today)
        for row in rows:
            if right_name(row.get("right")) != "put":
                continue
            strike = finite_number(row.get("strike"))
            premium = finite_number(row.get("ask"))
            underlying = stock_price if stock_price is not None else finite_number(row.get("underlying_price"))
            if strike is None or premium is None or premium <= 0 or underlying is None or underlying <= 0:
                continue
            hedge_cost_percent = premium / underlying
            if max_hedge_cost_percent is not None and hedge_cost_percent > max_hedge_cost_percent:
                continue
            protected_floor = strike - premium
            protected_floor_percent = protected_floor / underlying
            max_loss_percent = 1 - protected_floor_percent
            protection_efficiency = protected_floor_percent / hedge_cost_percent if hedge_cost_percent > 0 else None
            output.append({
                "root": row.get("root", ticker),
                "expiration": expiration,
                "strategy": "protective_put",
                "strike": strike,
                "premium": premium,
                "hedge_cost": premium * 100,
                "hedge_cost_percent": hedge_cost_percent,
                "protected_floor": protected_floor,
                "protected_floor_percent": protected_floor_percent,
                "max_loss_percent": max_loss_percent,
                "protection_efficiency": protection_efficiency,
                "dte": days,
                "underlying_price": underlying,
                "delta": finite_number(row.get("delta")),
                "implied_vol": finite_number(row.get("implied_vol")),
                "theta": finite_number(row.get("theta")),
                "vega": finite_number(row.get("vega")),
                "timestamp": row.get("timestamp"),
            })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


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
        data=empty_frame(_OUTPUT_COLUMNS),
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


def _sort_protective_puts(puts: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [
        column
        for column in ["protected_floor_percent", "hedge_cost_percent"]
        if column != rank_by
    ]
    descending = [rank_by != "hedge_cost_percent"]
    descending.extend(column != "hedge_cost_percent" for column in tie_breakers)
    return sort_and_limit(
        puts,
        rank_by=rank_by,
        tie_breakers=tie_breakers,
        descending=descending,
        top_n=top_n,
    )


def _execute_protective_put_request(
    request: ProtectivePutRequest,
    *,
    client: Client | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_protective_puts(request, conn=conn)

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
            endpoint="screen_protective_puts",
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

    puts = _build_rows(
        ticker,
        chain,
        today=today,
        stock_price=request.stock_price,
        max_hedge_cost_percent=request.max_hedge_cost_percent,
    )
    candidate_rows = len(puts)
    puts = _sort_protective_puts(puts, rank_by=request.rank_by, top_n=request.top_n)

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


def screen_protective_puts(
    request: ProtectivePutRequest,
    client: Client | None = None,
    *,
    conn=None,
) -> ScreenerResult:
    """Screen protective puts from a typed request and return diagnostics."""
    return _execute_protective_put_request(request, client=client, conn=conn)


def _attempt_screen(
    request: ProtectivePutRequest,
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
            result = screen_protective_puts(request, client=client, conn=conn)
            if (
                timeout_policy is not None
                and timeout_policy.per_ticker_seconds is not None
                and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds
            ):
                raise ThetaTimeoutError(
                    "Protective-put screening exceeded the per-ticker timeout.",
                    ticker=request.ticker,
                    endpoint="screen_protective_puts",
                    params=_request_params(request),
                    retryable=True,
                    user_message="Protective-put screening exceeded the per-ticker timeout.",
                )
            return result
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ThetaDataError)
                else classify_thetadata_error(
                    exc,
                    ticker=request.ticker,
                    endpoint="screen_protective_puts",
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


def screen_protective_put_watchlist(
    tickers: list[str],
    request: ProtectivePutRequest,
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
    """Screen a watchlist for protective puts with per-ticker failures."""
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_protective_puts")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
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
                endpoint="screen_protective_puts",
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


def _warm_protective_put_request(
    request: ProtectivePutRequest,
    *,
    client: Client | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request, endpoint="warm_protective_put_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_protective_puts(request, conn=conn)

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
            "Protective-put cache warmup exceeded the per-ticker timeout.",
            ticker=ticker,
            endpoint="warm_protective_put_cache",
            params=_request_params(request),
            retryable=True,
            user_message="Protective-put cache warmup exceeded the per-ticker timeout.",
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
    request: ProtectivePutRequest,
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
            return _warm_protective_put_request(
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
                    endpoint="warm_protective_put_cache",
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


def warm_protective_put_cache(
    tickers: list[str],
    request: ProtectivePutRequest,
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
    """Warm cache inputs for protective-put screening without ranking candidates."""
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
            return ticker, _circuit_open_failure(ticker, endpoint="warm_protective_put_cache")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
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
                endpoint="warm_protective_put_cache",
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


def get_best_protective_puts(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    max_hedge_cost_percent: float | None = None,
    top_n: int | None = 25,
    rank_by: RankBy = "protection_efficiency",
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
    """Find protective puts ranked by protection efficiency, floor, cost, or delta."""
    request = ProtectivePutRequest(
        ticker=ticker,
        expiration=expiration,
        min_dte=min_dte,
        max_dte=max_dte,
        max_hedge_cost_percent=max_hedge_cost_percent,
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
    return _execute_protective_put_request(request, client=client, conn=conn).data


find_best_protective_puts = get_best_protective_puts

__all__ = [
    "ProtectivePutRequest",
    "get_best_protective_puts",
    "find_best_protective_puts",
    "plan_protective_puts",
    "screen_protective_puts",
    "screen_protective_put_watchlist",
    "warm_protective_put_cache",
]
