import datetime as dt
import logging
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from ._common import GreekSource, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import dte, empty_frame, grouped_by_expiration, normalize_rows, row_by_strike, sort_and_limit
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

Right = Literal["call", "put", "both"]
RankBy = Literal["theta_edge", "vega_per_debit", "near_credit_to_debit", "debit", "calendar_days"]

_RIGHT_VALUES = {"call", "put", "both"}
_RANK_BY_VALUES = {"theta_edge", "vega_per_debit", "near_credit_to_debit", "debit", "calendar_days"}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}

_OUTPUT_COLUMNS = [
    "root", "strategy", "right", "near_expiration", "far_expiration", "strike",
    "debit", "max_loss", "near_credit", "far_ask", "near_dte", "far_dte",
    "calendar_days", "near_credit_to_debit", "net_delta", "net_theta",
    "net_vega", "theta_edge", "vega_per_debit", "underlying_price",
]


@dataclass(frozen=True, slots=True)
class CalendarSpreadRequest:
    ticker: str | None = None
    near_expiration: dt.date | str = "*"
    far_expiration: dt.date | str = "*"
    right: Right = "both"
    min_debit: float = 0.01
    top_n: int | None = 25
    rank_by: RankBy = "vega_per_debit"
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
    max_candidates_per_expiration: int | None = None
    max_candidates_total: int | None = None

    def for_ticker(self, ticker: str) -> "CalendarSpreadRequest":
        return replace(self, ticker=ticker)


def _request_params(request: CalendarSpreadRequest) -> dict[str, object]:
    return {
        "near_expiration": request.near_expiration,
        "far_expiration": request.far_expiration,
        "right": request.right,
        "min_debit": request.min_debit,
        "strike_range": request.strike_range,
        "top_n": request.top_n,
        "rank_by": request.rank_by,
        "greeks_source": request.greeks_source,
        "cache_policy": request.cache_policy,
        "allow_full_chain": request.allow_full_chain,
        "max_candidates_per_expiration": request.max_candidates_per_expiration,
        "max_candidates_total": request.max_candidates_total,
    }


def _invalid_request(message: str, *, ticker: str | None, params: dict[str, object], endpoint: str) -> InvalidRequestError:
    return InvalidRequestError(message, ticker=ticker, endpoint=endpoint, params=params, retryable=False, user_message=message)


def _validate_request(
    request: CalendarSpreadRequest,
    *,
    endpoint: str = "screen_calendar_spreads",
) -> tuple[dt.date | str, dt.date | str, tuple[ScreenerWarning, ...]]:
    params = _request_params(request)
    if not request.ticker:
        raise _invalid_request("ticker is required", ticker=None, params=params, endpoint=endpoint)
    try:
        near_expiration = parse_expiration(request.near_expiration)
        far_expiration = parse_expiration(request.far_expiration)
    except (TypeError, ValueError) as exc:
        raise _invalid_request("near_expiration and far_expiration must be '*' or YYYY-MM-DD dates", ticker=request.ticker, params=params, endpoint=endpoint) from exc
    if request.right not in _RIGHT_VALUES:
        raise _invalid_request("right must be one of: call, put, both", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.rank_by not in _RANK_BY_VALUES:
        raise _invalid_request(f"rank_by must be one of: {', '.join(sorted(_RANK_BY_VALUES))}", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.greeks_source not in _GREEKS_SOURCE_VALUES:
        raise _invalid_request("greeks_source must be one of: auto, thetadata, local, none", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.cache_policy not in _CACHE_POLICY_VALUES:
        raise _invalid_request("cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_candidates_per_expiration is not None and request.max_candidates_per_expiration <= 0:
        raise _invalid_request("max_candidates_per_expiration must be greater than zero", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_candidates_total is not None and request.max_candidates_total <= 0:
        raise _invalid_request("max_candidates_total must be greater than zero", ticker=request.ticker, params=params, endpoint=endpoint)

    warnings = []
    if near_expiration == "*" or far_expiration == "*":
        message = "calendar spread typed requests with wildcard expirations can fetch broad option chains."
        if not request.allow_full_chain:
            raise _invalid_request(f"{message} Set allow_full_chain=True.", ticker=request.ticker, params=params, endpoint=endpoint)
        warnings.append(ScreenerWarning(code="full_chain", message=message))
    elif far_expiration <= near_expiration:
        raise _invalid_request("far_expiration must be after near_expiration", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.greeks_source == "local":
        warnings.append(ScreenerWarning(code="local_greeks", message="local Greek calculation can be slow across two chains.", severity="info"))
    return near_expiration, far_expiration, tuple(warnings)


def _planned_endpoint_and_params(request: CalendarSpreadRequest, expiration: dt.date | str) -> tuple[str, dict[str, object]]:
    params = {
        "expiration": expiration,
        "strike": "*",
        "right": request.right,
        "strike_range": request.strike_range,
        "min_time": request.min_time,
    }
    if request.greeks_source in {"local", "none"}:
        return "option_snapshot_quote", params
    return "option_snapshot_greeks_first_order", {
        **params,
        "annual_dividend": request.annual_dividend,
        "rate_type": request.rate_type,
        "rate_value": request.rate_value,
        "stock_price": request.stock_price,
        "version": request.version,
        "use_market_value": request.use_market_value,
    }


def _planned_cost(request: CalendarSpreadRequest, near_expiration: dt.date | str, far_expiration: dt.date | str) -> PlanCost:
    if (near_expiration == "*" or far_expiration == "*") and request.strike_range is None:
        return "high"
    if near_expiration == "*" or far_expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: CalendarSpreadRequest, near_expiration: dt.date | str, far_expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if (near_expiration == "*" or far_expiration == "*") and request.strike_range is None:
        return "high"
    return "medium"


def plan_calendar_spreads(request: CalendarSpreadRequest, *, conn=None) -> ScreenerPlan:
    near_expiration, far_expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    if near_expiration == "*" or far_expiration == "*":
        return ScreenerPlan(
            ticker=ticker,
            strategy="calendar_spread",
            expected_endpoint=_planned_endpoint_and_params(request, near_expiration)[0],
            upstream_calls=None,
            cache_hits=0,
            cache_misses=0,
            cost=_planned_cost(request, near_expiration, far_expiration),
            local_computation=_planned_local_computation(request, near_expiration, far_expiration),
            warnings=(
                *warnings,
                ScreenerWarning(
                    code="wildcard_plan",
                    message=(
                        "Wildcard expiration plans cannot estimate concrete "
                        "per-expiration cache coverage without expanding expirations."
                    ),
                    severity="info",
                ),
            ),
            cache_coverage=(),
        )
    planned = [_planned_endpoint_and_params(request, near_expiration), _planned_endpoint_and_params(request, far_expiration)]
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        coverages = []
        cache_hits = 0
        cache_misses = 0
        upstream_calls = 0
        for endpoint, params in planned:
            if request.cache_policy == "no_cache":
                coverage = CacheCoverage(endpoint=endpoint, root=ticker, params=params, covered=False, fresh=False, reason="no_cache")
                hit = False
            else:
                coverage = inspect_cache_coverage(conn=conn, endpoint=endpoint, root=ticker, params=params, stale_threshold=request.stale_threshold)
                hit = coverage.covered and coverage.fresh and request.cache_policy != "refresh"
            coverages.append(coverage)
            cache_hits += 1 if hit else 0
            cache_misses += 0 if hit else 1
            upstream_calls += 0 if hit or request.cache_policy == "cache_only" else 1
        return ScreenerPlan(
            ticker=ticker,
            strategy="calendar_spread",
            expected_endpoint=planned[0][0],
            upstream_calls=upstream_calls,
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            cost=_planned_cost(request, near_expiration, far_expiration),
            local_computation=_planned_local_computation(request, near_expiration, far_expiration),
            warnings=warnings,
            cache_coverage=tuple(coverages),
        )
    finally:
        if own_conn:
            conn.close()


def _build_rows(ticker: str, near_chain: pl.DataFrame, far_chain: pl.DataFrame, *, today: dt.date, right: Right, min_debit: float) -> pl.DataFrame:
    near_by_expiration = grouped_by_expiration(normalize_rows(near_chain))
    far_by_expiration = grouped_by_expiration(normalize_rows(far_chain))
    output = []
    for near_exp, near_rows in near_by_expiration.items():
        for far_exp, far_rows in far_by_expiration.items():
            if far_exp <= near_exp:
                continue
            near_days = dte(near_exp, today)
            far_days = dte(far_exp, today)
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


def _sort_calendar_spreads(spreads: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [column for column in ["near_credit_to_debit", "calendar_days"] if column != rank_by]
    return sort_and_limit(spreads, rank_by=rank_by, tie_breakers=tie_breakers, descending=[True] * (1 + len(tie_breakers)), top_n=top_n)


def _apply_candidate_caps(spreads: pl.DataFrame, *, rank_by: str, max_candidates_per_expiration: int | None, max_candidates_total: int | None) -> tuple[pl.DataFrame, int]:
    if spreads.is_empty():
        return spreads, 0
    original_count = len(spreads)
    capped = spreads
    if max_candidates_per_expiration is not None:
        frames = [
            _sort_calendar_spreads(group, rank_by=rank_by, top_n=max_candidates_per_expiration)
            for group in capped.partition_by(["near_expiration", "far_expiration"], maintain_order=True)
        ]
        capped = pl.concat(frames, how="diagonal_relaxed") if frames else empty_frame(_OUTPUT_COLUMNS)
    if max_candidates_total is not None:
        capped = _sort_calendar_spreads(capped, rank_by=rank_by, top_n=max_candidates_total)
    return capped, original_count - len(capped)


def _fetch_chain(request: CalendarSpreadRequest, *, expiration: dt.date | str, client: Client, today: dt.date, diagnostics: dict[str, object], conn=None) -> pl.DataFrame:
    return get_first_order_chain(
        ticker=request.ticker or "",
        expiration=expiration,
        client=client,
        log=log,
        right=request.right,
        today=today,
        annual_dividend=request.annual_dividend,
        rate_type=request.rate_type,
        rate_value=request.rate_value,
        stock_price=request.stock_price,
        version=request.version,
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


def screen_calendar_spreads(request: CalendarSpreadRequest, client: Client | None = None, *, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    near_expiration, far_expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_calendar_spreads(request, conn=conn)
    if client is None:
        client = create_client()
    diagnostics: dict[str, object] = {}
    try:
        near_chain = _fetch_chain(request, expiration=near_expiration, client=client, today=today, diagnostics=diagnostics, conn=conn)
        far_chain = _fetch_chain(request, expiration=far_expiration, client=client, today=today, diagnostics=diagnostics, conn=conn)
    except ThetaDataError:
        raise
    except Exception as exc:
        raise classify_thetadata_error(exc, ticker=ticker, endpoint="screen_calendar_spreads", params=_request_params(request)) from exc
    fetched_rows = len(near_chain) + len(far_chain)
    filtered_rows = fetched_rows
    spreads = _build_rows(ticker, near_chain, far_chain, today=today, right=request.right, min_debit=request.min_debit)
    spreads, pruned_candidate_rows = _apply_candidate_caps(
        spreads,
        rank_by=request.rank_by,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(spreads)
    if pruned_candidate_rows:
        warnings = (*warnings, ScreenerWarning(code="candidate_limit", message=f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible calendar spread candidates."))
    spreads = _sort_calendar_spreads(spreads, rank_by=request.rank_by, top_n=request.top_n)
    return ScreenerResult(
        data=spreads,
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=fetched_rows,
            filtered_rows=filtered_rows,
            candidate_rows=candidate_rows,
            returned_rows=len(spreads),
            duration_seconds=time.perf_counter() - started_at,
            greeks_source=str(diagnostics.get("greeks_source", request.greeks_source)),
            cache_hits=plan.cache_hits,
            cache_misses=plan.cache_misses,
            upstream_calls=plan.upstream_calls,
            cache_policy=request.cache_policy,
            pruned_candidate_rows=pruned_candidate_rows,
        ),
        warnings=warnings,
    )


def _attempt_screen(request: CalendarSpreadRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            attempt_started = time.perf_counter()
            result = screen_calendar_spreads(request, client=client, conn=conn)
            if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds:
                raise ThetaTimeoutError("Calendar spread screening exceeded the per-ticker timeout.", ticker=request.ticker, endpoint="screen_calendar_spreads", params=_request_params(request), retryable=True, user_message="Calendar spread screening exceeded the per-ticker timeout.")
            return result
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="screen_calendar_spreads", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_calendar_spread_watchlist(
    tickers: list[str],
    request: CalendarSpreadRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_calendar_spreads")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_screen(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="screen_calendar_spreads")

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
            ticker, item = run_one(ticker)
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
    return BatchResult(data=data, successes=successes, failures=failures, stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at))


def _warm_calendar_spread_request(request: CalendarSpreadRequest, *, client: Client | None = None, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    near_expiration, far_expiration, warnings = _validate_request(request, endpoint="warm_calendar_spread_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_calendar_spreads(request, conn=conn)
    if client is None:
        client = create_client()
    diagnostics: dict[str, object] = {}
    near_chain = _fetch_chain(request, expiration=near_expiration, client=client, today=today, diagnostics=diagnostics, conn=conn)
    far_chain = _fetch_chain(request, expiration=far_expiration, client=client, today=today, diagnostics=diagnostics, conn=conn)
    duration = time.perf_counter() - started_at
    if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and duration > timeout_policy.per_ticker_seconds:
        raise ThetaTimeoutError("Calendar spread cache warmup exceeded the per-ticker timeout.", ticker=ticker, endpoint="warm_calendar_spread_cache", params=_request_params(request), retryable=True, user_message="Calendar spread cache warmup exceeded the per-ticker timeout.")
    fetched_rows = len(near_chain) + len(far_chain)
    return ScreenerResult(
        data=pl.DataFrame(),
        stats=ScreenerStats(ticker=ticker, fetched_rows=fetched_rows, filtered_rows=fetched_rows, candidate_rows=0, returned_rows=0, duration_seconds=duration, greeks_source=str(diagnostics.get("greeks_source", request.greeks_source)), cache_hits=plan.cache_hits, cache_misses=plan.cache_misses, upstream_calls=plan.upstream_calls, cache_policy=request.cache_policy),
        warnings=warnings,
    )


def _attempt_warm(request: CalendarSpreadRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            return _warm_calendar_spread_request(request, client=client, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="warm_calendar_spread_cache", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_calendar_spread_cache(
    tickers: list[str],
    request: CalendarSpreadRequest,
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
    started_at = time.perf_counter()
    retry_policy = retry_policy or RetryPolicy()
    successes: list[TickerResult] = []
    failures: list[TickerFailure] = []
    tickers = list(tickers)
    rate_limiter = _RateLimiter(rate_limit_policy)
    circuit_breaker = _CircuitBreaker(circuit_breaker_policy)

    def run_one(ticker: str) -> tuple[str, ScreenerResult | TickerFailure]:
        if circuit_breaker.is_open():
            return ticker, _circuit_open_failure(ticker, endpoint="warm_calendar_spread_cache")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_warm(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="warm_calendar_spread_cache")

    if concurrency > 1 and len(tickers) > 1 and conn is None:
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
            ticker, item = run_one(ticker)
            if isinstance(item, TickerFailure):
                failures.append(item)
                if on_progress is not None:
                    on_progress(item)
                continue
            successes.append(TickerResult(ticker=ticker, stats=item.stats))
            if on_progress is not None:
                on_progress(successes[-1])
    return WarmCacheResult(successes=successes, failures=failures, stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at))


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
    greeks_source: GreekSource = "auto",
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int | Literal["fast", "balanced", "accurate"] = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    max_candidates_per_expiration: int | None = None,
    max_candidates_total: int | None = None,
    conn=None,
) -> pl.DataFrame:
    request = CalendarSpreadRequest(
        ticker=ticker,
        near_expiration=near_expiration,
        far_expiration=far_expiration,
        right=right,
        min_debit=min_debit,
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
        max_candidates_per_expiration=max_candidates_per_expiration,
        max_candidates_total=max_candidates_total,
    )
    return screen_calendar_spreads(request, client=client, conn=conn).data


find_best_calendar_spreads = get_best_calendar_spreads

__all__ = [
    "CalendarSpreadRequest",
    "get_best_calendar_spreads",
    "find_best_calendar_spreads",
    "plan_calendar_spreads",
    "screen_calendar_spreads",
    "screen_calendar_spread_watchlist",
    "warm_calendar_spread_cache",
]
