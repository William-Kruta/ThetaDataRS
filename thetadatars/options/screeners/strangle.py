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
from ._common import (
    GreekSource,
    filter_expiration_dte,
    finite_number,
    get_first_order_chain,
    parse_expiration,
    probability_otm_from_delta,
    right_name,
)
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

Side = Literal["long", "short"]
RankBy = Literal["vega_per_dollar", "theta_income", "premium", "probability_range", "breakeven_width_percent"]

_SIDE_VALUES = {"long", "short"}
_RANK_BY_VALUES = {"vega_per_dollar", "theta_income", "premium", "probability_range", "breakeven_width_percent"}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "side", "put_strike", "call_strike",
    "premium", "max_loss", "lower_breakeven", "upper_breakeven",
    "breakeven_width_percent", "probability_range", "dte", "underlying_price",
    "put_delta", "call_delta", "net_delta", "net_theta", "net_vega",
    "vega_per_dollar", "theta_income",
]


@dataclass(frozen=True, slots=True)
class StrangleRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    side: Side = "long"
    min_dte: int | None = None
    max_dte: int | None = None
    min_premium: float = 0.01
    top_n: int | None = 25
    rank_by: RankBy = "vega_per_dollar"
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

    def for_ticker(self, ticker: str) -> "StrangleRequest":
        return replace(self, ticker=ticker)


def _request_params(request: StrangleRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "right": "both",
        "side": request.side,
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
        "min_premium": request.min_premium,
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


def _validate_request(request: StrangleRequest, *, endpoint: str = "screen_strangles") -> tuple[dt.date | str, tuple[ScreenerWarning, ...]]:
    params = _request_params(request)
    if not request.ticker:
        raise _invalid_request("ticker is required", ticker=None, params=params, endpoint=endpoint)
    try:
        expiration = parse_expiration(request.expiration)
    except (TypeError, ValueError) as exc:
        raise _invalid_request("expiration must be '*' or a YYYY-MM-DD date", ticker=request.ticker, params=params, endpoint=endpoint) from exc
    if request.side not in _SIDE_VALUES:
        raise _invalid_request("side must be one of: long, short", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.rank_by not in _RANK_BY_VALUES:
        raise _invalid_request(f"rank_by must be one of: {', '.join(sorted(_RANK_BY_VALUES))}", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.greeks_source not in _GREEKS_SOURCE_VALUES:
        raise _invalid_request("greeks_source must be one of: auto, thetadata, local, none", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.cache_policy not in _CACHE_POLICY_VALUES:
        raise _invalid_request("cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.min_dte is not None and request.max_dte is not None and request.min_dte > request.max_dte:
        raise _invalid_request("min_dte cannot be greater than max_dte", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_candidates_per_expiration is not None and request.max_candidates_per_expiration <= 0:
        raise _invalid_request("max_candidates_per_expiration must be greater than zero", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_candidates_total is not None and request.max_candidates_total <= 0:
        raise _invalid_request("max_candidates_total must be greater than zero", ticker=request.ticker, params=params, endpoint=endpoint)

    warnings = []
    if expiration == "*" and request.max_dte is None:
        message = "expiration='*' without max_dte can fetch a full option chain."
        if not request.allow_full_chain:
            raise _invalid_request(f"{message} Set max_dte or allow_full_chain=True.", ticker=request.ticker, params=params, endpoint=endpoint)
        warnings.append(ScreenerWarning(code="full_chain", message=message))
    if expiration == "*" and request.strike_range is None:
        warnings.append(ScreenerWarning(code="unbounded_strikes", message="expiration='*' without strike_range can fetch far out-of-the-money contracts."))
    if request.greeks_source == "local":
        warnings.append(ScreenerWarning(code="local_greeks", message="local Greek calculation can be slow across broad chains.", severity="info"))
    return expiration, tuple(warnings)


def _planned_endpoint_and_params(request: StrangleRequest, expiration: dt.date | str) -> tuple[str, dict[str, object]]:
    params = {
        "expiration": expiration,
        "strike": "*",
        "right": "both",
        "max_dte": request.max_dte,
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


def _planned_cost(request: StrangleRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: StrangleRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_strangles(request: StrangleRequest, *, conn=None) -> ScreenerPlan:
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    endpoint, params = _planned_endpoint_and_params(request, expiration)
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        if request.cache_policy == "no_cache":
            coverage = CacheCoverage(endpoint=endpoint, root=ticker, params=params, covered=False, fresh=False, reason="no_cache")
            cache_hits = 0
            cache_misses = 1
            upstream_calls: int | None = 1
        else:
            coverage = inspect_cache_coverage(conn=conn, endpoint=endpoint, root=ticker, params=params, stale_threshold=request.stale_threshold)
            cache_hits = 1 if coverage.covered and coverage.fresh and request.cache_policy != "refresh" else 0
            cache_misses = 0 if cache_hits else 1
            upstream_calls = 0 if cache_hits or request.cache_policy == "cache_only" else 1
        return ScreenerPlan(
            ticker=ticker,
            strategy="strangle",
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


def _build_rows(ticker: str, chain: pl.DataFrame, *, today: dt.date, side: Side, min_premium: float) -> pl.DataFrame:
    output = []
    for expiration, rows in grouped_by_expiration(normalize_rows(chain)).items():
        puts = sorted([r for r in rows if right_name(r.get("right")) == "put"], key=lambda r: finite_number(r.get("strike")) or 0)
        calls = sorted([r for r in rows if right_name(r.get("right")) == "call"], key=lambda r: finite_number(r.get("strike")) or 0)
        days = dte(expiration, today)
        for put in puts:
            ps = finite_number(put.get("strike"))
            put_price = finite_number(put.get("ask" if side == "long" else "bid"))
            if ps is None or put_price is None:
                continue
            for call in calls:
                cs = finite_number(call.get("strike"))
                call_price = finite_number(call.get("ask" if side == "long" else "bid"))
                if cs is None or call_price is None or cs <= ps:
                    continue
                premium = put_price + call_price
                if premium < min_premium:
                    continue
                underlying = finite_number(call.get("underlying_price"))
                lower_breakeven = ps - premium
                upper_breakeven = cs + premium
                breakeven_width_percent = (upper_breakeven - lower_breakeven) / underlying if underlying is not None and underlying > 0 else None
                put_delta = finite_number(put.get("delta"))
                call_delta = finite_number(call.get("delta"))
                put_prob = probability_otm_from_delta(put_delta)
                call_prob = probability_otm_from_delta(call_delta)
                probability_range = max(0.0, min(1.0, put_prob + call_prob - 1)) if put_prob is not None and call_prob is not None else None
                net_delta = (put_delta or 0) + (call_delta or 0)
                net_theta = (finite_number(put.get("theta")) or 0) + (finite_number(call.get("theta")) or 0)
                net_vega = (finite_number(put.get("vega")) or 0) + (finite_number(call.get("vega")) or 0)
                if side == "short":
                    net_delta *= -1
                    net_theta *= -1
                    net_vega *= -1
                output.append({
                    "root": put.get("root", ticker),
                    "expiration": expiration,
                    "strategy": f"{side}_strangle",
                    "side": side,
                    "put_strike": ps,
                    "call_strike": cs,
                    "premium": premium,
                    "max_loss": premium * 100 if side == "long" else None,
                    "lower_breakeven": lower_breakeven,
                    "upper_breakeven": upper_breakeven,
                    "breakeven_width_percent": breakeven_width_percent,
                    "probability_range": probability_range,
                    "dte": days,
                    "underlying_price": underlying,
                    "put_delta": put_delta,
                    "call_delta": call_delta,
                    "net_delta": net_delta,
                    "net_theta": net_theta,
                    "net_vega": net_vega,
                    "vega_per_dollar": abs(net_vega) / premium if premium > 0 else None,
                    "theta_income": net_theta if side == "short" else -net_theta,
                })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def _sort_strangles(strangles: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [column for column in ["premium", "probability_range"] if column != rank_by]
    descending = [True]
    descending.extend(True for _ in tie_breakers)
    return sort_and_limit(strangles, rank_by=rank_by, tie_breakers=tie_breakers, descending=descending, top_n=top_n)


def _apply_candidate_caps(strangles: pl.DataFrame, *, rank_by: str, max_candidates_per_expiration: int | None, max_candidates_total: int | None) -> tuple[pl.DataFrame, int]:
    if strangles.is_empty():
        return strangles, 0
    original_count = len(strangles)
    capped = strangles
    if max_candidates_per_expiration is not None:
        frames = [_sort_strangles(group, rank_by=rank_by, top_n=max_candidates_per_expiration) for group in capped.partition_by("expiration", maintain_order=True)]
        capped = pl.concat(frames, how="diagonal_relaxed") if frames else empty_frame(_OUTPUT_COLUMNS)
    if max_candidates_total is not None:
        capped = _sort_strangles(capped, rank_by=rank_by, top_n=max_candidates_total)
    return capped, original_count - len(capped)


def _empty_result(*, ticker: str, started_at: float, greeks_source: str | None, plan: ScreenerPlan, cache_policy: CachePolicy, warnings: tuple[ScreenerWarning, ...]) -> ScreenerResult:
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


def screen_strangles(request: StrangleRequest, client: Client | None = None, *, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_strangles(request, conn=conn)
    if expiration != "*":
        days_to_expiration = dte(expiration, today)
        if request.min_dte is not None and days_to_expiration < request.min_dte:
            return _empty_result(ticker=ticker, started_at=started_at, greeks_source=None, plan=plan, cache_policy=request.cache_policy, warnings=warnings)
        if request.max_dte is not None and days_to_expiration > request.max_dte:
            return _empty_result(ticker=ticker, started_at=started_at, greeks_source=None, plan=plan, cache_policy=request.cache_policy, warnings=warnings)
    if client is None:
        client = create_client()
    diagnostics: dict[str, object] = {}
    try:
        chain = get_first_order_chain(
            ticker=ticker, expiration=expiration, client=client, log=log, right="both", today=today,
            annual_dividend=request.annual_dividend, rate_type=request.rate_type, rate_value=request.rate_value,
            stock_price=request.stock_price, version=request.version, max_dte=request.max_dte, strike_range=request.strike_range,
            min_time=request.min_time, use_market_value=request.use_market_value, greeks_source=request.greeks_source,
            fallback_to_local_greeks=request.fallback_to_local_greeks, local_greeks_steps=request.local_greeks_steps,
            stale_threshold=request.stale_threshold, cache_policy=request.cache_policy, diagnostics=diagnostics, conn=conn,
        )
    except ThetaDataError:
        raise
    except Exception as exc:
        raise classify_thetadata_error(exc, ticker=ticker, endpoint="screen_strangles", params=_request_params(request)) from exc
    fetched_rows = len(chain)
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=request.min_dte, max_dte=request.max_dte)
    filtered_rows = len(chain)
    strangles = _build_rows(ticker, chain, today=today, side=request.side, min_premium=request.min_premium)
    strangles, pruned_candidate_rows = _apply_candidate_caps(
        strangles,
        rank_by=request.rank_by,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(strangles)
    if pruned_candidate_rows:
        warnings = (*warnings, ScreenerWarning(code="candidate_limit", message=f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible strangle candidates."))
    strangles = _sort_strangles(strangles, rank_by=request.rank_by, top_n=request.top_n)
    return ScreenerResult(
        data=strangles,
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=fetched_rows,
            filtered_rows=filtered_rows,
            candidate_rows=candidate_rows,
            returned_rows=len(strangles),
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


def _attempt_screen(request: StrangleRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            attempt_started = time.perf_counter()
            result = screen_strangles(request, client=client, conn=conn)
            if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds:
                raise ThetaTimeoutError("Strangle screening exceeded the per-ticker timeout.", ticker=request.ticker, endpoint="screen_strangles", params=_request_params(request), retryable=True, user_message="Strangle screening exceeded the per-ticker timeout.")
            return result
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="screen_strangles", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_strangle_watchlist(
    tickers: list[str],
    request: StrangleRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_strangles")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_screen(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="screen_strangles")

    if concurrency > 1 and len(tickers) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                _, item = future.result()
                if isinstance(item, TickerFailure):
                    failures.append(item)
                    if on_progress is not None:
                        on_progress(item)
                    continue
                successes.append(TickerResult(ticker=item.stats.ticker, stats=item.stats))
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
    return BatchResult(data=data, successes=successes, failures=failures, stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at))


def _warm_strangle_request(request: StrangleRequest, *, client: Client | None = None, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request, endpoint="warm_strangle_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_strangles(request, conn=conn)
    if client is None:
        client = create_client()
    diagnostics: dict[str, object] = {}
    chain = get_first_order_chain(
        ticker=ticker, expiration=expiration, client=client, log=log, right="both", today=today,
        annual_dividend=request.annual_dividend, rate_type=request.rate_type, rate_value=request.rate_value,
        stock_price=request.stock_price, version=request.version, max_dte=request.max_dte, strike_range=request.strike_range,
        min_time=request.min_time, use_market_value=request.use_market_value, greeks_source=request.greeks_source,
        fallback_to_local_greeks=request.fallback_to_local_greeks, local_greeks_steps=request.local_greeks_steps,
        stale_threshold=request.stale_threshold, cache_policy=request.cache_policy, diagnostics=diagnostics, conn=conn,
    )
    duration = time.perf_counter() - started_at
    if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and duration > timeout_policy.per_ticker_seconds:
        raise ThetaTimeoutError("Strangle cache warmup exceeded the per-ticker timeout.", ticker=ticker, endpoint="warm_strangle_cache", params=_request_params(request), retryable=True, user_message="Strangle cache warmup exceeded the per-ticker timeout.")
    filtered = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=request.min_dte, max_dte=request.max_dte)
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


def _attempt_warm(request: StrangleRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            return _warm_strangle_request(request, client=client, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="warm_strangle_cache", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_strangle_cache(
    tickers: list[str],
    request: StrangleRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="warm_strangle_cache")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_warm(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="warm_strangle_cache")

    if concurrency > 1 and len(tickers) > 1 and conn is None:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, ticker): ticker for ticker in tickers}
            for future in as_completed(futures):
                _, item = future.result()
                if isinstance(item, TickerFailure):
                    failures.append(item)
                    if on_progress is not None:
                        on_progress(item)
                    continue
                successes.append(TickerResult(ticker=item.stats.ticker, stats=item.stats))
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
    return WarmCacheResult(successes=successes, failures=failures, stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at))


def get_best_strangles(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    side: Side = "long",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_premium: float = 0.01,
    top_n: int | None = 25,
    rank_by: RankBy = "vega_per_dollar",
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
    request = StrangleRequest(
        ticker=ticker,
        expiration=expiration,
        side=side,
        min_dte=min_dte,
        max_dte=max_dte,
        min_premium=min_premium,
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
    return screen_strangles(request, client=client, conn=conn).data


find_best_strangles = get_best_strangles

__all__ = [
    "StrangleRequest",
    "get_best_strangles",
    "find_best_strangles",
    "plan_strangles",
    "screen_strangles",
    "screen_strangle_watchlist",
    "warm_strangle_cache",
]
