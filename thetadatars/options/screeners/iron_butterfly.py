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
from ._common import GreekSource, filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, right_name
from ._strategy_utils import annualize, dte, empty_frame, grouped_by_expiration, normalize_rows, row_by_strike, sort_and_limit
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

RankBy = Literal["annualized_return_on_risk", "return_on_risk", "credit", "body_distance_percent"]
_RANK_BY_VALUES = {"annualized_return_on_risk", "return_on_risk", "credit", "body_distance_percent"}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "long_put_strike", "body_strike",
    "long_call_strike", "put_width", "call_width", "credit", "max_loss",
    "return_on_risk", "annualized_return_on_risk", "lower_breakeven",
    "upper_breakeven", "body_distance_percent", "dte", "underlying_price",
    "short_put_delta", "short_call_delta",
]


@dataclass(frozen=True, slots=True)
class IronButterflyRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    min_dte: int | None = None
    max_dte: int | None = None
    min_credit: float = 0.01
    min_width: float | None = None
    max_width: float | None = None
    min_short_delta: float | None = None
    max_short_delta: float | None = None
    top_n: int | None = 25
    rank_by: RankBy = "annualized_return_on_risk"
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

    def for_ticker(self, ticker: str) -> "IronButterflyRequest":
        return replace(self, ticker=ticker)


def _request_params(request: IronButterflyRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "right": "both",
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
        "min_credit": request.min_credit,
        "min_width": request.min_width,
        "max_width": request.max_width,
        "min_short_delta": request.min_short_delta,
        "max_short_delta": request.max_short_delta,
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


def _validate_request(request: IronButterflyRequest, *, endpoint: str = "screen_iron_butterflies") -> tuple[dt.date | str, tuple[ScreenerWarning, ...]]:
    params = _request_params(request)
    if not request.ticker:
        raise _invalid_request("ticker is required", ticker=None, params=params, endpoint=endpoint)
    try:
        expiration = parse_expiration(request.expiration)
    except (TypeError, ValueError) as exc:
        raise _invalid_request("expiration must be '*' or a YYYY-MM-DD date", ticker=request.ticker, params=params, endpoint=endpoint) from exc
    if request.rank_by not in _RANK_BY_VALUES:
        raise _invalid_request(f"rank_by must be one of: {', '.join(sorted(_RANK_BY_VALUES))}", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.greeks_source not in _GREEKS_SOURCE_VALUES:
        raise _invalid_request("greeks_source must be one of: auto, thetadata, local, none", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.cache_policy not in _CACHE_POLICY_VALUES:
        raise _invalid_request("cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.min_dte is not None and request.max_dte is not None and request.min_dte > request.max_dte:
        raise _invalid_request("min_dte cannot be greater than max_dte", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.min_width is not None and request.max_width is not None and request.min_width > request.max_width:
        raise _invalid_request("min_width cannot be greater than max_width", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.min_short_delta is not None and request.max_short_delta is not None and request.min_short_delta > request.max_short_delta:
        raise _invalid_request("min_short_delta cannot be greater than max_short_delta", ticker=request.ticker, params=params, endpoint=endpoint)
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


def _planned_endpoint_and_params(request: IronButterflyRequest, expiration: dt.date | str) -> tuple[str, dict[str, object]]:
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


def _planned_cost(request: IronButterflyRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: IronButterflyRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_iron_butterflies(request: IronButterflyRequest, *, conn=None) -> ScreenerPlan:
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    endpoint, params = _planned_endpoint_and_params(request, expiration)
    if expiration == "*":
        return ScreenerPlan(
            ticker=ticker,
            strategy="iron_butterfly",
            expected_endpoint=endpoint,
            upstream_calls=None,
            cache_hits=0,
            cache_misses=0,
            cost=_planned_cost(request, expiration),
            local_computation=_planned_local_computation(request, expiration),
            warnings=(
                *warnings,
                ScreenerWarning(
                    code="wildcard_plan",
                    message="Wildcard expiration plans cannot estimate concrete per-expiration cache coverage without expanding expirations.",
                    severity="info",
                ),
            ),
            cache_coverage=(),
        )
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
            upstream_calls = 0 if (cache_hits or request.cache_policy == "cache_only") else 1
        return ScreenerPlan(
            ticker=ticker,
            strategy="iron_butterfly",
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


def _short_delta_allowed(delta: float | None, *, min_short_delta: float | None, max_short_delta: float | None) -> bool:
    if delta is None:
        return min_short_delta is None and max_short_delta is None
    absolute = abs(delta)
    if min_short_delta is not None and absolute < min_short_delta:
        return False
    if max_short_delta is not None and absolute > max_short_delta:
        return False
    return True


def _build_rows(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    min_credit: float,
    min_width: float | None,
    max_width: float | None,
    min_short_delta: float | None = None,
    max_short_delta: float | None = None,
) -> pl.DataFrame:
    output = []
    for expiration, expiration_rows in grouped_by_expiration(normalize_rows(chain)).items():
        puts = row_by_strike([r for r in expiration_rows if right_name(r.get("right")) == "put"])
        calls = row_by_strike([r for r in expiration_rows if right_name(r.get("right")) == "call"])
        body_strikes = sorted(set(puts) & set(calls))
        days = dte(expiration, today)
        for body in body_strikes:
            short_put = puts[body]
            short_call = calls[body]
            sp_bid = finite_number(short_put.get("bid"))
            sc_bid = finite_number(short_call.get("bid"))
            if sp_bid is None or sc_bid is None:
                continue
            short_put_delta = finite_number(short_put.get("delta"))
            short_call_delta = finite_number(short_call.get("delta"))
            if not _short_delta_allowed(short_put_delta, min_short_delta=min_short_delta, max_short_delta=max_short_delta):
                continue
            if not _short_delta_allowed(short_call_delta, min_short_delta=min_short_delta, max_short_delta=max_short_delta):
                continue
            for lp, long_put in puts.items():
                lp_ask = finite_number(long_put.get("ask"))
                if lp >= body or lp_ask is None:
                    continue
                put_width = body - lp
                if min_width is not None and put_width < min_width:
                    continue
                if max_width is not None and put_width > max_width:
                    continue
                for lc, long_call in calls.items():
                    lc_ask = finite_number(long_call.get("ask"))
                    if lc <= body or lc_ask is None:
                        continue
                    call_width = lc - body
                    if min_width is not None and call_width < min_width:
                        continue
                    if max_width is not None and call_width > max_width:
                        continue
                    credit = sp_bid + sc_bid - lp_ask - lc_ask
                    max_loss = max(put_width, call_width) - credit
                    if credit < min_credit or max_loss <= 0:
                        continue
                    underlying = finite_number(short_put.get("underlying_price"))
                    body_distance_percent = abs(body - underlying) / underlying if underlying is not None and underlying > 0 else None
                    return_on_risk = credit / max_loss
                    output.append({
                        "root": short_put.get("root", ticker),
                        "expiration": expiration,
                        "strategy": "iron_butterfly",
                        "long_put_strike": lp,
                        "body_strike": body,
                        "long_call_strike": lc,
                        "put_width": put_width,
                        "call_width": call_width,
                        "credit": credit,
                        "max_loss": max_loss * 100,
                        "return_on_risk": return_on_risk,
                        "annualized_return_on_risk": annualize(return_on_risk, days),
                        "lower_breakeven": body - credit,
                        "upper_breakeven": body + credit,
                        "body_distance_percent": body_distance_percent,
                        "dte": days,
                        "underlying_price": underlying,
                        "short_put_delta": short_put_delta,
                        "short_call_delta": short_call_delta,
                    })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def _sort_iron_butterflies(flies: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [column for column in ["credit", "body_distance_percent"] if column != rank_by]
    descending = [rank_by != "body_distance_percent"]
    descending.extend(False if column == "body_distance_percent" else True for column in tie_breakers)
    return sort_and_limit(flies, rank_by=rank_by, tie_breakers=tie_breakers, descending=descending, top_n=top_n)


def _apply_candidate_caps(flies: pl.DataFrame, *, rank_by: str, max_candidates_per_expiration: int | None, max_candidates_total: int | None) -> tuple[pl.DataFrame, int]:
    if flies.is_empty():
        return flies, 0
    original_count = len(flies)
    capped = flies
    if max_candidates_per_expiration is not None:
        frames = [_sort_iron_butterflies(group, rank_by=rank_by, top_n=max_candidates_per_expiration) for group in capped.partition_by("expiration", maintain_order=True)]
        capped = pl.concat(frames, how="diagonal_relaxed") if frames else empty_frame(_OUTPUT_COLUMNS)
    if max_candidates_total is not None:
        capped = _sort_iron_butterflies(capped, rank_by=rank_by, top_n=max_candidates_total)
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


def screen_iron_butterflies(request: IronButterflyRequest, client: Client | None = None, *, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_iron_butterflies(request, conn=conn)
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
        raise classify_thetadata_error(exc, ticker=ticker, endpoint="screen_iron_butterflies", params=_request_params(request)) from exc
    fetched_rows = len(chain)
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=request.min_dte, max_dte=request.max_dte)
    filtered_rows = len(chain)
    flies = _build_rows(
        ticker,
        chain,
        today=today,
        min_credit=request.min_credit,
        min_width=request.min_width,
        max_width=request.max_width,
        min_short_delta=request.min_short_delta,
        max_short_delta=request.max_short_delta,
    )
    flies, pruned_candidate_rows = _apply_candidate_caps(
        flies,
        rank_by=request.rank_by,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(flies)
    if pruned_candidate_rows:
        warnings = (*warnings, ScreenerWarning(code="candidate_limit", message=f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible iron butterfly candidates."))
    flies = _sort_iron_butterflies(flies, rank_by=request.rank_by, top_n=request.top_n)
    return ScreenerResult(
        data=flies,
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=fetched_rows,
            filtered_rows=filtered_rows,
            candidate_rows=candidate_rows,
            returned_rows=len(flies),
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


def _attempt_screen(request: IronButterflyRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            attempt_started = time.perf_counter()
            result = screen_iron_butterflies(request, client=client, conn=conn)
            if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds:
                raise ThetaTimeoutError("Iron butterfly screening exceeded the per-ticker timeout.", ticker=request.ticker, endpoint="screen_iron_butterflies", params=_request_params(request), retryable=True, user_message="Iron butterfly screening exceeded the per-ticker timeout.")
            return result
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="screen_iron_butterflies", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_iron_butterfly_watchlist(
    tickers: list[str],
    request: IronButterflyRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_iron_butterflies")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_screen(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="screen_iron_butterflies")

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


def _warm_iron_butterfly_request(request: IronButterflyRequest, *, client: Client | None = None, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request, endpoint="warm_iron_butterfly_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_iron_butterflies(request, conn=conn)
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
        raise ThetaTimeoutError("Iron butterfly cache warmup exceeded the per-ticker timeout.", ticker=ticker, endpoint="warm_iron_butterfly_cache", params=_request_params(request), retryable=True, user_message="Iron butterfly cache warmup exceeded the per-ticker timeout.")
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


def _attempt_warm(request: IronButterflyRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            return _warm_iron_butterfly_request(request, client=client, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="warm_iron_butterfly_cache", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_iron_butterfly_cache(
    tickers: list[str],
    request: IronButterflyRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="warm_iron_butterfly_cache")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_warm(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="warm_iron_butterfly_cache")

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


def get_best_iron_butterflies(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_credit: float = 0.01,
    min_width: float | None = None,
    max_width: float | None = None,
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
    expiration = parse_expiration(expiration)
    today = dt.date.today()
    if expiration != "*":
        days = dte(expiration, today)
        if min_dte is not None and days < min_dte:
            return empty_frame(_OUTPUT_COLUMNS)
        if max_dte is not None and days > max_dte:
            return empty_frame(_OUTPUT_COLUMNS)
    if client is None:
        client = create_client()
    chain = get_first_order_chain(
        ticker=ticker, expiration=expiration, client=client, log=log, right="both", today=today,
        annual_dividend=annual_dividend, rate_type=rate_type, rate_value=rate_value,
        stock_price=stock_price, version=version, max_dte=max_dte, strike_range=strike_range,
        min_time=min_time, use_market_value=use_market_value, fallback_to_local_greeks=fallback_to_local_greeks,
        local_greeks_steps=local_greeks_steps, stale_threshold=stale_threshold, conn=conn,
    )
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=min_dte, max_dte=max_dte)
    flies = _build_rows(ticker, chain, today=today, min_credit=min_credit, min_width=min_width, max_width=max_width)
    return _sort_iron_butterflies(flies, rank_by=rank_by, top_n=top_n)


find_best_iron_butterflies = get_best_iron_butterflies

__all__ = [
    "IronButterflyRequest",
    "find_best_iron_butterflies",
    "get_best_iron_butterflies",
    "plan_iron_butterflies",
    "screen_iron_butterflies",
    "screen_iron_butterfly_watchlist",
    "warm_iron_butterfly_cache",
]
