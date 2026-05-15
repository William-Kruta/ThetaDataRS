import bisect
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
from ._common import GreekSource, filter_expiration_dte, finite_number, get_first_order_chain, parse_expiration, probability_otm_from_delta, right_name
from ._strategy_utils import annualize, dte, empty_frame, grouped_by_expiration, normalize_rows, sort_and_limit
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

RankBy = Literal["annualized_risk_adjusted_return", "annualized_return_on_risk", "return_on_risk", "credit", "probability_range"]
_RANK_BY_VALUES = {"annualized_risk_adjusted_return", "annualized_return_on_risk", "return_on_risk", "credit", "probability_range"}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}

_OUTPUT_COLUMNS = [
    "root", "expiration", "strategy", "long_put_strike", "short_put_strike",
    "short_call_strike", "long_call_strike", "put_width", "call_width", "credit",
    "max_loss", "return_on_risk", "annualized_return_on_risk", "probability_range",
    "risk_adjusted_return", "annualized_risk_adjusted_return", "lower_breakeven",
    "upper_breakeven", "dte", "underlying_price", "short_put_delta", "short_call_delta",
]


@dataclass(frozen=True, slots=True)
class IronCondorRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    min_dte: int | None = None
    max_dte: int | None = None
    min_credit: float = 0.01
    min_width: float | None = None
    max_width: float | None = None
    min_short_delta: float | None = None
    max_short_delta: float | None = None
    min_probability_range: float | None = None
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
    max_candidates_per_expiration: int | None = None
    max_candidates_total: int | None = None

    def for_ticker(self, ticker: str) -> "IronCondorRequest":
        return replace(self, ticker=ticker)


def _request_params(request: IronCondorRequest) -> dict[str, object]:
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
        "min_probability_range": request.min_probability_range,
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


def _validate_request(request: IronCondorRequest, *, endpoint: str = "screen_iron_condors") -> tuple[dt.date | str, tuple[ScreenerWarning, ...]]:
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


def _planned_endpoint_and_params(request: IronCondorRequest, expiration: dt.date | str) -> tuple[str, dict[str, object]]:
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


def _planned_cost(request: IronCondorRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: IronCondorRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_iron_condors(request: IronCondorRequest, *, conn=None) -> ScreenerPlan:
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    endpoint, params = _planned_endpoint_and_params(request, expiration)
    if expiration == "*":
        return ScreenerPlan(
            ticker=ticker,
            strategy="iron_condor",
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
            strategy="iron_condor",
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
    min_probability_range: float | None = None,
) -> pl.DataFrame:
    output = []
    for expiration, expiration_rows in grouped_by_expiration(normalize_rows(chain)).items():
        puts = sorted([r for r in expiration_rows if right_name(r.get("right")) == "put"], key=lambda r: finite_number(r.get("strike")) or 0)
        calls = sorted([r for r in expiration_rows if right_name(r.get("right")) == "call"], key=lambda r: finite_number(r.get("strike")) or 0)
        call_strikes = [finite_number(r.get("strike")) for r in calls]
        days = dte(expiration, today)

        # Pre-compute valid put-spread pairs (long_put, short_put) so the call-side
        # loops don't redundantly re-evaluate put-side constraints on every iteration.
        valid_put_pairs: list[tuple[float, float, float, float, float | None]] = []
        for long_put in puts:
            lp = finite_number(long_put.get("strike"))
            lp_ask = finite_number(long_put.get("ask"))
            if lp is None or lp_ask is None:
                continue
            for short_put in puts:
                sp = finite_number(short_put.get("strike"))
                sp_bid = finite_number(short_put.get("bid"))
                if sp is None or sp_bid is None or sp <= lp:
                    continue
                short_put_delta = finite_number(short_put.get("delta"))
                if not _short_delta_allowed(short_put_delta, min_short_delta=min_short_delta, max_short_delta=max_short_delta):
                    continue
                put_width = sp - lp
                if min_width is not None and put_width < min_width:
                    continue
                if max_width is not None and put_width > max_width:
                    break  # strikes are sorted; wider pairs only get wider from here
                valid_put_pairs.append((lp, lp_ask, sp, sp_bid, short_put_delta, short_put.get("underlying_price"), short_put.get("root")))

        for short_call in calls:
            sc = finite_number(short_call.get("strike"))
            sc_bid = finite_number(short_call.get("bid"))
            if sc is None or sc_bid is None:
                continue
            short_call_delta = finite_number(short_call.get("delta"))
            if not _short_delta_allowed(short_call_delta, min_short_delta=min_short_delta, max_short_delta=max_short_delta):
                continue

            # Only consider long_calls strictly above sc, within max_width.
            lc_start = bisect.bisect_right(call_strikes, sc)
            lc_end = len(calls)
            if max_width is not None:
                lc_end = bisect.bisect_right(call_strikes, sc + max_width)
            if min_width is not None:
                lc_start = bisect.bisect_left(call_strikes, sc + min_width, lc_start)

            for long_call in calls[lc_start:lc_end]:
                lc = finite_number(long_call.get("strike"))
                lc_ask = finite_number(long_call.get("ask"))
                if lc is None or lc_ask is None:
                    continue
                call_width = lc - sc

                for lp, lp_ask, sp, sp_bid, short_put_delta, underlying_price, root in valid_put_pairs:
                    if sc <= sp:
                        continue
                    credit = sp_bid - lp_ask + sc_bid - lc_ask
                    max_loss = max(sp - lp, call_width) - credit
                    if credit < min_credit or max_loss <= 0:
                        continue
                    return_on_risk = credit / max_loss
                    call_prob = probability_otm_from_delta(finite_number(short_call.get("delta")))
                    sp_prob = probability_otm_from_delta(short_put_delta)
                    probability_range = None
                    if sp_prob is not None and call_prob is not None:
                        probability_range = max(0.0, min(1.0, sp_prob + call_prob - 1))
                    if min_probability_range is not None and (probability_range is None or probability_range < min_probability_range):
                        continue
                    risk_adjusted_return = return_on_risk * probability_range if probability_range is not None else None
                    output.append({
                        "root": root if root is not None else ticker,
                        "expiration": expiration,
                        "strategy": "iron_condor",
                        "long_put_strike": lp,
                        "short_put_strike": sp,
                        "short_call_strike": sc,
                        "long_call_strike": lc,
                        "put_width": sp - lp,
                        "call_width": call_width,
                        "credit": credit,
                        "max_loss": max_loss * 100,
                        "return_on_risk": return_on_risk,
                        "annualized_return_on_risk": annualize(return_on_risk, days),
                        "probability_range": probability_range,
                        "risk_adjusted_return": risk_adjusted_return,
                        "annualized_risk_adjusted_return": annualize(risk_adjusted_return, days),
                        "lower_breakeven": sp - credit,
                        "upper_breakeven": sc + credit,
                        "dte": days,
                        "underlying_price": finite_number(underlying_price),
                        "short_put_delta": short_put_delta,
                        "short_call_delta": short_call_delta,
                    })
    return pl.DataFrame(output).select(_OUTPUT_COLUMNS) if output else empty_frame(_OUTPUT_COLUMNS)


def _sort_iron_condors(condors: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [column for column in ["credit", "probability_range"] if column != rank_by]
    return sort_and_limit(condors, rank_by=rank_by, tie_breakers=tie_breakers, descending=[True] * (1 + len(tie_breakers)), top_n=top_n)


def _apply_candidate_caps(condors: pl.DataFrame, *, rank_by: str, max_candidates_per_expiration: int | None, max_candidates_total: int | None) -> tuple[pl.DataFrame, int]:
    if condors.is_empty():
        return condors, 0
    original_count = len(condors)
    capped = condors
    if max_candidates_per_expiration is not None:
        frames = [_sort_iron_condors(group, rank_by=rank_by, top_n=max_candidates_per_expiration) for group in capped.partition_by("expiration", maintain_order=True)]
        capped = pl.concat(frames, how="diagonal_relaxed") if frames else empty_frame(_OUTPUT_COLUMNS)
    if max_candidates_total is not None:
        capped = _sort_iron_condors(capped, rank_by=rank_by, top_n=max_candidates_total)
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


def screen_iron_condors(request: IronCondorRequest, client: Client | None = None, *, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_iron_condors(request, conn=conn)
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
        raise classify_thetadata_error(exc, ticker=ticker, endpoint="screen_iron_condors", params=_request_params(request)) from exc
    fetched_rows = len(chain)
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=request.min_dte, max_dte=request.max_dte)
    filtered_rows = len(chain)
    condors = _build_rows(
        ticker,
        chain,
        today=today,
        min_credit=request.min_credit,
        min_width=request.min_width,
        max_width=request.max_width,
        min_short_delta=request.min_short_delta,
        max_short_delta=request.max_short_delta,
        min_probability_range=request.min_probability_range,
    )
    condors, pruned_candidate_rows = _apply_candidate_caps(
        condors,
        rank_by=request.rank_by,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(condors)
    if pruned_candidate_rows:
        warnings = (*warnings, ScreenerWarning(code="candidate_limit", message=f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible iron condor candidates."))
    condors = _sort_iron_condors(condors, rank_by=request.rank_by, top_n=request.top_n)
    return ScreenerResult(
        data=condors,
        stats=ScreenerStats(
            ticker=ticker,
            fetched_rows=fetched_rows,
            filtered_rows=filtered_rows,
            candidate_rows=candidate_rows,
            returned_rows=len(condors),
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


def _attempt_screen(request: IronCondorRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            attempt_started = time.perf_counter()
            result = screen_iron_condors(request, client=client, conn=conn)
            if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds:
                raise ThetaTimeoutError("Iron condor screening exceeded the per-ticker timeout.", ticker=request.ticker, endpoint="screen_iron_condors", params=_request_params(request), retryable=True, user_message="Iron condor screening exceeded the per-ticker timeout.")
            return result
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="screen_iron_condors", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_iron_condor_watchlist(
    tickers: list[str],
    request: IronCondorRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_iron_condors")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_screen(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="screen_iron_condors")

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


def _warm_iron_condor_request(request: IronCondorRequest, *, client: Client | None = None, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request, endpoint="warm_iron_condor_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_iron_condors(request, conn=conn)
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
        raise ThetaTimeoutError("Iron condor cache warmup exceeded the per-ticker timeout.", ticker=ticker, endpoint="warm_iron_condor_cache", params=_request_params(request), retryable=True, user_message="Iron condor cache warmup exceeded the per-ticker timeout.")
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


def _attempt_warm(request: IronCondorRequest, *, client: Client | None, retry_policy: RetryPolicy, timeout_policy: TimeoutPolicy | None = None, conn=None) -> ScreenerResult:
    attempts = 0
    last_error: ThetaDataError | None = None
    while attempts < max(retry_policy.max_attempts, 1):
        attempts += 1
        try:
            return _warm_iron_condor_request(request, client=client, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(exc, ticker=request.ticker, endpoint="warm_iron_condor_cache", params=_request_params(request))
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_iron_condor_cache(
    tickers: list[str],
    request: IronCondorRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="warm_iron_condor_cache")
        rate_limiter.wait()
        ticker_request = request.for_ticker(ticker)
        try:
            task_client = client if client is not None else (client_factory() if client_factory else create_client())
            return ticker, _attempt_warm(ticker_request, client=task_client, retry_policy=retry_policy, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            circuit_breaker.record_failure()
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="warm_iron_condor_cache")

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


def get_best_iron_condors(
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
    condors = _build_rows(ticker, chain, today=today, min_credit=min_credit, min_width=min_width, max_width=max_width)
    return sort_and_limit(condors, rank_by=rank_by, tie_breakers=["credit", "probability_range"], descending=[True, True, True], top_n=top_n)


find_best_iron_condors = get_best_iron_condors

__all__ = [
    "IronCondorRequest",
    "find_best_iron_condors",
    "get_best_iron_condors",
    "plan_iron_condors",
    "screen_iron_condor_watchlist",
    "screen_iron_condors",
    "warm_iron_condor_cache",
]
