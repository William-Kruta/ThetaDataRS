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
    right_name,
)
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

Right = Literal["call", "put", "both"]
RankBy = Literal["annualized_return_on_risk", "return_on_risk", "max_profit", "debit", "probability_itm"]

_RIGHT_VALUES = {"call", "put", "both"}
_RANK_BY_VALUES = {"annualized_return_on_risk", "return_on_risk", "max_profit", "debit", "probability_itm"}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}
_OUTPUT_COLUMNS = [
    "root", "expiration", "right", "spread_type", "long_strike", "short_strike",
    "width", "debit", "max_profit", "max_loss", "return_on_risk",
    "annualized_return_on_risk", "breakeven", "probability_itm", "dte",
    "underlying_price", "long_ask", "short_bid", "long_delta", "short_delta",
    "long_implied_vol", "short_implied_vol", "long_timestamp", "short_timestamp",
]


@dataclass(frozen=True, slots=True)
class DebitSpreadRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    right: Right = "both"
    min_dte: int | None = None
    max_dte: int | None = None
    min_debit: float = 0.01
    max_debit: float | None = None
    min_width: float | None = None
    max_width: float | None = None
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

    def for_ticker(self, ticker: str) -> "DebitSpreadRequest":
        return replace(self, ticker=ticker)


def _invalid_request(
    message: str,
    *,
    ticker: str | None,
    params: dict[str, object],
    endpoint: str = "screen_debit_spreads",
) -> InvalidRequestError:
    return InvalidRequestError(
        message,
        ticker=ticker,
        endpoint=endpoint,
        params=params,
        retryable=False,
        user_message=message,
    )


def _request_params(request: DebitSpreadRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "right": request.right,
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
        "min_debit": request.min_debit,
        "max_debit": request.max_debit,
        "min_width": request.min_width,
        "max_width": request.max_width,
        "strike_range": request.strike_range,
        "top_n": request.top_n,
        "rank_by": request.rank_by,
        "greeks_source": request.greeks_source,
        "cache_policy": request.cache_policy,
        "allow_full_chain": request.allow_full_chain,
        "max_candidates_per_expiration": request.max_candidates_per_expiration,
        "max_candidates_total": request.max_candidates_total,
    }


def _validate_request(
    request: DebitSpreadRequest,
    *,
    endpoint: str = "screen_debit_spreads",
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
    if request.right not in _RIGHT_VALUES:
        raise _invalid_request("right must be one of: call, put, both", ticker=request.ticker, params=params, endpoint=endpoint)
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
    if request.min_dte is not None and request.max_dte is not None and request.min_dte > request.max_dte:
        raise _invalid_request("min_dte cannot be greater than max_dte", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.min_width is not None and request.max_width is not None and request.min_width > request.max_width:
        raise _invalid_request("min_width cannot be greater than max_width", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_debit is not None and request.min_debit > request.max_debit:
        raise _invalid_request("min_debit cannot be greater than max_debit", ticker=request.ticker, params=params, endpoint=endpoint)
    if request.max_candidates_per_expiration is not None and request.max_candidates_per_expiration <= 0:
        raise _invalid_request(
            "max_candidates_per_expiration must be greater than zero",
            ticker=request.ticker,
            params=params,
            endpoint=endpoint,
        )
    if request.max_candidates_total is not None and request.max_candidates_total <= 0:
        raise _invalid_request("max_candidates_total must be greater than zero", ticker=request.ticker, params=params, endpoint=endpoint)

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


def _planned_endpoint_and_params(request: DebitSpreadRequest, expiration: dt.date | str) -> tuple[str, dict[str, object]]:
    if request.greeks_source in {"local", "none"}:
        return (
            "option_snapshot_quote",
            {
                "expiration": expiration,
                "strike": "*",
                "right": request.right,
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
            "right": request.right,
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


def _planned_cost(request: DebitSpreadRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: DebitSpreadRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_debit_spreads(request: DebitSpreadRequest, *, conn=None) -> ScreenerPlan:
    """Explain the expected debit-spread screening cost before fetching data."""
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
            strategy="debit_spread",
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


def _probability_itm(long_leg: dict, short_leg: dict) -> float | None:
    long_delta = finite_number(long_leg.get("delta"))
    short_delta = finite_number(short_leg.get("delta"))
    if long_delta is None and short_delta is None:
        return None
    if long_delta is None:
        return abs(short_delta)
    if short_delta is None:
        return abs(long_delta)
    return max(0.0, min(1.0, (abs(long_delta) + abs(short_delta)) / 2))


def _build_rows(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    right: Right,
    min_debit: float,
    max_debit: float | None,
    min_width: float | None,
    max_width: float | None,
) -> pl.DataFrame:
    rows = normalize_rows(chain)
    if not rows:
        return empty_frame(_OUTPUT_COLUMNS)
    output = []
    for expiration, expiration_rows in grouped_by_expiration(rows).items():
        days = dte(expiration, today)
        for option_right in ("call", "put"):
            if right != "both" and right != option_right:
                continue
            side_rows = sorted([row for row in expiration_rows if right_name(row.get("right")) == option_right], key=lambda row: finite_number(row.get("strike")) or 0)
            by_strike = row_by_strike(side_rows)
            strikes = sorted(by_strike)
            for long_strike in strikes:
                for short_strike in strikes:
                    if option_right == "call" and short_strike <= long_strike:
                        continue
                    if option_right == "put" and short_strike >= long_strike:
                        continue
                    long_leg = by_strike[long_strike]
                    short_leg = by_strike[short_strike]
                    long_ask = finite_number(long_leg.get("ask"))
                    short_bid = finite_number(short_leg.get("bid"))
                    if long_ask is None or short_bid is None:
                        continue
                    width = abs(short_strike - long_strike)
                    debit = long_ask - short_bid
                    max_profit = width - debit
                    if debit < min_debit or max_profit <= 0:
                        continue
                    if max_debit is not None and debit > max_debit:
                        continue
                    if min_width is not None and width < min_width:
                        continue
                    if max_width is not None and width > max_width:
                        continue
                    spread_type = "bull_call_debit" if option_right == "call" else "bear_put_debit"
                    breakeven = long_strike + debit if option_right == "call" else long_strike - debit
                    return_on_risk = max_profit / debit
                    output.append({
                        "root": long_leg.get("root", ticker),
                        "expiration": expiration,
                        "right": option_right,
                        "spread_type": spread_type,
                        "long_strike": long_strike,
                        "short_strike": short_strike,
                        "width": width,
                        "debit": debit,
                        "max_profit": max_profit,
                        "max_loss": debit * 100,
                        "return_on_risk": return_on_risk,
                        "annualized_return_on_risk": annualize(return_on_risk, days),
                        "breakeven": breakeven,
                        "probability_itm": _probability_itm(long_leg, short_leg),
                        "dte": days,
                        "underlying_price": finite_number(long_leg.get("underlying_price")),
                        "long_ask": long_ask,
                        "short_bid": short_bid,
                        "long_delta": finite_number(long_leg.get("delta")),
                        "short_delta": finite_number(short_leg.get("delta")),
                        "long_implied_vol": finite_number(long_leg.get("implied_vol")),
                        "short_implied_vol": finite_number(short_leg.get("implied_vol")),
                        "long_timestamp": long_leg.get("timestamp"),
                        "short_timestamp": short_leg.get("timestamp"),
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


def _sort_debit_spreads(spreads: pl.DataFrame, *, rank_by: str, top_n: int | None) -> pl.DataFrame:
    tie_breakers = [column for column in ["max_profit", "probability_itm"] if column != rank_by]
    descending = [True]
    descending.extend(True for _ in tie_breakers)
    return sort_and_limit(spreads, rank_by=rank_by, tie_breakers=tie_breakers, descending=descending, top_n=top_n)


def _apply_candidate_caps(
    spreads: pl.DataFrame,
    *,
    rank_by: str,
    max_candidates_per_expiration: int | None,
    max_candidates_total: int | None,
) -> tuple[pl.DataFrame, int]:
    if spreads.is_empty():
        return spreads, 0

    original_count = len(spreads)
    capped = spreads
    if max_candidates_per_expiration is not None:
        frames = [
            _sort_debit_spreads(group, rank_by=rank_by, top_n=max_candidates_per_expiration)
            for group in capped.partition_by("expiration", maintain_order=True)
        ]
        capped = pl.concat(frames, how="diagonal_relaxed") if frames else empty_frame(_OUTPUT_COLUMNS)
    if max_candidates_total is not None:
        capped = _sort_debit_spreads(capped, rank_by=rank_by, top_n=max_candidates_total)
    return capped, original_count - len(capped)


def _execute_debit_spread_request(
    request: DebitSpreadRequest,
    *,
    client: Client | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_debit_spreads(request, conn=conn)

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
            ticker=ticker,
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
            endpoint="screen_debit_spreads",
            params=_request_params(request),
        ) from exc

    fetched_rows = len(chain)
    chain = filter_expiration_dte(chain, expiration=expiration, today=today, min_dte=request.min_dte, max_dte=request.max_dte)
    filtered_rows = len(chain)
    spreads = _build_rows(
        ticker,
        chain,
        today=today,
        right=request.right,
        min_debit=request.min_debit,
        max_debit=request.max_debit,
        min_width=request.min_width,
        max_width=request.max_width,
    )
    spreads, pruned_candidate_rows = _apply_candidate_caps(
        spreads,
        rank_by=request.rank_by,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(spreads)
    if pruned_candidate_rows:
        warnings = (
            *warnings,
            ScreenerWarning(
                code="candidate_limit",
                message=f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible debit spread candidates.",
            ),
        )
    spreads = _sort_debit_spreads(spreads, rank_by=request.rank_by, top_n=request.top_n)

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


def screen_debit_spreads(request: DebitSpreadRequest, client: Client | None = None, *, conn=None) -> ScreenerResult:
    """Screen debit spreads from a typed request and return diagnostics."""
    return _execute_debit_spread_request(request, client=client, conn=conn)


def _attempt_screen(
    request: DebitSpreadRequest,
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
            result = screen_debit_spreads(request, client=client, conn=conn)
            if (
                timeout_policy is not None
                and timeout_policy.per_ticker_seconds is not None
                and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds
            ):
                raise ThetaTimeoutError(
                    "Debit spread screening exceeded the per-ticker timeout.",
                    ticker=request.ticker,
                    endpoint="screen_debit_spreads",
                    params=_request_params(request),
                    retryable=True,
                    user_message="Debit spread screening exceeded the per-ticker timeout.",
                )
            return result
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(
                exc,
                ticker=request.ticker,
                endpoint="screen_debit_spreads",
                params=_request_params(request),
            )
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def screen_debit_spread_watchlist(
    tickers: list[str],
    request: DebitSpreadRequest,
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
    """Screen a watchlist while returning per-ticker partial failures."""
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_debit_spreads")
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
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="screen_debit_spreads")

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
        stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at),
    )


def _warm_debit_spread_request(
    request: DebitSpreadRequest,
    *,
    client: Client | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request, endpoint="warm_debit_spread_cache")
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_debit_spreads(request, conn=conn)

    if client is None:
        client = create_client()

    diagnostics: dict[str, object] = {}
    chain = get_first_order_chain(
        ticker=ticker,
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
    if timeout_policy is not None and timeout_policy.per_ticker_seconds is not None and duration > timeout_policy.per_ticker_seconds:
        raise ThetaTimeoutError(
            "Debit spread cache warmup exceeded the per-ticker timeout.",
            ticker=ticker,
            endpoint="warm_debit_spread_cache",
            params=_request_params(request),
            retryable=True,
            user_message="Debit spread cache warmup exceeded the per-ticker timeout.",
        )
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


def _attempt_warm(
    request: DebitSpreadRequest,
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
            return _warm_debit_spread_request(request, client=client, timeout_policy=timeout_policy, conn=conn)
        except Exception as exc:
            error = exc if isinstance(exc, ThetaDataError) else classify_thetadata_error(
                exc,
                ticker=request.ticker,
                endpoint="warm_debit_spread_cache",
                params=_request_params(request),
            )
            last_error = error
            if not error.retryable or attempts >= max(retry_policy.max_attempts, 1) or retry_policy.backoff_seconds <= 0:
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def warm_debit_spread_cache(
    tickers: list[str],
    request: DebitSpreadRequest,
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
    """Warm cache inputs used by debit-spread screening without ranking spreads."""
    started_at = time.perf_counter()
    retry_policy = retry_policy or RetryPolicy()
    successes: list[TickerResult] = []
    failures: list[TickerFailure] = []
    tickers = list(tickers)
    rate_limiter = _RateLimiter(rate_limit_policy)
    circuit_breaker = _CircuitBreaker(circuit_breaker_policy)

    def run_one(ticker: str) -> tuple[str, ScreenerResult | TickerFailure]:
        if circuit_breaker.is_open():
            return ticker, _circuit_open_failure(ticker, endpoint="warm_debit_spread_cache")
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
            return ticker, _failure_for(ticker, exc, attempts=max(retry_policy.max_attempts, 1), endpoint="warm_debit_spread_cache")

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
        stats=BatchStats(total=len(tickers), succeeded=len(successes), failed=len(failures), duration_seconds=time.perf_counter() - started_at),
    )


def get_best_debit_spreads(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "both",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_debit: float = 0.01,
    max_debit: float | None = None,
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
    greeks_source: GreekSource = "auto",
    fallback_to_local_greeks: bool = True,
    local_greeks_steps: int | Literal["fast", "balanced", "accurate"] = 150,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    max_candidates_per_expiration: int | None = None,
    max_candidates_total: int | None = None,
    conn=None,
) -> pl.DataFrame:
    request = DebitSpreadRequest(
        ticker=ticker,
        expiration=expiration,
        right=right,
        min_dte=min_dte,
        max_dte=max_dte,
        min_debit=min_debit,
        max_debit=max_debit,
        min_width=min_width,
        max_width=max_width,
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
    return screen_debit_spreads(request, client=client, conn=conn).data


find_best_debit_spreads = get_best_debit_spreads

__all__ = [
    "DebitSpreadRequest",
    "get_best_debit_spreads",
    "find_best_debit_spreads",
    "plan_debit_spreads",
    "screen_debit_spreads",
    "screen_debit_spread_watchlist",
    "warm_debit_spread_cache",
]
