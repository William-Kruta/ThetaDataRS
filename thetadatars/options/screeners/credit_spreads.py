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
from ._strategy_utils import expiration_date as _norm_expiration_date
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
RankBy = Literal[
    "annualized_return_on_risk",
    "annualized_risk_adjusted_return",
    "return_on_risk",
    "risk_adjusted_return",
    "credit_to_width",
    "credit",
]

_RIGHT_VALUES = {"call", "put", "both"}
_RANK_BY_VALUES = {
    "annualized_return_on_risk",
    "annualized_risk_adjusted_return",
    "return_on_risk",
    "risk_adjusted_return",
    "credit_to_width",
    "credit",
}
_GREEKS_SOURCE_VALUES = {"auto", "thetadata", "local", "none"}
_CACHE_POLICY_VALUES = {"prefer_cache", "cache_only", "refresh", "no_cache"}
_OUTPUT_COLUMNS = [
    "root",
    "expiration",
    "right",
    "spread_type",
    "short_strike",
    "long_strike",
    "width",
    "credit",
    "max_loss",
    "return_on_risk",
    "annualized_return_on_risk",
    "probability_otm",
    "risk_adjusted_return",
    "annualized_risk_adjusted_return",
    "credit_to_width",
    "breakeven",
    "dte",
    "underlying_price",
    "short_bid",
    "long_ask",
    "short_delta",
    "long_delta",
    "short_implied_vol",
    "long_implied_vol",
    "short_timestamp",
    "long_timestamp",
]


@dataclass(frozen=True, slots=True)
class CreditSpreadRequest:
    ticker: str | None = None
    expiration: dt.date | str = "*"
    right: Right = "both"
    min_dte: int | None = None
    max_dte: int | None = None
    min_width: float | None = None
    max_width: float | None = None
    min_credit: float = 0.01
    min_credit_to_width: float | None = None
    min_bid: float = 0.01
    max_long_ask: float | None = None
    min_short_delta: float | None = None
    max_short_delta: float | None = None
    include_itm: bool = False
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

    def for_ticker(self, ticker: str) -> "CreditSpreadRequest":
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
        endpoint="screen_credit_spreads",
        params=params,
        retryable=False,
        user_message=message,
    )


def _request_params(request: CreditSpreadRequest) -> dict[str, object]:
    return {
        "expiration": request.expiration,
        "right": request.right,
        "min_dte": request.min_dte,
        "max_dte": request.max_dte,
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
    request: CreditSpreadRequest,
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
    if request.right not in _RIGHT_VALUES:
        raise _invalid_request(
            "right must be one of: call, put, both",
            ticker=request.ticker,
            params=params,
        )
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
        request.min_width is not None
        and request.max_width is not None
        and request.min_width > request.max_width
    ):
        raise _invalid_request(
            "min_width cannot be greater than max_width",
            ticker=request.ticker,
            params=params,
        )
    if (
        request.max_candidates_per_expiration is not None
        and request.max_candidates_per_expiration <= 0
    ):
        raise _invalid_request(
            "max_candidates_per_expiration must be greater than zero",
            ticker=request.ticker,
            params=params,
        )
    if request.max_candidates_total is not None and request.max_candidates_total <= 0:
        raise _invalid_request(
            "max_candidates_total must be greater than zero",
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
    request: CreditSpreadRequest,
    expiration: dt.date | str,
) -> tuple[str, dict[str, object]]:
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


def _planned_cost(request: CreditSpreadRequest, expiration: dt.date | str) -> PlanCost:
    if expiration == "*" and request.strike_range is None:
        return "high"
    if expiration == "*" or request.greeks_source == "local":
        return "medium"
    return "low"


def _planned_local_computation(request: CreditSpreadRequest, expiration: dt.date | str) -> PlanCost:
    if request.greeks_source != "local":
        return "low"
    if expiration == "*" and request.strike_range is None:
        return "high"
    return "medium"


def plan_credit_spreads(
    request: CreditSpreadRequest,
    *,
    conn=None,
) -> ScreenerPlan:
    """Explain the expected credit-spread screening cost before fetching data."""
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
            strategy="credit_spread",
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


def plan_screener(
    request: CreditSpreadRequest,
    *,
    conn=None,
) -> ScreenerPlan:
    if isinstance(request, CreditSpreadRequest):
        return plan_credit_spreads(request, conn=conn)
    raise TypeError("plan_screener currently supports CreditSpreadRequest")


def _latest_underlying_price(rows: list[dict]) -> float | None:
    for row in rows:
        price = finite_number(row.get("underlying_price"))
        if price is not None and price > 0:
            return price
    return None


def _passes_delta_filter(
    delta: float | None,
    min_short_delta: float | None,
    max_short_delta: float | None,
) -> bool:
    if delta is None:
        return min_short_delta is None and max_short_delta is None
    abs_delta = abs(delta)
    if min_short_delta is not None and abs_delta < min_short_delta:
        return False
    if max_short_delta is not None and abs_delta > max_short_delta:
        return False
    return True


def _build_spread_row(
    ticker: str,
    expiration: dt.date,
    right: Literal["call", "put"],
    short_leg: dict,
    long_leg: dict,
    dte: int,
) -> dict | None:
    short_strike = finite_number(short_leg.get("strike"))
    long_strike = finite_number(long_leg.get("strike"))
    short_bid = finite_number(short_leg.get("bid"))
    long_ask = finite_number(long_leg.get("ask"))

    if short_strike is None or long_strike is None or short_bid is None or long_ask is None:
        return None

    width = abs(short_strike - long_strike)
    credit = short_bid - long_ask
    max_loss = width - credit
    if width <= 0 or max_loss <= 0:
        return None

    short_delta = finite_number(short_leg.get("delta"))
    probability_otm = probability_otm_from_delta(short_delta)
    return_on_risk = credit / max_loss
    annualized_return_on_risk = return_on_risk * (365 / dte) if dte > 0 else None
    risk_adjusted_return = (
        return_on_risk * probability_otm if probability_otm is not None else None
    )
    annualized_risk_adjusted_return = (
        annualized_return_on_risk * probability_otm
        if annualized_return_on_risk is not None and probability_otm is not None
        else None
    )
    credit_to_width = credit / width
    breakeven = short_strike + credit if right == "call" else short_strike - credit
    spread_type = "bear_call_credit" if right == "call" else "bull_put_credit"

    return {
        "root": short_leg.get("root", ticker),
        "expiration": expiration,
        "right": right,
        "spread_type": spread_type,
        "short_strike": short_strike,
        "long_strike": long_strike,
        "width": width,
        "credit": credit,
        "max_loss": max_loss * 100,
        "return_on_risk": return_on_risk,
        "annualized_return_on_risk": annualized_return_on_risk,
        "probability_otm": probability_otm,
        "risk_adjusted_return": risk_adjusted_return,
        "annualized_risk_adjusted_return": annualized_risk_adjusted_return,
        "credit_to_width": credit_to_width,
        "breakeven": breakeven,
        "dte": dte,
        "underlying_price": finite_number(short_leg.get("underlying_price")),
        "short_bid": short_bid,
        "long_ask": long_ask,
        "short_delta": short_delta,
        "long_delta": finite_number(long_leg.get("delta")),
        "short_implied_vol": finite_number(short_leg.get("implied_vol")),
        "long_implied_vol": finite_number(long_leg.get("implied_vol")),
        "short_timestamp": short_leg.get("timestamp"),
        "long_timestamp": long_leg.get("timestamp"),
    }


def _build_credit_spreads(
    ticker: str,
    chain: pl.DataFrame,
    *,
    today: dt.date,
    include_itm: bool,
    min_width: float | None,
    max_width: float | None,
    min_credit: float,
    min_credit_to_width: float | None,
    min_bid: float,
    max_long_ask: float | None,
    min_short_delta: float | None,
    max_short_delta: float | None,
    max_candidates_per_expiration: int | None,
    max_candidates_total: int | None,
) -> tuple[pl.DataFrame, int]:
    if chain.is_empty():
        return pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS}), 0

    rows = chain.to_dicts()
    underlying_price = _latest_underlying_price(rows)
    spreads = []
    pruned_candidate_rows = 0

    expirations = sorted({_norm_expiration_date(row.get("expiration")) for row in rows if row.get("expiration") is not None})
    for expiration_date in expirations:
        dte = (expiration_date - today).days
        expiration_rows = [row for row in rows if row.get("expiration") is not None and _norm_expiration_date(row.get("expiration")) == expiration_date]
        expiration_candidates = 0

        for option_right in ("call", "put"):
            right_rows = [
                row for row in expiration_rows
                if right_name(row.get("right")) == option_right
            ]
            if not right_rows:
                continue

            if underlying_price is not None and not include_itm:
                if option_right == "call":
                    right_rows = [
                        row for row in right_rows
                        if (finite_number(row.get("strike")) or 0) > underlying_price
                    ]
                else:
                    right_rows = [
                        row for row in right_rows
                        if (finite_number(row.get("strike")) or 0) < underlying_price
                    ]

            right_rows = sorted(right_rows, key=lambda row: finite_number(row.get("strike")) or 0)
            for short_leg in right_rows:
                short_strike = finite_number(short_leg.get("strike"))
                short_bid = finite_number(short_leg.get("bid"))
                short_delta = finite_number(short_leg.get("delta"))
                if short_strike is None or short_bid is None:
                    continue
                if short_bid < min_bid:
                    continue
                if not _passes_delta_filter(short_delta, min_short_delta, max_short_delta):
                    continue

                for long_leg in right_rows:
                    long_strike = finite_number(long_leg.get("strike"))
                    long_ask = finite_number(long_leg.get("ask"))
                    if long_strike is None or long_ask is None:
                        continue
                    if max_long_ask is not None and long_ask > max_long_ask:
                        continue
                    if option_right == "call" and long_strike <= short_strike:
                        continue
                    if option_right == "put" and long_strike >= short_strike:
                        continue

                    width = abs(short_strike - long_strike)
                    if min_width is not None and width < min_width:
                        continue
                    if max_width is not None and width > max_width:
                        continue

                    spread = _build_spread_row(ticker, expiration_date, option_right, short_leg, long_leg, dte)
                    if spread is None:
                        continue
                    if spread["credit"] < min_credit:
                        continue
                    if min_credit_to_width is not None and spread["credit_to_width"] < min_credit_to_width:
                        continue
                    expiration_limit_reached = (
                        max_candidates_per_expiration is not None
                        and expiration_candidates >= max_candidates_per_expiration
                    )
                    total_limit_reached = (
                        max_candidates_total is not None
                        and len(spreads) >= max_candidates_total
                    )
                    if expiration_limit_reached or total_limit_reached:
                        pruned_candidate_rows += 1
                        continue
                    spreads.append(spread)
                    expiration_candidates += 1

    if not spreads:
        return (
            pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS}),
            pruned_candidate_rows,
        )

    return pl.DataFrame(spreads).select(_OUTPUT_COLUMNS), pruned_candidate_rows


def _execute_credit_spread_request(
    request: CreditSpreadRequest,
    *,
    client: Client | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_credit_spreads(request, conn=conn)

    if expiration != "*":
        days_to_expiration = (expiration - today).days
        if request.min_dte is not None and days_to_expiration < request.min_dte:
            data = pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})
            return ScreenerResult(
                data=data,
                stats=ScreenerStats(
                    ticker=ticker,
                    fetched_rows=0,
                    filtered_rows=0,
                    candidate_rows=0,
                    returned_rows=0,
                    duration_seconds=time.perf_counter() - started_at,
                    greeks_source=None,
                    cache_hits=plan.cache_hits,
                    cache_misses=plan.cache_misses,
                    upstream_calls=plan.upstream_calls,
                    cache_policy=request.cache_policy,
                ),
                warnings=warnings,
            )
        if request.max_dte is not None and days_to_expiration > request.max_dte:
            data = pl.DataFrame(schema={column: pl.Null for column in _OUTPUT_COLUMNS})
            return ScreenerResult(
                data=data,
                stats=ScreenerStats(
                    ticker=ticker,
                    fetched_rows=0,
                    filtered_rows=0,
                    candidate_rows=0,
                    returned_rows=0,
                    duration_seconds=time.perf_counter() - started_at,
                    greeks_source=None,
                    cache_hits=plan.cache_hits,
                    cache_misses=plan.cache_misses,
                    upstream_calls=plan.upstream_calls,
                    cache_policy=request.cache_policy,
                ),
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
            endpoint="screen_credit_spreads",
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

    spreads, pruned_candidate_rows = _build_credit_spreads(
        ticker,
        chain,
        today=today,
        include_itm=request.include_itm,
        min_width=request.min_width,
        max_width=request.max_width,
        min_credit=request.min_credit,
        min_credit_to_width=request.min_credit_to_width,
        min_bid=request.min_bid,
        max_long_ask=request.max_long_ask,
        min_short_delta=request.min_short_delta,
        max_short_delta=request.max_short_delta,
        max_candidates_per_expiration=request.max_candidates_per_expiration,
        max_candidates_total=request.max_candidates_total,
    )
    candidate_rows = len(spreads)
    if pruned_candidate_rows:
        warnings = (
            *warnings,
            ScreenerWarning(
                code="candidate_limit",
                message=(
                    f"Candidate limits pruned {pruned_candidate_rows} otherwise eligible "
                    "credit spread candidates."
                ),
            ),
        )

    if not spreads.is_empty():
        spreads = spreads.sort(
            [request.rank_by, "credit", "return_on_risk"],
            descending=[True, True, True],
            nulls_last=True,
        )
        if request.top_n is not None and request.top_n > 0:
            spreads = spreads.head(request.top_n)

    stats = ScreenerStats(
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
    )
    return ScreenerResult(data=spreads, stats=stats, warnings=warnings)


def screen_credit_spreads(
    request: CreditSpreadRequest,
    client: Client | None = None,
    *,
    conn=None,
) -> ScreenerResult:
    """Screen credit spreads from a typed request and return data plus diagnostics."""
    return _execute_credit_spread_request(request, client=client, conn=conn)


def _attempt_screen(
    request: CreditSpreadRequest,
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
            result = screen_credit_spreads(request, client=client, conn=conn)
            if (
                timeout_policy is not None
                and timeout_policy.per_ticker_seconds is not None
                and time.perf_counter() - attempt_started > timeout_policy.per_ticker_seconds
            ):
                raise ThetaTimeoutError(
                    "Credit spread screening exceeded the per-ticker timeout.",
                    ticker=request.ticker,
                    endpoint="screen_credit_spreads",
                    params=_request_params(request),
                    retryable=True,
                    user_message="Credit spread screening exceeded the per-ticker timeout.",
                )
            return result
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ThetaDataError)
                else classify_thetadata_error(
                    exc,
                    ticker=request.ticker,
                    endpoint="screen_credit_spreads",
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


def screen_credit_spread_watchlist(
    tickers: list[str],
    request: CreditSpreadRequest,
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
            return ticker, _circuit_open_failure(ticker, endpoint="screen_credit_spreads")
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
                endpoint="screen_credit_spreads",
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


def _warm_credit_spread_request(
    request: CreditSpreadRequest,
    *,
    client: Client | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    conn=None,
) -> ScreenerResult:
    started_at = time.perf_counter()
    expiration, warnings = _validate_request(request)
    ticker = request.ticker or ""
    today = dt.date.today()
    plan = plan_credit_spreads(request, conn=conn)

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
    if (
        timeout_policy is not None
        and timeout_policy.per_ticker_seconds is not None
        and duration > timeout_policy.per_ticker_seconds
    ):
        raise ThetaTimeoutError(
            "Credit spread cache warmup exceeded the per-ticker timeout.",
            ticker=ticker,
            endpoint="warm_credit_spread_cache",
            params=_request_params(request),
            retryable=True,
            user_message="Credit spread cache warmup exceeded the per-ticker timeout.",
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
    request: CreditSpreadRequest,
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
            return _warm_credit_spread_request(
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
                    endpoint="warm_credit_spread_cache",
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


def warm_credit_spread_cache(
    tickers: list[str],
    request: CreditSpreadRequest,
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
    """Warm the cache inputs used by credit-spread screening without ranking spreads."""
    started_at = time.perf_counter()
    retry_policy = retry_policy or RetryPolicy()
    successes: list[TickerResult] = []
    failures: list[TickerFailure] = []
    tickers = list(tickers)
    rate_limiter = _RateLimiter(rate_limit_policy)
    circuit_breaker = _CircuitBreaker(circuit_breaker_policy)

    def run_one(ticker: str) -> tuple[str, ScreenerResult | TickerFailure]:
        if circuit_breaker.is_open():
            return ticker, _circuit_open_failure(ticker, endpoint="screen_credit_spreads")
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
                endpoint="screen_credit_spreads",
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


def get_best_credit_spreads(
    ticker: str,
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    right: Right = "both",
    min_dte: int | None = None,
    max_dte: int | None = None,
    min_width: float | None = None,
    max_width: float | None = None,
    min_credit: float = 0.01,
    min_credit_to_width: float | None = None,
    min_bid: float = 0.01,
    max_long_ask: float | None = None,
    min_short_delta: float | None = None,
    max_short_delta: float | None = None,
    include_itm: bool = False,
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
    """Find the best yielding vertical option credit spreads for an expiration.

    The screener sells the richer near-the-money leg at bid and buys the farther
    out-of-the-money hedge at ask. Results are ranked by annualized return on
    risk by default.
    """
    request = CreditSpreadRequest(
        ticker=ticker,
        expiration=expiration,
        right=right,
        min_dte=min_dte,
        max_dte=max_dte,
        min_width=min_width,
        max_width=max_width,
        min_credit=min_credit,
        min_credit_to_width=min_credit_to_width,
        min_bid=min_bid,
        max_long_ask=max_long_ask,
        min_short_delta=min_short_delta,
        max_short_delta=max_short_delta,
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
        max_candidates_per_expiration=max_candidates_per_expiration,
        max_candidates_total=max_candidates_total,
        allow_full_chain=True,
    )
    return _execute_credit_spread_request(request, client=client, conn=conn).data


find_best_credit_spreads = get_best_credit_spreads


__all__ = [
    "BatchResult",
    "BatchStats",
    "CircuitBreakerPolicy",
    "CreditSpreadRequest",
    "PlanCost",
    "RateLimitPolicy",
    "RetryPolicy",
    "ScreenerResult",
    "ScreenerStats",
    "ScreenerWarning",
    "ScreenerPlan",
    "TimeoutPolicy",
    "TickerFailure",
    "TickerResult",
    "WarmCacheResult",
    "get_best_credit_spreads",
    "find_best_credit_spreads",
    "plan_credit_spreads",
    "plan_screener",
    "screen_credit_spreads",
    "screen_credit_spread_watchlist",
    "warm_credit_spread_cache",
]
