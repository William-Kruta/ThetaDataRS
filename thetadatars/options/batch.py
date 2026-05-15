from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from ..client import Client, create_client
from ..data.cache import CachePolicy
from ..errors import (
    ThetaDataError,
    TimeoutError as ThetaTimeoutError,
    classify_thetadata_error,
)
from .snapshot.greeks_first_order import RateType, get_snapshot_greeks_first_order
from .snapshot.quote import get_snapshot_quote

Right = Literal["call", "put", "both"]


@dataclass(frozen=True, slots=True)
class DataBatchStats:
    total: int
    succeeded: int
    failed: int
    duration_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True, slots=True)
class DataTickerResult:
    ticker: str
    rows: int
    duration_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True, slots=True)
class DataTickerFailure:
    ticker: str
    error: ThetaDataError
    retryable: bool
    attempts: int


@dataclass(frozen=True, slots=True)
class DataBatchResult:
    data: pl.DataFrame
    successes: list[DataTickerResult]
    failures: list[DataTickerFailure]
    stats: DataBatchStats


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class TimeoutPolicy:
    per_ticker_seconds: float | None = None


@dataclass(frozen=True, slots=True)
class RateLimitPolicy:
    min_interval_seconds: float = 0.0


class _RateLimiter:
    def __init__(self, policy: RateLimitPolicy | None) -> None:
        self.min_interval_seconds = (
            max(policy.min_interval_seconds, 0.0)
            if policy is not None
            else 0.0
        )
        self._last_start = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        if self.min_interval_seconds <= 0:
            return
        with self._lock:
            now = time.perf_counter()
            wait_seconds = self.min_interval_seconds - (now - self._last_start)
            if wait_seconds > 0:
                time.sleep(wait_seconds)
                now = time.perf_counter()
            self._last_start = now


def _as_tickers(tickers: Sequence[str]) -> list[str]:
    cleaned = [ticker.strip().upper() for ticker in tickers if ticker and ticker.strip()]
    if len(cleaned) != len(set(cleaned)):
        deduped: list[str] = []
        seen: set[str] = set()
        for ticker in cleaned:
            if ticker in seen:
                continue
            seen.add(ticker)
            deduped.append(ticker)
        return deduped
    return cleaned


def _failure_for(
    ticker: str,
    error: Exception,
    attempts: int,
    *,
    endpoint: str,
    params: dict[str, object],
) -> DataTickerFailure:
    structured = (
        error
        if isinstance(error, ThetaDataError)
        else classify_thetadata_error(
            error,
            ticker=ticker,
            endpoint=endpoint,
            params=params,
        )
    )
    return DataTickerFailure(
        ticker=ticker,
        error=structured,
        retryable=structured.retryable,
        attempts=attempts,
    )


def _attempt_fetch(
    ticker: str,
    fetch_one: Callable[[str], pl.DataFrame],
    *,
    endpoint: str,
    params: dict[str, object],
    retry_policy: RetryPolicy,
    timeout_policy: TimeoutPolicy | None,
) -> tuple[pl.DataFrame, DataTickerResult]:
    attempts = 0
    last_error: ThetaDataError | None = None
    max_attempts = max(retry_policy.max_attempts, 1)
    while attempts < max_attempts:
        attempts += 1
        try:
            started_at = time.perf_counter()
            frame = fetch_one(ticker)
            duration = time.perf_counter() - started_at
            if (
                timeout_policy is not None
                and timeout_policy.per_ticker_seconds is not None
                and duration > timeout_policy.per_ticker_seconds
            ):
                raise ThetaTimeoutError(
                    "Batch data request exceeded the per-ticker timeout.",
                    ticker=ticker,
                    endpoint=endpoint,
                    params=params,
                    retryable=True,
                    user_message="Batch data request exceeded the per-ticker timeout.",
                )
            return frame, DataTickerResult(
                ticker=ticker,
                rows=len(frame),
                duration_seconds=duration,
            )
        except Exception as exc:
            error = (
                exc
                if isinstance(exc, ThetaDataError)
                else classify_thetadata_error(
                    exc,
                    ticker=ticker,
                    endpoint=endpoint,
                    params=params,
                )
            )
            last_error = error
            if (
                not error.retryable
                or attempts >= max_attempts
                or retry_policy.backoff_seconds <= 0
            ):
                break
            time.sleep(retry_policy.backoff_seconds)
    assert last_error is not None
    raise last_error


def _run_ticker_batch(
    tickers: Sequence[str],
    *,
    endpoint: str,
    params: dict[str, object],
    fetch_one: Callable[[str], pl.DataFrame],
    concurrency: int,
    retry_policy: RetryPolicy | None,
    timeout_policy: TimeoutPolicy | None,
    rate_limit_policy: RateLimitPolicy | None,
    on_progress: Callable[[DataTickerResult | DataTickerFailure], None] | None,
) -> DataBatchResult:
    started_at = time.perf_counter()
    ticker_list = _as_tickers(tickers)
    retry_policy = retry_policy or RetryPolicy()
    rate_limiter = _RateLimiter(rate_limit_policy)
    successes: list[DataTickerResult] = []
    failures: list[DataTickerFailure] = []
    frames: list[pl.DataFrame] = []

    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    def run_one(
        ticker: str,
    ) -> tuple[str, pl.DataFrame | None, DataTickerResult | DataTickerFailure]:
        request_params = {**params, "ticker": ticker}
        rate_limiter.wait()
        try:
            frame, result = _attempt_fetch(
                ticker,
                fetch_one,
                endpoint=endpoint,
                params=request_params,
                retry_policy=retry_policy,
                timeout_policy=timeout_policy,
            )
            return ticker, frame, result
        except Exception as exc:
            return ticker, None, _failure_for(
                ticker,
                exc,
                attempts=max(retry_policy.max_attempts, 1),
                endpoint=endpoint,
                params=request_params,
            )

    def handle_item(frame: pl.DataFrame | None, item: DataTickerResult | DataTickerFailure) -> None:
        if isinstance(item, DataTickerFailure):
            failures.append(item)
            if on_progress is not None:
                on_progress(item)
            return
        successes.append(item)
        if frame is not None and not frame.is_empty():
            frames.append(frame)
        if on_progress is not None:
            on_progress(item)

    if concurrency > 1 and len(ticker_list) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(run_one, ticker): ticker for ticker in ticker_list}
            for future in as_completed(futures):
                _, frame, item = future.result()
                handle_item(frame, item)
    else:
        for ticker in ticker_list:
            _, frame, item = run_one(ticker)
            handle_item(frame, item)

    data = pl.concat(frames, how="diagonal_relaxed") if frames else pl.DataFrame()
    return DataBatchResult(
        data=data,
        successes=successes,
        failures=failures,
        stats=DataBatchStats(
            total=len(ticker_list),
            succeeded=len(successes),
            failed=len(failures),
            duration_seconds=time.perf_counter() - started_at,
        ),
    )


def _resolve_client(
    client: Client | None,
    client_factory: Callable[[], Client] | None,
) -> Client:
    if client is not None:
        return client
    if client_factory is not None:
        return client_factory()
    return create_client()


def get_snapshot_quote_batch(
    tickers: Sequence[str],
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    strike: str = "*",
    right: Right = "both",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    conn=None,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[DataTickerResult | DataTickerFailure], None] | None = None,
) -> DataBatchResult:
    """Fetch snapshot quote data for a ticker set using the endpoint cache policy."""
    effective_concurrency = 1 if conn is not None else concurrency
    params: dict[str, object] = {
        "expiration": expiration,
        "strike": strike,
        "right": right,
        "max_dte": max_dte,
        "strike_range": strike_range,
        "min_time": min_time,
        "cache_policy": cache_policy,
    }

    def fetch_one(ticker: str) -> pl.DataFrame:
        task_client = _resolve_client(client, client_factory)
        return get_snapshot_quote(
            ticker,
            expiration,
            task_client,
            strike=strike,
            right=right,
            max_dte=max_dte,
            strike_range=strike_range,
            min_time=min_time,
            stale_threshold=stale_threshold,
            cache_policy=cache_policy,
            conn=conn,
        )

    return _run_ticker_batch(
        tickers,
        endpoint="option_snapshot_quote",
        params=params,
        fetch_one=fetch_one,
        concurrency=effective_concurrency,
        retry_policy=retry_policy,
        timeout_policy=timeout_policy,
        rate_limit_policy=rate_limit_policy,
        on_progress=on_progress,
    )


def get_snapshot_greeks_first_order_batch(
    tickers: Sequence[str],
    expiration: dt.date | str,
    client: Client | None = None,
    *,
    strike: str = "*",
    right: Right = "both",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    conn=None,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[DataTickerResult | DataTickerFailure], None] | None = None,
) -> DataBatchResult:
    """Fetch first-order Greeks for a ticker set using the endpoint cache policy."""
    effective_concurrency = 1 if conn is not None else concurrency
    params: dict[str, object] = {
        "expiration": expiration,
        "strike": strike,
        "right": right,
        "annual_dividend": annual_dividend,
        "rate_type": rate_type,
        "rate_value": rate_value,
        "stock_price": stock_price,
        "version": version,
        "max_dte": max_dte,
        "strike_range": strike_range,
        "min_time": min_time,
        "use_market_value": use_market_value,
        "cache_policy": cache_policy,
    }

    def fetch_one(ticker: str) -> pl.DataFrame:
        task_client = _resolve_client(client, client_factory)
        return get_snapshot_greeks_first_order(
            ticker,
            expiration,
            task_client,
            strike=strike,
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

    return _run_ticker_batch(
        tickers,
        endpoint="option_snapshot_greeks_first_order",
        params=params,
        fetch_one=fetch_one,
        concurrency=effective_concurrency,
        retry_policy=retry_policy,
        timeout_policy=timeout_policy,
        rate_limit_policy=rate_limit_policy,
        on_progress=on_progress,
    )


def warm_snapshot_quote_cache(
    tickers: Sequence[str],
    expiration: dt.date | str,
    client: Client | None = None,
    **kwargs,
) -> DataBatchResult:
    """Refresh snapshot quote cache rows for a ticker set."""
    kwargs["cache_policy"] = "refresh"
    return get_snapshot_quote_batch(tickers, expiration, client=client, **kwargs)


def warm_snapshot_greeks_first_order_cache(
    tickers: Sequence[str],
    expiration: dt.date | str,
    client: Client | None = None,
    **kwargs,
) -> DataBatchResult:
    """Refresh first-order Greeks cache rows for a ticker set."""
    kwargs["cache_policy"] = "refresh"
    return get_snapshot_greeks_first_order_batch(tickers, expiration, client=client, **kwargs)


__all__ = [
    "DataBatchResult",
    "DataBatchStats",
    "DataTickerFailure",
    "DataTickerResult",
    "RateLimitPolicy",
    "RetryPolicy",
    "TimeoutPolicy",
    "get_snapshot_greeks_first_order_batch",
    "get_snapshot_quote_batch",
    "warm_snapshot_greeks_first_order_cache",
    "warm_snapshot_quote_cache",
]
