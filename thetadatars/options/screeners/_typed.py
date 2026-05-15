import threading
import time
from dataclasses import dataclass, field
from typing import Literal

import polars as pl

from ...data.cache import CacheCoverage
from ...errors import (
    ThetaDataError,
    TransientNetworkError,
    classify_thetadata_error,
)

PlanCost = Literal["low", "medium", "high"]


@dataclass(frozen=True, slots=True)
class ScreenerWarning:
    code: str
    message: str
    severity: Literal["info", "warning"] = "warning"


@dataclass(frozen=True, slots=True)
class ScreenerStats:
    ticker: str
    fetched_rows: int
    filtered_rows: int
    candidate_rows: int
    returned_rows: int
    duration_seconds: float = field(default=0.0, compare=False)
    greeks_source: str | None = None
    cache_hits: int = 0
    cache_misses: int = 0
    upstream_calls: int | None = None
    cache_policy: str | None = None
    pruned_candidate_rows: int = 0


@dataclass(frozen=True, slots=True)
class ScreenerResult:
    data: pl.DataFrame
    stats: ScreenerStats
    warnings: tuple[ScreenerWarning, ...] = ()


@dataclass(frozen=True, slots=True)
class ScreenerPlan:
    ticker: str
    strategy: str
    expected_endpoint: str
    upstream_calls: int | None
    cache_hits: int
    cache_misses: int
    cost: PlanCost
    local_computation: PlanCost
    warnings: tuple[ScreenerWarning, ...]
    cache_coverage: tuple[CacheCoverage, ...] = ()
    estimated_contracts: int | None = None


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


@dataclass(frozen=True, slots=True)
class CircuitBreakerPolicy:
    max_failures: int | None = None


@dataclass(frozen=True, slots=True)
class TickerResult:
    ticker: str
    stats: ScreenerStats


@dataclass(frozen=True, slots=True)
class TickerFailure:
    ticker: str
    error: ThetaDataError
    retryable: bool
    attempts: int


@dataclass(frozen=True, slots=True)
class BatchStats:
    total: int
    succeeded: int
    failed: int
    duration_seconds: float = field(default=0.0, compare=False)


@dataclass(frozen=True, slots=True)
class BatchResult:
    data: pl.DataFrame
    successes: list[TickerResult]
    failures: list[TickerFailure]
    stats: BatchStats


@dataclass(frozen=True, slots=True)
class WarmCacheResult:
    successes: list[TickerResult]
    failures: list[TickerFailure]
    stats: BatchStats


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


class _CircuitBreaker:
    def __init__(self, policy: CircuitBreakerPolicy | None) -> None:
        self.max_failures = (
            policy.max_failures
            if policy is not None and policy.max_failures is not None and policy.max_failures > 0
            else None
        )
        self.failures = 0
        self._lock = threading.Lock()

    def is_open(self) -> bool:
        with self._lock:
            return self.max_failures is not None and self.failures >= self.max_failures

    def record_failure(self) -> None:
        with self._lock:
            if self.max_failures is not None:
                self.failures += 1


def _failure_for(
    ticker: str,
    error: Exception,
    attempts: int,
    *,
    endpoint: str,
) -> TickerFailure:
    structured = (
        error
        if isinstance(error, ThetaDataError)
        else classify_thetadata_error(
            error,
            ticker=ticker,
            endpoint=endpoint,
            params={"ticker": ticker},
        )
    )
    return TickerFailure(
        ticker=ticker,
        error=structured,
        retryable=structured.retryable,
        attempts=attempts,
    )


def _circuit_open_failure(ticker: str, *, endpoint: str) -> TickerFailure:
    error = TransientNetworkError(
        "Circuit breaker is open after repeated watchlist failures.",
        ticker=ticker,
        endpoint=endpoint,
        retryable=True,
        user_message="Screening is temporarily paused after repeated failures.",
    )
    return TickerFailure(ticker=ticker, error=error, retryable=True, attempts=0)


__all__ = [
    "BatchResult",
    "BatchStats",
    "CircuitBreakerPolicy",
    "PlanCost",
    "RateLimitPolicy",
    "RetryPolicy",
    "ScreenerPlan",
    "ScreenerResult",
    "ScreenerStats",
    "ScreenerWarning",
    "TickerFailure",
    "TickerResult",
    "TimeoutPolicy",
    "WarmCacheResult",
]
