from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from ...client import Client
from .call import (
    CallRequest,
    plan_calls,
    screen_call_watchlist,
    screen_calls,
    warm_call_cache,
)
from .cash_secured_put import (
    CashSecuredPutRequest,
    plan_cash_secured_puts,
    screen_cash_secured_put_watchlist,
    screen_cash_secured_puts,
    warm_cash_secured_put_cache,
)
from .covered_call import (
    CoveredCallRequest,
    plan_covered_calls,
    screen_covered_call_watchlist,
    screen_covered_calls,
    warm_covered_call_cache,
)
from .protective_put import (
    ProtectivePutRequest,
    plan_protective_puts,
    screen_protective_put_watchlist,
    screen_protective_puts,
    warm_protective_put_cache,
)
from .put import (
    LongPutRequest,
    plan_long_puts,
    screen_long_put_watchlist,
    screen_long_puts,
    warm_long_put_cache,
)
from ._typed import (
    BatchResult,
    CircuitBreakerPolicy,
    RateLimitPolicy,
    RetryPolicy,
    ScreenerPlan,
    ScreenerResult,
    TimeoutPolicy,
    TickerFailure,
    TickerResult,
    WarmCacheResult,
)
from .credit_spreads import (
    CreditSpreadRequest,
    plan_credit_spreads,
    screen_credit_spread_watchlist,
    screen_credit_spreads,
    warm_credit_spread_cache,
)
from .calendar_spread import (
    CalendarSpreadRequest,
    plan_calendar_spreads,
    screen_calendar_spread_watchlist,
    screen_calendar_spreads,
    warm_calendar_spread_cache,
)
from .debit_spreads import (
    DebitSpreadRequest,
    plan_debit_spreads,
    screen_debit_spread_watchlist,
    screen_debit_spreads,
    warm_debit_spread_cache,
)
from .diagonal_spread import (
    DiagonalSpreadRequest,
    plan_diagonal_spreads,
    screen_diagonal_spread_watchlist,
    screen_diagonal_spreads,
    warm_diagonal_spread_cache,
)
from .iron_butterfly import (
    IronButterflyRequest,
    plan_iron_butterflies,
    screen_iron_butterflies,
    screen_iron_butterfly_watchlist,
    warm_iron_butterfly_cache,
)
from .iron_condor import (
    IronCondorRequest,
    plan_iron_condors,
    screen_iron_condor_watchlist,
    screen_iron_condors,
    warm_iron_condor_cache,
)
from .straddle import (
    StraddleRequest,
    plan_straddles,
    screen_straddle_watchlist,
    screen_straddles,
    warm_straddle_cache,
)
from .strangle import (
    StrangleRequest,
    plan_strangles,
    screen_strangle_watchlist,
    screen_strangles,
    warm_strangle_cache,
)

StrategyName = Literal[
    "credit_spread",
    "cash_secured_put",
    "debit_spread",
    "iron_condor",
    "iron_butterfly",
    "straddle",
    "strangle",
    "covered_call",
    "calendar_spread",
    "diagonal_spread",
    "protective_put",
    "call",
    "put",
]
StrategyRequest = (
    CreditSpreadRequest
    | DebitSpreadRequest
    | CashSecuredPutRequest
    | CoveredCallRequest
    | ProtectivePutRequest
    | CallRequest
    | LongPutRequest
    | StraddleRequest
    | StrangleRequest
    | CalendarSpreadRequest
    | DiagonalSpreadRequest
    | IronCondorRequest
    | IronButterflyRequest
)

_STRATEGIES: dict[str, dict[str, object]] = {
    "credit_spread": {
        "display_name": "Credit spread",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_credit_spreads",
    },
    "cash_secured_put": {
        "display_name": "Cash-secured put",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_cash_secured_puts",
    },
    "debit_spread": {
        "display_name": "Debit spread",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_debit_spreads",
    },
    "iron_condor": {
        "display_name": "Iron condor",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_iron_condors",
    },
    "iron_butterfly": {
        "display_name": "Iron butterfly",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_iron_butterflies",
    },
    "straddle": {
        "display_name": "Straddle",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_straddles",
    },
    "strangle": {
        "display_name": "Strangle",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_strangles",
    },
    "covered_call": {
        "display_name": "Covered call",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_covered_calls",
    },
    "calendar_spread": {
        "display_name": "Calendar spread",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_calendar_spreads",
    },
    "diagonal_spread": {
        "display_name": "Diagonal spread",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_diagonal_spreads",
    },
    "protective_put": {
        "display_name": "Protective put",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_protective_puts",
    },
    "call": {
        "display_name": "Call",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_calls",
    },
    "put": {
        "display_name": "Put",
        "typed_request": True,
        "watchlist": True,
        "warm_cache": True,
        "legacy_method": "get_best_puts",
    },
}


def get_available_strategies() -> dict[str, dict[str, object]]:
    """Return strategy metadata for UI/backend discovery."""
    return {name: dict(metadata) for name, metadata in _STRATEGIES.items()}


def _known_strategy(strategy: str) -> None:
    if strategy not in _STRATEGIES:
        names = ", ".join(sorted(_STRATEGIES))
        raise ValueError(f"strategy must be one of: {names}")


def _credit_request(request: StrategyRequest | None) -> CreditSpreadRequest:
    if request is None:
        return CreditSpreadRequest()
    if not isinstance(request, CreditSpreadRequest):
        raise TypeError("credit_spread requires CreditSpreadRequest")
    return request


def _debit_request(request: StrategyRequest | None) -> DebitSpreadRequest:
    if request is None:
        return DebitSpreadRequest()
    if not isinstance(request, DebitSpreadRequest):
        raise TypeError("debit_spread requires DebitSpreadRequest")
    return request


def _cash_secured_put_request(request: StrategyRequest | None) -> CashSecuredPutRequest:
    if request is None:
        return CashSecuredPutRequest()
    if not isinstance(request, CashSecuredPutRequest):
        raise TypeError("cash_secured_put requires CashSecuredPutRequest")
    return request


def _covered_call_request(request: StrategyRequest | None) -> CoveredCallRequest:
    if request is None:
        return CoveredCallRequest()
    if not isinstance(request, CoveredCallRequest):
        raise TypeError("covered_call requires CoveredCallRequest")
    return request


def _protective_put_request(request: StrategyRequest | None) -> ProtectivePutRequest:
    if request is None:
        return ProtectivePutRequest()
    if not isinstance(request, ProtectivePutRequest):
        raise TypeError("protective_put requires ProtectivePutRequest")
    return request


def _call_request(request: StrategyRequest | None) -> CallRequest:
    if request is None:
        return CallRequest()
    if not isinstance(request, CallRequest):
        raise TypeError("call requires CallRequest")
    return request


def _long_put_request(request: StrategyRequest | None) -> LongPutRequest:
    if request is None:
        return LongPutRequest()
    if not isinstance(request, LongPutRequest):
        raise TypeError("put requires LongPutRequest")
    return request


def _straddle_request(request: StrategyRequest | None) -> StraddleRequest:
    if request is None:
        return StraddleRequest()
    if not isinstance(request, StraddleRequest):
        raise TypeError("straddle requires StraddleRequest")
    return request


def _strangle_request(request: StrategyRequest | None) -> StrangleRequest:
    if request is None:
        return StrangleRequest()
    if not isinstance(request, StrangleRequest):
        raise TypeError("strangle requires StrangleRequest")
    return request


def _calendar_request(request: StrategyRequest | None) -> CalendarSpreadRequest:
    if request is None:
        return CalendarSpreadRequest()
    if not isinstance(request, CalendarSpreadRequest):
        raise TypeError("calendar_spread requires CalendarSpreadRequest")
    return request


def _diagonal_request(request: StrategyRequest | None) -> DiagonalSpreadRequest:
    if request is None:
        return DiagonalSpreadRequest()
    if not isinstance(request, DiagonalSpreadRequest):
        raise TypeError("diagonal_spread requires DiagonalSpreadRequest")
    return request


def _iron_condor_request(request: StrategyRequest | None) -> IronCondorRequest:
    if request is None:
        return IronCondorRequest()
    if not isinstance(request, IronCondorRequest):
        raise TypeError("iron_condor requires IronCondorRequest")
    return request


def _iron_butterfly_request(request: StrategyRequest | None) -> IronButterflyRequest:
    if request is None:
        return IronButterflyRequest()
    if not isinstance(request, IronButterflyRequest):
        raise TypeError("iron_butterfly requires IronButterflyRequest")
    return request


def plan_screener(
    request: StrategyRequest,
    *,
    conn=None,
) -> ScreenerPlan:
    """Dispatch request planning for any first-class typed strategy."""
    if isinstance(request, CreditSpreadRequest):
        return plan_credit_spreads(request, conn=conn)
    if isinstance(request, DebitSpreadRequest):
        return plan_debit_spreads(request, conn=conn)
    if isinstance(request, CashSecuredPutRequest):
        return plan_cash_secured_puts(request, conn=conn)
    if isinstance(request, CoveredCallRequest):
        return plan_covered_calls(request, conn=conn)
    if isinstance(request, ProtectivePutRequest):
        return plan_protective_puts(request, conn=conn)
    if isinstance(request, CallRequest):
        return plan_calls(request, conn=conn)
    if isinstance(request, LongPutRequest):
        return plan_long_puts(request, conn=conn)
    if isinstance(request, StraddleRequest):
        return plan_straddles(request, conn=conn)
    if isinstance(request, StrangleRequest):
        return plan_strangles(request, conn=conn)
    if isinstance(request, CalendarSpreadRequest):
        return plan_calendar_spreads(request, conn=conn)
    if isinstance(request, DiagonalSpreadRequest):
        return plan_diagonal_spreads(request, conn=conn)
    if isinstance(request, IronCondorRequest):
        return plan_iron_condors(request, conn=conn)
    if isinstance(request, IronButterflyRequest):
        return plan_iron_butterflies(request, conn=conn)
    raise TypeError("plan_screener requires a typed screener request")


def screen_strategy(
    request: StrategyRequest,
    client: Client | None = None,
    *,
    conn=None,
) -> ScreenerResult:
    """Run one typed strategy request and return data plus diagnostics."""
    if isinstance(request, CreditSpreadRequest):
        return screen_credit_spreads(request, client=client, conn=conn)
    if isinstance(request, DebitSpreadRequest):
        return screen_debit_spreads(request, client=client, conn=conn)
    if isinstance(request, CashSecuredPutRequest):
        return screen_cash_secured_puts(request, client=client, conn=conn)
    if isinstance(request, CoveredCallRequest):
        return screen_covered_calls(request, client=client, conn=conn)
    if isinstance(request, ProtectivePutRequest):
        return screen_protective_puts(request, client=client, conn=conn)
    if isinstance(request, CallRequest):
        return screen_calls(request, client=client, conn=conn)
    if isinstance(request, LongPutRequest):
        return screen_long_puts(request, client=client, conn=conn)
    if isinstance(request, StraddleRequest):
        return screen_straddles(request, client=client, conn=conn)
    if isinstance(request, StrangleRequest):
        return screen_strangles(request, client=client, conn=conn)
    if isinstance(request, CalendarSpreadRequest):
        return screen_calendar_spreads(request, client=client, conn=conn)
    if isinstance(request, DiagonalSpreadRequest):
        return screen_diagonal_spreads(request, client=client, conn=conn)
    if isinstance(request, IronCondorRequest):
        return screen_iron_condors(request, client=client, conn=conn)
    if isinstance(request, IronButterflyRequest):
        return screen_iron_butterflies(request, client=client, conn=conn)
    raise TypeError("screen_strategy requires a typed screener request")


def screen_watchlist(
    tickers: list[str],
    *,
    strategy: StrategyName,
    request: StrategyRequest | None = None,
    client: Client | None = None,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    circuit_breaker_policy: CircuitBreakerPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[TickerResult | TickerFailure], None] | None = None,
) -> BatchResult:
    """Screen a watchlist for a supported first-class strategy."""
    _known_strategy(strategy)
    if strategy == "credit_spread":
        return screen_credit_spread_watchlist(
            tickers,
            _credit_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "debit_spread":
        return screen_debit_spread_watchlist(
            tickers,
            _debit_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "cash_secured_put":
        return screen_cash_secured_put_watchlist(
            tickers,
            _cash_secured_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "covered_call":
        return screen_covered_call_watchlist(
            tickers,
            _covered_call_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "protective_put":
        return screen_protective_put_watchlist(
            tickers,
            _protective_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "call":
        return screen_call_watchlist(
            tickers,
            _call_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "put":
        return screen_long_put_watchlist(
            tickers,
            _long_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "straddle":
        return screen_straddle_watchlist(
            tickers,
            _straddle_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "strangle":
        return screen_strangle_watchlist(
            tickers,
            _strangle_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "calendar_spread":
        return screen_calendar_spread_watchlist(
            tickers,
            _calendar_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "diagonal_spread":
        return screen_diagonal_spread_watchlist(
            tickers,
            _diagonal_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "iron_condor":
        return screen_iron_condor_watchlist(
            tickers,
            _iron_condor_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    if strategy == "iron_butterfly":
        return screen_iron_butterfly_watchlist(
            tickers,
            _iron_butterfly_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
        )
    raise NotImplementedError(
        f"{strategy!r} has a legacy DataFrame screener, but no typed watchlist API yet"
    )


def warm_cache(
    tickers: list[str],
    *,
    strategy: StrategyName,
    request: StrategyRequest | None = None,
    client: Client | None = None,
    concurrency: int = 1,
    retry_policy: RetryPolicy | None = None,
    timeout_policy: TimeoutPolicy | None = None,
    rate_limit_policy: RateLimitPolicy | None = None,
    circuit_breaker_policy: CircuitBreakerPolicy | None = None,
    client_factory: Callable[[], Client] | None = None,
    on_progress: Callable[[TickerResult | TickerFailure], None] | None = None,
    conn=None,
) -> WarmCacheResult:
    """Warm cache inputs for a supported first-class strategy."""
    _known_strategy(strategy)
    if strategy == "credit_spread":
        return warm_credit_spread_cache(
            tickers,
            _credit_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "debit_spread":
        return warm_debit_spread_cache(
            tickers,
            _debit_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "cash_secured_put":
        return warm_cash_secured_put_cache(
            tickers,
            _cash_secured_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "covered_call":
        return warm_covered_call_cache(
            tickers,
            _covered_call_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "protective_put":
        return warm_protective_put_cache(
            tickers,
            _protective_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "call":
        return warm_call_cache(
            tickers,
            _call_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "put":
        return warm_long_put_cache(
            tickers,
            _long_put_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "straddle":
        return warm_straddle_cache(
            tickers,
            _straddle_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "strangle":
        return warm_strangle_cache(
            tickers,
            _strangle_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "calendar_spread":
        return warm_calendar_spread_cache(
            tickers,
            _calendar_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "diagonal_spread":
        return warm_diagonal_spread_cache(
            tickers,
            _diagonal_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "iron_condor":
        return warm_iron_condor_cache(
            tickers,
            _iron_condor_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    if strategy == "iron_butterfly":
        return warm_iron_butterfly_cache(
            tickers,
            _iron_butterfly_request(request),
            client=client,
            concurrency=concurrency,
            retry_policy=retry_policy,
            timeout_policy=timeout_policy,
            rate_limit_policy=rate_limit_policy,
            circuit_breaker_policy=circuit_breaker_policy,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )
    raise NotImplementedError(
        f"{strategy!r} has a legacy DataFrame screener, but no warm-cache API yet"
    )


__all__ = [
    "StrategyName",
    "StrategyRequest",
    "get_available_strategies",
    "plan_screener",
    "screen_strategy",
    "screen_watchlist",
    "warm_cache",
]
