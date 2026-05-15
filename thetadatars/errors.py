from __future__ import annotations

import builtins
from typing import Any


class ThetaDataError(Exception):
    """Base package error with structured context for application callers."""

    def __init__(
        self,
        message: str | None = None,
        *,
        ticker: str | None = None,
        endpoint: str | None = None,
        params: dict[str, Any] | None = None,
        status_code: int | None = None,
        retryable: bool = False,
        user_message: str | None = None,
        debug_message: str | None = None,
    ) -> None:
        self.ticker = ticker
        self.endpoint = endpoint
        self.params = params or {}
        self.status_code = status_code
        self.retryable = retryable
        self.user_message = user_message or message or "ThetaData request failed"
        self.debug_message = debug_message or message or self.user_message
        super().__init__(self.debug_message)


class NoDataError(ThetaDataError):
    """Raised when ThetaData has no rows for the requested symbol or endpoint."""


class SubscriptionError(ThetaDataError):
    """Raised when an endpoint is unavailable for the account subscription."""


class RateLimitError(ThetaDataError):
    """Raised when an upstream request is rate limited."""


class TimeoutError(ThetaDataError, builtins.TimeoutError):
    """Raised when an upstream request times out."""


class TransientNetworkError(ThetaDataError):
    """Raised for retryable upstream transport or server failures."""


class InvalidRequestError(ThetaDataError, ValueError):
    """Raised before network calls when request validation fails."""


class CacheError(ThetaDataError):
    """Raised for local cache read/write failures."""


class CacheMissError(CacheError):
    """Raised when cache-only access cannot satisfy a request."""


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    if status is not None:
        return int(status)
    response = getattr(error, "response", None)
    response_status = getattr(response, "status_code", None)
    return int(response_status) if response_status is not None else None


def classify_thetadata_error(
    error: Exception,
    *,
    ticker: str | None = None,
    endpoint: str | None = None,
    params: dict[str, Any] | None = None,
) -> ThetaDataError:
    """Convert raw upstream errors into package errors with context."""
    if isinstance(error, ThetaDataError):
        return error

    status_code = _status_code(error)
    message = str(error)
    lowered = message.lower()
    context = {
        "ticker": ticker,
        "endpoint": endpoint,
        "params": params,
        "status_code": status_code,
        "debug_message": message,
    }

    if "no data found" in lowered or lowered.startswith("no data"):
        return NoDataError(
            message,
            retryable=False,
            user_message="No data was returned for the request.",
            **context,
        )
    if "subscription" in lowered or (
        "standard subscription" in lowered and "value subscription" in lowered
    ):
        return SubscriptionError(
            message,
            retryable=False,
            user_message="The requested ThetaData endpoint is not available for this subscription.",
            **context,
        )
    if status_code == 429 or "rate limit" in lowered or "too many requests" in lowered:
        return RateLimitError(
            message,
            retryable=True,
            user_message="ThetaData rate limited the request.",
            **context,
        )
    if isinstance(error, builtins.TimeoutError) or "timeout" in lowered or "timed out" in lowered:
        return TimeoutError(
            message,
            retryable=True,
            user_message="The ThetaData request timed out.",
            **context,
        )
    if (
        status_code is not None
        and status_code >= 500
        or "connection" in lowered
        or "temporarily unavailable" in lowered
        or "reset by peer" in lowered
        or "unavailable" in lowered
    ):
        return TransientNetworkError(
            message,
            retryable=True,
            user_message="ThetaData request failed with a transient upstream error.",
            **context,
        )
    if status_code is not None and 400 <= status_code < 500:
        return InvalidRequestError(
            message,
            retryable=False,
            user_message="ThetaData rejected the request.",
            **context,
        )
    return ThetaDataError(message, retryable=False, **context)


__all__ = [
    "ThetaDataError",
    "NoDataError",
    "SubscriptionError",
    "RateLimitError",
    "TimeoutError",
    "TransientNetworkError",
    "InvalidRequestError",
    "CacheError",
    "CacheMissError",
    "classify_thetadata_error",
]
