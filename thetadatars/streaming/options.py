import datetime as dt
from collections.abc import AsyncIterator
from typing import Any

from .client import DEFAULT_STREAM_URL, StreamClient
from .payloads import OptionRight, bulk_stream_payload, option_contract, stream_payload


def option_trade_payload(
    root: str,
    expiration: dt.date | str | int,
    strike: float | int | str,
    right: OptionRight,
    *,
    add: bool = True,
    request_id: int = 0,
) -> dict[str, object]:
    return stream_payload(
        "OPTION",
        "TRADE",
        add=add,
        request_id=request_id,
        contract=option_contract(root, expiration, strike, right),
    )


def option_quote_payload(
    root: str,
    expiration: dt.date | str | int,
    strike: float | int | str,
    right: OptionRight,
    *,
    add: bool = True,
    request_id: int = 0,
) -> dict[str, object]:
    return stream_payload(
        "OPTION",
        "QUOTE",
        add=add,
        request_id=request_id,
        contract=option_contract(root, expiration, strike, right),
    )


def option_full_trade_payload(*, add: bool = True, request_id: int = 0) -> dict[str, object]:
    return bulk_stream_payload("OPTION", "TRADE", add=add, request_id=request_id)


def stream_option_trades(
    root: str,
    expiration: dt.date | str | int,
    strike: float | int | str,
    right: OptionRight,
    *,
    request_id: int = 0,
    url: str = DEFAULT_STREAM_URL,
    max_messages: int | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any] | str]:
    payload = option_trade_payload(root, expiration, strike, right, request_id=request_id)
    return StreamClient(url).subscribe(payload, max_messages=max_messages, timeout=timeout)


def stream_option_quotes(
    root: str,
    expiration: dt.date | str | int,
    strike: float | int | str,
    right: OptionRight,
    *,
    request_id: int = 0,
    url: str = DEFAULT_STREAM_URL,
    max_messages: int | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any] | str]:
    payload = option_quote_payload(root, expiration, strike, right, request_id=request_id)
    return StreamClient(url).subscribe(payload, max_messages=max_messages, timeout=timeout)


def stream_option_full_trades(
    *,
    request_id: int = 0,
    url: str = DEFAULT_STREAM_URL,
    max_messages: int | None = None,
    timeout: float | None = None,
) -> AsyncIterator[dict[str, Any] | str]:
    payload = option_full_trade_payload(request_id=request_id)
    return StreamClient(url).subscribe(payload, max_messages=max_messages, timeout=timeout)
