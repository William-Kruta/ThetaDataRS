from collections.abc import AsyncIterator
from typing import Any

from .client import DEFAULT_STREAM_URL, StreamClient
from .payloads import bulk_stream_payload, equity_contract, stream_payload


def stock_trade_payload(root: str, *, add: bool = True, request_id: int = 0) -> dict[str, object]:
    return stream_payload("STOCK", "TRADE", add=add, request_id=request_id, contract=equity_contract(root))


def stock_quote_payload(root: str, *, add: bool = True, request_id: int = 0) -> dict[str, object]:
    return stream_payload("STOCK", "QUOTE", add=add, request_id=request_id, contract=equity_contract(root))


def stock_full_trade_payload(*, add: bool = True, request_id: int = 0) -> dict[str, object]:
    return bulk_stream_payload("STOCK", "TRADE", add=add, request_id=request_id)


def stream_stock_trades(root: str, *, request_id: int = 0, url: str = DEFAULT_STREAM_URL, max_messages: int | None = None, timeout: float | None = None) -> AsyncIterator[dict[str, Any] | str]:
    return StreamClient(url).subscribe(stock_trade_payload(root, request_id=request_id), max_messages=max_messages, timeout=timeout)


def stream_stock_quotes(root: str, *, request_id: int = 0, url: str = DEFAULT_STREAM_URL, max_messages: int | None = None, timeout: float | None = None) -> AsyncIterator[dict[str, Any] | str]:
    return StreamClient(url).subscribe(stock_quote_payload(root, request_id=request_id), max_messages=max_messages, timeout=timeout)


def stream_stock_full_trades(*, request_id: int = 0, url: str = DEFAULT_STREAM_URL, max_messages: int | None = None, timeout: float | None = None) -> AsyncIterator[dict[str, Any] | str]:
    return StreamClient(url).subscribe(stock_full_trade_payload(request_id=request_id), max_messages=max_messages, timeout=timeout)
