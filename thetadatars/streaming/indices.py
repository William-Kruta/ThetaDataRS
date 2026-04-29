from collections.abc import AsyncIterator
from typing import Any

from .client import DEFAULT_STREAM_URL, StreamClient
from .payloads import equity_contract, stream_payload


def index_price_payload(root: str, *, add: bool = True, request_id: int = 0) -> dict[str, object]:
    return stream_payload("INDEX", "TRADE", add=add, request_id=request_id, contract=equity_contract(root))


def stream_index_prices(root: str, *, request_id: int = 0, url: str = DEFAULT_STREAM_URL, max_messages: int | None = None, timeout: float | None = None) -> AsyncIterator[dict[str, Any] | str]:
    return StreamClient(url).subscribe(index_price_payload(root, request_id=request_id), max_messages=max_messages, timeout=timeout)
