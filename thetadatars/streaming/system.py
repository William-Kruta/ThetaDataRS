from typing import Any

from .client import DEFAULT_STREAM_URL, StreamClient
from .payloads import stop_payload


async def stop_all_streams(*, url: str = DEFAULT_STREAM_URL) -> dict[str, Any] | str:
    return await StreamClient(url).send(stop_payload())


def is_request_response(message: dict[str, Any] | str, request_id: int | None = None) -> bool:
    if not isinstance(message, dict):
        return False
    header = message.get("header")
    if not isinstance(header, dict) or header.get("type") != "REQ_RESPONSE":
        return False
    return request_id is None or header.get("req_id") == request_id


def request_response_status(message: dict[str, Any] | str) -> str | None:
    if not isinstance(message, dict):
        return None
    header = message.get("header")
    if not isinstance(header, dict) or header.get("type") != "REQ_RESPONSE":
        return None
    return header.get("response") or header.get("status")
