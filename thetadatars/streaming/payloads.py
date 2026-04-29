import datetime as dt
from typing import Literal, TypedDict

SecurityType = Literal["OPTION", "STOCK", "INDEX"]
RequestType = Literal["TRADE", "QUOTE"]
OptionRight = Literal["call", "put", "C", "P"]


class StreamPayload(TypedDict, total=False):
    msg_type: str
    sec_type: SecurityType
    req_type: RequestType
    add: bool
    id: int
    contract: dict[str, object]


def option_contract(
    root: str,
    expiration: dt.date | str | int,
    strike: float | int | str,
    right: OptionRight,
) -> dict[str, object]:
    return {
        "root": root,
        "expiration": _format_expiration(expiration),
        "strike": _format_strike(strike),
        "right": _format_right(right),
    }


def equity_contract(root: str) -> dict[str, object]:
    return {"root": root}


def stream_payload(
    sec_type: SecurityType,
    req_type: RequestType,
    *,
    add: bool,
    request_id: int,
    contract: dict[str, object],
) -> StreamPayload:
    return {
        "msg_type": "STREAM",
        "sec_type": sec_type,
        "req_type": req_type,
        "add": add,
        "id": request_id,
        "contract": contract,
    }


def bulk_stream_payload(
    sec_type: Literal["OPTION", "STOCK"],
    req_type: Literal["TRADE"],
    *,
    add: bool,
    request_id: int,
) -> StreamPayload:
    return {
        "msg_type": "STREAM_BULK",
        "sec_type": sec_type,
        "req_type": req_type,
        "add": add,
        "id": request_id,
    }


def stop_payload() -> dict[str, str]:
    return {"msg_type": "STOP"}


def _format_expiration(expiration: dt.date | str | int) -> int:
    if isinstance(expiration, dt.datetime):
        expiration = expiration.date()
    if isinstance(expiration, dt.date):
        return int(expiration.strftime("%Y%m%d"))
    if isinstance(expiration, int):
        return expiration
    value = expiration.replace("-", "")
    if len(value) != 8 or not value.isdigit():
        raise ValueError("expiration must be YYYY-MM-DD, YYYYMMDD, datetime.date, or int")
    return int(value)


def _format_strike(strike: float | int | str) -> int:
    if isinstance(strike, str):
        if strike.isdigit():
            return int(strike)
        strike = float(strike)
    return int(round(float(strike) * 1000))


def _format_right(right: OptionRight) -> str:
    normalized = right.upper()
    if normalized in {"CALL", "C"}:
        return "C"
    if normalized in {"PUT", "P"}:
        return "P"
    raise ValueError("right must be call, put, C, or P")
