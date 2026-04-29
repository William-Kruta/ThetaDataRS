import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

DEFAULT_STREAM_URL = "ws://127.0.0.1:25520/v1/events"


@dataclass(slots=True)
class StreamClient:
    url: str = DEFAULT_STREAM_URL

    async def send(self, payload: dict[str, Any]) -> dict[str, Any] | str:
        async with self._connect() as websocket:
            await websocket.send(json.dumps(payload))
            return _decode_message(await websocket.recv())

    async def subscribe(
        self,
        payload: dict[str, Any],
        *,
        max_messages: int | None = None,
        timeout: float | None = None,
    ) -> AsyncIterator[dict[str, Any] | str]:
        async with self._connect() as websocket:
            await websocket.send(json.dumps(payload))
            count = 0
            while max_messages is None or count < max_messages:
                raw = await asyncio.wait_for(websocket.recv(), timeout=timeout) if timeout else await websocket.recv()
                count += 1
                yield _decode_message(raw)

    async def collect(
        self,
        payload: dict[str, Any],
        *,
        max_messages: int,
        timeout: float | None = None,
    ) -> list[dict[str, Any] | str]:
        messages: list[dict[str, Any] | str] = []
        async for message in self.subscribe(payload, max_messages=max_messages, timeout=timeout):
            messages.append(message)
        return messages

    async def run(
        self,
        payload: dict[str, Any],
        handler: Callable[[dict[str, Any] | str], Awaitable[None] | None],
        *,
        max_messages: int | None = None,
        timeout: float | None = None,
    ) -> None:
        async for message in self.subscribe(payload, max_messages=max_messages, timeout=timeout):
            result = handler(message)
            if result is not None:
                await result

    def _connect(self):
        try:
            import websockets
        except ImportError as exc:
            raise ImportError("Install the 'websockets' package to use the ThetaData streaming API") from exc
        return websockets.connect(self.url)


def _decode_message(raw: Any) -> dict[str, Any] | str:
    if isinstance(raw, bytes):
        raw = raw.decode()
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return raw
