import inspect
from collections.abc import Callable
from functools import cached_property, wraps
from importlib import import_module
from typing import Any, Literal

from .client import create_client
from .streaming.client import DEFAULT_STREAM_URL


_OPTION_MODULES = [
    "thetadatars.options.list.contracts",
    "thetadatars.options.list.dates",
    "thetadatars.options.list.expirations",
    "thetadatars.options.list.strikes",
    "thetadatars.options.list.symbols",
    "thetadatars.options.history.eod",
    "thetadatars.options.history.greeks_all",
    "thetadatars.options.history.greeks_eod",
    "thetadatars.options.history.greeks_first_order",
    "thetadatars.options.history.greeks_second_order",
    "thetadatars.options.history.greeks_third_order",
    "thetadatars.options.history.implied_volatility",
    "thetadatars.options.history.ohlcv",
    "thetadatars.options.history.open_interest",
    "thetadatars.options.history.quote",
    "thetadatars.options.history.trade",
    "thetadatars.options.history.trade_greeks_all",
    "thetadatars.options.history.trade_greeks_first_order",
    "thetadatars.options.history.trade_greeks_second_order",
    "thetadatars.options.history.trade_greeks_third_order",
    "thetadatars.options.history.trade_implied_volatility",
    "thetadatars.options.history.trade_quote",
    "thetadatars.options.snapshot.greeks_all",
    "thetadatars.options.snapshot.greeks_first_order",
    "thetadatars.options.snapshot.greeks_second_order",
    "thetadatars.options.snapshot.greeks_third_order",
    "thetadatars.options.snapshot.implied_volatility",
    "thetadatars.options.snapshot.market_value",
    "thetadatars.options.snapshot.ohlcv",
    "thetadatars.options.snapshot.open_interest",
    "thetadatars.options.snapshot.quote",
    "thetadatars.options.snapshot.trade",
]

_STREAMING_MODULES = [
    "thetadatars.streaming.indices",
    "thetadatars.streaming.options",
    "thetadatars.streaming.stocks",
    "thetadatars.streaming.system",
]


def _load_public_functions() -> dict[str, Callable[..., Any]]:
    functions: dict[str, Callable[..., Any]] = {}

    for module_name in _OPTION_MODULES:
        module = import_module(module_name)
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if name.startswith(("fetch_", "get_", "read_")):
                functions[name] = value

    for module_name in _STREAMING_MODULES:
        module = import_module(module_name)
        public_names = getattr(module, "__all__", None)
        members = (
            ((name, getattr(module, name)) for name in public_names)
            if public_names is not None
            else inspect.getmembers(module, inspect.isfunction)
        )
        for name, value in members:
            if not name.startswith("_") and inspect.isfunction(value):
                functions[name] = value

    return functions


class ThetaDataRS:
    """Facade over the package's option and streaming helper functions."""

    def __init__(
        self,
        email: str | None = None,
        passwd: str | None = None,
        dataframe_return_type: Literal["pandas", "polars"] = "polars",
        stream_url: str = DEFAULT_STREAM_URL,
    ) -> None:
        self.email = email
        self.passwd = passwd
        self.dataframe_return_type = dataframe_return_type
        self.stream_url = stream_url

    @cached_property
    def client(self):
        return create_client(
            email=self.email,
            passwd=self.passwd,
            dataframe_return_type=self.dataframe_return_type,
        )

    @cached_property
    def _functions(self) -> dict[str, Callable[..., Any]]:
        return _load_public_functions()

    def __getattr__(self, name: str) -> Callable[..., Any]:
        try:
            function = self._functions[name]
        except KeyError as exc:
            raise AttributeError(f"{type(self).__name__!s} has no method {name!r}") from exc

        signature = inspect.signature(function)

        @wraps(function)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            supplied_args = signature.bind_partial(*args, **kwargs).arguments
            if "client" in signature.parameters and "client" not in supplied_args:
                kwargs["client"] = self.client
            if "url" in signature.parameters and "url" not in supplied_args:
                kwargs["url"] = self.stream_url
            return function(*args, **kwargs)

        return wrapper

    def __dir__(self) -> list[str]:
        return sorted(set(super().__dir__()) | set(self._functions))


__all__ = ["ThetaDataRS"]
