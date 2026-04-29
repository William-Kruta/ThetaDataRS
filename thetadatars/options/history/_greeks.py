import datetime as dt
from typing import Callable

import polars as pl

from ...client import Client
from ._common import RateType, Right, get_history_table, read_history_table, require_date_range, require_expiration


def fetch_interval_greeks(
    client_method: Callable[..., pl.DataFrame],
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    client: Client,
    interval: str = "1s",
    strike: str = "*",
    right: Right = "both",
    start_time: dt.time | str = "09:30:00",
    end_time: dt.time | str = "16:00:00",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    version: str = "latest",
    strike_range: int | None = None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return client_method(
        symbol=ticker, expiration=expiration, interval=interval, strike=strike,
        right=right, start_time=start_time, end_time=end_time,
        annual_dividend=annual_dividend, rate_type=rate_type,
        rate_value=rate_value, version=version, strike_range=strike_range,
        start_date=start, end_date=end,
    )


def fetch_trade_greeks(
    client_method: Callable[..., pl.DataFrame],
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Right = "both",
    start_time: dt.time | str = "09:30:00",
    end_time: dt.time | str = "16:00:00",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    version: str = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return client_method(
        symbol=ticker, expiration=expiration, strike=strike, right=right,
        start_time=start_time, end_time=end_time,
        annual_dividend=annual_dividend, rate_type=rate_type,
        rate_value=rate_value, version=version, max_dte=max_dte,
        strike_range=strike_range, start_date=start, end_date=end,
    )


def fetch_eod_greeks(
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Right = "both",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    version: str = "latest",
    underlyer_use_nbbo: bool = False,
    max_dte: int | None = None,
    strike_range: int | None = None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return client.option_history_greeks_eod(
        symbol=ticker, expiration=expiration, start_date=start, end_date=end,
        strike=strike, right=right, annual_dividend=annual_dividend,
        rate_type=rate_type, rate_value=rate_value, version=version,
        underlyer_use_nbbo=underlyer_use_nbbo, max_dte=max_dte,
        strike_range=strike_range,
    )


def read_greeks(table: str, columns: list[str], root: str, start_date: dt.date | str, end_date: dt.date | str | None, expiration: dt.date | str | None, strike: float | str | None, right: Right | None, conn=None) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration) if expiration is not None else None
    return read_history_table(table, columns[:-1], root, start, end, expiration, strike, right, conn=conn, order_columns=["date", "timestamp", "expiration", "strike", "right"])


def get_greeks(
    *,
    table: str,
    columns: list[str],
    fetcher: Callable[[], pl.DataFrame],
    log,
    label: str,
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    strike: str,
    right: Right,
    stale_threshold: dt.timedelta,
    conn=None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return get_history_table(
        table=table, columns=columns, fetcher=fetcher, log=log, label=label,
        root=ticker, start_date=start, end_date=end, expiration=expiration,
        strike=strike, right=right, stale_threshold=stale_threshold, conn=conn,
        order_columns=["date", "timestamp", "expiration", "strike", "right"],
    )
