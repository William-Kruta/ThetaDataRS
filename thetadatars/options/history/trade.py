import datetime as dt
import logging

import polars as pl

from ...client import Client
from ._common import Right, get_history_table, read_history_table, require_date_range, require_expiration
from ._schemas import TRADE_COLS

log = logging.getLogger(__name__)
_TABLE = "option_trades"


def fetch_option_history_trade(
    ticker: str, start_date: dt.date | str, end_date: dt.date | str, expiration: dt.date | str,
    client: Client, strike: str = "*", right: Right = "both",
    start_time: dt.time | str = "09:30:00", end_time: dt.time | str = "16:00:00",
    max_dte: int | None = None, strike_range: int | None = None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return client.option_history_trade(
        symbol=ticker, expiration=expiration, strike=strike, right=right,
        start_time=start_time, end_time=end_time, max_dte=max_dte,
        strike_range=strike_range, start_date=start, end_date=end,
    )


def read_option_history_trade(root: str, start_date: dt.date | str, end_date: dt.date | str | None = None, expiration: dt.date | str | None = None, strike: float | str | None = None, right: Right | None = None, conn=None) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration) if expiration is not None else None
    return read_history_table(_TABLE, TRADE_COLS[:-1], root, start, end, expiration, strike, right, conn=conn, order_columns=["date", "timestamp", "expiration", "strike", "right"])


def get_option_history_trade(
    ticker: str, start_date: dt.date | str, end_date: dt.date | str, expiration: dt.date | str,
    client: Client, strike: str = "*", right: Right = "both",
    start_time: dt.time | str = "09:30:00", end_time: dt.time | str = "16:00:00",
    max_dte: int | None = None, strike_range: int | None = None,
    stale_threshold: dt.timedelta = dt.timedelta(days=7), conn=None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return get_history_table(
        table=_TABLE, columns=TRADE_COLS,
        fetcher=lambda: fetch_option_history_trade(ticker, start, end, expiration, client, strike, right, start_time, end_time, max_dte, strike_range),
        log=log, label="option history trade", root=ticker, start_date=start, end_date=end,
        expiration=expiration, strike=strike, right=right, stale_threshold=stale_threshold,
        conn=conn, order_columns=["date", "timestamp", "expiration", "strike", "right"],
    )
