import datetime as dt
import logging

import polars as pl

from ...client import Client
from ._common import Right, get_history_table, read_history_table, require_date_range, require_expiration
from ._schemas import EOD_COLS

log = logging.getLogger(__name__)
_TABLE = "option_eod"


def fetch_option_history_eod(
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Right = "both",
    max_dte: int | None = None,
    strike_range: int | None = None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return client.option_history_eod(
        start_date=start,
        end_date=end,
        symbol=ticker,
        expiration=expiration,
        strike=strike,
        right=right,
        max_dte=max_dte,
        strike_range=strike_range,
    )


def read_option_history_eod(
    root: str,
    start_date: dt.date | str,
    end_date: dt.date | str | None = None,
    expiration: dt.date | str | None = None,
    strike: float | str | None = None,
    right: Right | None = None,
    conn=None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration) if expiration is not None else None
    return read_history_table(_TABLE, EOD_COLS[:-1], root, start, end, expiration, strike, right, conn=conn)


def get_option_history_eod(
    ticker: str,
    start_date: dt.date | str,
    end_date: dt.date | str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Right = "both",
    max_dte: int | None = None,
    strike_range: int | None = None,
    stale_threshold: dt.timedelta = dt.timedelta(days=7),
    conn=None,
) -> pl.DataFrame:
    start, end = require_date_range(start_date, end_date)
    expiration = require_expiration(expiration)
    return get_history_table(
        table=_TABLE,
        columns=EOD_COLS,
        fetcher=lambda: fetch_option_history_eod(ticker, start, end, expiration, client, strike, right, max_dte, strike_range),
        log=log,
        label="option history eod",
        root=ticker,
        start_date=start,
        end_date=end,
        expiration=expiration,
        strike=strike,
        right=right,
        stale_threshold=stale_threshold,
        conn=conn,
        order_columns=["date", "expiration", "strike", "right"],
    )
