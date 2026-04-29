import datetime as dt
import logging
import polars as pl
from ...client import Client
from ._common import RateType, Right
from ._greeks import fetch_trade_greeks, get_greeks, read_greeks
from ._schemas import TRADE_GREEKS_COLS

log = logging.getLogger(__name__)
_TABLE = "option_trade_greeks"


def fetch_option_history_trade_greeks_third_order(ticker: str, start_date: dt.date | str, end_date: dt.date | str, expiration: dt.date | str, client: Client, strike: str = "*", right: Right = "both", start_time: dt.time | str = "09:30:00", end_time: dt.time | str = "16:00:00", annual_dividend: float | None = None, rate_type: RateType = "sofr", rate_value: float | None = None, version: str = "latest", max_dte: int | None = None, strike_range: int | None = None) -> pl.DataFrame:
    return fetch_trade_greeks(client.option_history_trade_greeks_third_order, ticker, start_date, end_date, expiration, client, strike, right, start_time, end_time, annual_dividend, rate_type, rate_value, version, max_dte, strike_range)


def read_option_history_trade_greeks_third_order(root: str, start_date: dt.date | str, end_date: dt.date | str | None = None, expiration: dt.date | str | None = None, strike: float | str | None = None, right: Right | None = None, conn=None) -> pl.DataFrame:
    return read_greeks(_TABLE, TRADE_GREEKS_COLS, root, start_date, end_date, expiration, strike, right, conn)


def get_option_history_trade_greeks_third_order(ticker: str, start_date: dt.date | str, end_date: dt.date | str, expiration: dt.date | str, client: Client, strike: str = "*", right: Right = "both", start_time: dt.time | str = "09:30:00", end_time: dt.time | str = "16:00:00", annual_dividend: float | None = None, rate_type: RateType = "sofr", rate_value: float | None = None, version: str = "latest", max_dte: int | None = None, strike_range: int | None = None, stale_threshold: dt.timedelta = dt.timedelta(days=7), conn=None) -> pl.DataFrame:
    return get_greeks(table=_TABLE, columns=TRADE_GREEKS_COLS, fetcher=lambda: fetch_option_history_trade_greeks_third_order(ticker, start_date, end_date, expiration, client, strike, right, start_time, end_time, annual_dividend, rate_type, rate_value, version, max_dte, strike_range), log=log, label="option history third order trade greeks", ticker=ticker, start_date=start_date, end_date=end_date, expiration=expiration, strike=strike, right=right, stale_threshold=stale_threshold, conn=conn)
