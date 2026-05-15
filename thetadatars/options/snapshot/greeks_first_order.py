import datetime as dt
import logging
import polars as pl
from typing import Literal

from ...client import Client
from ...data.cache import CachePolicy, get_or_fetch
from ...data.db import get_connection

log = logging.getLogger(__name__)

_TABLE = "snapshot_greeks_first_order"
_ENDPOINT = "option_snapshot_greeks_first_order"
_READ_COLS = ["root", "expiration", "strike", "right", "timestamp", "bid", "ask", "delta", "theta", "vega", "rho", "epsilon", "lambda", "implied_vol", "iv_error", "underlying_timestamp", "underlying_price"]
_DB_COLS = [*_READ_COLS, "fetched_at"]

RateType = Literal[
    "sofr",
    "treasury_m1", "treasury_m3", "treasury_m6",
    "treasury_y1", "treasury_y2", "treasury_y3", "treasury_y5",
    "treasury_y7", "treasury_y10", "treasury_y20", "treasury_y30",
]


def fetch_snapshot_greeks_first_order(
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Literal["call", "put", "both"] = "both",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
):
    if isinstance(expiration, str) and expiration != "*":
        expiration = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    return client.option_snapshot_greeks_first_order(
        symbol=ticker,
        expiration=expiration,
        strike=strike,
        right=right,
        annual_dividend=annual_dividend,
        rate_type=rate_type,
        rate_value=rate_value,
        stock_price=stock_price,
        version=version,
        max_dte=max_dte,
        strike_range=strike_range,
        min_time=min_time,
        use_market_value=use_market_value,
    )


def read_snapshot_greeks_first_order(
    root: str,
    expiration: dt.date | None = None,
    strike: float | None = None,
    right: Literal["call", "put"] | None = None,
    conn=None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        q = 'SELECT root, expiration, strike, "right", timestamp, bid, ask, delta, theta, vega, rho, epsilon, "lambda", implied_vol, iv_error, underlying_timestamp, underlying_price FROM snapshot_greeks_first_order WHERE root = ?'
        params = [root]
        if expiration is not None:
            q += " AND expiration = ?"
            params.append(expiration)
        if strike is not None:
            q += " AND strike = ?"
            params.append(strike)
        if right is not None:
            q += ' AND "right" = ?'
            params.append(right)
        return conn.execute(q + ' ORDER BY expiration, strike, "right"', params).pl()
    finally:
        if own_conn:
            conn.close()


def _specific_strike(strike: float | str) -> float | None:
    if strike == "*":
        return None
    return float(strike)


def _read_right(right: Literal["call", "put", "both"]) -> Literal["call", "put"] | None:
    return right if right in {"call", "put"} else None


def _normalize_snapshot_greeks_first_order(df: pl.DataFrame) -> pl.DataFrame:
    if "symbol" in df.columns:
        df = df.rename({"symbol": "root"})
    return df.select([column for column in _READ_COLS if column in df.columns])


def get_snapshot_greeks_first_order(
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Literal["call", "put", "both"] = "both",
    annual_dividend: float | None = None,
    rate_type: RateType = "sofr",
    rate_value: float | None = None,
    stock_price: float | None = None,
    version: Literal["latest", "1"] = "latest",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    use_market_value: bool = False,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    cache_policy: CachePolicy = "prefer_cache",
    conn=None,
) -> pl.DataFrame:
    if isinstance(expiration, str) and expiration != "*":
        expiration = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    exp_is_date = isinstance(expiration, dt.date)
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        params = {
            "expiration": expiration,
            "strike": strike,
            "right": right,
            "annual_dividend": annual_dividend,
            "rate_type": rate_type,
            "rate_value": rate_value,
            "stock_price": stock_price,
            "version": version,
            "max_dte": max_dte,
            "strike_range": strike_range,
            "min_time": min_time,
            "use_market_value": use_market_value,
        }

        def read_cached() -> pl.DataFrame:
            return read_snapshot_greeks_first_order(
                ticker,
                expiration if exp_is_date else None,
                strike=_specific_strike(strike),
                right=_read_right(right),
                conn=conn,
            )

        def fetch_upstream() -> pl.DataFrame:
            log.info("Fetching first-order greeks for %s exp=%s from API", ticker, expiration)
            df = fetch_snapshot_greeks_first_order(
                ticker, expiration, client, strike, right,
                annual_dividend, rate_type, rate_value, stock_price,
                version, max_dte, strike_range, min_time, use_market_value,
            )
            return _normalize_snapshot_greeks_first_order(df)

        def write_cached(rows: pl.DataFrame, fetched_at: dt.datetime) -> None:
            if rows.is_empty():
                return
            frame = rows.with_columns(pl.lit(fetched_at).alias("fetched_at"))
            frame = frame.select([column for column in _DB_COLS if column in frame.columns])
            conn.execute(f"INSERT OR REPLACE INTO {_TABLE} SELECT * FROM frame")
            log.info("Fetched and stored %d first-order greek snapshots for %s", len(frame), ticker)

        return get_or_fetch(
            conn=conn,
            endpoint=_ENDPOINT,
            root=ticker,
            params=params,
            cache_policy=cache_policy,
            stale_threshold=stale_threshold,
            read_cached=read_cached,
            fetch_upstream=fetch_upstream,
            write_cached=write_cached,
        )
    finally:
        if own_conn:
            conn.close()
