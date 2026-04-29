import datetime as dt
import logging
import polars as pl
from typing import Literal

from ...client import Client
from ...data.db import get_connection

log = logging.getLogger(__name__)

_TABLE = "snapshot_open_interest"


def fetch_snapshot_open_interest(
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Literal["call", "put", "both"] = "both",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
):
    if isinstance(expiration, str) and expiration != "*":
        expiration = dt.datetime.strptime(expiration, "%Y-%m-%d").date()
    return client.option_snapshot_open_interest(
        symbol=ticker,
        expiration=expiration,
        strike=strike,
        right=right,
        max_dte=max_dte,
        strike_range=strike_range,
        min_time=min_time,
    )


def read_snapshot_open_interest(
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
        q = 'SELECT root, expiration, strike, "right", timestamp, open_interest FROM snapshot_open_interest WHERE root = ?'
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


def get_snapshot_open_interest(
    ticker: str,
    expiration: dt.date | str,
    client: Client,
    strike: str = "*",
    right: Literal["call", "put", "both"] = "both",
    max_dte: int | None = None,
    strike_range: int | None = None,
    min_time: dt.time | None = None,
    stale_threshold: dt.timedelta = dt.timedelta(hours=1),
    conn=None,
) -> pl.DataFrame:
    exp_is_date = isinstance(expiration, dt.date)
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        stale_q = f"SELECT MAX(fetched_at) FROM {_TABLE} WHERE root = ?"
        stale_params = [ticker]
        if exp_is_date:
            stale_q += " AND expiration = ?"
            stale_params.append(expiration)

        row = conn.execute(stale_q, stale_params).fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale"
            log.info("Fetching snapshot open interest for %s exp=%s from API (%s)", ticker, expiration, reason)
            try:
                df = fetch_snapshot_open_interest(ticker, expiration, client, strike, right, max_dte, strike_range, min_time)
                now = dt.datetime.now()
                if "symbol" in df.columns:
                    df = df.rename({"symbol": "root"})
                df = df.with_columns(pl.lit(now).alias("fetched_at"))
                df = df.select([c for c in ["root", "expiration", "strike", "right", "timestamp", "open_interest", "fetched_at"] if c in df.columns])
                conn.execute(f"INSERT OR REPLACE INTO {_TABLE} SELECT * FROM df")
                log.info("Fetched and stored %d open interest snapshots for %s", len(df), ticker)
            except Exception:
                log.exception("Failed to fetch snapshot open interest for %s exp=%s", ticker, expiration)
                raise
        else:
            log.debug("Reading snapshot open interest for %s from local DB (fetched_at=%s)", ticker, last_fetched)

        return read_snapshot_open_interest(ticker, expiration if exp_is_date else None, conn=conn)
    finally:
        if own_conn:
            conn.close()
