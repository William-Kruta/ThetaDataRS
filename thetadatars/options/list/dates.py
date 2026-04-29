import datetime as dt
import logging
import polars as pl
from typing import Literal

from ...client import create_client, Client
from ...data.db import get_connection

log = logging.getLogger(__name__)


def fetch_options_dates_list(
    ticker: str,
    expiration_date: dt.date,
    client: Client,
    request_type: Literal["trade", "quote"] = "quote",
):
    if isinstance(expiration_date, str):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    df = client.option_list_dates(
        request_type=request_type,
        symbol=ticker,
        expiration=expiration_date,
    )
    return df


def read_options_dates_list(
    root: str,
    expiration: dt.date,
    request_type: Literal["trade", "quote"] = "quote",
    conn=None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT date FROM option_dates
            WHERE root = ? AND expiration = ? AND request_type = ?
            ORDER BY date
            """,
            [root, expiration, request_type],
        ).pl()
    finally:
        if own_conn:
            conn.close()


def get_options_dates_list(
    ticker: str,
    expiration_date: dt.date,
    client: Client,
    request_type: Literal["trade", "quote"] = "quote",
    stale_threshold: dt.timedelta = dt.timedelta(days=1),
    conn=None,
) -> pl.DataFrame:
    if isinstance(expiration_date, str):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        row = conn.execute(
            "SELECT MAX(fetched_at) FROM option_dates "
            "WHERE root = ? AND expiration = ? AND request_type = ?",
            [ticker, expiration_date, request_type],
        ).fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale"
            log.info(
                "Fetching %s dates for %s exp=%s from API (%s)",
                request_type, ticker, expiration_date, reason,
            )
            try:
                df = fetch_options_dates_list(ticker, expiration_date, client, request_type)
                now = dt.datetime.now()
                df = df.with_columns(
                    pl.lit(ticker).alias("root"),
                    pl.lit(expiration_date).alias("expiration"),
                    pl.lit(request_type).alias("request_type"),
                    pl.lit(now).alias("fetched_at"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO option_dates "
                    "(root, expiration, date, request_type, fetched_at) "
                    "SELECT root, expiration, date, request_type, fetched_at FROM df"
                )
                log.info(
                    "Fetched and stored %d %s dates for %s exp=%s",
                    len(df), request_type, ticker, expiration_date,
                )
            except Exception:
                log.exception(
                    "Failed to fetch %s dates for %s exp=%s from API",
                    request_type, ticker, expiration_date,
                )
                raise
        else:
            log.debug(
                "Reading %s dates for %s exp=%s from local DB (fetched_at=%s)",
                request_type, ticker, expiration_date, last_fetched,
            )

        return read_options_dates_list(ticker, expiration_date, request_type, conn=conn)
    finally:
        if own_conn:
            conn.close()
