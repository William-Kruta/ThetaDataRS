import datetime as dt
import logging
import polars as pl
from typing import Literal

from ...client import create_client, Client
from ...data.db import get_connection

log = logging.getLogger(__name__)


def fetch_options_contract_list(
    tickers: list[str],
    trading_date: dt.date,
    client: Client,
    request_type: Literal["trade", "quote"] = "trade",
):
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(trading_date, str):
        trading_date = dt.datetime.strptime(trading_date, "%Y-%m-%d").date()
    df = client.option_list_contracts(
        symbol=tickers, request_type=request_type.lower(), date=trading_date
    )
    return df


def read_options_contract_list(
    root: str,
    expiration: dt.date | None = None,
    conn=None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        if expiration is not None:
            return conn.execute(
                """
                SELECT root, expiration, strike, "right"
                FROM option_contracts
                WHERE root = ? AND expiration = ?
                ORDER BY expiration, strike, "right"
                """,
                [root, expiration],
            ).pl()
        return conn.execute(
            """
            SELECT root, expiration, strike, "right"
            FROM option_contracts
            WHERE root = ?
            ORDER BY expiration, strike, "right"
            """,
            [root],
        ).pl()
    finally:
        if own_conn:
            conn.close()


def get_options_contract_list(
    tickers: list[str] | str,
    trading_date: dt.date,
    client: Client,
    request_type: Literal["trade", "quote"] = "trade",
    stale_threshold: dt.timedelta = dt.timedelta(days=1),
    conn=None,
) -> pl.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(trading_date, str):
        trading_date = dt.datetime.strptime(trading_date, "%Y-%m-%d").date()
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT MAX(fetched_at) FROM option_contracts "
            f"WHERE root IN ({','.join('?' * len(tickers))}) AND as_of_date = ?",
            tickers + [trading_date],
        ).fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale"
            log.info(
                "Fetching contracts for %s on %s from API (%s)", tickers, trading_date, reason
            )
            try:
                df = fetch_options_contract_list(tickers, trading_date, client, request_type)
                now = dt.datetime.now()
                # API returns "symbol" for the underlying; map to "root"
                if "symbol" in df.columns:
                    df = df.rename({"symbol": "root"})
                df = df.with_columns(
                    pl.lit(trading_date).alias("as_of_date"),
                    pl.lit(now).alias("fetched_at"),
                )
                conn.execute(
                    "INSERT OR REPLACE INTO option_contracts "
                    '(root, expiration, strike, "right", as_of_date, fetched_at) '
                    'SELECT root, expiration, strike, "right", as_of_date, fetched_at FROM df'
                )
                log.info(
                    "Fetched and stored %d contracts for %s on %s", len(df), tickers, trading_date
                )
            except Exception:
                log.exception(
                    "Failed to fetch contracts for %s on %s from API", tickers, trading_date
                )
                raise
        else:
            log.debug(
                "Reading contracts for %s on %s from local DB (fetched_at=%s)",
                tickers, trading_date, last_fetched,
            )

        roots_filter = ",".join("?" * len(tickers))
        if len(tickers) == 1:
            return read_options_contract_list(tickers[0], conn=conn)
        return conn.execute(
            f"""
            SELECT root, expiration, strike, "right"
            FROM option_contracts
            WHERE root IN ({roots_filter}) AND as_of_date = ?
            ORDER BY root, expiration, strike, "right"
            """,
            tickers + [trading_date],
        ).pl()
    finally:
        if own_conn:
            conn.close()
