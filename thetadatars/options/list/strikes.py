import datetime as dt
import logging
import polars as pl

from ...client import create_client, Client
from ...data.db import get_connection

log = logging.getLogger(__name__)


def fetch_options_strike_list(
    tickers: list[str], expiration_date: dt.date, client: Client
):
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(expiration_date, str):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    df = client.option_list_strikes(symbol=tickers, expiration=expiration_date)
    return df


def read_options_strike_list(
    root: str,
    expiration: dt.date,
    conn=None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        return conn.execute(
            """
            SELECT DISTINCT strike FROM option_contracts
            WHERE root = ? AND expiration = ?
            ORDER BY strike
            """,
            [root, expiration],
        ).pl()
    finally:
        if own_conn:
            conn.close()


def get_options_strike_list(
    tickers: list[str] | str,
    expiration_date: dt.date,
    client: Client,
    stale_threshold: dt.timedelta = dt.timedelta(days=1),
    conn=None,
) -> pl.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    if isinstance(expiration_date, str):
        expiration_date = dt.datetime.strptime(expiration_date, "%Y-%m-%d").date()
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        # Strikes are derived from option_contracts; check freshness there
        row = conn.execute(
            f"SELECT MAX(fetched_at) FROM option_contracts "
            f"WHERE root IN ({','.join('?' * len(tickers))}) AND expiration = ?",
            tickers + [expiration_date],
        ).fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local contract data" if last_fetched is None else "stale"
            log.info(
                "Fetching strikes for %s exp=%s from API (%s)", tickers, expiration_date, reason
            )
            try:
                df = fetch_options_strike_list(tickers, expiration_date, client)
                log.info(
                    "Fetched %d strikes for %s exp=%s (not cached — populate contracts for local reads)",
                    len(df), tickers, expiration_date,
                )
                return df
            except Exception:
                log.exception(
                    "Failed to fetch strikes for %s exp=%s from API", tickers, expiration_date
                )
                raise

        log.debug(
            "Reading strikes for %s exp=%s from local DB (fetched_at=%s)",
            tickers, expiration_date, last_fetched,
        )
        return conn.execute(
            f"""
            SELECT DISTINCT strike FROM option_contracts
            WHERE root IN ({','.join('?' * len(tickers))}) AND expiration = ?
            ORDER BY strike
            """,
            tickers + [expiration_date],
        ).pl()
    finally:
        if own_conn:
            conn.close()
