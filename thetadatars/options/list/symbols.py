import datetime as dt
import logging
import polars as pl

from ...client import create_client, Client
from ...data.db import get_connection

log = logging.getLogger(__name__)


def fetch_options_symbols_list(client: Client):
    df = client.option_list_symbols()
    return df


def read_options_symbols_list(conn=None) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        return conn.execute("SELECT symbol FROM option_symbols ORDER BY symbol").pl()
    finally:
        if own_conn:
            conn.close()


def get_options_symbols_list(
    client: Client,
    stale_threshold: dt.timedelta = dt.timedelta(days=7),
    conn=None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        row = conn.execute("SELECT MAX(fetched_at) FROM option_symbols").fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale"
            log.info("Fetching option symbols from API (%s)", reason)
            try:
                df = fetch_options_symbols_list(client)
                now = dt.datetime.now()
                df = df.with_columns(pl.lit(now).alias("fetched_at"))
                conn.execute(
                    "INSERT OR REPLACE INTO option_symbols (symbol, fetched_at) SELECT symbol, fetched_at FROM df"
                )
                log.info("Fetched and stored %d option symbols", len(df))
            except Exception:
                log.exception("Failed to fetch option symbols from API")
                raise
        else:
            log.debug("Reading option symbols from local DB (fetched_at=%s)", last_fetched)

        return read_options_symbols_list(conn=conn)
    finally:
        if own_conn:
            conn.close()
