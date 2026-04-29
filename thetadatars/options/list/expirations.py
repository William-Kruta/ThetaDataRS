import datetime as dt
import logging
import polars as pl

from ...client import create_client, Client
from ...data.db import get_connection

log = logging.getLogger(__name__)


def fetch_options_expiration_list(tickers: list[str], client: Client):
    if isinstance(tickers, str):
        tickers = [tickers]
    df = client.option_list_expirations(symbol=tickers)
    return df


def read_options_expiration_list(root: str, conn=None) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        return conn.execute(
            "SELECT expiration FROM option_expirations WHERE root = ? ORDER BY expiration",
            [root],
        ).pl()
    finally:
        if own_conn:
            conn.close()


def get_options_expiration_list(
    tickers: list[str] | str,
    client: Client,
    stale_threshold: dt.timedelta = dt.timedelta(days=1),
    conn=None,
) -> pl.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        row = conn.execute(
            f"SELECT MAX(fetched_at) FROM option_expirations WHERE root IN ({','.join('?' * len(tickers))})",
            tickers,
        ).fetchone()
        last_fetched = row[0] if row else None
        is_stale = last_fetched is None or (dt.datetime.now() - last_fetched > stale_threshold)

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale"
            log.info("Fetching expirations for %s from API (%s)", tickers, reason)
            try:
                df = fetch_options_expiration_list(tickers, client)
                now = dt.datetime.now()
                # API returns "symbol" for the underlying; map to "root"
                df = df.rename({"symbol": "root"}).with_columns(pl.lit(now).alias("fetched_at"))
                conn.execute(
                    "INSERT OR REPLACE INTO option_expirations (root, expiration, fetched_at) "
                    "SELECT root, expiration, fetched_at FROM df"
                )
                log.info("Fetched and stored %d expirations for %s", len(df), tickers)
            except Exception:
                log.exception("Failed to fetch expirations for %s from API", tickers)
                raise
        else:
            log.debug("Reading expirations for %s from local DB (fetched_at=%s)", tickers, last_fetched)

        return conn.execute(
            f"SELECT root, expiration FROM option_expirations WHERE root IN ({','.join('?' * len(tickers))}) ORDER BY root, expiration",
            tickers,
        ).pl()
    finally:
        if own_conn:
            conn.close()
