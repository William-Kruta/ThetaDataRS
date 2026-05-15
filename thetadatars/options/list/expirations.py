import datetime as dt
import logging
import polars as pl

from ...client import create_client, Client
from ...data.cache import CachePolicy, normalize_cache_policy
from ...data.db import get_connection
from ...errors import CacheMissError

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
    cache_policy: CachePolicy = "prefer_cache",
) -> pl.DataFrame:
    if isinstance(tickers, str):
        tickers = [tickers]
    tickers = list(tickers)
    if not tickers:
        return pl.DataFrame(schema={"root": pl.Utf8, "expiration": pl.Date})
    policy = normalize_cache_policy(cache_policy)

    def normalize_fetched(df: pl.DataFrame) -> pl.DataFrame:
        if "symbol" in df.columns:
            df = df.rename({"symbol": "root"})
        return df.select(["root", "expiration"])

    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        placeholders = ",".join("?" * len(tickers))
        if policy == "no_cache":
            log.info("Fetching expirations for %s from API (no_cache)", tickers)
            return normalize_fetched(fetch_options_expiration_list(tickers, client))

        rows = conn.execute(
            f"""
            SELECT root, MAX(fetched_at) AS fetched_at, COUNT(*) AS row_count
            FROM option_expirations
            WHERE root IN ({placeholders})
            GROUP BY root
            """,
            tickers,
        ).fetchall()
        fetched_by_root = {root: fetched_at for root, fetched_at, row_count in rows if row_count > 0}
        now = dt.datetime.now()
        is_fresh = all(
            (fetched_at := fetched_by_root.get(ticker)) is not None
            and now - fetched_at <= stale_threshold
            for ticker in tickers
        )

        if policy == "cache_only" and not is_fresh:
            raise CacheMissError(
                "No fresh cached expiration list was found for the request.",
                ticker=",".join(tickers),
                endpoint="option_list_expirations",
                params={"tickers": tickers},
                retryable=False,
                user_message="No fresh cached expiration list is available for this request.",
            )

        if policy == "refresh" or not is_fresh:
            reason = "refresh" if policy == "refresh" else "no local data or stale"
            log.info("Fetching expirations for %s from API (%s)", tickers, reason)
            try:
                df = fetch_options_expiration_list(tickers, client)
                now = dt.datetime.now()
                # API returns "symbol" for the underlying; map to "root"
                df = normalize_fetched(df).with_columns(pl.lit(now).alias("fetched_at"))
                conn.execute(
                    f"DELETE FROM option_expirations WHERE root IN ({placeholders})",
                    tickers,
                )
                conn.execute(
                    "INSERT OR REPLACE INTO option_expirations (root, expiration, fetched_at) "
                    "SELECT root, expiration, fetched_at FROM df"
                )
                log.info("Fetched and stored %d expirations for %s", len(df), tickers)
            except Exception:
                log.exception("Failed to fetch expirations for %s from API", tickers)
                raise
        else:
            log.debug("Reading expirations for %s from local DB", tickers)

        return conn.execute(
            f"SELECT root, expiration FROM option_expirations WHERE root IN ({placeholders}) ORDER BY root, expiration",
            tickers,
        ).pl()
    finally:
        if own_conn:
            conn.close()
