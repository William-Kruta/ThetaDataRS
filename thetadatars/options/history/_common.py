import datetime as dt
import logging
from typing import Callable, Literal

import duckdb
import polars as pl

from ...client import Client
from ...data.db import get_connection

Right = Literal["call", "put", "both"]
RateType = Literal[
    "sofr",
    "treasury_m1",
    "treasury_m3",
    "treasury_m6",
    "treasury_y1",
    "treasury_y2",
    "treasury_y3",
    "treasury_y5",
    "treasury_y7",
    "treasury_y10",
    "treasury_y20",
    "treasury_y30",
]

_RESERVED = {"right", "open", "close", "lambda"}


def parse_date(value: dt.date | str | None) -> dt.date | str | None:
    if value is None or value == "*":
        return value
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.datetime.strptime(value, "%Y-%m-%d").date()


def quote_identifier(name: str) -> str:
    if name in _RESERVED:
        return f'"{name}"'
    return name


def select_sql(columns: list[str]) -> str:
    return ", ".join(quote_identifier(c) for c in columns)


def normalize_history_df(
    df: pl.DataFrame,
    columns: list[str],
    fetched_at: dt.datetime,
    start_date: dt.date,
    end_date: dt.date,
) -> pl.DataFrame:
    if "symbol" in df.columns and "root" not in df.columns:
        df = df.rename({"symbol": "root"})

    if "date" not in df.columns:
        if start_date == end_date:
            df = df.with_columns(pl.lit(start_date).alias("date"))
        elif "timestamp" in df.columns:
            df = df.with_columns(pl.col("timestamp").cast(pl.Date).alias("date"))
        elif "created" in df.columns:
            df = df.with_columns(pl.col("created").cast(pl.Date).alias("date"))
        elif "trade_timestamp" in df.columns:
            df = df.with_columns(pl.col("trade_timestamp").cast(pl.Date).alias("date"))

    df = df.with_columns(pl.lit(fetched_at).alias("fetched_at"))

    for column in columns:
        if column not in df.columns:
            df = df.with_columns(pl.lit(None).alias(column))

    return df.select(columns)


def read_history_table(
    table: str,
    columns: list[str],
    root: str,
    start_date: dt.date,
    end_date: dt.date,
    expiration: dt.date | str | None = None,
    strike: float | str | None = None,
    right: Right | None = None,
    conn: duckdb.DuckDBPyConnection | None = None,
    order_columns: list[str] | None = None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        where, params = _build_where(root, start_date, end_date, expiration, strike, right)
        order_by = ", ".join(quote_identifier(c) for c in (order_columns or columns[:5]))
        return conn.execute(
            f"""
            SELECT {select_sql(columns)}
            FROM {table}
            WHERE {where}
            ORDER BY {order_by}
            """,
            params,
        ).pl()
    finally:
        if own_conn:
            conn.close()


def get_history_table(
    *,
    table: str,
    columns: list[str],
    fetcher: Callable[[], pl.DataFrame],
    log: logging.Logger,
    label: str,
    root: str,
    start_date: dt.date,
    end_date: dt.date,
    expiration: dt.date | str,
    strike: str = "*",
    right: Right = "both",
    stale_threshold: dt.timedelta = dt.timedelta(days=1),
    conn: duckdb.DuckDBPyConnection | None = None,
    order_columns: list[str] | None = None,
) -> pl.DataFrame:
    own_conn = conn is None
    if own_conn:
        conn = get_connection()
    try:
        where, params = _build_where(root, start_date, end_date, expiration, strike, right)
        row = conn.execute(
            f"""
            SELECT MIN(date), MAX(date), MAX(fetched_at)
            FROM {table}
            WHERE {where}
            """,
            params,
        ).fetchone()
        min_date, max_date, last_fetched = row if row else (None, None, None)
        missing_range = min_date is None or min_date > start_date or max_date < end_date
        is_stale = (
            missing_range
            or last_fetched is None
            or (dt.datetime.now() - last_fetched > stale_threshold)
        )

        if is_stale:
            reason = "no local data" if last_fetched is None else "stale or incomplete"
            log.info("Fetching %s for %s %s..%s from API (%s)", label, root, start_date, end_date, reason)
            try:
                now = dt.datetime.now()
                df = normalize_history_df(fetcher(), columns, now, start_date, end_date)
                insert_cols = select_sql(columns)
                conn.execute(
                    f"""
                    INSERT OR REPLACE INTO {table} ({insert_cols})
                    SELECT {insert_cols} FROM df
                    """
                )
                log.info("Fetched and stored %d %s rows for %s", len(df), label, root)
            except Exception:
                log.exception("Failed to fetch %s for %s %s..%s", label, root, start_date, end_date)
                raise
        else:
            log.debug("Reading %s for %s from local DB (fetched_at=%s)", label, root, last_fetched)

        return read_history_table(
            table,
            columns[:-1] if columns and columns[-1] == "fetched_at" else columns,
            root,
            start_date,
            end_date,
            expiration,
            strike,
            right,
            conn=conn,
            order_columns=order_columns,
        )
    finally:
        if own_conn:
            conn.close()


def _build_where(
    root: str,
    start_date: dt.date,
    end_date: dt.date,
    expiration: dt.date | str | None,
    strike: float | str | None,
    right: Right | None,
) -> tuple[str, list[object]]:
    where = ["root = ?", "date BETWEEN ? AND ?"]
    params: list[object] = [root, start_date, end_date]
    if expiration is not None and expiration != "*":
        where.append("expiration = ?")
        params.append(expiration)
    if strike is not None and strike != "*":
        where.append("strike = ?")
        params.append(float(strike))
    if right is not None and right != "both":
        where.append('"right" = ?')
        params.append(right)
    return " AND ".join(where), params


def require_date_range(
    start_date: dt.date | str,
    end_date: dt.date | str | None = None,
) -> tuple[dt.date, dt.date]:
    start = parse_date(start_date)
    end = parse_date(end_date) if end_date is not None else start
    if not isinstance(start, dt.date) or not isinstance(end, dt.date):
        raise ValueError("start_date and end_date must be concrete dates")
    return start, end


def require_expiration(expiration: dt.date | str) -> dt.date | str:
    parsed = parse_date(expiration)
    if parsed is None:
        raise ValueError("expiration is required")
    return parsed
