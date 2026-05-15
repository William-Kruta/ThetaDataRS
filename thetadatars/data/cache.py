from __future__ import annotations

import datetime as dt
import hashlib
import json
import threading
import time
import uuid
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

import polars as pl

from ..errors import CacheMissError

CachePolicy = Literal["prefer_cache", "cache_only", "refresh", "no_cache"]

_VALID_POLICIES = {"prefer_cache", "cache_only", "refresh", "no_cache"}
_LOCKS: weakref.WeakValueDictionary[str, threading.Lock] = weakref.WeakValueDictionary()
_LOCKS_GUARD = threading.Lock()


@dataclass(frozen=True, slots=True)
class CacheHit:
    endpoint: str
    root: str
    params: dict[str, Any]
    fetched_at: dt.datetime
    row_count: int | None = None


@dataclass(frozen=True, slots=True)
class CacheCoverage:
    endpoint: str
    root: str
    params: dict[str, Any]
    covered: bool
    fresh: bool
    reason: Literal["fresh_hit", "stale_hit", "miss", "no_cache"]
    fetched_at: dt.datetime | None = None


def _json_default(value: object) -> object:
    if isinstance(value, dt.datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return str(value)


def _canonical_value(value: object) -> object:
    if isinstance(value, dt.datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, dt.date | dt.time):
        return value.isoformat()
    if isinstance(value, dt.timedelta):
        return value.total_seconds()
    return value


def canonical_params(params: dict[str, object]) -> dict[str, object]:
    return {key: _canonical_value(params[key]) for key in sorted(params)}


def params_json(params: dict[str, object]) -> str:
    return json.dumps(
        canonical_params(params),
        default=_json_default,
        sort_keys=True,
        separators=(",", ":"),
    )


def params_hash(params: dict[str, object]) -> str:
    return hashlib.sha256(params_json(params).encode("utf-8")).hexdigest()


def _lock_for(key: str) -> threading.Lock:
    with _LOCKS_GUARD:
        lock = _LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _LOCKS[key] = lock
        return lock


def normalize_cache_policy(policy: str) -> CachePolicy:
    if policy not in _VALID_POLICIES:
        raise ValueError(
            "cache_policy must be one of: prefer_cache, cache_only, refresh, no_cache"
        )
    return policy  # type: ignore[return-value]


# Params where a cached value of None means "no restriction" (all rows were fetched),
# so the cache covers any specific requested value for that param.
_SUPERSET_NONE_PARAMS = {"max_dte", "strike_range", "min_time"}


def _covers_right(cached: object, requested: object) -> bool:
    return cached == requested or cached == "both" and requested in {"call", "put"}


def _covers_strike(cached: object, requested: object) -> bool:
    return cached == requested or cached == "*"


def params_cover(cached: dict[str, object], requested: dict[str, object]) -> bool:
    requested = canonical_params(requested)
    cached = canonical_params(cached)
    for key, requested_value in requested.items():
        cached_value = cached.get(key)
        if key == "right":
            if not _covers_right(cached_value, requested_value):
                return False
        elif key == "strike":
            if not _covers_strike(cached_value, requested_value):
                return False
        elif key == "expiration":
            if cached_value != requested_value and cached_value != "*":
                return False
        elif key in _SUPERSET_NONE_PARAMS and cached_value is None:
            # cached fetch had no restriction on this param → superset of any specific value
            pass
        elif cached_value != requested_value:
            return False
    return True


def find_fresh_cache_hit(
    *,
    conn,
    endpoint: str,
    root: str,
    params: dict[str, object],
    stale_threshold: dt.timedelta,
    now: dt.datetime | None = None,
) -> CacheHit | None:
    now = now or dt.datetime.now()
    earliest = now - stale_threshold
    rows = conn.execute(
        """
        SELECT params_json, fetched_at, row_count
        FROM cache_fetches
        WHERE endpoint = ? AND root = ? AND status = 'success'
          AND fetched_at >= ?
        ORDER BY fetched_at DESC
        LIMIT 200
        """,
        [endpoint, root, earliest],
    ).fetchall()
    for cached_params_json, fetched_at, row_count in rows:
        cached_params = json.loads(cached_params_json)
        if params_cover(cached_params, params):
            return CacheHit(
                endpoint=endpoint,
                root=root,
                params=cached_params,
                fetched_at=fetched_at,
                row_count=row_count,
            )
    return None


def inspect_cache_coverage(
    *,
    conn,
    endpoint: str,
    root: str,
    params: dict[str, object],
    stale_threshold: dt.timedelta,
    now: dt.datetime | None = None,
) -> CacheCoverage:
    now = now or dt.datetime.now()
    rows = conn.execute(
        """
        SELECT params_json, fetched_at
        FROM cache_fetches
        WHERE endpoint = ? AND root = ? AND status = 'success'
        ORDER BY fetched_at DESC
        LIMIT 500
        """,
        [endpoint, root],
    ).fetchall()
    stale_match: dt.datetime | None = None
    for cached_params_json, fetched_at in rows:
        cached_params = json.loads(cached_params_json)
        if not params_cover(cached_params, params):
            continue
        if fetched_at is not None and now - fetched_at <= stale_threshold:
            return CacheCoverage(
                endpoint=endpoint,
                root=root,
                params=canonical_params(params),
                covered=True,
                fresh=True,
                reason="fresh_hit",
                fetched_at=fetched_at,
            )
        stale_match = stale_match or fetched_at

    if stale_match is not None:
        return CacheCoverage(
            endpoint=endpoint,
            root=root,
            params=canonical_params(params),
            covered=True,
            fresh=False,
            reason="stale_hit",
            fetched_at=stale_match,
        )

    return CacheCoverage(
        endpoint=endpoint,
        root=root,
        params=canonical_params(params),
        covered=False,
        fresh=False,
        reason="miss",
    )


def record_cache_fetch(
    *,
    conn,
    endpoint: str,
    root: str,
    params: dict[str, object],
    cache_policy: CachePolicy,
    status: Literal["success", "error"],
    row_count: int | None,
    fetched_at: dt.datetime,
    duration_seconds: float,
    error: Exception | None = None,
) -> None:
    serialized_params = params_json(params)
    serialized_coverage = serialized_params
    conn.execute(
        """
        INSERT INTO cache_fetches (
            fetch_id,
            endpoint,
            root,
            params_hash,
            params_json,
            coverage_json,
            cache_policy,
            status,
            row_count,
            fetched_at,
            duration_seconds,
            error_type,
            error_message
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            str(uuid.uuid4()),
            endpoint,
            root,
            params_hash(params),
            serialized_params,
            serialized_coverage,
            cache_policy,
            status,
            row_count,
            fetched_at,
            duration_seconds,
            type(error).__name__ if error is not None else None,
            str(error) if error is not None else None,
        ],
    )


def get_or_fetch(
    *,
    conn,
    endpoint: str,
    root: str,
    params: dict[str, object],
    cache_policy: str,
    stale_threshold: dt.timedelta,
    read_cached: Callable[[], pl.DataFrame],
    fetch_upstream: Callable[[], pl.DataFrame],
    write_cached: Callable[[pl.DataFrame, dt.datetime], None],
) -> pl.DataFrame:
    policy = normalize_cache_policy(cache_policy)
    if policy == "no_cache":
        return fetch_upstream()

    lock_key = f"{endpoint}:{root}:{params_hash(params)}"
    with _lock_for(lock_key):
        if policy != "refresh":
            hit = find_fresh_cache_hit(
                conn=conn,
                endpoint=endpoint,
                root=root,
                params=params,
                stale_threshold=stale_threshold,
            )
            if hit is not None:
                # A previous fetch returned zero rows (e.g. no options for this ticker).
                # Return empty immediately rather than re-querying the DB and then falling
                # through to a redundant upstream fetch on every call within the stale window.
                if hit.row_count == 0:
                    return pl.DataFrame()
                cached = read_cached()
                if not cached.is_empty():
                    return cached

            if policy == "cache_only":
                raise CacheMissError(
                    "No fresh cached data was found for the request.",
                    ticker=root,
                    endpoint=endpoint,
                    params=canonical_params(params),
                    retryable=False,
                    user_message="No fresh cached data is available for this request.",
                )

        started_at = time.perf_counter()
        fetched_at = dt.datetime.now()
        try:
            rows = fetch_upstream()
            write_cached(rows, fetched_at)
        except Exception as exc:
            record_cache_fetch(
                conn=conn,
                endpoint=endpoint,
                root=root,
                params=params,
                cache_policy=policy,
                status="error",
                row_count=None,
                fetched_at=fetched_at,
                duration_seconds=time.perf_counter() - started_at,
                error=exc,
            )
            raise

        record_cache_fetch(
            conn=conn,
            endpoint=endpoint,
            root=root,
            params=params,
            cache_policy=policy,
            status="success",
            row_count=len(rows),
            fetched_at=fetched_at,
            duration_seconds=time.perf_counter() - started_at,
        )
        # Return the freshly-fetched rows directly rather than re-reading from the DB.
        # Re-reading could return a different (potentially empty) result if write_cached
        # skipped writing empty rows, and adds an unnecessary DB round-trip.
        return rows
