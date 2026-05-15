import datetime as dt

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.options.batch import (
    get_snapshot_greeks_first_order_batch,
    get_snapshot_quote_batch,
)


class FakeBatchClient:
    def __init__(self, quote_failures=None):
        self.quote_calls = []
        self.greeks_calls = []
        self.quote_failures = set(quote_failures or ())

    def option_snapshot_quote(
        self,
        *,
        symbol,
        expiration,
        strike="*",
        right="both",
        **kwargs,
    ):
        self.quote_calls.append(
            {
                "root": symbol,
                "exp": expiration,
                "right": right,
                "strike": strike,
                **kwargs,
            }
        )
        if symbol in self.quote_failures:
            raise RuntimeError(f"no data found for {symbol}")
        return pl.DataFrame(
            {
                "root": [symbol],
                "expiration": [expiration],
                "strike": [strike if strike != "*" else 100000],
                "right": [right],
                "timestamp": [dt.datetime(2025, 1, 2, 9, 30)],
                "bid_size": [10],
                "bid_exchange": [1],
                "bid": [1.0],
                "bid_condition": [0],
                "ask_size": [11],
                "ask_exchange": [2],
                "ask": [1.2],
                "ask_condition": [0],
            }
        )

    def option_snapshot_greeks_first_order(
        self,
        *,
        symbol,
        expiration,
        strike="*",
        right="both",
        **kwargs,
    ):
        self.greeks_calls.append(
            {
                "root": symbol,
                "exp": expiration,
                "right": right,
                "strike": strike,
                **kwargs,
            }
        )
        return pl.DataFrame(
            {
                "root": [symbol],
                "expiration": [expiration],
                "strike": [strike if strike != "*" else 100000],
                "right": [right],
                "timestamp": [dt.datetime(2025, 1, 2, 9, 30)],
                "bid": [1.0],
                "ask": [1.2],
                "delta": [0.42],
                "theta": [-0.03],
                "vega": [0.12],
                "rho": [0.02],
                "epsilon": [0.01],
                "lambda": [1.0],
                "implied_vol": [0.25],
                "iv_error": [0.0],
                "underlying_timestamp": [dt.datetime(2025, 1, 2, 9, 30)],
                "underlying_price": [100.0],
            }
        )


def test_quote_batch_uses_endpoint_cache_policy_and_combines_data(tmp_path):
    conn = get_connection(str(tmp_path / "batch.duckdb"))
    client = FakeBatchClient()

    result = get_snapshot_quote_batch(
        ["AAA", "BBB"],
        dt.date(2025, 1, 17),
        client=client,
        right="put",
        cache_policy="prefer_cache",
        conn=conn,
    )

    assert result.stats.total == 2
    assert result.stats.succeeded == 2
    assert result.stats.failed == 0
    assert sorted(result.data["root"].to_list()) == ["AAA", "BBB"]
    assert len(client.quote_calls) == 2

    cached = get_snapshot_quote_batch(
        ["AAA", "BBB"],
        dt.date(2025, 1, 17),
        client=client,
        right="put",
        cache_policy="cache_only",
        conn=conn,
    )

    assert cached.stats.succeeded == 2
    assert cached.stats.failed == 0
    assert sorted(cached.data["root"].to_list()) == ["AAA", "BBB"]
    assert len(client.quote_calls) == 2


def test_quote_batch_cache_only_miss_returns_structured_failures(tmp_path):
    conn = get_connection(str(tmp_path / "empty.duckdb"))
    client = FakeBatchClient()

    result = get_snapshot_quote_batch(
        ["AAA", "BBB"],
        dt.date(2025, 1, 17),
        client=client,
        right="call",
        cache_policy="cache_only",
        conn=conn,
    )

    assert result.stats.total == 2
    assert result.stats.succeeded == 0
    assert result.stats.failed == 2
    assert result.data.is_empty()
    assert client.quote_calls == []
    assert {failure.ticker for failure in result.failures} == {"AAA", "BBB"}
    assert {failure.error.endpoint for failure in result.failures} == {"option_snapshot_quote"}


def test_quote_batch_partial_failure_reports_progress(tmp_path):
    conn = get_connection(str(tmp_path / "partial.duckdb"))
    client = FakeBatchClient(quote_failures={"BAD"})
    progress = []

    result = get_snapshot_quote_batch(
        ["GOOD", "BAD"],
        dt.date(2025, 1, 17),
        client=client,
        right="call",
        cache_policy="prefer_cache",
        conn=conn,
        on_progress=progress.append,
    )

    assert result.stats.succeeded == 1
    assert result.stats.failed == 1
    assert result.data["root"].to_list() == ["GOOD"]
    assert [item.ticker for item in progress] == ["GOOD", "BAD"]
    assert result.failures[0].ticker == "BAD"
    assert result.failures[0].error.endpoint == "option_snapshot_quote"


def test_greeks_batch_passes_controls_and_combines_data(tmp_path):
    conn = get_connection(str(tmp_path / "greeks.duckdb"))
    client = FakeBatchClient()

    result = get_snapshot_greeks_first_order_batch(
        ["AAA", "BBB"],
        dt.date(2025, 1, 17),
        client=client,
        right="put",
        annual_dividend=1.25,
        rate_type="sofr",
        version="1",
        strike_range=10,
        cache_policy="refresh",
        conn=conn,
    )

    assert result.stats.succeeded == 2
    assert sorted(result.data["root"].to_list()) == ["AAA", "BBB"]
    assert len(client.greeks_calls) == 2
    assert all(call["rate_type"] == "sofr" for call in client.greeks_calls)
    assert all(call["version"] == "1" for call in client.greeks_calls)
    assert all(call["strike_range"] == 10 for call in client.greeks_calls)


def test_batch_helpers_are_exposed_on_facade():
    from thetadatars.thetadata import ThetaDataRS

    theta = ThetaDataRS()

    assert hasattr(theta, "get_snapshot_quote_batch")
    assert hasattr(theta, "get_snapshot_greeks_first_order_batch")
