import datetime as dt
from pathlib import Path
import tempfile
import unittest

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.data.cache import inspect_cache_coverage
from thetadatars.errors import CacheMissError
from thetadatars.options.list.expirations import get_options_expiration_list
from thetadatars.options.snapshot.greeks_first_order import get_snapshot_greeks_first_order
from thetadatars.options.snapshot.quote import get_snapshot_quote


class FakeSnapshotClient:
    def __init__(self):
        self.quote_calls = []
        self.greeks_calls = []

    def option_snapshot_quote(
        self,
        *,
        symbol,
        expiration,
        strike="*",
        right="both",
        max_dte=None,
        strike_range=None,
        min_time=None,
    ):
        self.quote_calls.append(
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "max_dte": max_dte,
                "strike_range": strike_range,
                "min_time": min_time,
            }
        )
        option_right = "call" if right == "call" else "put"
        strike_value = 105.0 if option_right == "call" else 95.0
        return pl.DataFrame(
            [
                {
                    "root": symbol,
                    "expiration": expiration,
                    "strike": strike_value,
                    "right": option_right,
                    "timestamp": dt.datetime(2026, 5, 13, 13, 0),
                    "bid_size": 10,
                    "bid_exchange": 1,
                    "bid": 1.20,
                    "bid_condition": 0,
                    "ask_size": 12,
                    "ask_exchange": 2,
                    "ask": 1.30,
                    "ask_condition": 0,
                }
            ]
        )

    def option_snapshot_greeks_first_order(
        self,
        *,
        symbol,
        expiration,
        strike="*",
        right="both",
        annual_dividend=None,
        rate_type="sofr",
        rate_value=None,
        stock_price=None,
        version="latest",
        max_dte=None,
        strike_range=None,
        min_time=None,
        use_market_value=False,
    ):
        self.greeks_calls.append(
            {
                "symbol": symbol,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "version": version,
                "strike_range": strike_range,
            }
        )
        return pl.DataFrame(
            [
                {
                    "root": symbol,
                    "expiration": expiration,
                    "strike": 95.0,
                    "right": "put",
                    "timestamp": dt.datetime(2026, 5, 13, 13, 0),
                    "bid": 1.20,
                    "ask": 1.30,
                    "delta": -0.25,
                    "theta": -0.01,
                    "vega": 0.05,
                    "rho": -0.02,
                    "epsilon": None,
                    "lambda": None,
                    "implied_vol": 0.45,
                    "iv_error": None,
                    "underlying_timestamp": dt.datetime(2026, 5, 13, 13, 0),
                    "underlying_price": 100.0,
                }
            ]
        )


class FakeExpirationClient:
    def __init__(self, expirations: list[dt.date] | None = None):
        self.expiration_calls = []
        self.expirations = expirations or [dt.date(2026, 6, 19)]

    def option_list_expirations(self, *, symbol):
        self.expiration_calls.append({"symbol": symbol})
        symbols = symbol if isinstance(symbol, list) else [symbol]
        return pl.DataFrame(
            [
                {
                    "symbol": ticker,
                    "expiration": expiration,
                }
                for ticker in symbols
                for expiration in self.expirations
            ]
        )


class CachePolicyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "cache-test.duckdb")
        self.conn = get_connection(self.db_path)
        self.client = FakeSnapshotClient()
        self.expiration = dt.date(2026, 6, 19)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_cache_only_miss_does_not_fetch(self):
        with self.assertRaises(CacheMissError) as raised:
            get_snapshot_quote(
                "XYZ",
                self.expiration,
                client=self.client,
                right="put",
                cache_policy="cache_only",
                conn=self.conn,
            )

        self.assertEqual(self.client.quote_calls, [])
        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "option_snapshot_quote")
        self.assertFalse(raised.exception.retryable)

    def test_expiration_list_cache_policies_control_upstream_and_writes(self):
        expiration_client = FakeExpirationClient()

        with self.assertRaises(CacheMissError):
            get_options_expiration_list(
                "XYZ",
                client=expiration_client,
                cache_policy="cache_only",
                conn=self.conn,
            )
        self.assertEqual(expiration_client.expiration_calls, [])

        no_cache_rows = get_options_expiration_list(
            "XYZ",
            client=expiration_client,
            cache_policy="no_cache",
            conn=self.conn,
        )
        cached_count = self.conn.execute(
            "SELECT COUNT(*) FROM option_expirations WHERE root = 'XYZ'"
        ).fetchone()[0]
        self.assertEqual(len(no_cache_rows), 1)
        self.assertEqual(cached_count, 0)
        self.assertEqual(len(expiration_client.expiration_calls), 1)

        cached_rows = get_options_expiration_list(
            "XYZ",
            client=expiration_client,
            cache_policy="prefer_cache",
            conn=self.conn,
        )
        cache_only_rows = get_options_expiration_list(
            "XYZ",
            client=expiration_client,
            cache_policy="cache_only",
            conn=self.conn,
        )
        self.assertEqual(cached_rows.to_dicts(), cache_only_rows.to_dicts())
        self.assertEqual(len(expiration_client.expiration_calls), 2)

    def test_expiration_refresh_replaces_stale_root_expirations(self):
        old_fetched_at = dt.datetime.now() - dt.timedelta(days=10)
        self.conn.execute(
            """
            INSERT INTO option_expirations (root, expiration, fetched_at)
            VALUES ('XYZ', DATE '2026-07-17', ?)
            """,
            [old_fetched_at],
        )
        expiration_client = FakeExpirationClient(expirations=[dt.date(2026, 6, 19)])

        refreshed = get_options_expiration_list(
            "XYZ",
            client=expiration_client,
            cache_policy="refresh",
            conn=self.conn,
        )
        cached = get_options_expiration_list(
            "XYZ",
            client=expiration_client,
            cache_policy="cache_only",
            conn=self.conn,
        )

        self.assertEqual(refreshed["expiration"].to_list(), [dt.date(2026, 6, 19)])
        self.assertEqual(cached["expiration"].to_list(), [dt.date(2026, 6, 19)])
        self.assertEqual(len(expiration_client.expiration_calls), 1)

    def test_prefer_cache_uses_request_coverage_not_just_root_expiration(self):
        put_rows = get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            cache_policy="prefer_cache",
            conn=self.conn,
        )
        call_rows = get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="call",
            cache_policy="prefer_cache",
            conn=self.conn,
        )

        self.assertEqual(len(self.client.quote_calls), 2)
        self.assertEqual(put_rows["right"].to_list(), ["put"])
        self.assertEqual(call_rows["right"].to_list(), ["call"])

    def test_fetch_records_cache_metadata(self):
        get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            strike_range=10,
            cache_policy="prefer_cache",
            conn=self.conn,
        )

        rows = self.conn.execute(
            """
            SELECT endpoint, root, row_count, status, cache_policy, params_json
            FROM cache_fetches
            WHERE endpoint = 'option_snapshot_quote'
            """
        ).fetchall()
        self.assertEqual(len(rows), 1)
        endpoint, root, row_count, status, cache_policy, params_json = rows[0]
        self.assertEqual(endpoint, "option_snapshot_quote")
        self.assertEqual(root, "XYZ")
        self.assertEqual(row_count, 1)
        self.assertEqual(status, "success")
        self.assertEqual(cache_policy, "prefer_cache")
        self.assertIn('"right":"put"', params_json)
        self.assertIn('"strike_range":10', params_json)

    def test_narrow_cache_entry_does_not_satisfy_broad_request(self):
        get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            strike_range=10,
            cache_policy="prefer_cache",
            conn=self.conn,
        )
        get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            strike_range=None,
            cache_policy="prefer_cache",
            conn=self.conn,
        )

        self.assertEqual(len(self.client.quote_calls), 2)
        self.assertEqual(self.client.quote_calls[0]["strike_range"], 10)
        self.assertIsNone(self.client.quote_calls[1]["strike_range"])

    def test_no_cache_fetches_without_writing_rows_or_metadata(self):
        rows = get_snapshot_greeks_first_order(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            cache_policy="no_cache",
            conn=self.conn,
        )

        cached_rows = self.conn.execute(
            "SELECT COUNT(*) FROM snapshot_greeks_first_order"
        ).fetchone()[0]
        metadata_rows = self.conn.execute(
            "SELECT COUNT(*) FROM cache_fetches WHERE endpoint = 'option_snapshot_greeks_first_order'"
        ).fetchone()[0]
        self.assertEqual(len(rows), 1)
        self.assertEqual(len(self.client.greeks_calls), 1)
        self.assertEqual(cached_rows, 0)
        self.assertEqual(metadata_rows, 0)

    def test_inspect_cache_coverage_reports_fresh_hit(self):
        get_snapshot_quote(
            "XYZ",
            self.expiration,
            client=self.client,
            right="put",
            strike_range=10,
            cache_policy="prefer_cache",
            conn=self.conn,
        )

        coverage = inspect_cache_coverage(
            conn=self.conn,
            endpoint="option_snapshot_quote",
            root="XYZ",
            params={
                "expiration": self.expiration,
                "strike": "*",
                "right": "put",
                "max_dte": None,
                "strike_range": 10,
                "min_time": None,
            },
            stale_threshold=dt.timedelta(hours=1),
        )

        self.assertTrue(coverage.covered)
        self.assertTrue(coverage.fresh)
        self.assertEqual(coverage.reason, "fresh_hit")
        self.assertEqual(coverage.endpoint, "option_snapshot_quote")

    def test_inspect_cache_coverage_reports_miss(self):
        coverage = inspect_cache_coverage(
            conn=self.conn,
            endpoint="option_snapshot_quote",
            root="XYZ",
            params={
                "expiration": self.expiration,
                "strike": "*",
                "right": "put",
            },
            stale_threshold=dt.timedelta(hours=1),
        )

        self.assertFalse(coverage.covered)
        self.assertFalse(coverage.fresh)
        self.assertEqual(coverage.reason, "miss")


if __name__ == "__main__":
    unittest.main()
