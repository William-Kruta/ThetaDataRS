import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import debit_spreads
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerResult,
    ScreenerStats,
    WarmCacheResult,
)
from thetadatars.options.screeners.debit_spreads import (
    DebitSpreadRequest,
    get_best_debit_spreads,
    plan_debit_spreads,
    screen_debit_spread_watchlist,
    screen_debit_spreads,
    warm_debit_spread_cache,
)


def sample_chain(root: str = "XYZ", expiration: dt.date = dt.date(2026, 6, 19)) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "root": root,
                "expiration": expiration,
                "strike": 100.0,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 4.90,
                "ask": 5.10,
                "delta": 0.55,
                "implied_vol": 0.35,
                "underlying_price": 101.0,
            },
            {
                "root": root,
                "expiration": expiration,
                "strike": 105.0,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 2.00,
                "ask": 2.15,
                "delta": 0.35,
                "implied_vol": 0.38,
                "underlying_price": 101.0,
            },
        ]
    )


def multi_spread_chain(root: str = "XYZ", expiration: dt.date = dt.date(2026, 6, 19)) -> pl.DataFrame:
    rows = []
    for strike, bid, ask, delta in [
        (100.0, 4.90, 5.10, 0.55),
        (105.0, 2.00, 2.15, 0.35),
        (110.0, 0.70, 0.80, 0.20),
        (115.0, 0.20, 0.30, 0.10),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "implied_vol": 0.35,
                "underlying_price": 101.0,
            }
        )
    return pl.DataFrame(rows)


def cap_regression_chain(root: str = "XYZ", expiration: dt.date = dt.date(2026, 6, 19)) -> pl.DataFrame:
    rows = []
    for strike, bid, ask, delta in [
        (100.0, 4.90, 5.10, 0.55),
        (105.0, 2.00, 2.15, 0.35),
        (110.0, 0.70, 0.80, 0.20),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "implied_vol": 0.35,
                "underlying_price": 101.0,
            }
        )
    return pl.DataFrame(rows)


class DebitSpreadScreeningTests(unittest.TestCase):
    def test_typed_screen_returns_result_and_passes_fetch_controls(self):
        seen = {}

        def fake_chain(**kwargs):
            seen.update(kwargs)
            kwargs["diagnostics"]["greeks_source"] = "local"
            return sample_chain(root=kwargs["ticker"])

        request = DebitSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="call",
            greeks_source="local",
            cache_policy="refresh",
            strike_range=20,
            top_n=5,
        )

        with patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_debit_spreads(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(seen["right"], "call")
        self.assertEqual(seen["greeks_source"], "local")
        self.assertEqual(seen["cache_policy"], "refresh")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "local")
        self.assertEqual(result.data["spread_type"][0], "bull_call_debit")

    def test_full_chain_guard_rejects_before_fetch(self):
        request = DebitSpreadRequest(ticker="XYZ", expiration="*")

        with patch.object(debit_spreads, "get_first_order_chain") as fetch:
            with self.assertRaises(InvalidRequestError) as raised:
                screen_debit_spreads(request, client=object())

        fetch.assert_not_called()
        self.assertEqual(raised.exception.endpoint, "screen_debit_spreads")

    def test_invalid_ranges_reject_before_fetch(self):
        requests = [
            DebitSpreadRequest(ticker="XYZ", expiration="2026-06-19", min_debit=2.0, max_debit=1.0),
            DebitSpreadRequest(ticker="XYZ", expiration="2026-06-19", min_width=10.0, max_width=5.0),
        ]

        with patch.object(debit_spreads, "get_first_order_chain") as fetch:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(InvalidRequestError):
                        screen_debit_spreads(request, client=object())

        fetch.assert_not_called()

    def test_planner_reports_endpoint_right_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "debit-plan.duckdb"))
            try:
                request = DebitSpreadRequest(
                    ticker="XYZ",
                    expiration="2026-06-19",
                    right="put",
                    greeks_source="none",
                    strike_range=10,
                )

                plan = plan_debit_spreads(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.strategy, "debit_spread")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_coverage[0].params["right"], "put")
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_watchlist_partial_failure_and_client_factory_failure(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError("No data", ticker="BAD", endpoint="screen_debit_spreads", retryable=False)
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "debit": [3.10]}),
                stats=ScreenerStats(ticker=request.ticker, fetched_rows=2, filtered_rows=2, candidate_rows=1, returned_rows=1),
            )

        request = DebitSpreadRequest(expiration="*", max_dte=45, strike_range=20)
        with patch.object(debit_spreads, "screen_debit_spreads", side_effect=fake_screen):
            batch = screen_debit_spread_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].error.endpoint, "screen_debit_spreads")

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_debit_spread_watchlist(["BAD"], request, client_factory=failing_client_factory)

        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_debit_spreads")

    def test_warm_cache_fetches_chain_without_building_or_ranking_candidates(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_chain(root=kwargs["ticker"])

        request = DebitSpreadRequest(expiration="*", max_dte=45, strike_range=20, greeks_source="none")

        with (
            patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain),
            patch.object(debit_spreads, "_build_rows", side_effect=AssertionError("warm cache built candidates")),
        ):
            warmup = warm_debit_spread_cache(["GOOD"], request, client=object())

        self.assertIsInstance(warmup, WarmCacheResult)
        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 2)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_failures_use_warm_endpoint(self):
        warmup = warm_debit_spread_cache(["BAD"], DebitSpreadRequest(expiration="*"), client=object())

        self.assertEqual(warmup.stats.failed, 1)
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_debit_spread_cache")

        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError("No data", ticker=request.ticker, endpoint="warm_debit_spread_cache", retryable=False)

        request = DebitSpreadRequest(expiration="*", max_dte=45, strike_range=20)
        with patch.object(debit_spreads, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_debit_spread_cache(
                ["BAD", "SKIPPED"],
                request,
                client=object(),
                circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1),
            )

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].ticker, "SKIPPED")
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_debit_spread_cache")

    def test_candidate_caps_prune_rows_and_emit_warning(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return multi_spread_chain()

        request = DebitSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="call",
            max_candidates_total=2,
            top_n=None,
        )

        with patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_debit_spreads(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertGreater(result.stats.pruned_candidate_rows, 0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_candidate_caps_keep_best_ranked_candidate_not_first_traversed(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return cap_regression_chain()

        request = DebitSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="call",
            max_candidates_total=1,
            top_n=None,
        )

        with patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_debit_spreads(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertEqual(result.stats.pruned_candidate_rows, 2)
        self.assertEqual(result.data["long_strike"][0], 105.0)
        self.assertEqual(result.data["short_strike"][0], 110.0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_rank_by_debit_preserves_legacy_descending_order(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return multi_spread_chain()

        request = DebitSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="call",
            rank_by="debit",
            top_n=1,
        )

        with patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_debit_spreads(request, client=object())

        self.assertAlmostEqual(result.data["debit"][0], 4.90)

    def test_warm_cache_with_supplied_connection_does_not_use_thread_pool(self):
        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            return ScreenerResult(
                data=pl.DataFrame(),
                stats=ScreenerStats(
                    ticker=request.ticker,
                    fetched_rows=0,
                    filtered_rows=0,
                    candidate_rows=0,
                    returned_rows=0,
                ),
            )

        request = DebitSpreadRequest(expiration="*", max_dte=45, strike_range=20)

        with (
            patch.object(debit_spreads, "_attempt_warm", side_effect=fake_warm),
            patch.object(debit_spreads, "ThreadPoolExecutor") as executor,
        ):
            warmup = warm_debit_spread_cache(
                ["A", "B"],
                request,
                client=object(),
                concurrency=2,
                conn=object(),
            )

        executor.assert_not_called()
        self.assertEqual(warmup.stats.succeeded, 2)

    def test_legacy_get_best_debit_spreads_returns_dataframe_and_sorts_debit_descending(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return multi_spread_chain()

        with patch.object(debit_spreads, "get_first_order_chain", side_effect=fake_chain):
            df = get_best_debit_spreads(
                "XYZ",
                "*",
                client=object(),
                right="call",
                rank_by="debit",
                top_n=1,
            )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["root"][0], "XYZ")
        self.assertAlmostEqual(df["debit"][0], 4.90)


if __name__ == "__main__":
    unittest.main()
