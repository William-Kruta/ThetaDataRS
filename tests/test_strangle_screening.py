import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import strangle
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerResult,
    ScreenerStats,
    WarmCacheResult,
)
from thetadatars.options.screeners.strangle import (
    StrangleRequest,
    get_best_strangles,
    plan_strangles,
    screen_strangle_watchlist,
    screen_strangles,
    warm_strangle_cache,
)


def sample_chain(root: str = "XYZ", expiration: dt.date = dt.date(2026, 6, 19)) -> pl.DataFrame:
    rows = []
    for strike, right, delta in [
        (90.0, "put", -0.50),
        (95.0, "put", -0.10),
        (105.0, "call", 0.10),
        (110.0, "call", 0.50),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "bid": 0.90,
                "ask": 1.00,
                "delta": delta,
                "theta": -0.02,
                "vega": 0.40,
                "underlying_price": 100.0,
            }
        )
    return pl.DataFrame(rows)


class StrangleScreeningTests(unittest.TestCase):
    def test_typed_screen_returns_result_and_passes_fetch_controls(self):
        seen = {}

        def fake_chain(**kwargs):
            seen.update(kwargs)
            kwargs["diagnostics"]["greeks_source"] = "local"
            return sample_chain(root=kwargs["ticker"])

        request = StrangleRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            greeks_source="local",
            cache_policy="refresh",
            strike_range=20,
            top_n=1,
        )

        with patch.object(strangle, "get_first_order_chain", side_effect=fake_chain):
            result = screen_strangles(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(seen["right"], "both")
        self.assertEqual(seen["greeks_source"], "local")
        self.assertEqual(seen["cache_policy"], "refresh")
        self.assertEqual(result.stats.fetched_rows, 4)
        self.assertEqual(result.stats.candidate_rows, 4)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.data["strategy"][0], "long_strangle")

    def test_full_chain_guard_rejects_before_fetch(self):
        request = StrangleRequest(ticker="XYZ", expiration="*")

        with patch.object(strangle, "get_first_order_chain") as fetch:
            with self.assertRaises(InvalidRequestError) as raised:
                screen_strangles(request, client=object())

        fetch.assert_not_called()
        self.assertEqual(raised.exception.endpoint, "screen_strangles")

    def test_invalid_side_rank_and_range_reject_before_fetch(self):
        requests = [
            StrangleRequest(ticker="XYZ", expiration=None),
            StrangleRequest(ticker="XYZ", expiration="2026-06-19", side="bad"),
            StrangleRequest(ticker="XYZ", expiration="2026-06-19", rank_by="bad"),
            StrangleRequest(ticker="XYZ", expiration="2026-06-19", min_dte=20, max_dte=10),
            StrangleRequest(ticker="XYZ", expiration="2026-06-19", max_candidates_per_expiration=0),
        ]

        with patch.object(strangle, "get_first_order_chain") as fetch:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(InvalidRequestError):
                        screen_strangles(request, client=object())

        fetch.assert_not_called()

    def test_planner_reports_endpoint_right_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "strangle-plan.duckdb"))
            try:
                request = StrangleRequest(ticker="XYZ", expiration="2026-06-19", greeks_source="none", strike_range=10)
                plan = plan_strangles(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.strategy, "strangle")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_coverage[0].params["right"], "both")
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_watchlist_partial_failure_and_client_factory_failure(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError("No data", ticker="BAD", endpoint="screen_strangles", retryable=False)
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "premium": [2.0]}),
                stats=ScreenerStats(ticker=request.ticker, fetched_rows=4, filtered_rows=4, candidate_rows=1, returned_rows=1),
            )

        request = StrangleRequest(expiration="*", max_dte=45, strike_range=20)
        with patch.object(strangle, "screen_strangles", side_effect=fake_screen):
            batch = screen_strangle_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_strangles")

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_strangle_watchlist(["BAD"], request, client_factory=failing_client_factory)
        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_strangles")

    def test_warm_cache_fetches_chain_without_building_candidates(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_chain(root=kwargs["ticker"])

        request = StrangleRequest(expiration="*", max_dte=45, strike_range=20, greeks_source="none")
        with (
            patch.object(strangle, "get_first_order_chain", side_effect=fake_chain),
            patch.object(strangle, "_build_rows", side_effect=AssertionError("warm cache built candidates")),
        ):
            warmup = warm_strangle_cache(["GOOD"], request, client=object())

        self.assertIsInstance(warmup, WarmCacheResult)
        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_failures_use_warm_endpoint(self):
        warmup = warm_strangle_cache(["BAD"], StrangleRequest(expiration="*"), client=object())
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_strangle_cache")

        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError("No data", ticker=request.ticker, endpoint="warm_strangle_cache", retryable=False)

        request = StrangleRequest(expiration="*", max_dte=45, strike_range=20)
        with patch.object(strangle, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_strangle_cache(["BAD", "SKIPPED"], request, client=object(), circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1))

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_strangle_cache")

    def test_candidate_caps_prune_and_keep_best_ranked_candidate(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain()

        request = StrangleRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            rank_by="probability_range",
            max_candidates_total=1,
            top_n=None,
        )
        with patch.object(strangle, "get_first_order_chain", side_effect=fake_chain):
            result = screen_strangles(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertEqual(result.stats.pruned_candidate_rows, 3)
        self.assertEqual(result.data["put_strike"][0], 95.0)
        self.assertEqual(result.data["call_strike"][0], 105.0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_legacy_get_best_strangles_returns_dataframe(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain(root=kwargs["ticker"])

        with patch.object(strangle, "get_first_order_chain", side_effect=fake_chain):
            df = get_best_strangles("XYZ", "*", client=object(), top_n=1)

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["root"][0], "XYZ")


if __name__ == "__main__":
    unittest.main()
