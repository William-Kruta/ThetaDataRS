import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import iron_butterfly
from thetadatars.options.screeners import _common
from thetadatars.options.screeners._typed import (
    BatchStats,
    ScreenerResult,
    ScreenerStats,
    WarmCacheResult,
)
from thetadatars.options.screeners.iron_butterfly import (
    IronButterflyRequest,
    get_best_iron_butterflies,
    plan_iron_butterflies,
    screen_iron_butterflies,
    screen_iron_butterfly_watchlist,
    warm_iron_butterfly_cache,
)


def sample_chain(root: str = "XYZ", expiration: dt.date = dt.date(2026, 6, 19)) -> pl.DataFrame:
    rows = []
    for strike, right, bid, ask, delta in [
        (90.0, "put", 0.10, 0.20, -0.50),
        (95.0, "put", 0.25, 0.35, -0.25),
        (100.0, "put", 2.20, 2.30, -0.10),
        (100.0, "call", 2.20, 2.30, 0.10),
        (105.0, "call", 0.25, 0.35, 0.25),
        (110.0, "call", 0.10, 0.20, 0.50),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": right,
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "theta": -0.02,
                "vega": 0.40,
                "underlying_price": 100.0,
            }
        )
    return pl.DataFrame(rows)


class IronButterflyScreeningTests(unittest.TestCase):
    def test_typed_screen_returns_result_and_passes_fetch_controls(self):
        seen = {}

        def fake_chain(**kwargs):
            seen.update(kwargs)
            kwargs["diagnostics"]["greeks_source"] = "local"
            return sample_chain(root=kwargs["ticker"])

        request = IronButterflyRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            greeks_source="local",
            cache_policy="refresh",
            strike_range=20,
            top_n=1,
        )

        with patch.object(iron_butterfly, "get_first_order_chain", side_effect=fake_chain):
            result = screen_iron_butterflies(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(seen["right"], "both")
        self.assertEqual(seen["greeks_source"], "local")
        self.assertEqual(seen["cache_policy"], "refresh")
        self.assertEqual(result.stats.fetched_rows, 6)
        self.assertGreater(result.stats.candidate_rows, 1)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.data["strategy"][0], "iron_butterfly")

    def test_invalid_requests_reject_before_fetch(self):
        requests = [
            IronButterflyRequest(ticker="XYZ", expiration="*"),
            IronButterflyRequest(ticker="XYZ", expiration=None),
            IronButterflyRequest(ticker="XYZ", expiration="2026-06-19", rank_by="bad"),
            IronButterflyRequest(ticker="XYZ", expiration="2026-06-19", min_dte=20, max_dte=10),
            IronButterflyRequest(ticker="XYZ", expiration="2026-06-19", min_width=10, max_width=5),
            IronButterflyRequest(ticker="XYZ", expiration="2026-06-19", min_short_delta=0.5, max_short_delta=0.1),
            IronButterflyRequest(ticker="XYZ", expiration="2026-06-19", max_candidates_per_expiration=0),
        ]

        with patch.object(iron_butterfly, "get_first_order_chain") as fetch:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(InvalidRequestError):
                        screen_iron_butterflies(request, client=object())

        fetch.assert_not_called()

    def test_planner_cache_only_miss_counts_no_upstream_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "iron-butterfly-plan.duckdb"))
            try:
                request = IronButterflyRequest(
                    ticker="XYZ",
                    expiration="2026-06-19",
                    greeks_source="none",
                    strike_range=10,
                    cache_policy="cache_only",
                )
                plan = plan_iron_butterflies(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.strategy, "iron_butterfly")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_coverage[0].params["right"], "both")
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 0)

    def test_wildcard_planner_does_not_report_precise_cache_coverage(self):
        request = IronButterflyRequest(
            ticker="XYZ",
            expiration="*",
            allow_full_chain=True,
            max_dte=45,
            strike_range=20,
        )
        plan = plan_iron_butterflies(request, conn=object())

        self.assertIsNone(plan.upstream_calls)
        self.assertEqual(plan.cache_coverage, ())
        self.assertIn("wildcard_plan", {warning.code for warning in plan.warnings})

    def test_watchlist_partial_failure_and_client_factory_failure(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError("No data", ticker="BAD", endpoint="screen_iron_butterflies", retryable=False)
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "credit": [4.0]}),
                stats=ScreenerStats(ticker=request.ticker, fetched_rows=6, filtered_rows=6, candidate_rows=1, returned_rows=1),
            )

        request = IronButterflyRequest(expiration="*", max_dte=45, strike_range=20)
        with patch.object(iron_butterfly, "screen_iron_butterflies", side_effect=fake_screen):
            batch = screen_iron_butterfly_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_iron_butterflies")

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_iron_butterfly_watchlist(["BAD"], request, client_factory=failing_client_factory)
        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_iron_butterflies")

    def test_warm_cache_fetches_chain_without_building_candidates(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_chain(root=kwargs["ticker"])

        request = IronButterflyRequest(expiration="*", max_dte=45, strike_range=20, greeks_source="none")
        with (
            patch.object(iron_butterfly, "get_first_order_chain", side_effect=fake_chain),
            patch.object(iron_butterfly, "_build_rows", side_effect=AssertionError("warm cache built candidates")),
        ):
            warmup = warm_iron_butterfly_cache(["GOOD"], request, client=object())

        self.assertIsInstance(warmup, WarmCacheResult)
        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_candidate_caps_prune_after_filtering_and_keep_best_ranked_candidate(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain()

        request = IronButterflyRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            rank_by="return_on_risk",
            min_short_delta=0.05,
            max_short_delta=0.20,
            max_candidates_total=1,
            top_n=None,
        )
        with patch.object(iron_butterfly, "get_first_order_chain", side_effect=fake_chain):
            result = screen_iron_butterflies(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertGreater(result.stats.pruned_candidate_rows, 0)
        self.assertEqual(result.data["body_strike"][0], 100.0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_body_distance_rank_keeps_nearest_body_when_capping(self):
        rows = []
        for strike, right, bid, ask in [
            (90.0, "put", 0.10, 0.20),
            (95.0, "put", 0.60, 0.70),
            (100.0, "put", 2.20, 2.30),
            (105.0, "put", 0.60, 0.70),
            (100.0, "call", 2.20, 2.30),
            (105.0, "call", 0.60, 0.70),
            (110.0, "call", 0.10, 0.20),
            (115.0, "call", 0.05, 0.15),
        ]:
            rows.append(
                {
                    "root": "XYZ",
                    "expiration": dt.date(2026, 6, 19),
                    "strike": strike,
                    "right": right,
                    "bid": bid,
                    "ask": ask,
                    "delta": -0.10 if right == "put" else 0.10,
                    "theta": -0.02,
                    "vega": 0.40,
                    "underlying_price": 100.0,
                }
            )

        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return pl.DataFrame(rows)

        request = IronButterflyRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            rank_by="body_distance_percent",
            max_candidates_total=1,
            top_n=None,
        )
        with patch.object(iron_butterfly, "get_first_order_chain", side_effect=fake_chain):
            result = screen_iron_butterflies(request, client=object())

        self.assertEqual(result.data["body_strike"][0], 100.0)
        self.assertEqual(result.data["body_distance_percent"][0], 0.0)

    def test_wildcard_warm_cache_propagates_cache_policy_to_expiration_discovery(self):
        seen = {}

        def fake_expirations(ticker, *, client, stale_threshold, cache_policy, conn=None):
            seen["ticker"] = ticker
            seen["stale_threshold"] = stale_threshold
            seen["cache_policy"] = cache_policy
            return pl.DataFrame(schema={"root": pl.Utf8, "expiration": pl.Date})

        request = IronButterflyRequest(
            expiration="*",
            allow_full_chain=True,
            max_dte=45,
            strike_range=20,
            greeks_source="none",
            cache_policy="cache_only",
            stale_threshold=dt.timedelta(minutes=5),
        )
        with patch.object(_common, "get_options_expiration_list", side_effect=fake_expirations):
            warmup = warm_iron_butterfly_cache(["XYZ"], request, client=object())

        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(seen["ticker"], "XYZ")
        self.assertEqual(seen["cache_policy"], "cache_only")
        self.assertEqual(seen["stale_threshold"], dt.timedelta(minutes=5))

    def test_legacy_get_best_iron_butterflies_returns_dataframe(self):
        def fake_chain(**kwargs):
            return sample_chain(root=kwargs["ticker"])

        with patch.object(iron_butterfly, "get_first_order_chain", side_effect=fake_chain):
            df = get_best_iron_butterflies("XYZ", "*", client=object(), top_n=1)

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["root"][0], "XYZ")


if __name__ == "__main__":
    unittest.main()
