import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import diagonal_spread
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerResult,
    ScreenerStats,
    WarmCacheResult,
)
from thetadatars.options.screeners.diagonal_spread import (
    DiagonalSpreadRequest,
    get_best_diagonal_spreads,
    plan_diagonal_spreads,
    screen_diagonal_spread_watchlist,
    screen_diagonal_spreads,
    warm_diagonal_spread_cache,
)


NEAR = dt.date(2026, 6, 19)
FAR = dt.date(2026, 7, 17)


def chain(expiration: dt.date, root: str = "XYZ") -> pl.DataFrame:
    rows = []
    for strike, bid, ask, delta in [
        (100.0, 4.0, 5.0, 0.55),
        (105.0, 2.0, 3.0, 0.45),
        (110.0, 0.1, 2.2, 0.35),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": "call",
                "bid": bid if expiration == NEAR else bid + 0.2,
                "ask": ask if expiration == FAR else ask - 0.5,
                "delta": delta,
                "theta": -0.02,
                "vega": 0.1 + strike / 1000,
                "underlying_price": 103.0,
            }
        )
    return pl.DataFrame(rows)


class DiagonalSpreadScreeningTests(unittest.TestCase):
    def test_typed_screen_returns_result_and_passes_fetch_controls_to_both_chains(self):
        calls = []

        def fake_chain(**kwargs):
            calls.append(kwargs)
            kwargs["diagnostics"]["greeks_source"] = "local"
            return chain(kwargs["expiration"], root=kwargs["ticker"])

        request = DiagonalSpreadRequest(
            ticker="XYZ",
            near_expiration=NEAR,
            far_expiration=FAR,
            right="call",
            greeks_source="local",
            cache_policy="refresh",
            strike_range=10,
        )

        with patch.object(diagonal_spread, "get_first_order_chain", side_effect=fake_chain):
            result = screen_diagonal_spreads(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual([call["expiration"] for call in calls], [NEAR, FAR])
        self.assertEqual([call["right"] for call in calls], ["call", "call"])
        self.assertEqual([call["greeks_source"] for call in calls], ["local", "local"])
        self.assertEqual([call["cache_policy"] for call in calls], ["refresh", "refresh"])
        self.assertEqual(result.stats.fetched_rows, 6)
        self.assertEqual(result.stats.filtered_rows, 6)
        self.assertGreater(result.stats.candidate_rows, 0)

    def test_invalid_requests_reject_before_fetch(self):
        requests = [
            DiagonalSpreadRequest(near_expiration=NEAR, far_expiration=FAR),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration="bad", far_expiration=FAR),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=FAR, far_expiration=NEAR),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, right="bad"),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, rank_by="bad"),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, max_candidates_total=0),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, max_candidates_per_expiration=0),
        ]

        with patch.object(diagonal_spread, "get_first_order_chain") as fetch:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(InvalidRequestError):
                        screen_diagonal_spreads(request, client=object())

        fetch.assert_not_called()

    def test_planner_reports_two_coverage_records_and_cache_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "diagonal-plan.duckdb"))
            try:
                request = DiagonalSpreadRequest(
                    ticker="XYZ",
                    near_expiration=NEAR,
                    far_expiration=FAR,
                    right="put",
                    greeks_source="none",
                    strike_range=10,
                )
                plan = plan_diagonal_spreads(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.strategy, "diagonal_spread")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(len(plan.cache_coverage), 2)
        self.assertEqual([coverage.params["right"] for coverage in plan.cache_coverage], ["put", "put"])
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 2)
        self.assertEqual(plan.upstream_calls, 2)

    def test_planner_reports_cache_only_misses_without_upstream_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "diagonal-cache-only-plan.duckdb"))
            try:
                request = DiagonalSpreadRequest(
                    ticker="XYZ",
                    near_expiration=NEAR,
                    far_expiration=FAR,
                    greeks_source="none",
                    cache_policy="cache_only",
                    strike_range=10,
                )
                plan = plan_diagonal_spreads(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 2)
        self.assertEqual(plan.upstream_calls, 0)

    def test_planner_marks_wildcard_expiration_cost_as_unknown(self):
        request = DiagonalSpreadRequest(
            ticker="XYZ",
            near_expiration="*",
            far_expiration=FAR,
            allow_full_chain=True,
            max_candidates_total=1,
            strike_range=10,
        )

        plan = plan_diagonal_spreads(request, conn=object())

        self.assertIsNone(plan.upstream_calls)
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 0)
        self.assertEqual(plan.cache_coverage, ())
        self.assertIn("wildcard_plan", {warning.code for warning in plan.warnings})

    def test_watchlist_partial_failure_and_client_factory_failure(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError("No data", ticker="BAD", endpoint="screen_diagonal_spreads", retryable=False)
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "debit": [1.0]}),
                stats=ScreenerStats(ticker=request.ticker, fetched_rows=2, filtered_rows=2, candidate_rows=1, returned_rows=1),
            )

        request = DiagonalSpreadRequest(near_expiration=NEAR, far_expiration=FAR, strike_range=10)
        with patch.object(diagonal_spread, "screen_diagonal_spreads", side_effect=fake_screen):
            batch = screen_diagonal_spread_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].error.endpoint, "screen_diagonal_spreads")

        batch = screen_diagonal_spread_watchlist(["BAD"], request, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_diagonal_spreads")

    def test_warm_cache_fetches_both_chains_without_building_candidates(self):
        request = DiagonalSpreadRequest(near_expiration=NEAR, far_expiration=FAR, greeks_source="none", strike_range=10)

        with (
            patch.object(diagonal_spread, "get_first_order_chain", side_effect=lambda **kwargs: chain(kwargs["expiration"])),
            patch.object(diagonal_spread, "_build_rows", side_effect=AssertionError("warm cache built candidates")),
        ):
            warmup = warm_diagonal_spread_cache(["XYZ"], request, client=object())

        self.assertIsInstance(warmup, WarmCacheResult)
        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 6)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_failures_use_warm_endpoint(self):
        warmup = warm_diagonal_spread_cache(["BAD"], DiagonalSpreadRequest(near_expiration=NEAR, far_expiration=NEAR), client=object())
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_diagonal_spread_cache")

        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError("No data", ticker=request.ticker, endpoint="warm_diagonal_spread_cache", retryable=False)

        request = DiagonalSpreadRequest(near_expiration=NEAR, far_expiration=FAR, strike_range=10)
        with patch.object(diagonal_spread, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_diagonal_spread_cache(["BAD", "SKIP"], request, client=object(), circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1))

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_diagonal_spread_cache")

    def test_candidate_caps_keep_best_ranked_candidate_and_warn(self):
        def cap_chain(expiration: dt.date) -> pl.DataFrame:
            rows = []
            for strike, near_bid, far_ask in [
                (100.0, 0.1, 5.0),
                (105.0, 2.9, 3.0),
                (110.0, 4.0, 2.0),
            ]:
                rows.append(
                    {
                        "root": "XYZ",
                        "expiration": expiration,
                        "strike": strike,
                        "right": "call",
                        "bid": near_bid if expiration == NEAR else near_bid + 0.1,
                        "ask": far_ask if expiration == FAR else far_ask - 0.5,
                        "delta": 0.4,
                        "theta": -0.02,
                        "vega": 0.1,
                        "underlying_price": 103.0,
                    }
                )
            return pl.DataFrame(rows)

        request = DiagonalSpreadRequest(
            ticker="XYZ",
            near_expiration=NEAR,
            far_expiration=FAR,
            right="call",
            rank_by="return_if_assigned",
            max_candidates_total=1,
            top_n=None,
        )

        with patch.object(diagonal_spread, "get_first_order_chain", side_effect=lambda **kwargs: cap_chain(kwargs["expiration"])):
            result = screen_diagonal_spreads(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertGreater(result.stats.pruned_candidate_rows, 0)
        self.assertEqual(result.data["long_strike"][0], 100.0)
        self.assertEqual(result.data["short_strike"][0], 110.0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_wildcard_full_chain_excludes_invalid_expiration_pairs(self):
        expirations = [dt.date(2026, 6, 19), dt.date(2026, 7, 17), dt.date(2026, 8, 21)]

        def full_chain(**kwargs):
            rows = []
            for expiration in expirations:
                for strike, bid, ask in [(100.0, 0.5, 3.0), (105.0, 1.0, 2.0)]:
                    rows.append(
                        {
                            "root": "XYZ",
                            "expiration": expiration,
                            "strike": strike,
                            "right": "call",
                            "bid": bid,
                            "ask": ask,
                            "delta": 0.4,
                            "theta": -0.02,
                            "vega": 0.1,
                            "underlying_price": 103.0,
                        }
                    )
            return pl.DataFrame(rows)

        request = DiagonalSpreadRequest(
            ticker="XYZ",
            near_expiration="*",
            far_expiration="*",
            right="call",
            allow_full_chain=True,
            top_n=None,
        )

        with patch.object(diagonal_spread, "get_first_order_chain", side_effect=full_chain):
            result = screen_diagonal_spreads(request, client=object())

        pairs = [
            (row["near_expiration"], row["far_expiration"])
            for row in result.data.select(["near_expiration", "far_expiration"]).to_dicts()
        ]

        self.assertTrue(pairs)
        self.assertTrue(all(far > near for near, far in pairs))

    def test_legacy_wrapper_returns_dataframe(self):
        with patch.object(diagonal_spread, "get_first_order_chain", side_effect=lambda **kwargs: chain(kwargs["expiration"])):
            data = get_best_diagonal_spreads("XYZ", NEAR, FAR, client=object(), right="call")

        self.assertIsInstance(data, pl.DataFrame)
        self.assertFalse(data.is_empty())


if __name__ == "__main__":
    unittest.main()
