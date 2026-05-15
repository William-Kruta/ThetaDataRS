import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import calendar_spread
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerResult,
    ScreenerStats,
    WarmCacheResult,
)
from thetadatars.options.screeners.calendar_spread import (
    CalendarSpreadRequest,
    get_best_calendar_spreads,
    plan_calendar_spreads,
    screen_calendar_spread_watchlist,
    screen_calendar_spreads,
    warm_calendar_spread_cache,
)


NEAR = dt.date(2026, 6, 19)
FAR = dt.date(2026, 7, 17)


def chain(expiration: dt.date, bids: list[float] | None = None, root: str = "XYZ") -> pl.DataFrame:
    bids = bids or [2.0, 1.0, 0.5]
    rows = []
    for strike, bid in zip([100.0, 105.0, 110.0], bids, strict=True):
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": "call",
                "bid": bid,
                "ask": bid + (2.0 if expiration == FAR else 0.2),
                "delta": 0.4 + strike / 1000,
                "theta": -0.02,
                "vega": 0.1 + strike / 1000,
                "underlying_price": 103.0,
            }
        )
    return pl.DataFrame(rows)


class CalendarSpreadScreeningTests(unittest.TestCase):
    def test_typed_screen_returns_result_and_passes_fetch_controls_to_both_chains(self):
        calls = []

        def fake_chain(**kwargs):
            calls.append(kwargs)
            kwargs["diagnostics"]["greeks_source"] = "local"
            return chain(kwargs["expiration"], root=kwargs["ticker"])

        request = CalendarSpreadRequest(
            ticker="XYZ",
            near_expiration=NEAR,
            far_expiration=FAR,
            right="call",
            greeks_source="local",
            cache_policy="refresh",
            strike_range=10,
        )

        with patch.object(calendar_spread, "get_first_order_chain", side_effect=fake_chain):
            result = screen_calendar_spreads(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual([call["expiration"] for call in calls], [NEAR, FAR])
        self.assertEqual([call["right"] for call in calls], ["call", "call"])
        self.assertEqual([call["greeks_source"] for call in calls], ["local", "local"])
        self.assertEqual([call["cache_policy"] for call in calls], ["refresh", "refresh"])
        self.assertEqual(result.stats.fetched_rows, 6)
        self.assertEqual(result.stats.filtered_rows, 6)
        self.assertEqual(result.stats.candidate_rows, 3)
        self.assertEqual(result.stats.returned_rows, 3)

    def test_invalid_requests_reject_before_fetch(self):
        requests = [
            CalendarSpreadRequest(near_expiration=NEAR, far_expiration=FAR),
            CalendarSpreadRequest(ticker="XYZ", near_expiration="bad", far_expiration=FAR),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=FAR, far_expiration=NEAR),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, right="bad"),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, rank_by="bad"),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, max_candidates_total=0),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=NEAR, far_expiration=FAR, max_candidates_per_expiration=0),
        ]

        with patch.object(calendar_spread, "get_first_order_chain") as fetch:
            for request in requests:
                with self.subTest(request=request):
                    with self.assertRaises(InvalidRequestError):
                        screen_calendar_spreads(request, client=object())

        fetch.assert_not_called()

    def test_planner_reports_two_coverage_records_and_cache_misses(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "calendar-plan.duckdb"))
            try:
                request = CalendarSpreadRequest(
                    ticker="XYZ",
                    near_expiration=NEAR,
                    far_expiration=FAR,
                    right="put",
                    greeks_source="none",
                    strike_range=10,
                )
                plan = plan_calendar_spreads(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.strategy, "calendar_spread")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(len(plan.cache_coverage), 2)
        self.assertEqual([coverage.params["right"] for coverage in plan.cache_coverage], ["put", "put"])
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 2)
        self.assertEqual(plan.upstream_calls, 2)

    def test_planner_reports_cache_only_misses_without_upstream_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "calendar-cache-only-plan.duckdb"))
            try:
                request = CalendarSpreadRequest(
                    ticker="XYZ",
                    near_expiration=NEAR,
                    far_expiration=FAR,
                    greeks_source="none",
                    cache_policy="cache_only",
                    strike_range=10,
                )
                plan = plan_calendar_spreads(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 2)
        self.assertEqual(plan.upstream_calls, 0)

    def test_planner_marks_wildcard_expiration_cost_as_unknown(self):
        request = CalendarSpreadRequest(
            ticker="XYZ",
            near_expiration="*",
            far_expiration=FAR,
            allow_full_chain=True,
            max_candidates_total=1,
            strike_range=10,
        )

        plan = plan_calendar_spreads(request, conn=object())

        self.assertIsNone(plan.upstream_calls)
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 0)
        self.assertEqual(plan.cache_coverage, ())
        self.assertIn("wildcard_plan", {warning.code for warning in plan.warnings})

    def test_watchlist_partial_failure_and_client_factory_failure(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError("No data", ticker="BAD", endpoint="screen_calendar_spreads", retryable=False)
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "debit": [1.0]}),
                stats=ScreenerStats(ticker=request.ticker, fetched_rows=2, filtered_rows=2, candidate_rows=1, returned_rows=1),
            )

        request = CalendarSpreadRequest(near_expiration=NEAR, far_expiration=FAR, strike_range=10)
        with patch.object(calendar_spread, "screen_calendar_spreads", side_effect=fake_screen):
            batch = screen_calendar_spread_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].error.endpoint, "screen_calendar_spreads")

        batch = screen_calendar_spread_watchlist(["BAD"], request, client_factory=lambda: (_ for _ in ()).throw(RuntimeError("boom")))
        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].error.endpoint, "screen_calendar_spreads")

    def test_warm_cache_fetches_both_chains_without_building_candidates(self):
        request = CalendarSpreadRequest(near_expiration=NEAR, far_expiration=FAR, greeks_source="none", strike_range=10)

        with (
            patch.object(calendar_spread, "get_first_order_chain", side_effect=lambda **kwargs: chain(kwargs["expiration"])),
            patch.object(calendar_spread, "_build_rows", side_effect=AssertionError("warm cache built candidates")),
        ):
            warmup = warm_calendar_spread_cache(["XYZ"], request, client=object())

        self.assertIsInstance(warmup, WarmCacheResult)
        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 6)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_failures_use_warm_endpoint(self):
        warmup = warm_calendar_spread_cache(["BAD"], CalendarSpreadRequest(near_expiration=NEAR, far_expiration=NEAR), client=object())
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_calendar_spread_cache")

        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError("No data", ticker=request.ticker, endpoint="warm_calendar_spread_cache", retryable=False)

        request = CalendarSpreadRequest(near_expiration=NEAR, far_expiration=FAR, strike_range=10)
        with patch.object(calendar_spread, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_calendar_spread_cache(["BAD", "SKIP"], request, client=object(), circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1))

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_calendar_spread_cache")

    def test_candidate_caps_keep_best_ranked_candidate_and_warn(self):
        def fake_chain(**kwargs):
            return chain(kwargs["expiration"], bids=[0.2, 1.0, 2.0] if kwargs["expiration"] == NEAR else None)

        request = CalendarSpreadRequest(
            ticker="XYZ",
            near_expiration=NEAR,
            far_expiration=FAR,
            right="call",
            rank_by="near_credit_to_debit",
            max_candidates_total=1,
            top_n=None,
        )

        with patch.object(calendar_spread, "get_first_order_chain", side_effect=fake_chain):
            result = screen_calendar_spreads(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertGreater(result.stats.pruned_candidate_rows, 0)
        self.assertEqual(result.data["strike"][0], 110.0)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_wildcard_full_chain_retains_every_valid_expiration_pair(self):
        expirations = [dt.date(2026, 6, 19), dt.date(2026, 7, 17), dt.date(2026, 8, 21)]

        def full_chain(**kwargs):
            rows = []
            for index, expiration in enumerate(expirations):
                rows.append(
                    {
                        "root": "XYZ",
                        "expiration": expiration,
                        "strike": 100.0,
                        "right": "call",
                        "bid": 1.0 + index,
                        "ask": 5.0 + index,
                        "delta": 0.4,
                        "theta": -0.02,
                        "vega": 0.1,
                        "underlying_price": 103.0,
                    }
                )
            return pl.DataFrame(rows)

        request = CalendarSpreadRequest(
            ticker="XYZ",
            near_expiration="*",
            far_expiration="*",
            right="call",
            allow_full_chain=True,
            top_n=None,
        )

        with patch.object(calendar_spread, "get_first_order_chain", side_effect=full_chain):
            result = screen_calendar_spreads(request, client=object())

        pairs = {
            (row["near_expiration"], row["far_expiration"])
            for row in result.data.select(["near_expiration", "far_expiration"]).to_dicts()
        }

        self.assertEqual(
            pairs,
            {
                (expirations[0], expirations[1]),
                (expirations[0], expirations[2]),
                (expirations[1], expirations[2]),
            },
        )

    def test_legacy_wrapper_returns_dataframe(self):
        with patch.object(calendar_spread, "get_first_order_chain", side_effect=lambda **kwargs: chain(kwargs["expiration"])):
            data = get_best_calendar_spreads("XYZ", NEAR, FAR, client=object(), right="call")

        self.assertIsInstance(data, pl.DataFrame)
        self.assertFalse(data.is_empty())


if __name__ == "__main__":
    unittest.main()
