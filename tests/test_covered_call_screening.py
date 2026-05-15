import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from thetadatars import ThetaDataRS
from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import covered_call, strategies
from thetadatars.options.screeners.covered_call import (
    CoveredCallRequest,
    plan_covered_calls,
    screen_covered_call_watchlist,
    screen_covered_calls,
    warm_covered_call_cache,
)
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
)


def sample_call_chain(
    root: str = "XYZ",
    expiration: dt.date = dt.date(2026, 6, 19),
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "root": root,
                "expiration": expiration,
                "strike": 105.0,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 1.20,
                "ask": 1.30,
                "delta": 0.25,
                "implied_vol": 0.45,
                "underlying_price": 100.0,
            },
            {
                "root": root,
                "expiration": expiration,
                "strike": 110.0,
                "right": "call",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 0.55,
                "ask": 0.60,
                "delta": 0.12,
                "implied_vol": 0.48,
                "underlying_price": 100.0,
            },
        ]
    )


def sample_plan(request: CoveredCallRequest, **kwargs) -> ScreenerPlan:
    return ScreenerPlan(
        ticker=request.ticker or "",
        strategy="covered_call",
        expected_endpoint="option_snapshot_quote",
        upstream_calls=1,
        cache_hits=0,
        cache_misses=1,
        cost="medium",
        local_computation="low",
        warnings=(),
    )


class CoveredCallScreeningTests(unittest.TestCase):
    def test_screen_covered_calls_returns_result_and_passes_fetch_controls(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "call")
            self.assertEqual(kwargs["greeks_source"], "none")
            self.assertEqual(kwargs["cache_policy"], "prefer_cache")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_call_chain()

        request = CoveredCallRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            top_n=1,
            greeks_source="none",
        )

        with (
            patch.object(covered_call, "plan_covered_calls", return_value=sample_plan(request)),
            patch.object(covered_call, "get_first_order_chain", side_effect=fake_chain),
        ):
            result = screen_covered_calls(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.stats.ticker, "XYZ")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "none")
        self.assertEqual(result.data["strategy"][0], "covered_call")

    def test_request_rejects_unbounded_full_chain_by_default(self):
        request = CoveredCallRequest(ticker="XYZ", expiration="*")

        with self.assertRaises(InvalidRequestError) as raised:
            screen_covered_calls(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_covered_calls")
        self.assertFalse(raised.exception.retryable)

    def test_plan_covered_calls_reports_call_endpoint_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "covered-call-plan.duckdb"))
            try:
                request = CoveredCallRequest(
                    ticker="XYZ",
                    expiration=dt.date(2026, 6, 19),
                    strike_range=10,
                    greeks_source="none",
                )

                plan = plan_covered_calls(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.ticker, "XYZ")
        self.assertEqual(plan.strategy, "covered_call")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_coverage[0].params["right"], "call")
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_watchlist_returns_partial_failures(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError(
                    "No option data for BAD",
                    ticker="BAD",
                    endpoint="screen_covered_calls",
                    retryable=False,
                )
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker], "premium": [0.50]}),
                stats=ScreenerStats(
                    ticker=request.ticker,
                    fetched_rows=2,
                    filtered_rows=2,
                    candidate_rows=1,
                    returned_rows=1,
                ),
            )

        request = CoveredCallRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(covered_call, "screen_covered_calls", side_effect=fake_screen):
            batch = screen_covered_call_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertFalse(batch.failures[0].retryable)

    def test_watchlist_client_factory_failure_returns_ticker_failure(self):
        request = CoveredCallRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_covered_call_watchlist(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertEqual(batch.failures[0].error.endpoint, "screen_covered_calls")

    def test_warm_cache_fetches_chain_without_building_or_ranking_candidates(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "call")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_call_chain(root=kwargs["ticker"])

        request = CoveredCallRequest(
            expiration="*",
            max_dte=45,
            strike_range=20,
            greeks_source="none",
        )

        with (
            patch.object(covered_call, "plan_covered_calls", side_effect=sample_plan),
            patch.object(covered_call, "get_first_order_chain", side_effect=fake_chain),
            patch.object(covered_call, "_build_rows", side_effect=AssertionError("warm cache ranked candidates")),
        ):
            warmup = warm_covered_call_cache(["GOOD"], request, client=object())

        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.stats.failed, 0)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 2)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_client_factory_failure_returns_ticker_failure(self):
        request = CoveredCallRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        warmup = warm_covered_call_cache(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_covered_call_cache")

    def test_warm_cache_validation_failure_uses_warm_endpoint(self):
        warmup = warm_covered_call_cache(
            ["BAD"],
            CoveredCallRequest(expiration="*"),
            client=object(),
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_covered_call_cache")

    def test_warm_cache_circuit_breaker_skipped_failure_uses_warm_endpoint(self):
        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError(
                "No option data",
                ticker=request.ticker,
                endpoint="warm_covered_call_cache",
                retryable=False,
            )

        request = CoveredCallRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(covered_call, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_covered_call_cache(
                ["BAD", "SKIPPED"],
                request,
                client=object(),
                circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1),
            )

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].ticker, "SKIPPED")
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_covered_call_cache")

    def test_facade_exposes_screen_covered_calls(self):
        theta = ThetaDataRS()

        self.assertIn("screen_covered_calls", dir(theta))

    def test_strategy_metadata_marks_covered_call_typed_helpers(self):
        covered = strategies.get_available_strategies()["covered_call"]

        self.assertTrue(covered["typed_request"])
        self.assertTrue(covered["watchlist"])
        self.assertTrue(covered["warm_cache"])


if __name__ == "__main__":
    unittest.main()
