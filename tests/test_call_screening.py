import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from thetadatars import ThetaDataRS
from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import call, strategies
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
)
from thetadatars.options.screeners.call import (
    CallRequest,
    plan_calls,
    screen_call_watchlist,
    screen_calls,
    warm_call_cache,
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
                "open_interest": 100,
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
                "open_interest": 80,
            },
        ]
    )


def sample_plan(request: CallRequest, **kwargs) -> ScreenerPlan:
    return ScreenerPlan(
        ticker=request.ticker or "",
        strategy="call",
        expected_endpoint="option_snapshot_quote",
        upstream_calls=1,
        cache_hits=0,
        cache_misses=1,
        cost="medium",
        local_computation="low",
        warnings=(),
    )


class CallScreeningTests(unittest.TestCase):
    def test_screen_calls_returns_result_and_passes_fetch_controls(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "call")
            self.assertEqual(kwargs["greeks_source"], "none")
            self.assertEqual(kwargs["cache_policy"], "prefer_cache")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_call_chain()

        request = CallRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            top_n=1,
            greeks_source="none",
        )

        with (
            patch.object(call, "plan_calls", return_value=sample_plan(request)),
            patch.object(call, "get_first_order_chain", side_effect=fake_chain),
        ):
            result = screen_calls(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.stats.ticker, "XYZ")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "none")
        self.assertEqual(result.data["strategy"][0], "long_call")

    def test_request_rejects_unbounded_full_chain_by_default(self):
        request = CallRequest(ticker="XYZ", expiration="*")

        with self.assertRaises(InvalidRequestError) as raised:
            screen_calls(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_calls")
        self.assertFalse(raised.exception.retryable)

    def test_request_rejects_min_delta_greater_than_max_delta_before_fetch(self):
        request = CallRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            min_delta=0.60,
            max_delta=0.30,
        )

        with patch.object(call, "get_first_order_chain", side_effect=AssertionError("fetch called")):
            with self.assertRaises(InvalidRequestError) as raised:
                screen_calls(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_calls")
        self.assertFalse(raised.exception.retryable)

    def test_screen_calls_breaks_rank_ties_with_lower_premium(self):
        chain = pl.DataFrame(
            [
                {
                    "root": "XYZ",
                    "expiration": dt.date(2026, 6, 19),
                    "strike": 110.0,
                    "right": "call",
                    "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                    "bid": 1.90,
                    "ask": 2.00,
                    "delta": 0.20,
                    "vega": 0.20,
                    "implied_vol": 0.48,
                    "underlying_price": 100.0,
                },
                {
                    "root": "XYZ",
                    "expiration": dt.date(2026, 6, 19),
                    "strike": 105.0,
                    "right": "call",
                    "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                    "bid": 0.90,
                    "ask": 1.00,
                    "delta": 0.20,
                    "vega": 0.10,
                    "implied_vol": 0.45,
                    "underlying_price": 100.0,
                },
            ]
        )
        request = CallRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            rank_by="vega_per_dollar",
            greeks_source="none",
        )

        with (
            patch.object(call, "plan_calls", return_value=sample_plan(request)),
            patch.object(call, "get_first_order_chain", return_value=chain),
        ):
            result = screen_calls(request, client=object())

        self.assertEqual(result.data["premium"].to_list(), [1.0, 2.0])

    def test_plan_calls_reports_call_endpoint_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "call-plan.duckdb"))
            try:
                request = CallRequest(
                    ticker="XYZ",
                    expiration=dt.date(2026, 6, 19),
                    strike_range=10,
                    greeks_source="none",
                )

                plan = plan_calls(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.ticker, "XYZ")
        self.assertEqual(plan.strategy, "call")
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
                    endpoint="screen_calls",
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

        request = CallRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(call, "screen_calls", side_effect=fake_screen):
            batch = screen_call_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertFalse(batch.failures[0].retryable)

    def test_watchlist_client_factory_failure_returns_ticker_failure(self):
        request = CallRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_call_watchlist(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertEqual(batch.failures[0].error.endpoint, "screen_calls")

    def test_warm_cache_fetches_chain_without_building_or_ranking_candidates(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "call")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_call_chain(root=kwargs["ticker"])

        request = CallRequest(
            expiration="*",
            max_dte=45,
            strike_range=20,
            greeks_source="none",
        )

        with (
            patch.object(call, "plan_calls", side_effect=sample_plan),
            patch.object(call, "get_first_order_chain", side_effect=fake_chain),
            patch.object(call, "build_single_leg_options", side_effect=AssertionError("warm cache ranked candidates")),
        ):
            warmup = warm_call_cache(["GOOD"], request, client=object())

        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.stats.failed, 0)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 2)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_warm_cache_client_factory_failure_returns_ticker_failure(self):
        request = CallRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        warmup = warm_call_cache(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_call_cache")

    def test_warm_cache_validation_failure_uses_warm_endpoint(self):
        warmup = warm_call_cache(
            ["BAD"],
            CallRequest(expiration="*"),
            client=object(),
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_call_cache")

    def test_warm_cache_circuit_breaker_skipped_failure_uses_warm_endpoint(self):
        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError(
                "No option data",
                ticker=request.ticker,
                endpoint="warm_call_cache",
                retryable=False,
            )

        request = CallRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(call, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_call_cache(
                ["BAD", "SKIPPED"],
                request,
                client=object(),
                circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1),
            )

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].ticker, "SKIPPED")
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_call_cache")

    def test_facade_exposes_screen_calls(self):
        theta = ThetaDataRS()

        self.assertIn("screen_calls", dir(theta))

    def test_strategy_metadata_marks_call_typed_helpers(self):
        metadata = strategies.get_available_strategies()["call"]

        self.assertTrue(metadata["typed_request"])
        self.assertTrue(metadata["watchlist"])
        self.assertTrue(metadata["warm_cache"])


if __name__ == "__main__":
    unittest.main()
