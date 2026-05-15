import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from thetadatars import ThetaDataRS
from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import protective_put, strategies
from thetadatars.options.screeners._typed import (
    BatchStats,
    CircuitBreakerPolicy,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
)
from thetadatars.options.screeners.protective_put import (
    ProtectivePutRequest,
    plan_protective_puts,
    screen_protective_put_watchlist,
    screen_protective_puts,
    warm_protective_put_cache,
)


def sample_put_chain(
    root: str = "XYZ",
    expiration: dt.date = dt.date(2026, 6, 19),
) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "root": root,
                "expiration": expiration,
                "strike": 95.0,
                "right": "put",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 1.10,
                "ask": 1.20,
                "delta": -0.25,
                "implied_vol": 0.45,
                "underlying_price": 100.0,
            },
            {
                "root": root,
                "expiration": expiration,
                "strike": 90.0,
                "right": "put",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": 0.50,
                "ask": 0.60,
                "delta": -0.12,
                "implied_vol": 0.48,
                "underlying_price": 100.0,
            },
        ]
    )


def sample_plan(request: ProtectivePutRequest, **kwargs) -> ScreenerPlan:
    return ScreenerPlan(
        ticker=request.ticker or "",
        strategy="protective_put",
        expected_endpoint="option_snapshot_quote",
        upstream_calls=1,
        cache_hits=0,
        cache_misses=1,
        cost="medium",
        local_computation="low",
        warnings=(),
    )


class ProtectivePutScreeningTests(unittest.TestCase):
    def test_screen_protective_puts_returns_result_and_passes_fetch_controls(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "put")
            self.assertEqual(kwargs["greeks_source"], "none")
            self.assertEqual(kwargs["cache_policy"], "prefer_cache")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_put_chain()

        request = ProtectivePutRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            top_n=1,
            greeks_source="none",
        )

        with (
            patch.object(protective_put, "plan_protective_puts", return_value=sample_plan(request)),
            patch.object(protective_put, "get_first_order_chain", side_effect=fake_chain),
        ):
            result = screen_protective_puts(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.stats.ticker, "XYZ")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "none")
        self.assertEqual(result.data["strategy"][0], "protective_put")

    def test_screen_protective_puts_ranks_hedge_cost_percent_lowest_first(self):
        request = ProtectivePutRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            top_n=1,
            rank_by="hedge_cost_percent",
            greeks_source="none",
        )

        with (
            patch.object(protective_put, "plan_protective_puts", return_value=sample_plan(request)),
            patch.object(protective_put, "get_first_order_chain", return_value=sample_put_chain()),
        ):
            result = screen_protective_puts(request, client=object())

        self.assertEqual(result.data["strike"][0], 90.0)
        self.assertEqual(result.data["hedge_cost_percent"][0], 0.006)

    def test_request_rejects_unbounded_full_chain_by_default(self):
        request = ProtectivePutRequest(ticker="XYZ", expiration="*")

        with self.assertRaises(InvalidRequestError) as raised:
            screen_protective_puts(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_protective_puts")
        self.assertFalse(raised.exception.retryable)

    def test_plan_protective_puts_reports_put_endpoint_and_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "protective-put-plan.duckdb"))
            try:
                request = ProtectivePutRequest(
                    ticker="XYZ",
                    expiration=dt.date(2026, 6, 19),
                    strike_range=10,
                    greeks_source="none",
                )

                plan = plan_protective_puts(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.ticker, "XYZ")
        self.assertEqual(plan.strategy, "protective_put")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_coverage[0].params["right"], "put")
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_watchlist_returns_partial_failures(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError(
                    "No option data for BAD",
                    ticker="BAD",
                    endpoint="screen_protective_puts",
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

        request = ProtectivePutRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(protective_put, "screen_protective_puts", side_effect=fake_screen):
            batch = screen_protective_put_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertFalse(batch.failures[0].retryable)

    def test_warm_cache_fetches_chain_without_building_or_ranking_candidates(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "put")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_put_chain(root=kwargs["ticker"])

        request = ProtectivePutRequest(
            expiration="*",
            max_dte=45,
            strike_range=20,
            greeks_source="none",
        )

        with (
            patch.object(protective_put, "plan_protective_puts", side_effect=sample_plan),
            patch.object(protective_put, "get_first_order_chain", side_effect=fake_chain),
            patch.object(protective_put, "_build_rows", side_effect=AssertionError("warm cache ranked candidates")),
        ):
            warmup = warm_protective_put_cache(["GOOD"], request, client=object())

        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.stats.failed, 0)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 2)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)

    def test_watchlist_client_factory_failure_returns_ticker_failure(self):
        request = ProtectivePutRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        batch = screen_protective_put_watchlist(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(batch.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertEqual(batch.failures[0].error.endpoint, "screen_protective_puts")

    def test_warm_cache_client_factory_failure_returns_ticker_failure(self):
        request = ProtectivePutRequest(expiration="*", max_dte=45, strike_range=20)

        def failing_client_factory():
            raise RuntimeError("client setup failed")

        warmup = warm_protective_put_cache(
            ["BAD"],
            request,
            client_factory=failing_client_factory,
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_protective_put_cache")

    def test_warm_cache_validation_failure_uses_warm_endpoint(self):
        warmup = warm_protective_put_cache(
            ["BAD"],
            ProtectivePutRequest(expiration="*"),
            client=object(),
        )

        self.assertEqual(warmup.stats, BatchStats(total=1, succeeded=0, failed=1))
        self.assertEqual(warmup.failures[0].ticker, "BAD")
        self.assertEqual(warmup.failures[0].error.endpoint, "warm_protective_put_cache")

    def test_warm_cache_circuit_breaker_skipped_failure_uses_warm_endpoint(self):
        def fake_warm(request, client=None, retry_policy=None, timeout_policy=None, conn=None):
            raise NoDataError(
                "No option data",
                ticker=request.ticker,
                endpoint="warm_protective_put_cache",
                retryable=False,
            )

        request = ProtectivePutRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(protective_put, "_attempt_warm", side_effect=fake_warm):
            warmup = warm_protective_put_cache(
                ["BAD", "SKIPPED"],
                request,
                client=object(),
                circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1),
            )

        self.assertEqual(warmup.stats.failed, 2)
        self.assertEqual(warmup.failures[1].ticker, "SKIPPED")
        self.assertEqual(warmup.failures[1].attempts, 0)
        self.assertEqual(warmup.failures[1].error.endpoint, "warm_protective_put_cache")

    def test_facade_exposes_screen_protective_puts(self):
        theta = ThetaDataRS()

        self.assertIn("screen_protective_puts", dir(theta))

    def test_strategy_metadata_marks_protective_put_typed_helpers(self):
        protective = strategies.get_available_strategies()["protective_put"]

        self.assertTrue(protective["typed_request"])
        self.assertTrue(protective["watchlist"])
        self.assertTrue(protective["warm_cache"])


if __name__ == "__main__":
    unittest.main()
