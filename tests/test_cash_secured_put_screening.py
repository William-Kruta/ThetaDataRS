import datetime as dt
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import polars as pl

from thetadatars.data.db import get_connection
from thetadatars.errors import InvalidRequestError, NoDataError
from thetadatars.options.screeners import cash_secured_put
from thetadatars.options.screeners.cash_secured_put import (
    CashSecuredPutRequest,
    plan_cash_secured_puts,
    screen_cash_secured_puts,
    screen_cash_secured_put_watchlist,
    warm_cash_secured_put_cache,
)
from thetadatars.options.screeners.credit_spreads import (
    BatchStats,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
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
                "bid": 1.20,
                "ask": 1.30,
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
                "bid": 0.55,
                "ask": 0.60,
                "delta": -0.12,
                "implied_vol": 0.48,
                "underlying_price": 100.0,
            },
        ]
    )


def sample_plan(request: CashSecuredPutRequest, **kwargs) -> ScreenerPlan:
    return ScreenerPlan(
        ticker=request.ticker or "",
        strategy="cash_secured_put",
        expected_endpoint="option_snapshot_quote",
        upstream_calls=1,
        cache_hits=0,
        cache_misses=1,
        cost="medium",
        local_computation="low",
        warnings=(),
    )


class CashSecuredPutScreeningTests(unittest.TestCase):
    def test_screen_cash_secured_puts_returns_data_stats_and_warnings(self):
        def fake_chain(**kwargs):
            self.assertEqual(kwargs["right"], "put")
            self.assertEqual(kwargs["greeks_source"], "none")
            self.assertEqual(kwargs["cache_policy"], "prefer_cache")
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_put_chain()

        request = CashSecuredPutRequest(
            ticker="XYZ",
            expiration="*",
            max_dte=45,
            strike_range=20,
            top_n=1,
            greeks_source="none",
        )

        with (
            patch.object(cash_secured_put, "plan_cash_secured_puts", return_value=sample_plan(request)),
            patch.object(cash_secured_put, "get_first_order_chain", side_effect=fake_chain),
        ):
            result = screen_cash_secured_puts(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.stats.ticker, "XYZ")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "none")
        self.assertEqual(result.data["strategy"][0], "cash_secured_put")

    def test_request_api_rejects_unbounded_full_chain_by_default(self):
        request = CashSecuredPutRequest(ticker="XYZ", expiration="*")

        with self.assertRaises(InvalidRequestError) as raised:
            screen_cash_secured_puts(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_cash_secured_puts")
        self.assertFalse(raised.exception.retryable)

    def test_plan_cash_secured_puts_reports_cache_miss(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "cash-secured-put-plan.duckdb"))
            try:
                request = CashSecuredPutRequest(
                    ticker="XYZ",
                    expiration=dt.date(2026, 6, 19),
                    strike_range=10,
                    greeks_source="none",
                )

                plan = plan_cash_secured_puts(request, conn=conn)
            finally:
                conn.close()

        self.assertEqual(plan.ticker, "XYZ")
        self.assertEqual(plan.strategy, "cash_secured_put")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_watchlist_returns_partial_failures(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError(
                    "No option data for BAD",
                    ticker="BAD",
                    endpoint="screen_cash_secured_puts",
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

        request = CashSecuredPutRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(cash_secured_put, "screen_cash_secured_puts", side_effect=fake_screen):
            batch = screen_cash_secured_put_watchlist(["GOOD", "BAD"], request, client=object())

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertFalse(batch.failures[0].retryable)

    def test_warm_cache_fetches_chain_without_building_candidates(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "none"
            return sample_put_chain(root=kwargs["ticker"])

        request = CashSecuredPutRequest(
            expiration="*",
            max_dte=45,
            strike_range=20,
            greeks_source="none",
        )

        with (
            patch.object(cash_secured_put, "plan_cash_secured_puts", side_effect=sample_plan),
            patch.object(cash_secured_put, "get_first_order_chain", side_effect=fake_chain),
        ):
            warmup = warm_cash_secured_put_cache(["GOOD"], request, client=object())

        self.assertEqual(warmup.stats.succeeded, 1)
        self.assertEqual(warmup.stats.failed, 0)
        self.assertEqual(warmup.successes[0].stats.fetched_rows, 2)
        self.assertEqual(warmup.successes[0].stats.candidate_rows, 0)


if __name__ == "__main__":
    unittest.main()
