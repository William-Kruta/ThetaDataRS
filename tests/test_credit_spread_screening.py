import datetime as dt
import time
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars import ThetaDataRS
from thetadatars.errors import InvalidRequestError, NoDataError, classify_thetadata_error
from thetadatars.options.screeners import _common as screener_common
from thetadatars.options.screeners import credit_spreads
from thetadatars.options.screeners.credit_spreads import (
    BatchStats,
    CircuitBreakerPolicy,
    CreditSpreadRequest,
    RateLimitPolicy,
    ScreenerPlan,
    ScreenerResult,
    ScreenerStats,
    TimeoutPolicy,
    WarmCacheResult,
    screen_credit_spread_watchlist,
    screen_credit_spreads,
    warm_credit_spread_cache,
)


def sample_chain(
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
                "bid": 1.00,
                "ask": 1.10,
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
                "bid": 0.35,
                "ask": 0.40,
                "delta": -0.12,
                "implied_vol": 0.48,
                "underlying_price": 100.0,
            },
        ]
    )


def multi_spread_chain(
    root: str = "XYZ",
    expiration: dt.date = dt.date(2026, 6, 19),
) -> pl.DataFrame:
    rows = []
    for strike, bid, ask, delta in [
        (99.0, 1.50, 1.60, -0.30),
        (98.0, 1.10, 1.20, -0.25),
        (97.0, 0.80, 0.90, -0.20),
        (96.0, 0.55, 0.60, -0.16),
        (95.0, 0.35, 0.40, -0.12),
    ]:
        rows.append(
            {
                "root": root,
                "expiration": expiration,
                "strike": strike,
                "right": "put",
                "timestamp": dt.datetime(2026, 5, 12, 13, 0),
                "bid": bid,
                "ask": ask,
                "delta": delta,
                "implied_vol": 0.45,
                "underlying_price": 100.0,
            }
        )
    return pl.DataFrame(rows)


class CreditSpreadScreeningTests(unittest.TestCase):
    def test_screen_credit_spreads_returns_data_stats_and_warnings(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain()

        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration="*",
            right="put",
            max_dte=45,
            strike_range=20,
            top_n=5,
        )

        with patch.object(credit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_credit_spreads(request, client=object())

        self.assertIsInstance(result, ScreenerResult)
        self.assertEqual(result.stats.ticker, "XYZ")
        self.assertEqual(result.stats.fetched_rows, 2)
        self.assertEqual(result.stats.filtered_rows, 2)
        self.assertEqual(result.stats.candidate_rows, 1)
        self.assertEqual(result.stats.returned_rows, 1)
        self.assertEqual(result.stats.greeks_source, "thetadata")
        self.assertEqual(result.warnings, ())
        self.assertEqual(result.data["spread_type"][0], "bull_put_credit")

    def test_screen_credit_spreads_includes_cache_plan_stats(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain()

        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="put",
            strike_range=20,
            top_n=5,
        )
        plan = ScreenerPlan(
            ticker="XYZ",
            strategy="credit_spread",
            expected_endpoint="option_snapshot_greeks_first_order",
            upstream_calls=0,
            cache_hits=1,
            cache_misses=0,
            cost="low",
            local_computation="low",
            warnings=(),
        )

        with (
            patch.object(credit_spreads, "plan_credit_spreads", return_value=plan),
            patch.object(credit_spreads, "get_first_order_chain", side_effect=fake_chain),
        ):
            result = screen_credit_spreads(request, client=object())

        self.assertEqual(result.stats.cache_hits, 1)
        self.assertEqual(result.stats.cache_misses, 0)
        self.assertEqual(result.stats.upstream_calls, 0)
        self.assertEqual(result.stats.cache_policy, "prefer_cache")

    def test_candidate_caps_prune_results_and_emit_warning(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return multi_spread_chain()

        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            right="put",
            max_candidates_total=2,
            top_n=None,
        )

        with patch.object(credit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = screen_credit_spreads(request, client=object())

        self.assertEqual(result.stats.candidate_rows, 2)
        self.assertEqual(result.stats.pruned_candidate_rows, 8)
        self.assertEqual(len(result.data), 2)
        self.assertIn("candidate_limit", {warning.code for warning in result.warnings})

    def test_warm_credit_spread_cache_fetches_without_building_candidates(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain(root=kwargs["ticker"])

        request = CreditSpreadRequest(
            expiration="*",
            max_dte=45,
            strike_range=20,
            greeks_source="none",
        )

        with patch.object(credit_spreads, "get_first_order_chain", side_effect=fake_chain):
            result = warm_credit_spread_cache(["GOOD"], request, client=object())

        self.assertIsInstance(result, WarmCacheResult)
        self.assertEqual(result.stats.succeeded, 1)
        self.assertEqual(result.stats.failed, 0)
        self.assertEqual(result.successes[0].stats.fetched_rows, 2)
        self.assertEqual(result.successes[0].stats.candidate_rows, 0)

    def test_watchlist_timeout_marks_ticker_failure(self):
        def fake_screen(request, client=None, conn=None):
            time.sleep(0.03)
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

        request = CreditSpreadRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(credit_spreads, "screen_credit_spreads", side_effect=fake_screen):
            result = screen_credit_spread_watchlist(
                ["SLOW"],
                request,
                client=object(),
                timeout_policy=TimeoutPolicy(per_ticker_seconds=0.01),
            )

        self.assertEqual(result.stats.failed, 1)
        self.assertTrue(result.failures[0].retryable)
        self.assertEqual(result.failures[0].error.endpoint, "screen_credit_spreads")

    def test_watchlist_rate_limit_spaces_requests(self):
        starts = []

        def fake_screen(request, client=None, conn=None):
            starts.append(time.perf_counter())
            return ScreenerResult(
                data=pl.DataFrame({"root": [request.ticker]}),
                stats=ScreenerStats(
                    ticker=request.ticker,
                    fetched_rows=0,
                    filtered_rows=0,
                    candidate_rows=0,
                    returned_rows=1,
                ),
            )

        request = CreditSpreadRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(credit_spreads, "screen_credit_spreads", side_effect=fake_screen):
            result = screen_credit_spread_watchlist(
                ["A", "B"],
                request,
                client=object(),
                rate_limit_policy=RateLimitPolicy(min_interval_seconds=0.02),
            )

        self.assertEqual(result.stats.succeeded, 2)
        self.assertGreaterEqual(starts[1] - starts[0], 0.018)

    def test_watchlist_circuit_breaker_skips_remaining_tickers(self):
        calls = []

        def fake_screen(request, client=None, conn=None):
            calls.append(request.ticker)
            raise NoDataError(
                "No data",
                ticker=request.ticker,
                endpoint="screen_credit_spreads",
                retryable=False,
            )

        request = CreditSpreadRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(credit_spreads, "screen_credit_spreads", side_effect=fake_screen):
            result = screen_credit_spread_watchlist(
                ["BAD", "SKIPPED"],
                request,
                client=object(),
                circuit_breaker_policy=CircuitBreakerPolicy(max_failures=1),
            )

        self.assertEqual(calls, ["BAD"])
        self.assertEqual(result.stats.failed, 2)
        self.assertEqual(result.failures[1].ticker, "SKIPPED")
        self.assertEqual(result.failures[1].attempts, 0)

    def test_request_api_rejects_unbounded_full_chain_by_default(self):
        request = CreditSpreadRequest(ticker="XYZ", expiration="*")

        with self.assertRaises(InvalidRequestError) as raised:
            screen_credit_spreads(request, client=object())

        self.assertEqual(raised.exception.ticker, "XYZ")
        self.assertEqual(raised.exception.endpoint, "screen_credit_spreads")
        self.assertFalse(raised.exception.retryable)

    def test_legacy_dataframe_function_allows_existing_full_chain_usage(self):
        def fake_chain(**kwargs):
            kwargs["diagnostics"]["greeks_source"] = "thetadata"
            return sample_chain()

        with patch.object(credit_spreads, "get_first_order_chain", side_effect=fake_chain):
            df = credit_spreads.get_best_credit_spreads(
                "XYZ",
                "*",
                client=object(),
                right="put",
                top_n=1,
            )

        self.assertIsInstance(df, pl.DataFrame)
        self.assertEqual(len(df), 1)
        self.assertEqual(df["root"][0], "XYZ")

    def test_watchlist_returns_partial_failures(self):
        def fake_screen(request, client=None, conn=None):
            if request.ticker == "BAD":
                raise NoDataError(
                    "No option data for BAD",
                    ticker="BAD",
                    endpoint="screen_credit_spreads",
                    retryable=False,
                )
            data = pl.DataFrame({"root": [request.ticker], "credit": [0.50]})
            stats = ScreenerStats(
                ticker=request.ticker,
                fetched_rows=2,
                filtered_rows=2,
                candidate_rows=1,
                returned_rows=1,
                duration_seconds=0.01,
                greeks_source="thetadata",
            )
            return ScreenerResult(data=data, stats=stats)

        request = CreditSpreadRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(credit_spreads, "screen_credit_spreads", side_effect=fake_screen):
            batch = screen_credit_spread_watchlist(
                ["GOOD", "BAD"],
                request,
                client=object(),
            )

        self.assertEqual(batch.stats, BatchStats(total=2, succeeded=1, failed=1))
        self.assertEqual(len(batch.successes), 1)
        self.assertEqual(len(batch.failures), 1)
        self.assertEqual(batch.failures[0].ticker, "BAD")
        self.assertFalse(batch.failures[0].retryable)
        self.assertEqual(batch.data["root"].to_list(), ["GOOD"])

    def test_error_classifier_preserves_context_for_no_data(self):
        error = classify_thetadata_error(
            Exception("No data found for: option_snapshot_quote(MDA,*,*,both,None,None,None)"),
            ticker="MDA",
            endpoint="option_snapshot_quote",
            params={"expiration": "*"},
        )

        self.assertIsInstance(error, NoDataError)
        self.assertEqual(error.ticker, "MDA")
        self.assertEqual(error.endpoint, "option_snapshot_quote")
        self.assertEqual(error.params, {"expiration": "*"})
        self.assertFalse(error.retryable)

    def test_facade_exposes_screening_helpers(self):
        theta = ThetaDataRS()

        self.assertIn("screen_credit_spreads", dir(theta))
        self.assertIn("screen_credit_spread_watchlist", dir(theta))

    def test_wildcard_expiration_expands_before_snapshot_quote_call(self):
        today = dt.date.today()
        near_expiration = today + dt.timedelta(days=7)
        far_expiration = today + dt.timedelta(days=30)
        expirations = pl.DataFrame(
            {
                "root": ["XYZ", "XYZ"],
                "expiration": [near_expiration, far_expiration],
            }
        )
        seen_expirations = []

        def fake_quote(**kwargs):
            expiration = kwargs["expiration"]
            seen_expirations.append(expiration)
            self.assertIsInstance(expiration, dt.date)
            self.assertNotEqual(expiration, "*")
            return sample_chain(expiration=expiration)

        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration="*",
            right="put",
            max_dte=14,
            strike_range=5,
            top_n=1,
            greeks_source="none",
        )

        with (
            patch.object(screener_common, "get_options_expiration_list", return_value=expirations, create=True),
            patch.object(screener_common, "get_snapshot_quote", side_effect=fake_quote),
        ):
            result = screen_credit_spreads(request, client=object())

        self.assertEqual(seen_expirations, [near_expiration])
        self.assertEqual(result.stats.fetched_rows, 2)


if __name__ == "__main__":
    unittest.main()
