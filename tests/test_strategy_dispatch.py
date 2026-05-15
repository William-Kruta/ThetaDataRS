import datetime as dt
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import polars as pl

from thetadatars import ThetaDataRS
from thetadatars.data.db import get_connection
from thetadatars.options.screeners.call import CallRequest
from thetadatars.options.screeners.cash_secured_put import CashSecuredPutRequest
from thetadatars.options.screeners.covered_call import CoveredCallRequest
from thetadatars.options.screeners.put import LongPutRequest
from thetadatars.options.screeners.protective_put import ProtectivePutRequest
from thetadatars.options.screeners._typed import WarmCacheResult
from thetadatars.options.screeners._typed import (
    CircuitBreakerPolicy,
    RateLimitPolicy,
    RetryPolicy,
    TimeoutPolicy,
)
from thetadatars.options.screeners.credit_spreads import (
    BatchResult,
    BatchStats,
    CreditSpreadRequest,
    ScreenerStats,
    TickerResult,
)
from thetadatars.options.screeners.debit_spreads import DebitSpreadRequest
from thetadatars.options.screeners.iron_butterfly import IronButterflyRequest
from thetadatars.options.screeners.iron_condor import IronCondorRequest
from thetadatars.options.screeners.straddle import StraddleRequest
from thetadatars.options.screeners.strangle import StrangleRequest
import thetadatars.options.screeners as screeners
from thetadatars.options.screeners import strategies
from thetadatars.options.screeners.calendar_spread import CalendarSpreadRequest
from thetadatars.options.screeners.diagonal_spread import DiagonalSpreadRequest


class StrategyDispatchTests(unittest.TestCase):
    def test_available_strategies_marks_first_class_strategy_support(self):
        available = strategies.get_available_strategies()

        self.assertIn("credit_spread", available)
        self.assertIn("cash_secured_put", available)
        self.assertTrue(available["credit_spread"]["typed_request"])
        self.assertTrue(available["cash_secured_put"]["typed_request"])
        self.assertTrue(available["debit_spread"]["typed_request"])
        self.assertTrue(available["debit_spread"]["watchlist"])
        self.assertTrue(available["debit_spread"]["warm_cache"])
        self.assertTrue(available["straddle"]["typed_request"])
        self.assertTrue(available["straddle"]["watchlist"])
        self.assertTrue(available["straddle"]["warm_cache"])
        self.assertTrue(available["strangle"]["typed_request"])
        self.assertTrue(available["strangle"]["watchlist"])
        self.assertTrue(available["strangle"]["warm_cache"])
        self.assertTrue(available["calendar_spread"]["typed_request"])
        self.assertTrue(available["calendar_spread"]["watchlist"])
        self.assertTrue(available["calendar_spread"]["warm_cache"])
        self.assertTrue(available["diagonal_spread"]["typed_request"])
        self.assertTrue(available["diagonal_spread"]["watchlist"])
        self.assertTrue(available["diagonal_spread"]["warm_cache"])
        self.assertTrue(available["iron_condor"]["typed_request"])
        self.assertTrue(available["iron_condor"]["watchlist"])
        self.assertTrue(available["iron_condor"]["warm_cache"])
        self.assertTrue(available["iron_butterfly"]["typed_request"])
        self.assertTrue(available["iron_butterfly"]["watchlist"])
        self.assertTrue(available["iron_butterfly"]["warm_cache"])

    def test_screen_watchlist_dispatches_cash_secured_puts(self):
        expected = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[
                TickerResult(
                    ticker="XYZ",
                    stats=ScreenerStats(
                        ticker="XYZ",
                        fetched_rows=1,
                        filtered_rows=1,
                        candidate_rows=1,
                        returned_rows=1,
                    ),
                )
            ],
            failures=[],
            stats=BatchStats(total=1, succeeded=1, failed=0),
        )
        request = CashSecuredPutRequest(expiration="*", max_dte=45, strike_range=20)

        with patch.object(strategies, "screen_cash_secured_put_watchlist", return_value=expected) as screen:
            result = strategies.screen_watchlist(
                ["XYZ"],
                strategy="cash_secured_put",
                request=request,
                client=object(),
            )

        self.assertIs(result, expected)
        screen.assert_called_once()

    def test_plan_screener_dispatches_by_request_type(self):
        request = CashSecuredPutRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            strike_range=20,
            greeks_source="none",
        )
        conn = object()

        with patch.object(strategies, "plan_cash_secured_puts") as plan:
            strategies.plan_screener(request, conn=conn)

        plan.assert_called_once_with(request, conn=conn)

    def test_plan_screener_cache_only_misses_do_not_report_upstream_calls(self):
        expiration = dt.date(2026, 6, 19)
        far_expiration = dt.date(2026, 7, 17)
        requests = [
            CallRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            LongPutRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            CoveredCallRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            ProtectivePutRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            CashSecuredPutRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            CreditSpreadRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            DebitSpreadRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            StraddleRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            StrangleRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            CalendarSpreadRequest(ticker="XYZ", near_expiration=expiration, far_expiration=far_expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            DiagonalSpreadRequest(ticker="XYZ", near_expiration=expiration, far_expiration=far_expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            IronCondorRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
            IronButterflyRequest(ticker="XYZ", expiration=expiration, strike_range=20, greeks_source="none", cache_policy="cache_only"),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            conn = get_connection(str(Path(tmp) / "strategy-cache-only-plan.duckdb"))
            try:
                for request in requests:
                    with self.subTest(request=type(request).__name__):
                        plan = strategies.plan_screener(request, conn=conn)
                        self.assertGreater(plan.cache_misses, 0)
                        self.assertEqual(plan.upstream_calls, 0)
            finally:
                conn.close()

    def test_screen_watchlist_rejects_mismatched_request_type(self):
        with self.assertRaises(TypeError):
            strategies.screen_watchlist(
                ["XYZ"],
                strategy="cash_secured_put",
                request=CreditSpreadRequest(expiration="*", max_dte=45),
                client=object(),
            )

    def test_protective_put_generic_dispatches(self):
        request = ProtectivePutRequest(expiration="*", max_dte=45, strike_range=20)
        client = object()
        batch = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )
        warmup = WarmCacheResult(
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )

        with patch.object(strategies, "screen_protective_put_watchlist", return_value=batch) as screen:
            result = strategies.screen_watchlist(
                ["XYZ"],
                strategy="protective_put",
                request=request,
                client=client,
            )

        self.assertIs(result, batch)
        screen.assert_called_once()

        with patch.object(strategies, "warm_protective_put_cache", return_value=warmup) as warm:
            result = strategies.warm_cache(
                ["XYZ"],
                strategy="protective_put",
                request=request,
                client=client,
            )

        self.assertIs(result, warmup)
        warm.assert_called_once()

        with patch.object(strategies, "plan_protective_puts") as plan:
            strategies.plan_screener(request, conn=object())

        plan.assert_called_once()

        with patch.object(strategies, "screen_protective_puts") as screen_one:
            strategies.screen_strategy(request, client=client)

        screen_one.assert_called_once()

    def test_call_watchlist_and_warm_cache_dispatch(self):
        request = CallRequest(expiration="*", max_dte=45, strike_range=20)
        client = object()
        batch = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )
        warmup = WarmCacheResult(
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )

        with patch.object(strategies, "screen_call_watchlist", return_value=batch) as screen:
            result = strategies.screen_watchlist(
                ["XYZ"],
                strategy="call",
                request=request,
                client=client,
            )

        self.assertIs(result, batch)
        screen.assert_called_once()

        with patch.object(strategies, "warm_call_cache", return_value=warmup) as warm:
            result = strategies.warm_cache(
                ["XYZ"],
                strategy="call",
                request=request,
                client=client,
            )

        self.assertIs(result, warmup)
        warm.assert_called_once()

    def test_put_watchlist_and_warm_cache_dispatch(self):
        request = LongPutRequest(expiration="*", max_dte=45, strike_range=20)
        client = object()
        batch = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )
        warmup = WarmCacheResult(
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )

        with patch.object(strategies, "screen_long_put_watchlist", return_value=batch) as screen:
            result = strategies.screen_watchlist(
                ["XYZ"],
                strategy="put",
                request=request,
                client=client,
            )

        self.assertIs(result, batch)
        screen.assert_called_once()

        with patch.object(strategies, "warm_long_put_cache", return_value=warmup) as warm:
            result = strategies.warm_cache(
                ["XYZ"],
                strategy="put",
                request=request,
                client=client,
            )

        self.assertIs(result, warmup)
        warm.assert_called_once()

    def test_plan_screener_dispatches_calls(self):
        request = CallRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            strike_range=20,
            greeks_source="none",
        )
        conn = object()

        with patch.object(strategies, "plan_calls") as plan:
            strategies.plan_screener(request, conn=conn)

        plan.assert_called_once_with(request, conn=conn)

    def test_plan_screener_dispatches_puts(self):
        request = LongPutRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            strike_range=20,
            greeks_source="none",
        )
        conn = object()

        with patch.object(strategies, "plan_long_puts") as plan:
            strategies.plan_screener(request, conn=conn)

        plan.assert_called_once_with(request, conn=conn)

    def test_screen_strategy_dispatches_calls(self):
        request = CallRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            strike_range=20,
            greeks_source="none",
        )
        client = object()

        with patch.object(strategies, "screen_calls") as screen_one:
            strategies.screen_strategy(request, client=client)

        screen_one.assert_called_once_with(request, client=client, conn=None)

    def test_debit_spread_generic_dispatches(self):
        request = DebitSpreadRequest(expiration="*", max_dte=45, strike_range=20)
        client = object()
        retry = RetryPolicy(max_attempts=2)
        timeout = TimeoutPolicy(per_ticker_seconds=1.0)
        rate = RateLimitPolicy(min_interval_seconds=0.01)
        circuit = CircuitBreakerPolicy(max_failures=1)
        client_factory = object()
        on_progress = object()
        conn = object()
        batch = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )
        warmup = WarmCacheResult(
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )

        with patch.object(strategies, "screen_debit_spread_watchlist", return_value=batch) as screen:
            result = strategies.screen_watchlist(
                ["XYZ"],
                strategy="debit_spread",
                request=request,
                client=client,
                concurrency=3,
                retry_policy=retry,
                timeout_policy=timeout,
                rate_limit_policy=rate,
                circuit_breaker_policy=circuit,
                client_factory=client_factory,
                on_progress=on_progress,
            )

        self.assertIs(result, batch)
        screen.assert_called_once_with(
            ["XYZ"],
            request,
            client=client,
            concurrency=3,
            retry_policy=retry,
            timeout_policy=timeout,
            rate_limit_policy=rate,
            circuit_breaker_policy=circuit,
            client_factory=client_factory,
            on_progress=on_progress,
        )

        with patch.object(strategies, "warm_debit_spread_cache", return_value=warmup) as warm:
            result = strategies.warm_cache(
                ["XYZ"],
                strategy="debit_spread",
                request=request,
                client=client,
                concurrency=3,
                retry_policy=retry,
                timeout_policy=timeout,
                rate_limit_policy=rate,
                circuit_breaker_policy=circuit,
                client_factory=client_factory,
                on_progress=on_progress,
                conn=conn,
            )

        self.assertIs(result, warmup)
        warm.assert_called_once_with(
            ["XYZ"],
            request,
            client=client,
            concurrency=3,
            retry_policy=retry,
            timeout_policy=timeout,
            rate_limit_policy=rate,
            circuit_breaker_policy=circuit,
            client_factory=client_factory,
            on_progress=on_progress,
            conn=conn,
        )

        with patch.object(strategies, "plan_debit_spreads") as plan:
            strategies.plan_screener(request.for_ticker("XYZ"), conn=object())

        plan.assert_called_once()

        with patch.object(strategies, "screen_debit_spreads") as screen_one:
            strategies.screen_strategy(request.for_ticker("XYZ"), client=client)

        screen_one.assert_called_once()

    def test_straddle_and_strangle_generic_dispatches(self):
        cases = [
            (
                "straddle",
                StraddleRequest(expiration="*", max_dte=45, strike_range=20),
                "screen_straddle_watchlist",
                "warm_straddle_cache",
                "plan_straddles",
                "screen_straddles",
            ),
            (
                "strangle",
                StrangleRequest(expiration="*", max_dte=45, strike_range=20),
                "screen_strangle_watchlist",
                "warm_strangle_cache",
                "plan_strangles",
                "screen_strangles",
            ),
            (
                "calendar_spread",
                CalendarSpreadRequest(near_expiration="2026-06-19", far_expiration="2026-07-17", strike_range=20),
                "screen_calendar_spread_watchlist",
                "warm_calendar_spread_cache",
                "plan_calendar_spreads",
                "screen_calendar_spreads",
            ),
            (
                "diagonal_spread",
                DiagonalSpreadRequest(near_expiration="2026-06-19", far_expiration="2026-07-17", strike_range=20),
                "screen_diagonal_spread_watchlist",
                "warm_diagonal_spread_cache",
                "plan_diagonal_spreads",
                "screen_diagonal_spreads",
            ),
            (
                "iron_condor",
                IronCondorRequest(expiration="*", max_dte=45, strike_range=20),
                "screen_iron_condor_watchlist",
                "warm_iron_condor_cache",
                "plan_iron_condors",
                "screen_iron_condors",
            ),
            (
                "iron_butterfly",
                IronButterflyRequest(expiration="*", max_dte=45, strike_range=20),
                "screen_iron_butterfly_watchlist",
                "warm_iron_butterfly_cache",
                "plan_iron_butterflies",
                "screen_iron_butterflies",
            ),
        ]
        client = object()
        retry = RetryPolicy(max_attempts=2)
        timeout = TimeoutPolicy(per_ticker_seconds=1.0)
        rate = RateLimitPolicy(min_interval_seconds=0.01)
        circuit = CircuitBreakerPolicy(max_failures=1)
        client_factory = object()
        on_progress = object()
        conn = object()
        batch = BatchResult(
            data=pl.DataFrame({"root": ["XYZ"]}),
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )
        warmup = WarmCacheResult(
            successes=[],
            failures=[],
            stats=BatchStats(total=1, succeeded=0, failed=0),
        )

        for strategy, request, watchlist_name, warm_name, plan_name, screen_name in cases:
            with self.subTest(strategy=strategy):
                with patch.object(strategies, watchlist_name, return_value=batch) as screen:
                    result = strategies.screen_watchlist(
                        ["XYZ"],
                        strategy=strategy,
                        request=request,
                        client=client,
                        concurrency=3,
                        retry_policy=retry,
                        timeout_policy=timeout,
                        rate_limit_policy=rate,
                        circuit_breaker_policy=circuit,
                        client_factory=client_factory,
                        on_progress=on_progress,
                    )

                self.assertIs(result, batch)
                screen.assert_called_once_with(
                    ["XYZ"],
                    request,
                    client=client,
                    concurrency=3,
                    retry_policy=retry,
                    timeout_policy=timeout,
                    rate_limit_policy=rate,
                    circuit_breaker_policy=circuit,
                    client_factory=client_factory,
                    on_progress=on_progress,
                )

                with patch.object(strategies, warm_name, return_value=warmup) as warm:
                    result = strategies.warm_cache(
                        ["XYZ"],
                        strategy=strategy,
                        request=request,
                        client=client,
                        concurrency=3,
                        retry_policy=retry,
                        timeout_policy=timeout,
                        rate_limit_policy=rate,
                        circuit_breaker_policy=circuit,
                        client_factory=client_factory,
                        on_progress=on_progress,
                        conn=conn,
                    )

                self.assertIs(result, warmup)
                warm.assert_called_once_with(
                    ["XYZ"],
                    request,
                    client=client,
                    concurrency=3,
                    retry_policy=retry,
                    timeout_policy=timeout,
                    rate_limit_policy=rate,
                    circuit_breaker_policy=circuit,
                    client_factory=client_factory,
                    on_progress=on_progress,
                    conn=conn,
                )

                with patch.object(strategies, plan_name) as plan:
                    strategies.plan_screener(request.for_ticker("XYZ"), conn=conn)

                plan.assert_called_once_with(request.for_ticker("XYZ"), conn=conn)

                with patch.object(strategies, screen_name) as screen_one:
                    strategies.screen_strategy(request.for_ticker("XYZ"), client=client)

                screen_one.assert_called_once_with(request.for_ticker("XYZ"), client=client, conn=None)

    def test_screen_strategy_dispatches_puts(self):
        request = LongPutRequest(
            ticker="XYZ",
            expiration="2026-06-19",
            strike_range=20,
            greeks_source="none",
        )
        client = object()

        with patch.object(strategies, "screen_long_puts") as screen_one:
            strategies.screen_strategy(request, client=client)

        screen_one.assert_called_once_with(request, client=client, conn=None)

    def test_facade_exposes_generic_strategy_helpers(self):
        theta = ThetaDataRS()

        self.assertIn("screen_watchlist", dir(theta))
        self.assertIn("warm_cache", dir(theta))
        self.assertIn("plan_screener", dir(theta))
        self.assertIn("screen_cash_secured_puts", dir(theta))
        self.assertIn("screen_iron_condors", dir(theta))
        self.assertIn("warm_iron_butterfly_cache", dir(theta))

    def test_screeners_package_exports_typed_requests_and_generic_helpers(self):
        self.assertIs(screeners.CreditSpreadRequest, CreditSpreadRequest)
        self.assertIs(screeners.IronCondorRequest, IronCondorRequest)
        self.assertIs(screeners.IronButterflyRequest, IronButterflyRequest)
        self.assertIs(screeners.screen_iron_condors, strategies.screen_iron_condors)
        self.assertIs(screeners.plan_cash_secured_puts, strategies.plan_cash_secured_puts)
        self.assertTrue(callable(screeners.get_best_iron_butterflies))
        self.assertIn("iron_condor", screeners.get_available_strategies())


if __name__ == "__main__":
    unittest.main()
