import datetime as dt
from pathlib import Path
import tempfile
import unittest

from thetadatars import ThetaDataRS
from thetadatars.data.cache import record_cache_fetch
from thetadatars.data.db import get_connection
from thetadatars.options.screeners.credit_spreads import (
    CreditSpreadRequest,
    ScreenerPlan,
    plan_credit_spreads,
)


class CreditSpreadPlanningTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = str(Path(self.tmp.name) / "planning-test.duckdb")
        self.conn = get_connection(self.db_path)
        self.expiration = dt.date(2026, 6, 19)

    def tearDown(self):
        self.conn.close()
        self.tmp.cleanup()

    def test_plan_reports_cache_hit_for_quote_based_request(self):
        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration=self.expiration,
            right="put",
            strike_range=10,
            greeks_source="none",
        )
        record_cache_fetch(
            conn=self.conn,
            endpoint="option_snapshot_quote",
            root="XYZ",
            params={
                "expiration": self.expiration,
                "strike": "*",
                "right": "put",
                "max_dte": None,
                "strike_range": 10,
                "min_time": None,
            },
            cache_policy="prefer_cache",
            status="success",
            row_count=2,
            fetched_at=dt.datetime.now(),
            duration_seconds=0.01,
        )

        plan = plan_credit_spreads(request, conn=self.conn)

        self.assertIsInstance(plan, ScreenerPlan)
        self.assertEqual(plan.ticker, "XYZ")
        self.assertEqual(plan.strategy, "credit_spread")
        self.assertEqual(plan.expected_endpoint, "option_snapshot_quote")
        self.assertEqual(plan.cache_hits, 1)
        self.assertEqual(plan.cache_misses, 0)
        self.assertEqual(plan.upstream_calls, 0)
        self.assertEqual(plan.cost, "low")

    def test_plan_reports_upstream_call_for_cache_miss(self):
        request = CreditSpreadRequest(
            ticker="XYZ",
            expiration=self.expiration,
            right="put",
            strike_range=10,
            greeks_source="thetadata",
        )

        plan = plan_credit_spreads(request, conn=self.conn)

        self.assertEqual(plan.expected_endpoint, "option_snapshot_greeks_first_order")
        self.assertEqual(plan.cache_hits, 0)
        self.assertEqual(plan.cache_misses, 1)
        self.assertEqual(plan.upstream_calls, 1)

    def test_facade_exposes_planning_helper(self):
        theta = ThetaDataRS()

        self.assertIn("plan_credit_spreads", dir(theta))


if __name__ == "__main__":
    unittest.main()
