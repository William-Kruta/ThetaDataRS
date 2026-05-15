import threading
import unittest

from thetadatars.options.screeners import credit_spreads
from thetadatars.options.screeners._typed import (
    CircuitBreakerPolicy,
    RateLimitPolicy,
    _CircuitBreaker,
    _RateLimiter,
)


class TypedScreenerFoundationTests(unittest.TestCase):
    def test_credit_spreads_star_import_exports_plan_cost(self):
        self.assertIn("PlanCost", credit_spreads.__all__)

    def test_rate_limiter_initializes_lock_deterministically(self):
        limiter = _RateLimiter(RateLimitPolicy(min_interval_seconds=0.0))

        self.assertIsInstance(limiter._lock, type(threading.Lock()))

    def test_circuit_breaker_initializes_lock_deterministically(self):
        breaker = _CircuitBreaker(CircuitBreakerPolicy(max_failures=1))

        self.assertIsInstance(breaker._lock, type(threading.Lock()))


if __name__ == "__main__":
    unittest.main()
