# ThetaDataRS

ThetaDataRS is a Python wrapper around the `thetadata` package. It adds a convenience facade, local DuckDB caching, typed option strategy screeners, structured errors, and cache-aware batch helpers for watchlist-scale workflows.

## Installation

This project uses Python 3.12 and `uv`.

```bash
uv sync
```

Create a local `.env` file with your ThetaData credentials:

```bash
EMAIL=you@example.com
PASSWD=your-password
```

Credentials are loaded by `thetadatars.client.create_client()` through `python-dotenv`.

## Quick Start

Use `ThetaDataRS` when you want one object that wraps the existing endpoint functions and automatically supplies the ThetaData client.

```python
from thetadatars.thetadata import ThetaDataRS

theta = ThetaDataRS()

contracts = theta.get_options_contract_list(
    "AAPL",
    "2026-04-24",
)

print(contracts)
```

The first call fetches from ThetaData when local data is missing or stale, writes the result to DuckDB, and returns a Polars DataFrame. Later calls read from the local cache until the endpoint's `stale_threshold` is exceeded.

## Option List Examples

```python
from thetadatars.thetadata import ThetaDataRS

theta = ThetaDataRS()

symbols = theta.get_options_symbols_list()
expirations = theta.get_options_expiration_list("AAPL")
strikes = theta.get_options_strike_list("AAPL", "2026-04-24")
contracts = theta.get_options_contract_list("AAPL", "2026-04-24")
```

## Snapshot Example

```python
from thetadatars.thetadata import ThetaDataRS

theta = ThetaDataRS()

quotes = theta.get_snapshot_quote(
    ticker="AAPL",
    expiration="2026-04-24",
    strike="*",
    right="both",
)

print(quotes.head())
```

Snapshot endpoints default to a shorter cache window than historical endpoints because the data changes during the trading day.

Snapshot quote, first-order Greek, and option-expiration list helpers also support explicit cache policies:

```python
quotes = theta.get_snapshot_quote(
    ticker="AAPL",
    expiration="2026-04-24",
    strike="*",
    right="put",
    strike_range=20,
    cache_policy="prefer_cache",
)
```

Supported cache policies:

- `prefer_cache`: use fresh matching cache data, otherwise fetch from ThetaData and store it.
- `cache_only`: read only from DuckDB and raise `CacheMissError` if no fresh matching cache entry exists.
- `refresh`: fetch from ThetaData even when cache data exists, then update DuckDB.
- `no_cache`: fetch from ThetaData without reading or writing DuckDB.

Cache matching is parameter-aware. For example, a cached `right="put"` request will not satisfy `right="call"`, and a narrow `strike_range=10` fetch will not satisfy a broader unbounded request. Cache fetch metadata is stored in the `cache_fetches` DuckDB table with endpoint, request params, status, row count, and fetch timing.

Wildcard strategy screens use the same cache policy for expiration discovery and snapshot data. `cache_only` will not fetch missing expirations, and `no_cache` will not read or write the expiration cache.

## Batch Snapshot Data

Use cache-aware batch helpers when you need the same snapshot request across a watchlist without running a strategy screen.

```python
from thetadatars.thetadata import ThetaDataRS
from thetadatars.options.batch import RateLimitPolicy, RetryPolicy, TimeoutPolicy

theta = ThetaDataRS()

quotes = theta.get_snapshot_quote_batch(
    ["AAPL", "MSFT", "NVDA"],
    expiration="2026-04-24",
    right="both",
    strike_range=20,
    cache_policy="prefer_cache",
    concurrency=3,
    retry_policy=RetryPolicy(max_attempts=2, backoff_seconds=0.5),
    timeout_policy=TimeoutPolicy(per_ticker_seconds=30),
    rate_limit_policy=RateLimitPolicy(min_interval_seconds=0.25),
)

print(quotes.data)
print(quotes.stats)
print(quotes.failures)
```

The batch helpers delegate to the normal endpoint functions for each ticker, so cache behavior is identical to direct calls:

- `prefer_cache`: use fresh matching cache data per ticker, otherwise fetch and store.
- `cache_only`: return per-ticker `CacheMissError` failures without fetching missing data.
- `refresh`: fetch each ticker and update DuckDB.
- `no_cache`: fetch each ticker without reading or writing DuckDB.

Batch results do not fail the whole request because one ticker fails. They return:

- `data`: one combined Polars DataFrame for successful tickers.
- `successes`: per-ticker row counts and timing.
- `failures`: structured errors with ticker, endpoint, params, retryability, and attempts.
- `stats`: total, succeeded, failed, and total duration.

Warm cache helpers force `cache_policy="refresh"` and are useful before running screeners or UI workflows:

```python
warmup = theta.warm_snapshot_greeks_first_order_cache(
    ["AAPL", "MSFT"],
    expiration="2026-04-24",
    strike_range=20,
)

print(warmup.stats)
```

If you pass a shared DuckDB `conn`, batch helpers run serially to avoid using the same connection from multiple worker threads. Leave `conn=None` when you want concurrent per-ticker endpoint calls.

Currently supported raw batch helpers are:

- `get_snapshot_quote_batch`
- `get_snapshot_greeks_first_order_batch`
- `warm_snapshot_quote_cache`
- `warm_snapshot_greeks_first_order_cache`

## Historical Option Example

```python
from thetadatars.thetadata import ThetaDataRS

theta = ThetaDataRS()

eod = theta.get_option_history_eod(
    ticker="AAPL",
    start_date="2026-04-01",
    end_date="2026-04-24",
    expiration="2026-04-24",
    strike="*",
    right="both",
)

print(eod)
```

## Credit Spread Screening

Credit spread screeners can be called through the legacy DataFrame API or through a request object that returns data, stats, and warnings.

```python
from thetadatars.options.screeners.credit_spreads import (
    CreditSpreadRequest,
    warm_credit_spread_cache,
    screen_credit_spreads,
    screen_credit_spread_watchlist,
    TimeoutPolicy,
    RateLimitPolicy,
    CircuitBreakerPolicy,
)

request = CreditSpreadRequest(
    ticker="AAPL",
    expiration="*",
    right="put",
    max_dte=45,
    strike_range=20,
    top_n=25,
    cache_policy="prefer_cache",
    max_candidates_per_expiration=500,
    max_candidates_total=2_000,
)

result = screen_credit_spreads(request)

print(result.data)
print(result.stats)
print(result.warnings)
```

Typed requests guard expensive broad scans. `expiration="*"` requires `max_dte` unless `allow_full_chain=True`. Greek source can be selected with `greeks_source="auto"`, `"thetadata"`, `"local"`, or `"none"`.

You can inspect the expected cache/upstream behavior before running a credit-spread screen:

```python
from thetadatars.options.screeners.credit_spreads import plan_credit_spreads

plan = plan_credit_spreads(request)

print(plan.expected_endpoint)
print(plan.cache_hits, plan.cache_misses, plan.upstream_calls)
print(plan.cost, plan.local_computation)
```

`ScreenerStats` includes cache fields such as `cache_hits`, `cache_misses`, `upstream_calls`, and `cache_policy` so app code can explain whether a result came from cache or required an upstream request.

Warm cache data for a watchlist without building spread candidates:

```python
warmup = warm_credit_spread_cache(
    ["AAPL", "MSFT"],
    request,
    concurrency=2,
    rate_limit_policy=RateLimitPolicy(min_interval_seconds=0.25),
)

print(warmup.stats)
print(warmup.failures)
```

Screen watchlists with bounded runtime and failure controls:

```python
batch = screen_credit_spread_watchlist(
    ["AAPL", "MSFT"],
    request,
    concurrency=2,
    timeout_policy=TimeoutPolicy(per_ticker_seconds=30),
    rate_limit_policy=RateLimitPolicy(min_interval_seconds=0.25),
    circuit_breaker_policy=CircuitBreakerPolicy(max_failures=3),
)

print(batch.data)
print(batch.failures)
print(batch.stats)
```

Candidate caps limit strategy-builder work after normal filters have selected otherwise eligible spreads. When caps prune candidates, the result includes a `candidate_limit` warning and `ScreenerStats.pruned_candidate_rows`.

## Multi-Strategy Screening

All bundled strategy screeners now have typed request/result APIs:

- `call`
- `put`
- `covered_call`
- `protective_put`
- `cash_secured_put`
- `credit_spread`
- `debit_spread`
- `calendar_spread`
- `diagonal_spread`
- `straddle`
- `strangle`
- `iron_condor`
- `iron_butterfly`

Single-leg and stock-linked strategies use the same request/result flow as spreads:

```python
from thetadatars.options.screeners.cash_secured_put import (
    CashSecuredPutRequest,
    screen_cash_secured_puts,
)

request = CashSecuredPutRequest(
    ticker="AAPL",
    expiration="*",
    max_dte=45,
    strike_range=20,
    min_delta=0.10,
    max_delta=0.35,
    top_n=25,
    greeks_source="none",
)

result = screen_cash_secured_puts(request)

print(result.data)
print(result.stats)
```

The generic strategy dispatcher lets backend applications use one watchlist API for first-class typed strategies:

```python
from thetadatars.options.screeners.cash_secured_put import CashSecuredPutRequest
from thetadatars.options.screeners.strategies import (
    get_available_strategies,
    plan_screener,
    screen_watchlist,
    warm_cache,
)

request = CashSecuredPutRequest(expiration="*", max_dte=45, strike_range=20)

print(get_available_strategies())
print(plan_screener(request))

warmup = warm_cache(["AAPL", "MSFT"], strategy="cash_secured_put", request=request)
batch = screen_watchlist(["AAPL", "MSFT"], strategy="cash_secured_put", request=request)
```

The same dispatcher can run multi-leg strategies:

```python
from thetadatars.options.screeners.iron_condor import IronCondorRequest
from thetadatars.options.screeners.strategies import screen_watchlist

request = IronCondorRequest(
    expiration="*",
    max_dte=45,
    strike_range=30,
    min_credit=0.50,
    max_candidates_per_expiration=500,
    max_candidates_total=2_000,
)

batch = screen_watchlist(["AAPL", "MSFT"], strategy="iron_condor", request=request)
print(batch.data)
print(batch.failures)
```

Legacy `get_best_*` and `find_best_*` DataFrame helpers remain available for compatibility. Typed APIs add planning, cache statistics, warm-cache helpers, watchlist batching, structured failures, and candidate pruning warnings.

## Streaming Example

Streaming helpers return async iterators. `ThetaDataRS` automatically applies its `stream_url`, which defaults to `ws://127.0.0.1:25520/v1/events`.

```python
import asyncio

from thetadatars.thetadata import ThetaDataRS


async def main():
    theta = ThetaDataRS()

    async for message in theta.stream_option_trades(
        "AAPL",
        "2026-04-24",
        200,
        "C",
        max_messages=5,
        timeout=10,
    ):
        print(message)


asyncio.run(main())
```

## Direct Function Usage

You can also import endpoint functions directly if you want explicit control over the client.

```python
from thetadatars.client import create_client
from thetadatars.options.list.contracts import get_options_contract_list

client = create_client()

contracts = get_options_contract_list(
    "AAPL",
    "2026-04-24",
    client=client,
)
```

## Local Data

Cached endpoint data is stored in DuckDB through `thetadatars.data.db`. The default database path is controlled by `THETADATARS_DB`, the user config file, or the platform config directory.

Do not commit local database files, `.env` files, credentials, or generated market data.

## Development

```bash
uv sync
uv run --with pytest pytest
uv build
```

- `uv sync` installs dependencies.
- `uv run --with pytest pytest` runs the offline test suite.
- `uv run python test.py` runs the live smoke-test script and may call ThetaData services.
- `uv build` creates package distributions in `dist/`.
