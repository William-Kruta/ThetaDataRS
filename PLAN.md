# ThetaDataRS / thetadatars Improvement Outline

## Purpose

This document outlines changes I would make in `thetadatars` to make the package more flexible, robust, and efficient for screeners, watchlist-scale workflows, and application integration.

The recommendations are based on the current package behavior observed from this project:

- `get_best_credit_spreads(..., expiration="*")` fetches a full first-order greeks chain, or falls back to a full quote chain plus local greek calculation.
- `top_n` is applied after fetching and building all candidates.
- Broad requests can pull hundreds or thousands of contracts per ticker.
- Strategy builders generate many local combinations, and some are combinatorial.
- Errors are mostly raw upstream exceptions.
- The facade dynamically exposes functions through `ThetaDataRS.__getattr__`, which keeps the package compact but makes discoverability, typing, and compatibility harder.

## High-Level Goals

1. Make expensive calls explicit and controllable.
2. Push filtering as early as possible, ideally before upstream fetches.
3. Provide structured errors and diagnostics instead of opaque strings.
4. Add first-class batch/watchlist workflows with rate limiting, retries, progress, and partial failure results.
5. Improve cache semantics so repeated broad queries are fast and concurrent calls do not stampede the same upstream endpoint.
6. Preserve the simple facade API while adding typed, inspectable request objects for serious application use.

## Current Pain Points

### 1. Broad Screeners Fetch Too Much Data

The biggest efficiency issue is that strategy screeners often start by loading the whole option chain for a ticker.

For example, a credit spread request with:

```python
get_best_credit_spreads("RKLB", expiration="*", top_n=50)
```

does not fetch only 50 spread candidates. It fetches a full chain snapshot, then computes all eligible spread candidates locally, then sorts and truncates to `top_n`.

This makes `top_n` a display/result limit, not a workload limit.

Recommended changes:

- Add explicit request planning before fetching.
- Treat `expiration="*"` as a heavy query requiring a strategy-specific default `max_dte`.
- Push `right`, `max_dte`, `strike_range`, `min_time`, and moneyness filters into the chain fetch whenever possible.
- Make broad requests opt-in with clear names such as `expiration="*"` plus `max_dte=None` requiring `allow_full_chain=True`.
- Return query metadata showing how many contracts were fetched, filtered, paired, and returned.

### 2. Local Greek Fallback Is Very Expensive

When first-order greeks are unavailable from the ThetaData subscription, the library falls back to quote snapshots and local American option greek calculations.

That fallback can be very expensive:

- It processes each option row.
- It solves implied volatility with iterative pricing.
- It computes first-order greeks with repeated binomial tree calculations.
- Default local tree steps are high enough to become expensive across thousands of contracts.

Recommended changes:

- Make greek source explicit:
  - `greeks_source="thetadata"`
  - `greeks_source="local"`
  - `greeks_source="auto"`
  - `greeks_source="none"`
- Let screeners that do not strictly require greeks run without calculating them.
- Support lower-cost approximate greeks for screening, with a separate high-accuracy mode.
- Cache locally calculated greeks separately and reuse them.
- Add `local_greeks_steps` presets such as `"fast"`, `"balanced"`, and `"accurate"`.
- Expose metadata indicating when a fallback happened.

### 3. Strategy Candidate Generation Can Be Combinatorial

Vertical spreads are roughly pairwise within expiration/right. Iron condors can become much worse because they combine put spreads with call spreads.

Current pattern:

- Fetch all relevant options.
- Group by expiration.
- Loop across all valid legs.
- Build all candidates.
- Sort and apply `top_n`.

Recommended changes:

- Pre-filter short-leg candidates before building long-leg combinations.
- Add `max_candidates_per_expiration` and `max_candidates_total`.
- Add strategy-specific pruning:
  - delta windows
  - minimum bid
  - maximum ask
  - width constraints
  - moneyness ranges
  - target probability ranges
- Use vectorized Polars joins where possible instead of nested Python loops.
- For complex strategies, build smaller candidate pools first, then combine.

### 4. Cache Semantics Need More Control

The package has a DuckDB cache, which is good, but application workflows need more control.

Current issues:

- Broad `expiration="*"` requests cache large root-level snapshots.
- Concurrent requests for the same ticker can trigger duplicate upstream fetches.
- Callers cannot easily choose cache-only, refresh, stale-while-revalidate, or no-cache behavior.
- Errors do not clearly state whether local cache was used or upstream was called.

Recommended changes:

- Add a `CachePolicy` option:
  - `cache_policy="prefer_cache"`
  - `cache_policy="cache_only"`
  - `cache_policy="refresh"`
  - `cache_policy="stale_while_revalidate"`
  - `cache_policy="no_cache"`
- Add per-key request locks so concurrent calls for the same root/expiration/right do not stampede ThetaData.
- Store fetch metadata:
  - endpoint
  - request params
  - fetched row count
  - fetch duration
  - upstream status
  - stale threshold used
- Add helpers to inspect cache coverage before screening.
- Add `warm_cache()` APIs for watchlists and common screeners.

### 5. Batch and Watchlist Workflows Should Be First-Class

Applications naturally screen multiple tickers. Today, callers must implement batching outside the package.

Recommended changes:

- Add batch APIs:

```python
screen_watchlist(
    tickers=["RKLB", "RDW", "ASTS"],
    strategy="credit_spread",
    request=CreditSpreadRequest(...),
    concurrency=2,
    retry_policy=RetryPolicy(...),
)
```

For raw market data, add endpoint-specific watchlist helpers that reuse the endpoint cache layer:

```python
quotes = get_snapshot_quote_batch(
    tickers=["RKLB", "RDW", "ASTS"],
    expiration="2026-06-19",
    right="both",
    strike_range=20,
    cache_policy="prefer_cache",
    concurrency=2,
)
```

- Return a structured result:

```python
BatchResult(
    rows=DataFrame,
    successes=[TickerResult(...)]
    failures=[TickerFailure(...)]
    stats=BatchStats(...)
)
```

- Include per-ticker timing, row counts, cache hits, and error details.
- Support progress callbacks.
- Support cancellation.
- Default to conservative concurrency.
- Add backoff/retry for transient network failures.
- Avoid failing a full batch because one ticker has no data.

### 6. Error Handling Should Be Structured

The current API often raises raw exception strings such as:

```text
No data found for: option_snapshot_quote(MDA,*,*,both,None,None,None)
```

That is useful for debugging but weak for app-level UX.

Recommended changes:

- Define package exceptions:
  - `ThetaDataError`
  - `NoDataError`
  - `SubscriptionError`
  - `RateLimitError`
  - `TimeoutError`
  - `TransientNetworkError`
  - `InvalidRequestError`
  - `CacheError`
- Include structured fields:
  - `ticker`
  - `endpoint`
  - `params`
  - `status_code`
  - `retryable`
  - `user_message`
  - `debug_message`
- Do not require callers to parse strings.
- Preserve original exceptions through `__cause__`.

### 7. The Facade Needs Better Typing and Discoverability

`ThetaDataRS` dynamically exposes functions with `__getattr__`. This keeps the facade small, but it makes static analysis and IDE discovery weaker.

Recommended changes:

- Add generated or explicit typed facade methods where they materially improve discoverability.
- Add explicit namespaces:

```python
theta.options.snapshot.quote(...)
theta.options.screeners.credit_spreads(...)
theta.streaming.options.subscribe(...)
```

- Keep flat methods for backward compatibility.
- Make `dir(theta)` and docs stable without importing every module eagerly.
- Expose request/response models for screeners.

### 8. Strategy APIs Should Use Request Objects

Most screeners currently expose many keyword arguments. This is flexible but hard to validate, serialize, reuse, and document.

Recommended changes:

- Introduce dataclass or Pydantic-style request objects:

```python
CreditSpreadRequest(
    ticker="RKLB",
    expiration="*",
    right="put",
    min_dte=7,
    max_dte=45,
    strike_range=20,
    top_n=50,
    rank_by="annualized_return_on_risk",
)
```

- Keep existing keyword functions as wrappers.
- Validate requests before any network call.
- Provide `.estimate_cost()` or `.plan()` before execution.
- Make request objects serializable for app/front-end integration.

### 9. Query Planning Should Be Built In

The package should be able to explain what it is about to do before it does it.

Recommended API:

```python
plan = theta.plan_screener(CreditSpreadRequest(...))
print(plan.estimated_contracts)
print(plan.upstream_calls)
print(plan.cache_hits)
print(plan.warnings)
```

Plan output should include:

- whether upstream fetch is needed
- cache coverage
- expected endpoint
- broad-query warnings
- estimated contract count if known
- estimated local computation class: low, medium, high
- suggested filters to reduce cost

This would make app behavior much more predictable.

### 10. Observability Should Be First-Class

For a data library, performance and source visibility matter.

Recommended changes:

- Add structured logging around:
  - cache hit/miss
  - upstream call start/end
  - row counts
  - local greek fallback
  - strategy candidate counts
  - pruning counts
- Add optional callbacks:

```python
on_progress(Event(...))
on_warning(Warning(...))
```

- Return stats with every screener result:

```python
ScreenerResult(
    data=df,
    stats=ScreenerStats(
        fetched_rows=1766,
        candidate_rows=12420,
        returned_rows=50,
        cache_hit=False,
        duration_seconds=92.4,
    ),
)
```

### 11. Timeouts, Retries, and Rate Limits Should Be Configurable

Long-running option chain requests are expected. The package should distinguish slow-but-valid from failed.

Recommended changes:

- Add timeout options per client and per request.
- Add retry policies for retryable errors.
- Add rate limiting around upstream requests.
- Add circuit-breaker behavior for repeated failures.
- Add a clear timeout error that tells callers whether work may still be ongoing upstream.

### 12. Package Compatibility Should Be Hardened

This project currently needs a compatibility shim because `thetadatars 0.2.0` imports `thetadata.client.Client`, while installed `thetadata` exposes `ThetaClient`.

Recommended changes:

- Pin compatible `thetadata` versions.
- Or support both names:

```python
try:
    from thetadata.client import Client
except ImportError:
    from thetadata.client import ThetaClient as Client
```

- Add import smoke tests in CI.
- Add dependency upper bounds when upstream APIs are unstable.

## Recommended New Public API Shape

### Basic Usage

```python
theta = ThetaDataRS(email=email, passwd=password)

result = theta.screeners.credit_spreads(
    CreditSpreadRequest(
        ticker="RKLB",
        expiration="*",
        right="put",
        min_dte=7,
        max_dte=45,
        strike_range=20,
        top_n=50,
    )
)

df = result.data
print(result.stats)
print(result.warnings)
```

### Batch Usage

```python
batch = theta.screeners.batch(
    tickers=["RKLB", "RDW", "ASTS", "RIVN", "MDA"],
    strategy="credit_spread",
    request=CreditSpreadRequest(
        expiration="*",
        max_dte=45,
        strike_range=20,
        top_n=25,
    ),
    concurrency=2,
    retry_policy=RetryPolicy(max_attempts=2),
)

print(batch.data)
print(batch.failures)
print(batch.stats)
```

### Planning Usage

```python
plan = theta.screeners.plan(
    CreditSpreadRequest(
        ticker="ASTS",
        expiration="*",
        max_dte=None,
        top_n=50,
    )
)

if plan.cost == "high":
    print(plan.warnings)
```

## Implementation Roadmap

### Phase 1: Safety and Compatibility

- Fix `Client` / `ThetaClient` import compatibility.
- Add structured exception classes.
- Add timeout and retry configuration.
- Add tests for no-data, subscription, timeout, and transient network failures.

### Phase 2: Request Models and Result Metadata

- Add request dataclasses for screeners.
- Add result wrappers with `data`, `stats`, and `warnings`.
- Keep backward-compatible keyword wrapper functions.
- Add validation before upstream calls.

### Phase 3: Query Planning and Cost Visibility

- Add `plan_screener()` APIs.
- Estimate broad query cost from cached/list endpoints where possible.
- Emit warnings for `expiration="*"` without `max_dte` or `strike_range`.
- Add cache coverage inspection.

### Phase 4: Efficient Chain Acquisition

- Push filters into upstream fetches.
- Add default heavy-query guards.
- Add cache policies.
- Add per-key cache locks.
- Add cache warmup helpers.

### Phase 5: Strategy Builder Optimization

- Refactor nested loops into reusable candidate builders.
- Add pre-pruning before pair/combination generation.
- Use Polars joins for pair generation where practical.
- Add candidate count limits and explain when limits prune results.

### Phase 6: Batch/Watchlist API

- Add multi-ticker screeners.
- Add raw snapshot batch helpers for quote and first-order Greeks.
- Add bounded concurrency.
- Add per-ticker partial failure reporting.
- Add progress callbacks.
- Add retry/backoff.
- Extend raw batch coverage to other endpoints after cache-policy behavior is standardized.

### Phase 7: Typed Facade and Namespaces

- Add explicit namespaces for options, screeners, snapshots, history, and streaming.
- Add generated or explicit typed facade methods only if the dynamic facade becomes a practical maintenance or discoverability problem.
- Keep flat `ThetaDataRS` method names as compatibility aliases.

## Updated Remaining Priorities

After the structured errors, credit-spread request/result API, cache policies, cache metadata, and credit-spread watchlist support, the remaining work should be tackled in this order:

1. Harden package compatibility.
   - Add a `Client` / `ThetaClient` compatibility shim.
   - Add import smoke tests.
   - Add dependency upper bounds if upstream `thetadata` APIs remain unstable.

2. Add query planning and cost visibility.
   - Add `plan_screener()` and request-level `.plan()` or `.estimate_cost()` helpers.
   - Report expected upstream calls, broad-query warnings, estimated local computation cost, and suggested filters.

3. Add cache coverage inspection APIs.
   - Let callers ask whether a request can be satisfied from cache before running a screener.
   - Reuse `cache_fetches` metadata and parameter-aware coverage matching.

4. Add cache source metadata to screener stats.
   - Include cache hit, miss, refresh, no-cache, and mixed-source details in `ScreenerStats`.
   - Surface whether upstream was called.

5. Add cache warmup helpers.
   - Support warming quotes, first-order Greeks, expirations, and common watchlist screener inputs.
   - Keep concurrency conservative by default.

6. Add request-level and client-level timeout configuration for REST calls.
   - Distinguish slow valid work from failed work.
   - Map timeouts to structured `TimeoutError` values.

7. Add rate limiting and circuit-breaker behavior.
   - Bound watchlist pressure against ThetaData.
   - Stop repeated failures from hammering the same endpoint.

8. Optimize strategy builders.
   - Add `max_candidates_per_expiration` and `max_candidates_total`.
   - Pre-prune short-leg candidates.
   - Use Polars joins where practical.
   - Return warnings or stats when candidate caps prune results.

9. Extend general batch API coverage.
   - Strategy `screen_watchlist(strategy=..., request=...)` and cache-aware snapshot quote/first-order Greek batch helpers are in place.
   - Add raw list/history/remaining snapshot endpoint batch wrappers as cache-policy support expands.
   - Keep per-ticker partial failures, stats, retries, and progress callbacks.

10. Extend cache policies across more endpoints.
    - Apply `CachePolicy` consistently to list, history, and remaining snapshot endpoints.
    - Preserve endpoint-specific stale defaults.

11. Add `stale_while_revalidate`.
    - Return stale cached data immediately when acceptable.
    - Refresh in the background or through an explicit revalidation hook.

12. Cache locally calculated Greeks separately.
    - Key local Greek cache by quote data plus model inputs such as rate, dividend, stock price, valuation date, and step preset.
    - Avoid recalculating local Greeks across repeated broad screens.

13. Add typed request/result objects for additional strategies.
    - Start with the screeners used by applications most often.
    - Keep existing keyword functions as compatibility wrappers.

14. Add explicit facade namespaces.
    - Support APIs such as `theta.options.snapshot.quote(...)` and `theta.options.screeners.credit_spreads(...)`.
    - Keep existing flat dynamic methods for backward compatibility.

## Highest-Impact Changes First

If I were improving the package incrementally, I would start with these:

1. Add structured errors and preserve ticker/endpoint/params.
2. Add request/result objects with stats and warnings.
3. Add heavy-query warnings for `expiration="*"` without `max_dte` or `strike_range`.
4. Add batch screeners with partial failures and bounded concurrency.
5. Add cache policies and per-key fetch locks.
6. Optimize screeners to prune candidates before combinatorial pairing.

These changes would immediately improve app integration and make watchlist screening safer without requiring a full rewrite.

## Notes for This App

For the current web app, the package changes that would help most are:

- a batch screener API returning per-ticker failures
- structured no-data vs retryable/transient errors
- progress and stats for long-running calls
- cache warmup for watchlists
- strategy request planning and warnings
- efficient defaults for broad watchlist scans, especially `max_dte` and `strike_range`

Those package-level changes would let the frontend show clear messages like:

```text
MDA: no option snapshot quote data
RKLB: fetched 1,766 contracts, screened 50 results in 94s
ASTS: transient fetch timeout, retry available
```

instead of a generic summary such as:

```text
5 tickers failed: RKLB, RDW, ASTS, RIVN, MDA
```
