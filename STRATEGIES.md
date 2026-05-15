# Option Strategy Screeners

This project includes typed screeners for long calls, long puts, covered calls, protective puts, cash-secured puts, credit spreads, debit spreads, calendar spreads, diagonal spreads, straddles, strangles, iron condors, and iron butterflies. The screeners try ThetaData first-order Greeks first. If the account only has VALUE-tier access and the Greeks endpoint is unavailable, the screeners fall back to snapshot quotes plus locally calculated American-option Greeks.

Every strategy keeps its legacy `get_best_*` / `find_best_*` DataFrame helper and also exposes a typed request/result API with `plan_*`, `screen_*`, `warm_*_cache`, and watchlist helpers. Generic dispatch is available through `thetadatars.options.screeners.strategies`.

These screeners are ranking tools, not trade recommendations. They use quotes and model-derived metrics, so review liquidity, assignment risk, earnings, dividends, and position sizing before trading.

## Long Calls

Function:

```python
get_best_calls(ticker, expiration, ...)
```

A long call buys upside exposure. The maximum loss is the premium paid, and the trade benefits from the underlying moving above the breakeven price before expiration.

The screener assumes entry at the ask:

```python
premium = ask
max_loss = premium * 100
breakeven = strike + premium
```

Default ranking:

```python
rank_by="delta_per_dollar"
```

Use it when you want defined-risk bullish exposure and prefer paying premium instead of buying shares. For example, if you expect a stock to rally after a catalyst but want to cap downside risk to the option premium, screen calls with a maximum premium and a breakeven move you are willing to underwrite.

Example:

```python
calls = get_best_calls(
    "SPY",
    "2026-06-19",
    stock_price=500,
    max_premium=10,
    max_breakeven_move_percent=0.08,
)
```

## Long Puts

Functions:

```python
get_best_puts(ticker, expiration, side="long", ...)
get_best_long_puts(ticker, expiration, ...)
```

A long put buys downside exposure. The maximum loss is the premium paid, and the trade benefits from the underlying moving below the breakeven price before expiration.

The screener assumes entry at the ask:

```python
premium = ask
max_loss = premium * 100
breakeven = strike - premium
```

Default ranking:

```python
rank_by="delta_per_dollar"
```

Use it when you want defined-risk bearish exposure or portfolio protection. For example, if you own shares and want temporary downside protection through an earnings event, screen puts by premium, delta, and breakeven move.

Example:

```python
puts = get_best_puts(
    "SPY",
    "2026-06-19",
    side="long",
    stock_price=500,
    max_premium=12,
    min_delta=0.20,
    max_delta=0.45,
)
```

## Cash-Secured Puts

Functions:

```python
get_best_cash_secured_puts(ticker, expiration, ...)
get_best_puts(ticker, expiration, side="cash_secured", ...)
```

A cash-secured put sells a put while reserving enough cash to buy the shares if assigned. The trade collects premium and is usually used when you are willing to own the stock at the breakeven price.

The screener assumes entry at the bid:

```python
premium = bid
cash_collateral = strike * 100
net_cash_at_risk = (strike - premium) * 100
breakeven = strike - premium
```

Default ranking:

```python
rank_by="annualized_risk_adjusted_return"
```

The risk-adjusted return uses short-put delta as a rough probability proxy:

```python
probability_otm = 1 - abs(delta)
risk_adjusted_return = return_on_risk * probability_otm
```

Use it when you are neutral-to-bullish and would be comfortable buying the shares below the current market price. For example, if you want to acquire a stock only on a pullback, screen cash-secured puts for high risk-adjusted yield, sufficient probability of expiring out-of-the-money, and a breakeven price you would actually accept.

Example:

```python
csp = get_best_puts(
    "AAPL",
    "2026-06-19",
    side="cash_secured",
    stock_price=190,
    min_probability_otm=0.70,
    min_discount_to_underlying=0.05,
)
```

## Credit Spreads

Function:

```python
get_best_credit_spreads(ticker, expiration, ...)
```

A credit spread sells a higher-premium option and buys a farther out-of-the-money option as a hedge. This creates a defined-risk short-premium position.

The screener supports both:

```python
bull_put_credit   # sell put, buy lower-strike put
bear_call_credit  # sell call, buy higher-strike call
```

The screener assumes entry at conservative bid/ask prices:

```python
credit = short_bid - long_ask
max_loss = width - credit
return_on_risk = credit / max_loss
```

Default ranking:

```python
rank_by="annualized_return_on_risk"
```

Risk-adjusted ranking is also available:

```python
rank_by="annualized_risk_adjusted_return"
```

Use a bull put credit spread when you are neutral-to-bullish but want less cash at risk than a cash-secured put. Use a bear call credit spread when you are neutral-to-bearish and want defined-risk short call exposure. For example, if you think an index will stay above a support level, screen put credit spreads below that level with a maximum width and delta range.

Example:

```python
spreads = get_best_credit_spreads(
    "SPY",
    "2026-06-19",
    right="put",
    stock_price=500,
    max_width=5,
    min_short_delta=0.15,
    max_short_delta=0.35,
    rank_by="annualized_risk_adjusted_return",
)
```

## Choosing Between Them

Use long calls when you want defined-risk upside exposure and are willing to pay premium.

Use long puts when you want defined-risk downside exposure or protection.

Use cash-secured puts when you are willing to own shares at a lower effective price and want to collect premium while waiting.

Use credit spreads when you want short-premium income with defined risk and less collateral than a cash-secured or naked short option.

Use debit spreads when you want defined-risk directional exposure with less premium outlay than buying a single long option.

Use covered calls when you own shares and want to collect call premium while accepting capped upside.

Use protective puts when you own shares and want defined downside protection.

Use straddles or strangles when you want long-volatility exposure around a move and are willing to pay two option premiums.

Use calendar or diagonal spreads when you want term-structure exposure across expirations.

Use iron condors or iron butterflies when you want defined-risk range-bound short-premium exposure with candidate caps to control combinatorial search cost.

## Notes On Ranking

The default ranking differs by strategy because the trade objectives differ:

```python
get_best_calls                 -> delta_per_dollar
get_best_puts(side="long")     -> delta_per_dollar
get_best_cash_secured_puts     -> annualized_risk_adjusted_return
get_best_credit_spreads        -> annualized_return_on_risk
```

Other strategy modules document their supported `rank_by` values on their typed request objects and legacy helper signatures.

Delta-based probability fields are approximations, not true probabilities. They are useful for sorting and filtering, but they should not be treated as precise forecasts.
