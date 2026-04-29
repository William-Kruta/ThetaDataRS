EOD_COLS = [
    "date", "root", "expiration", "strike", "right", "created", "last_trade",
    "open", "high", "low", "close", "volume", "count",
    "bid_size", "bid_exchange", "bid", "bid_condition",
    "ask_size", "ask_exchange", "ask", "ask_condition", "fetched_at",
]

OHLCV_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp",
    "open", "high", "low", "close", "volume", "count", "fetched_at",
]

TRADE_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp", "sequence",
    "ext_condition1", "ext_condition2", "ext_condition3", "ext_condition4",
    "condition", "size", "exchange", "price", "fetched_at",
]

QUOTE_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp",
    "bid", "ask", "bid_size", "ask_size", "bid_exchange", "ask_exchange",
    "bid_condition", "ask_condition", "fetched_at",
]

TRADE_QUOTE_COLS = [
    "date", "root", "expiration", "strike", "right", "trade_timestamp",
    "quote_timestamp", "sequence", "ext_condition1", "ext_condition2",
    "ext_condition3", "ext_condition4", "condition", "size", "exchange",
    "price", "bid_size", "bid_exchange", "bid", "bid_condition",
    "ask_size", "ask_exchange", "ask", "ask_condition", "fetched_at",
]

OPEN_INTEREST_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp",
    "open_interest", "fetched_at",
]

GREEKS_BASE_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp",
    "bid", "ask", "delta", "theta", "vega", "rho", "epsilon", "lambda",
    "gamma", "vanna", "charm", "vomma", "veta", "vera", "speed", "zomma",
    "color", "ultima", "d1", "d2", "dual_delta", "dual_gamma",
    "implied_vol", "iv_error", "underlying_timestamp", "underlying_price",
    "fetched_at",
]

GREEKS_EOD_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp",
    "open", "high", "low", "close", "volume", "count",
    "bid_size", "bid_exchange", "bid", "bid_condition",
    "ask_size", "ask_exchange", "ask", "ask_condition",
    "delta", "theta", "vega", "rho", "epsilon", "lambda",
    "gamma", "vanna", "charm", "vomma", "veta", "vera", "speed", "zomma",
    "color", "ultima", "d1", "d2", "dual_delta", "dual_gamma",
    "implied_vol", "iv_error", "underlying_timestamp", "underlying_price",
    "fetched_at",
]

TRADE_GREEKS_COLS = [
    "date", "root", "expiration", "strike", "right", "timestamp", "sequence",
    "ext_condition1", "ext_condition2", "ext_condition3", "ext_condition4",
    "condition", "size", "exchange", "price",
    "delta", "theta", "vega", "rho", "epsilon", "lambda",
    "gamma", "vanna", "charm", "vomma", "veta", "vera", "speed", "zomma",
    "color", "ultima", "d1", "d2", "dual_delta", "dual_gamma",
    "implied_vol", "iv_error", "underlying_timestamp", "underlying_price",
    "fetched_at",
]
