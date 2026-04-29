# ThetaDataRS

ThetaDataRS is a Python wrapper around the `thetadata` package. It adds a small convenience facade and local DuckDB caching for option list, history, and snapshot endpoints.

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

Cached endpoint data is stored in DuckDB through `thetadatars.data.db`. Do not commit local database files, `.env` files, credentials, or generated market data.

## Development

```bash
uv sync
uv run python test.py
uv run python -m py_compile thetadatars/thetadata.py
uv build
```

- `uv sync` installs dependencies.
- `uv run python test.py` runs the current smoke-test script.
- `uv run python -m py_compile ...` checks syntax for a module.
- `uv build` creates package distributions in `dist/`.
